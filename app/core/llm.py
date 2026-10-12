"""Chat model factory, following each provider's current docs.

`MODEL` / `CLASSIFIER_MODEL` are "provider:model" strings, split at the *first* colon so OpenRouter
ids like `nvidia/some-model:free` survive.

* ``openrouter:`` -> official `ChatOpenRouter`. Sets `provider.require_parameters` so a request that
  needs tool calling is only routed to providers that support it (otherwise OpenRouter may send it
  to one that silently ignores `tools`), an optional ordered `models` fallback list for when a free
  endpoint is rate-limited or down, and app attribution.
* ``groq:`` -> official `ChatGroq`, `reasoning_effort="low"` for gpt-oss models (this is only used
  for tiny classification calls, so reasoning tokens are wasted latency and TPM).
* ``google_genai:`` -> `ChatGoogleGenerativeAI` (Gemini via a Google AI Studio key) with a limiter.
* anything else -> LangChain `init_chat_model` (anthropic:, openai:, ...).

`resolve_models` drops models whose provider key is not set, so only one LLM key is required;
`build_graph` puts the remaining fallbacks in a `ModelPool` so a rate-limited or exhausted model
rolls over to the next provider.

Free tiers are rate-limited per minute (OpenRouter free models: 20 req/min; Groq gpt-oss: 30
req/min), so each model gets a client-side token-bucket limiter. That avoids most 429s instead of
reacting to them; the retry middleware in the agents handles whatever still gets through.
"""

import re
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from app.core.config import Settings

KEYED_PROVIDERS = {
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
}


class ConfigError(RuntimeError):
    pass


def is_auth_error(exc: BaseException) -> bool:
    """401/403 from any provider SDK: a retry cannot fix a missing or rejected key."""
    if getattr(exc, "status_code", None) in (401, 403):
        return True
    name = type(exc).__name__
    return "Unauthorized" in name or "Authentication" in name or "PermissionDenied" in name


AUTH_HINT = (
    " (the provider rejected the API key: check the key for that provider in .env or your "
    "environment, then restart the server, because settings are read once at startup)"
)


# Known key prefixes per provider. OpenRouter returns "Missing Authentication header" for any
# bearer value that is not key-shaped, so a prefix mismatch is reported before any request.
# Google issues keys in more than one format (AIza..., AQ....), so it has no prefix check.
KEY_PREFIXES = {"openrouter": "sk-or-", "groq": "gsk_"}


def key_problem(provider: str, key: str | None) -> str | None:
    """Why this key is obviously not valid for the provider, or None if it looks plausible.

    Only the first 4 characters are ever quoted back.
    """
    if not key:
        return None
    if key != key.strip() or key[0] in "\"'`" or key[-1] in "\"'`":
        return "the value contains quote marks or spaces; use KEY=value with nothing around it"
    prefix = KEY_PREFIXES.get(provider)
    if prefix and not key.startswith(prefix):
        return (
            f"{provider} keys start with '{prefix}' but this one starts with '{key[:4]}...'. "
            "Is it another provider's key, or a placeholder?"
        )
    return None


# A spent daily quota (OpenRouter "free-models-per-day", Groq "requests/tokens per day (RPD/TPD)")
# or a request bigger than a per-minute token allowance (Groq 413 "Request too large") cannot
# succeed on retry. Retrying only burns minutes of backoff; the fallback chain should take over.
_DAILY_QUOTA = re.compile(r"per[- ]day|\bRPD\b|\bTPD\b", re.IGNORECASE)


def is_quota_exhausted(exc: BaseException) -> bool:
    """A daily request or token quota is spent; it will not lift for hours."""
    return bool(_DAILY_QUOTA.search(str(exc)))


def is_request_too_large(exc: BaseException) -> bool:
    """The request exceeds the model's per-request or per-minute token allowance."""
    return getattr(exc, "status_code", None) == 413 or "request too large" in str(exc).lower()


def should_retry_model(exc: BaseException) -> bool:
    """Retry transient failures only; hand the rest straight to the fallback chain."""
    return not (is_auth_error(exc) or is_quota_exhausted(exc) or is_request_too_large(exc))


LIMIT_HINT = (
    " (the agent hit its per-run spend cap without finishing: a model that keeps calling tools "
    "instead of answering. Retry, switch MODEL, or raise MAX_AGENT_MODEL_CALLS / MAX_SEARCHES)"
)


QUOTA_HINT = (
    " (a provider's daily quota is used up. Add models from another provider to "
    "MODEL_FALLBACKS, or wait for the reset)"
)
TOO_LARGE_HINT = (
    " (the request is larger than the model's token allowance. Use a model with a larger "
    "allowance, or lower SEARCH_RESULT_CHARS / SEARCH_NUM_RESULTS)"
)


def is_limit_error(exc: BaseException) -> bool:
    return "CallLimitExceeded" in type(exc).__name__


def hint_for(exc: BaseException) -> str:
    """Plain-English next step for failures the user can act on, else empty."""
    if is_auth_error(exc):
        return AUTH_HINT
    if is_limit_error(exc):
        return LIMIT_HINT
    if is_quota_exhausted(exc):
        return QUOTA_HINT
    if is_request_too_large(exc):
        return TOO_LARGE_HINT
    return ""


def describe_keys(settings: Settings) -> list[str]:
    """One masked line per provider the configured models use. Never prints key material."""
    specs = [settings.model, *parse_specs(settings.model_fallbacks), settings.classifier_model]
    specs = [s for s in specs if s]
    lines = []
    for provider in dict.fromkeys(_provider(s) for s in specs):
        env = KEYED_PROVIDERS.get(provider)
        if not env:
            continue
        key = api_key_for(settings, provider)
        if not key:
            unused = ", ".join(dict.fromkeys(s for s in specs if _provider(s) == provider))
            lines.append(f"{env}: not set, skipping {unused}")
        else:
            line = f"{env}: set ({len(key)} chars, starts '{key[:5]}...')"
            if problem := key_problem(provider, key):
                line += f"  PROBLEM: {problem}"
            lines.append(line)
    try:
        chain = resolve_models(settings)
    except ConfigError as exc:
        lines.append(f"PROBLEM: {exc}")
    else:
        if chain.primary != settings.model:
            lines.append(f"Main model: {chain.primary} (MODEL has no key)")
        if settings.classifier_model and chain.classifier != settings.classifier_model:
            lines.append(f"Classifier: {chain.classifier} (CLASSIFIER_MODEL has no key)")
    return lines


def api_key_for(settings: Settings, provider: str) -> str | None:
    """The configured key for a keyed provider (setting name = env var name, lowercased)."""
    return getattr(settings, KEYED_PROVIDERS[provider].lower())


def has_key(settings: Settings, spec: str) -> bool:
    """False only for a keyed provider whose key is not set. Other providers read their own env."""
    provider = _provider(spec)
    return provider not in KEYED_PROVIDERS or bool(api_key_for(settings, provider))


@dataclass(frozen=True)
class ModelChain:
    primary: str
    fallbacks: list[str]
    classifier: str


def resolve_models(settings: Settings) -> ModelChain:
    """The models to use, given which keys are set.

    MODEL, MODEL_FALLBACKS and CLASSIFIER_MODEL may name providers the user has no key for, so one
    .env works with whichever keys are filled in. Models without a key are skipped; if MODEL is
    one of them, the first usable fallback becomes the main model, and a keyless CLASSIFIER_MODEL
    falls back to the main model. Only "no usable model at all" is an error.
    """
    chain = list(dict.fromkeys([settings.model, *parse_specs(settings.model_fallbacks)]))
    usable = [s for s in chain if has_key(settings, s)]
    if not usable:
        envs = sorted({KEYED_PROVIDERS[_provider(s)] for s in chain})
        raise ConfigError(
            f"no API key is set for any configured model. Set {' or '.join(envs)} "
            "(or change MODEL / MODEL_FALLBACKS to a provider you have a key for)"
        )
    classifier = settings.classifier_model
    if not classifier or not has_key(settings, classifier):
        classifier = usable[0]
    return ModelChain(primary=usable[0], fallbacks=usable[1:], classifier=classifier)


def parse_specs(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def _provider(spec: str) -> str:
    return spec.partition(":")[0]


def forces_tool_calling(spec: str) -> bool:
    """Free routers vary in JSON-schema support; tool calling is the common denominator."""
    return _provider(spec) in KEYED_PROVIDERS


def classifier_structured_kwargs(spec: str) -> dict:
    """How to ask this provider for a structured answer from the small classifier call."""
    provider = _provider(spec)
    if provider == "groq":
        # gpt-oss on Groq supports constrained decoding: the output cannot violate the schema.
        return {"method": "json_schema", "strict": True}
    if provider == "openrouter":
        return {"method": "function_calling"}
    return {}


def _limiter(requests_per_minute: int) -> InMemoryRateLimiter:
    return InMemoryRateLimiter(
        requests_per_second=requests_per_minute / 60,
        check_every_n_seconds=0.1,
        max_bucket_size=2,  # no big bursts: the provider limit is per minute
    )


def make_chat_model(spec: str, settings: Settings) -> BaseChatModel:
    provider, _, name = spec.partition(":")
    if provider in KEYED_PROVIDERS:
        if not name:
            raise ConfigError(f"'{spec}' is missing a model name, e.g. {provider}:<model-id>")
        if not api_key_for(settings, provider):
            raise ConfigError(f"{KEYED_PROVIDERS[provider]} is not set (required by '{spec}')")

    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        fallbacks = parse_specs(settings.openrouter_fallback_models)
        return ChatOpenRouter(
            model=name,
            openrouter_api_key=settings.openrouter_api_key,
            openrouter_provider={"require_parameters": True},
            model_kwargs={"models": fallbacks} if fallbacks else {},
            app_title="AI Travel Planner",
            temperature=0.2,
            request_timeout=120_000,  # milliseconds
            max_retries=1,  # the agents' retry middleware does the real backoff
            rate_limiter=_limiter(settings.openrouter_rpm),
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        extra = {"reasoning_effort": settings.groq_reasoning_effort} if "gpt-oss" in name else {}
        return ChatGroq(
            model=name,
            groq_api_key=settings.groq_api_key,
            temperature=0,
            request_timeout=60,
            max_retries=1,
            rate_limiter=_limiter(settings.groq_rpm),
            **extra,
        )

    if provider == "google_genai":
        return init_chat_model(
            spec,
            google_api_key=settings.google_api_key,
            max_retries=1,
            rate_limiter=_limiter(settings.google_rpm),
        )

    return init_chat_model(spec)
