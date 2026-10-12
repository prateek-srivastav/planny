import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.envdoc import reconcile_shell_overrides

log = logging.getLogger(__name__)

# The .env next to the code, wherever the server is started from (a relative ".env" would
# silently depend on the current directory).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    # LLMs: "provider:model". Providers: openrouter, groq, or anything langchain's
    # init_chat_model knows (anthropic, openai, ...). See app/core/llm.py.
    model: str = "anthropic:claude-sonnet-5-5"
    # Optional: model for the one-shot "research or planner?" routing call. Defaults to MODEL.
    classifier_model: str | None = None
    openrouter_api_key: str | None = None
    groq_api_key: str | None = None
    google_api_key: str | None = None  # Google AI Studio (Gemini) free tier
    # Comma-separated "provider:model" specs tried in order when MODEL fails (rate limit, daily
    # cap, outage) after its own retries. This is how load rotates across providers.
    model_fallbacks: str = ""
    # A model that fails (after its retries) sits out this long before it is tried again, so a
    # spent daily cap does not slow every call. See app/core/model_pool.py.
    model_cooldown_seconds: int = 300
    # Longer bench for a spent daily quota, which will not lift within minutes.
    model_quota_cooldown_seconds: int = 3600
    # Per-run spend caps for each agent (see app/graph/agents/limits.py). Searches are capped
    # softly (the agent is told to finish); model calls are a hard stop that fails the plan.
    max_searches: int = 6
    # Size of each search call's output, which stays in the agent's context for later calls.
    # Defaults keep a full research run inside small per-minute token allowances (Groq: 8K TPM).
    search_num_results: int = 4
    search_result_chars: int = 600
    max_agent_model_calls: int = 16
    # Comma-separated OpenRouter model ids tried in order if MODEL's endpoints are rate-limited
    # or down (OpenRouter's `models` fallback list).
    openrouter_fallback_models: str = ""
    # Client-side request caps, just under the free-tier limits (OpenRouter :free = 20/min,
    # Groq gpt-oss = 30/min). Raise them if you have paid credit.
    openrouter_rpm: int = 18
    groq_rpm: int = 25
    google_rpm: int = 10
    groq_reasoning_effort: str = "low"

    # Search. Results are localised per request to the traveller's country. SEARCH_COUNTRY
    # (ISO 3166-1 alpha-2) is only a fallback for requests with no recognised nationality.
    search_country: str | None = None
    search_provider: Literal["exa", "serper"] = "exa"
    exa_api_key: str | None = None
    serper_api_key: str | None = None

    # Persistence: SQLite by default (durable across restarts, zero infra).
    # Setting POSTGRES_URL switches to Postgres.
    sqlite_path: str = "data/checkpoints.sqlite"
    postgres_url: str | None = None

    # Review loop
    max_revisions: int = 5

    # On startup, re-invoke plans whose run was cut off by a crash or restart.
    resume_orphans_on_startup: bool = True

    # Initial backoff (seconds) for model/tool retries inside the agents. Free-tier LLM routers
    # rate-limit per minute, so the default is generous.
    retry_initial_delay: float = 2.0

    # Geocoding of individual attractions (Nominatim usage policy: identify yourself, 1 req/s).
    nominatim_url: str = "https://nominatim.openstreetmap.org"
    http_user_agent: str = "planny/0.1 (+https://github.com/prateek-srivastav/planny)"

    @field_validator(
        "openrouter_api_key",
        "groq_api_key",
        "google_api_key",
        "exa_api_key",
        "serper_api_key",
        mode="before",
    )
    @classmethod
    def _clean_key(cls, v: object) -> object:
        """Strip stray whitespace and quotes; a blank key is 'not set', not an empty header."""
        if isinstance(v, str):
            v = v.strip().strip("\"'").strip()
            return v or None
        return v


@lru_cache
def get_settings() -> Settings:
    # Export .env into os.environ too: pydantic-settings only fills Settings fields, but the
    # Anthropic/OpenAI SDKs read their keys straight from the environment.
    for note in reconcile_shell_overrides(ENV_FILE):
        log.warning(note)
    load_dotenv(ENV_FILE)
    return Settings()
