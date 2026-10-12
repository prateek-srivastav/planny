import pytest
from langchain_groq import ChatGroq
from langchain_openrouter import ChatOpenRouter

from app.core.config import Settings
from app.core.llm import (
    ConfigError,
    classifier_structured_kwargs,
    forces_tool_calling,
    make_chat_model,
    resolve_models,
)


def settings(**kw):
    return Settings(_env_file=None, **kw)


def test_openrouter_is_the_official_integration_with_safe_routing():
    s = settings(openrouter_api_key="k", openrouter_fallback_models="a/b:free, c/d:free")
    m = make_chat_model("openrouter:nvidia/some-model:free", s)
    assert isinstance(m, ChatOpenRouter)
    assert m.model_name == "nvidia/some-model:free"  # colons inside the id survive
    # route only to providers that support every request parameter (tools in particular)
    assert m.openrouter_provider == {"require_parameters": True}
    assert m.model_kwargs == {"models": ["a/b:free", "c/d:free"]}
    assert m.rate_limiter is not None


def test_openrouter_without_fallbacks_sends_no_models_list():
    m = make_chat_model("openrouter:x/y:free", settings(openrouter_api_key="k"))
    assert m.model_kwargs == {}


def test_groq_is_the_official_integration_with_low_reasoning_for_gpt_oss():
    m = make_chat_model("groq:openai/gpt-oss-20b", settings(groq_api_key="k"))
    assert isinstance(m, ChatGroq)
    assert m.model_name == "openai/gpt-oss-20b"
    assert m.reasoning_effort == "low"
    assert m.rate_limiter is not None


def test_groq_non_reasoning_models_get_no_reasoning_effort():
    m = make_chat_model("groq:llama-3.1-8b-instant", settings(groq_api_key="k"))
    assert m.reasoning_effort is None


def test_rate_limiter_follows_the_configured_requests_per_minute():
    m = make_chat_model("groq:openai/gpt-oss-20b", settings(groq_api_key="k", groq_rpm=30))
    assert m.rate_limiter.requests_per_second == pytest.approx(0.5)


def test_missing_key_is_a_clear_error():
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        make_chat_model("openrouter:x/y:free", settings(openrouter_api_key=None))
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        make_chat_model("groq:x", settings(groq_api_key=None))


def test_missing_model_name_is_a_clear_error():
    with pytest.raises(ConfigError, match="model name"):
        make_chat_model("groq:", settings(groq_api_key="k"))


def test_structured_output_strategy_per_provider():
    assert classifier_structured_kwargs("groq:openai/gpt-oss-20b") == {
        "method": "json_schema",
        "strict": True,
    }
    assert classifier_structured_kwargs("openrouter:x/y:free") == {"method": "function_calling"}
    assert classifier_structured_kwargs("anthropic:claude-sonnet-5-5") == {}


def test_only_the_free_routers_force_tool_calling():
    assert forces_tool_calling("openrouter:x") and forces_tool_calling("groq:x")
    assert not forces_tool_calling("anthropic:claude-sonnet-5-5")


def test_every_keyed_provider_maps_to_a_real_setting():
    from app.core.llm import KEYED_PROVIDERS, api_key_for

    s = settings(openrouter_api_key="a", groq_api_key="b", google_api_key="c")
    assert {p: api_key_for(s, p) for p in KEYED_PROVIDERS} == {
        "openrouter": "a",
        "groq": "b",
        "google_genai": "c",
    }


def test_gemini_model_is_built_with_the_google_key():
    m = make_chat_model("google_genai:gemini-3.8-flash", settings(google_api_key="k"))
    assert m.google_api_key.get_secret_value() == "k"
    assert m.rate_limiter is not None
    with pytest.raises(ConfigError, match="GOOGLE_API_KEY"):
        make_chat_model("google_genai:gemini-3.8-flash", settings(google_api_key=None))


FULL_CHAIN = dict(
    model="google_genai:gemini-3.8-flash",
    model_fallbacks="openrouter:a/b:free,google_genai:gemini-3.5-flash,groq:openai/gpt-oss-120b",
    classifier_model="groq:openai/gpt-oss-20b",
)


def test_all_keys_set_keeps_the_configured_chain():
    chain = resolve_models(
        settings(**FULL_CHAIN, google_api_key="g", openrouter_api_key="o", groq_api_key="q")
    )
    assert chain.primary == "google_genai:gemini-3.8-flash"
    assert chain.fallbacks == [
        "openrouter:a/b:free",
        "google_genai:gemini-3.5-flash",
        "groq:openai/gpt-oss-120b",
    ]
    assert chain.classifier == "groq:openai/gpt-oss-20b"


def test_models_without_a_key_are_skipped_and_the_classifier_reuses_the_main_model():
    chain = resolve_models(settings(**FULL_CHAIN, google_api_key="g"))
    assert chain.primary == "google_genai:gemini-3.8-flash"
    assert chain.fallbacks == ["google_genai:gemini-3.5-flash"]
    assert chain.classifier == "google_genai:gemini-3.8-flash"


def test_a_keyless_main_model_hands_over_to_the_first_usable_fallback():
    chain = resolve_models(settings(**FULL_CHAIN, groq_api_key="q"))
    assert chain.primary == "groq:openai/gpt-oss-120b"
    assert chain.fallbacks == []
    assert chain.classifier == "groq:openai/gpt-oss-20b"


def test_no_usable_model_names_the_keys_to_set():
    with pytest.raises(ConfigError, match="GOOGLE_API_KEY or GROQ_API_KEY or OPENROUTER_API_KEY"):
        resolve_models(settings(**FULL_CHAIN))


def test_providers_without_a_known_key_setting_are_kept():
    chain = resolve_models(settings(model="anthropic:claude-sonnet-5-5"))
    assert chain.primary == chain.classifier == "anthropic:claude-sonnet-5-5"
