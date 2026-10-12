"""A rejected or missing API key: no pointless retries, a message that says what to do."""

import pytest

from app.core.config import Settings
from app.core.llm import AUTH_HINT, describe_keys, is_auth_error
from app.graph.build import build_graph
from tests.conftest import open_saver
from tests.fakes import make_classifier, make_deps, make_model
from tests.test_graph_hitl import start


class UnauthorizedResponseError(Exception):
    """Same class name as OpenRouter's SDK error."""


class WithStatus(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (UnauthorizedResponseError("Missing Authentication header"), True),
        (WithStatus(401), True),
        (WithStatus(403), True),
        (WithStatus(429), False),
        (WithStatus(503), False),
        (ConnectionError("down"), False),
    ],
)
def test_is_auth_error(exc, expected):
    assert is_auth_error(exc) is expected


async def test_a_rejected_key_fails_fast_with_an_actionable_message(db_path):
    async with await open_saver(db_path) as saver:
        model = make_model()
        model.fail_times = 10_000
        model.error = UnauthorizedResponseError("Missing Authentication header")
        graph = build_graph(saver, make_deps(), model, make_classifier())
        _, out = await start(graph)
        assert "UnauthorizedResponseError: Missing Authentication header" in out["error"]
        assert AUTH_HINT in out["error"]
        assert "restart the server" in out["error"]
        assert model.failures[0] == 1  # one attempt: retrying a bad key only wastes time


async def test_transient_errors_are_still_retried(db_path):
    async with await open_saver(db_path) as saver:
        model = make_model()
        model.fail_times = 10_000
        graph = build_graph(saver, make_deps(), model, make_classifier())
        _, out = await start(graph)
        assert AUTH_HINT not in out["error"]
        assert model.failures[0] > 1


def test_key_report_is_masked_and_flags_missing_keys():
    s = Settings(
        _env_file=None,
        model="openrouter:a/b:free",
        model_fallbacks="groq:openai/gpt-oss-120b",
        classifier_model="groq:openai/gpt-oss-20b",
        openrouter_api_key="sk-or-v1-SECRETSECRETSECRET",
        groq_api_key=None,
    )
    report = "\n".join(describe_keys(s))
    assert "OPENROUTER_API_KEY: set (27 chars, starts 'sk-or...')" in report
    assert "GROQ_API_KEY: not set, skipping groq:openai/gpt-oss-120b" in report
    assert "Classifier: openrouter:a/b:free (CLASSIFIER_MODEL has no key)" in report
    assert "SECRETSECRET" not in report  # never print key material
    assert report.count("OPENROUTER_API_KEY") == 1  # one line per provider, not per model


def test_ui_sends_no_currency_when_the_box_is_blank():
    """A blank currency field is omitted so the server derives it from the nationality."""
    from pathlib import Path

    js = (Path(__file__).resolve().parent.parent / "app/static/app.js").read_text()
    assert "currencyInput.value.trim().toUpperCase() || null" in js


@pytest.mark.parametrize(
    ("provider", "key", "fragment"),
    [
        ("openrouter", "sk-or-v1-abcdef", None),
        ("openrouter", "gsk_AbCdEf123", "starts with 'gsk_"),  # a Groq key in the OpenRouter slot
        ("openrouter", "11111111-2222-3333-4444-555555555555", "starts with '1111"),  # an Exa key
        ("openrouter", "your_key_here", "starts with 'your"),
        ("openrouter", '"sk-or-v1-abc"', "quote marks or spaces"),
        ("openrouter", "sk-or-v1-abc ", "quote marks or spaces"),
        ("groq", "gsk_abc", None),
        ("groq", "sk-or-v1-abc", "starts with 'sk-o"),
        ("anthropic", "anything", None),  # no known shape: don't guess
        ("openrouter", None, None),  # missing keys are reported elsewhere
    ],
)
def test_key_problem(provider, key, fragment):
    from app.core.llm import key_problem

    problem = key_problem(provider, key)
    if fragment is None:
        assert problem is None
    else:
        assert fragment in problem
        assert (key or "")[8:] not in problem  # only the first few characters are ever quoted


def test_key_report_flags_a_wrong_providers_key():
    s = Settings(
        _env_file=None,
        model="openrouter:a/b:free",
        classifier_model="groq:openai/gpt-oss-20b",
        openrouter_api_key="gsk_ThisIsActuallyAGroqKey",
        groq_api_key="gsk_ThisIsActuallyAGroqKey",
    )
    report = describe_keys(s)
    assert any(ln.startswith("OPENROUTER_API_KEY") and "PROBLEM" in ln for ln in report)
    assert not any(ln.startswith("GROQ_API_KEY") and "PROBLEM" in ln for ln in report)


async def test_a_spend_cap_failure_explains_itself(db_path):
    def forever(messages):
        return [{"name": "web_search", "args": {"query": "again"}}]

    async with await open_saver(db_path) as saver:
        graph = build_graph(saver, make_deps(), make_model(ResearchBody=forever), make_classifier())
        _, out = await start(graph)
        assert "ModelCallLimitExceededError" in out["error"]
        assert "MAX_AGENT_MODEL_CALLS" in out["error"]
        assert AUTH_HINT not in out["error"]  # the right hint, not the key one


@pytest.mark.parametrize(
    ("message", "status", "retry"),
    [
        ("Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000", 429, False),
        (
            "Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): Limit 200000",
            429,
            False,
        ),
        ("Rate limit reached ... on requests per day (RPD): Limit 1000", 429, False),
        ("Request too large for model `qwen/qwen3.8-27b` on tokens per minute (TPM)", 413, False),
        ("Rate limit exceeded", 429, True),  # per-minute: worth a backoff
        ("provider overloaded", 503, True),
    ],
)
def test_only_transient_model_errors_are_retried(message, status, retry):
    from app.core.llm import QUOTA_HINT, TOO_LARGE_HINT, hint_for, should_retry_model

    exc = WithStatus(status)
    exc.args = (message,)
    assert should_retry_model(exc) is retry
    expected_hint = "" if retry else (TOO_LARGE_HINT if status == 413 else QUOTA_HINT)
    assert hint_for(exc) == expected_hint


async def test_a_spent_daily_quota_rolls_over_without_retrying(db_path):
    async with await open_saver(db_path) as saver:
        primary = make_model()
        primary.fail_times = 10_000
        primary.error = WithStatus(429)
        primary.error.args = ("Rate limit exceeded: free-models-per-day",)
        fallback = make_model()
        graph = build_graph(
            saver, make_deps(), primary, make_classifier(), fallback_models=[fallback]
        )
        _, out = await start(graph)
        assert out["__interrupt__"][0].value["draft"]["days"]
        assert primary.failures[0] == 1  # tried once, benched, never retried


@pytest.mark.parametrize("key", ["AIzaSyB2kq8Lm3Rt7Vw9Pz", "AQ.Ab8RN6Kq2mT9vL4sW7pZ"])
def test_both_google_key_formats_are_accepted(key):
    from app.core.envdoc import obviously_invalid
    from app.core.llm import key_problem

    assert key_problem("google_genai", key) is None
    assert not obviously_invalid("GOOGLE_API_KEY", key)
