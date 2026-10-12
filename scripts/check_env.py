"""Verify your .env against the real services: one tiny call each.

    uv run python scripts/check_env.py

Checks every model that will run (models whose provider key is blank are skipped, as the server
skips them), including that they can do tool calling, which the agents need; the search provider;
and the keyless data services.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

import httpx
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.core.llm import (
    KEYED_PROVIDERS,
    ConfigError,
    classifier_structured_kwargs,
    describe_keys,
    make_chat_model,
    parse_specs,
    resolve_models,
)  # noqa: E402
from app.tools.geo import Geocoder  # noqa: E402
from app.tools.search import make_provider  # noqa: E402


class Ping(BaseModel):
    word: str


async def step(name: str, coro):
    start = time.monotonic()
    try:
        detail = await coro
        print(f"  ok    {name:<28} {time.monotonic() - start:4.1f}s  {detail}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  {name:<28} {type(exc).__name__}: {str(exc)[:200]}")
        return False


async def llm(spec: str, settings, kwargs: dict) -> str:
    model = make_chat_model(spec, settings)
    out = await model.with_structured_output(Ping, **kwargs).ainvoke(
        "Call the tool with word='pong'."
    )
    return f"structured output works {kwargs or '(default)'} -> {out.word!r}"


def env_doctor() -> bool:
    """Check the .env file itself before any network call. Never prints key material."""
    from app.core.config import ENV_FILE, Settings
    from app.core.envdoc import diagnose

    env_before = dict(os.environ)  # what the shell exported, before .env is loaded
    s = Settings(_env_file=ENV_FILE)
    specs = [s.model, *parse_specs(s.model_fallbacks), s.classifier_model or s.model]
    # Each LLM key is optional (models without one are skipped); at least one is checked below.
    optional = {}
    for spec in specs:
        var = KEYED_PROVIDERS.get(spec.partition(":")[0])
        if var:
            optional[var] = "its models are skipped"
    search_var = "EXA_API_KEY" if s.search_provider == "exa" else "SERPER_API_KEY"
    optional[search_var] = "plans are made without web research"
    print(f".env diagnosis ({ENV_FILE})")
    findings = diagnose(ENV_FILE, env_before, [], optional)
    for f in findings:
        print(f"  {'ok   ' if f.level == 'ok' else f.level.upper():<5} {f.message}")
    print()
    return not any(f.level == "error" for f in findings)


async def main() -> int:
    env_ok = env_doctor()
    s = get_settings()
    try:
        chain = resolve_models(s)
    except ConfigError as exc:
        print(f"ERROR {exc}")
        return 1
    print(f"MODEL={s.model}")
    print(f"MODEL_FALLBACKS={s.model_fallbacks or '(none)'}")
    print(f"CLASSIFIER_MODEL={s.classifier_model}")
    print(
        f"Will run: {chain.primary} (main), {len(chain.fallbacks)} fallback(s), "
        f"classifier {chain.classifier}"
    )
    print(f"SEARCH_PROVIDER={s.search_provider}\n")
    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": s.http_user_agent}) as client:

        async def search():
            if not (s.exa_api_key if s.search_provider == "exa" else s.serper_api_key):
                return "skipped (no key: plans are made without web research)"
            res = await make_provider(s, client).search("best things to do in Lisbon", 2)
            return f"{len(res)} results, first: {res[0].url if res else '-'}"

        async def quota():
            if not s.openrouter_api_key:
                return "skipped (no OPENROUTER_API_KEY)"
            r = await client.get(
                "https://openrouter.ai/api/v1/key",
                headers={"Authorization": f"Bearer {s.openrouter_api_key}"},
            )
            r.raise_for_status()
            d = r.json()["data"]["free_model_daily_requests"]
            note = ""
            if d["limit"] <= 50:
                note = "  (free accounts without 10+ credits get 50/day: about 3-4 plans)"
            return f"{d['remaining']}/{d['limit']} free-model requests left today{note}"

        async def geocode():
            place = await Geocoder(client, s.nominatim_url).destination("Goa")
            if place is None or place.country != "India":
                raise RuntimeError(f"'Goa' resolved to {place}")
            return str(place)

        async def forecast():
            r = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": 38.7,
                    "longitude": -9.1,
                    "daily": "temperature_2m_max",
                    "forecast_days": 1,
                },
            )
            r.raise_for_status()
            return "forecast API reachable"

        async def key_formats():
            lines = describe_keys(s)
            bad = [ln for ln in lines if "PROBLEM" in ln]
            if bad:
                raise RuntimeError("; ".join(bad))
            keys = [ln.split(":")[0] for ln in lines if ": set" in ln]
            return "; ".join(keys) + " look well-formed"

        results = [
            await step("key formats", key_formats()),
            await step("main model", llm(chain.primary, s, {"method": "function_calling"})),
            *[
                await step(f"fallback {i}: {spec}", llm(spec, s, {"method": "function_calling"}))
                for i, spec in enumerate(chain.fallbacks, start=1)
            ],
            await step(
                "classifier model",
                llm(chain.classifier, s, classifier_structured_kwargs(chain.classifier)),
            ),
            await step("openrouter free quota", quota()),
            await step(f"search ({s.search_provider})", search()),
            await step("geocoding (nominatim)", geocode()),
            await step("open-meteo forecast", forecast()),
        ]
    ok = all(results) and env_ok
    print("\nAll good." if ok else "\nFix the FAIL/ERROR lines above, then re-run.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
