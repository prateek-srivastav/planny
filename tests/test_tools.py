"""Unit tests for the deterministic tools and the HTTP-facing ones (mock transport)."""

import json
from datetime import date

import httpx
import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from app.core.config import Settings
from app.graph.state import ItineraryDraft
from app.tools import budget
from app.tools.collector import ToolFailure
from app.tools.conditions import fetch_conditions
from app.tools.geo import (
    Place,
    cluster,
    haversine_km,
    order_route,
    plan_days,
    travel_minutes,
)
from app.tools.search import ExaSearch, SerperSearch, make_provider
from tests.fakes import _daily, make_request

# ---------------------------------------------------------------- budget


def test_allocation_sums_to_total_for_every_tier():
    for tier in budget.TIER_SPLITS:
        out = budget.allocate(2000, travelers=2, nights=3, cost_tier=tier)
        assert sum(out["envelopes"].values()) == pytest.approx(2000, abs=0.05)
        assert set(out["envelopes"]) == set(budget.CATEGORIES)


def test_allocation_unknown_tier_falls_back_to_mid():
    assert (
        budget.allocate(1000, 1, 2, "???")["envelopes"]
        == budget.allocate(1000, 1, 2, "mid")["envelopes"]
    )


def test_check_remaining_flags_overspent_categories():
    env = budget.allocate(1000, 1, 3, "mid")["envelopes"]
    ok = budget.check_remaining(
        env, {"lodging": 300, "food": 100, "activities": 100, "transit": 50}
    )
    assert ok["ok"] and ok["over_budget_categories"] == []
    bad = budget.check_remaining(env, {"lodging": 500, "food": 100})
    assert bad["over_budget_categories"] == ["lodging"] and not bad["ok"]


def test_summary_is_computed_not_trusted():
    from tests.fakes import planner_answer

    request = make_request()
    body = planner_answer(
        [HumanMessage(content="Plan"), ToolMessage(content="x", tool_call_id="1")]
    )
    draft = ItineraryDraft(**body)
    summary = budget.summarize(draft, request)
    assert summary.by_category == {"lodging": 600, "food": 360, "activities": 400, "transit": 60}
    assert summary.total_estimated == 1420
    assert summary.remaining == request.budget_max - 1420
    over = budget.summarize(draft.model_copy(update={"days": draft.days * 2}), request)
    assert not over.within_budget


# ---------------------------------------------------------------- geo

LISBON = {
    "belem": (38.6916, -9.2160),
    "jeronimos": (38.6979, -9.2068),
    "alfama": (38.7110, -9.1290),
    "castle": (38.7139, -9.1334),
    "market": (38.7068, -9.1456),
    "oceanarium": (38.7634, -9.0938),
}


def test_haversine_known_distance():
    assert haversine_km(LISBON["belem"], LISBON["castle"]) == pytest.approx(
        7.6, abs=0.1
    )  # ~2.5 km north + ~7.2 km east
    assert haversine_km(LISBON["belem"], LISBON["belem"]) == 0


def test_travel_minutes_walks_short_hops_and_rides_long_ones():
    assert travel_minutes(0.9) < 15
    assert travel_minutes(9) > 30


def test_clusters_keep_neighbours_together_and_are_deterministic():
    pts = list(LISBON.values())
    groups = cluster(pts, 2)
    names = list(LISBON)
    sets = [{names[i] for i in g} for g in groups]
    assert {"belem", "jeronimos"} in [s & {"belem", "jeronimos"} for s in sets] or any(
        {"belem", "jeronimos"} <= s for s in sets
    )
    assert any({"alfama", "castle"} <= s for s in sets)
    assert cluster(pts, 2) == groups
    assert sorted(i for g in groups for i in g) == list(range(len(pts)))


def test_cluster_edge_cases():
    assert cluster([(1.0, 1.0)], 5) == [[0]]
    assert len(cluster(list(LISBON.values()), 99)) <= len(LISBON)


def test_route_visits_everything_once():
    pts = list(LISBON.values())
    route = order_route(list(range(len(pts))), pts)
    assert sorted(route) == list(range(len(pts)))


def test_plan_days_reports_legs():
    days = plan_days(LISBON, 2)
    assert len(days) == 2
    assert all(d["total_travel_minutes"] == sum(leg["minutes"] for leg in d["legs"]) for d in days)
    assert sum(len(d["stops"]) for d in days) == len(LISBON)


# ---------------------------------------------------------------- conditions

PLACE = Place("Lisbon", 38.72, -9.14, "Portugal")


def client_for(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_near_trip_uses_forecast_endpoint():
    seen = []

    def handler(req):
        seen.append(req.url.host)
        return httpx.Response(
            200, json=_daily(req.url.params["start_date"], req.url.params["end_date"], rain=2)
        )

    today = date(2026, 10, 9)
    out = await fetch_conditions(
        client_for(handler), PLACE, date(2026, 10, 12), date(2026, 10, 14), today
    )
    assert seen == ["api.open-meteo.com"]
    assert out.basis == "forecast" and len(out.daily) == 3
    assert out.rainy_day_fraction == 1.0


async def test_forecast_is_clipped_at_the_14_day_horizon_and_says_so():
    def handler(req):
        return httpx.Response(
            200, json=_daily(req.url.params["start_date"], req.url.params["end_date"])
        )

    today = date(2026, 10, 9)
    out = await fetch_conditions(
        client_for(handler), PLACE, date(2026, 10, 20), date(2026, 10, 30), today
    )
    assert out.basis == "forecast"
    assert out.daily[-1].date == date(2026, 10, 23)  # today + 14, Open-Meteo's last valid date
    assert "forecast horizon" in out.note


async def test_far_trip_uses_three_years_of_climatology_and_says_so():
    calls = []

    def handler(req):
        calls.append((req.url.host, req.url.params["start_date"], req.url.params["end_date"]))
        return httpx.Response(
            200, json=_daily(req.url.params["start_date"], req.url.params["end_date"], hi=30, lo=20)
        )

    today = date(2026, 10, 9)
    out = await fetch_conditions(
        client_for(handler), PLACE, date(2027, 8, 10), date(2027, 8, 13), today
    )
    assert out.basis == "historical_climatology"
    assert sorted(c[1][:4] for c in calls) == ["2023", "2024", "2025"]
    assert all(c[0] == "archive-api.open-meteo.com" for c in calls)
    assert out.avg_high_c == 30 and out.avg_low_c == 20
    assert out.daily == []  # a season is not a day-by-day forecast
    assert "same dates" in out.note


async def test_climatology_handles_leap_day_and_new_year_crossing():
    calls = []

    def handler(req):
        calls.append((req.url.params["start_date"], req.url.params["end_date"]))
        return httpx.Response(
            200, json=_daily(req.url.params["start_date"], req.url.params["end_date"])
        )

    today = date(2026, 10, 9)
    await fetch_conditions(client_for(handler), PLACE, date(2028, 12, 30), date(2029, 1, 3), today)
    assert ("2025-12-30", "2026-01-03") in calls
    calls.clear()
    await fetch_conditions(client_for(handler), PLACE, date(2028, 2, 29), date(2028, 3, 2), today)
    assert all(c[0].endswith("02-28") or c[0].endswith("02-29") for c in calls)


async def test_weather_api_error_raises_tool_failure():
    c = client_for(lambda req: httpx.Response(503))
    with pytest.raises(ToolFailure, match="destination_conditions"):
        await fetch_conditions(c, PLACE, date(2026, 10, 12), date(2026, 10, 14), date(2026, 10, 9))


# ---------------------------------------------------------------- search providers


async def test_exa_uses_capped_highlights_and_objective():
    captured = {}

    def handler(req):
        captured["headers"], captured["json"] = req.headers, json.loads(req.read())
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "T",
                        "url": "https://x.test",
                        "publishedDate": "2026-01-01",
                        "highlights": ["first passage", "second passage"],
                    }
                ]
            },
        )

    out = await ExaSearch(client_for(handler), "k").search("lisbon", 3, objective="find areas")
    body = captured["json"]
    assert captured["headers"]["x-api-key"] == "k"
    assert body["numResults"] == 3 and body["type"] == "auto"
    assert body["contents"] == {"highlights": {"maxCharacters": 600}}  # capped, not Exa's default
    assert body["objective"] == "find areas"
    assert out[0].url == "https://x.test" and out[0].published_date == "2026-01-01"
    assert out[0].text == "first passage … second passage"


async def test_exa_omits_objective_when_not_given():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.read()))
        return httpx.Response(200, json={"results": []})

    await ExaSearch(client_for(handler), "k").search("x")
    assert "objective" not in seen


async def test_serper_parsing():
    def handler(req):
        return httpx.Response(
            200,
            json={
                "organic": [{"title": "T", "link": "https://y.test", "snippet": "s", "date": "d"}]
            },
        )

    out = await SerperSearch(client_for(handler), "k").search("lisbon")
    assert (out[0].url, out[0].text) == ("https://y.test", "s")


async def test_missing_key_and_http_errors_are_tool_failures():
    c = client_for(lambda req: httpx.Response(429))
    with pytest.raises(ToolFailure, match="EXA_API_KEY") as missing:
        await ExaSearch(c, None).search("x")
    assert missing.value.retryable is False
    with pytest.raises(ToolFailure, match="429") as limited:
        await ExaSearch(c, "k").search("x")
    assert limited.value.retryable is True  # rate limits clear up: let the middleware back off


@pytest.mark.parametrize(
    ("status", "retryable", "hint"),
    [
        (401, False, "invalid API key"),
        (402, False, "out of credits"),
        (500, True, ""),
        (503, True, ""),
    ],
)
async def test_http_errors_are_classified_for_retry(status, retryable, hint):
    c = client_for(lambda req: httpx.Response(status))
    with pytest.raises(ToolFailure) as exc:
        await ExaSearch(c, "k").search("x")
    assert exc.value.retryable is retryable and hint in str(exc.value)


async def test_permanent_tool_failure_degrades_immediately_without_retry():
    from app.tools import collector
    from app.tools.registry import soft_fail

    calls = []

    @soft_fail
    async def broken():
        calls.append(1)
        raise ToolFailure("web_search: exa returned HTTP 402 (out of credits)", retryable=False)

    @soft_fail
    async def flaky():
        raise ToolFailure("web_search: exa returned HTTP 503")

    with collector.collecting() as run:
        out = await broken()
    assert "TOOL UNAVAILABLE" in out["error"] and len(calls) == 1
    assert run.degraded == ["web_search: exa returned HTTP 402 (out of credits)"]
    with pytest.raises(ToolFailure):  # transient: left for the retry middleware
        await flaky()


def test_provider_is_chosen_by_setting():
    c = client_for(lambda req: httpx.Response(200))
    assert make_provider(Settings(_env_file=None, search_provider="serper"), c).name == "serper"
    assert make_provider(Settings(_env_file=None), c).name == "exa"


async def test_exa_localises_results_to_the_configured_country():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.read()))
        return httpx.Response(200, json={"results": []})

    await ExaSearch(client_for(handler), "k", country="IN").search("goa")
    assert seen["userLocation"] == "IN"
    seen.clear()
    await ExaSearch(client_for(handler), "k").search("goa")
    assert "userLocation" not in seen


async def test_serper_localises_with_gl():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.read()))
        return httpx.Response(200, json={"organic": []})

    await SerperSearch(client_for(handler), "k", country="IN").search("goa")
    assert seen["gl"] == "in"


async def test_per_request_country_overrides_the_configured_fallback():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.read()))
        return httpx.Response(200, json={"results": []})

    exa = ExaSearch(client_for(handler), "k", country="IN")  # SEARCH_COUNTRY fallback
    await exa.search("rio", country="BR")
    assert seen["userLocation"] == "BR"
    seen.clear()
    await exa.search("rio")
    assert seen["userLocation"] == "IN"


def test_search_country_has_no_default_market():
    s = Settings(_env_file=None)
    assert s.search_country is None
    c = client_for(lambda req: httpx.Response(200))
    assert make_provider(s, c)._country is None


def test_countries_resolve_names_codes_and_demonyms():
    from app.core.countries import resolve

    assert resolve("India").currency == "INR" and resolve(" india ").iso2 == "IN"
    assert resolve("br").name == "Brazil" and resolve("Indian").iso2 == "IN"
    assert resolve("U.K.").iso2 == "GB" and resolve("USA").currency == "USD"
    assert resolve("Narnia") is None and resolve(None) is None and resolve("") is None


def test_every_country_row_is_well_formed_and_unique():
    from app.core.countries import COUNTRIES

    assert len({c.iso2 for c in COUNTRIES}) == len(COUNTRIES) > 50
    assert all(len(c.iso2) == 2 and len(c.currency) == 3 for c in COUNTRIES)


def test_market_notes_exist_only_for_configured_markets():
    from app.core.markets import notes_for

    assert "rupees" in notes_for("IN")
    assert notes_for("BR") == "" and notes_for(None) == ""


async def test_the_travellers_market_reaches_the_search_tool_and_prompts():
    """Indian traveller -> userLocation=IN and market notes; Brazilian -> BR and none."""
    from app.core.markets import notes_for
    from app.tools import collector
    from app.tools.registry import build_research_tools
    from tests.fakes import FakeSearch, make_deps

    for iso, expect_notes in (("IN", True), ("BR", False)):
        fake = FakeSearch()
        deps = make_deps(search=fake)
        web_search = build_research_tools(deps)[0]
        with collector.collecting(search_country=iso):
            await web_search.ainvoke({"query": "x"})
        assert fake.countries == [iso]
        assert bool(notes_for(iso)) is expect_notes


async def test_search_output_size_follows_settings():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.read()))
        return httpx.Response(
            200,
            json={"results": [{"title": "T", "url": "https://x.test", "highlights": ["a" * 900]}]},
        )

    s = Settings(_env_file=None, exa_api_key="k", search_result_chars=250)
    out = await make_provider(s, client_for(handler)).search("x")
    assert seen["contents"]["highlights"]["maxCharacters"] == 250
    assert len(out[0].text) == 250  # also enforced locally, whatever the provider returns


async def test_web_search_defaults_to_the_configured_result_count():
    from app.tools import collector
    from app.tools.registry import build_research_tools
    from tests.fakes import FakeSearch, make_deps

    class CountingSearch(FakeSearch):
        async def search(self, query, num_results=5, objective=None, country=None):
            self.sizes = [*getattr(self, "sizes", []), num_results]
            return await super().search(query, num_results, objective, country)

    search = CountingSearch()
    deps = make_deps(search=search)
    deps.settings = deps.settings.model_copy(update={"search_num_results": 3})
    web_search = build_research_tools(deps)[0]
    with collector.collecting():
        await web_search.ainvoke({"query": "x"})
        await web_search.ainvoke({"query": "y", "num_results": 20})
    assert search.sizes == [3, 8]  # default from settings; explicit values capped at 8


# ---------------------------------------------------------------- destination geocoding


def nominatim(hits_by_query: dict[str, list], status: int = 200, openmeteo: dict | None = None):
    seen = []

    def handler(req):
        if req.url.host == "nominatim.test":
            q = req.url.params["q"]
            seen.append(q)
            return httpx.Response(status, json=hits_by_query.get(q, []))
        if req.url.host == "geocoding-api.open-meteo.com":
            seen.append("open-meteo")
            return httpx.Response(200, json=openmeteo or {})
        return httpx.Response(404)

    return client_for(handler), seen


GOA_HIT = {
    "lat": "15.3004",
    "lon": "74.0855",
    "name": "Goa",
    "display_name": "Goa, India",
    "address": {"state": "Goa", "country": "India"},
}


async def test_destination_resolves_regions_through_nominatim():
    from app.tools.geo import Geocoder

    client, seen = nominatim({"Goa": [GOA_HIT]})
    place = await Geocoder(client, "https://nominatim.test", min_interval=0).destination("Goa")
    assert (place.name, place.country) == ("Goa", "India")
    assert place.lat == pytest.approx(15.3004)
    assert seen == ["Goa"]


async def test_destination_is_cached():
    from app.tools.geo import Geocoder

    client, seen = nominatim({"Goa": [GOA_HIT]})
    geocoder = Geocoder(client, "https://nominatim.test", min_interval=0)
    await geocoder.destination("Goa")
    await geocoder.destination("goa")
    assert seen == ["Goa"]


@pytest.mark.parametrize(("hits", "status"), [({}, 200), ({"Lisbon": []}, 503)])
async def test_destination_falls_back_to_open_meteo(hits, status):
    from app.tools.geo import Geocoder

    om = {
        "results": [{"name": "Lisbon", "latitude": 38.7, "longitude": -9.1, "country": "Portugal"}]
    }
    client, seen = nominatim(hits, status=status, openmeteo=om)
    place = await Geocoder(client, "https://nominatim.test", min_interval=0).destination("Lisbon")
    assert place.country == "Portugal" and seen == ["Lisbon", "open-meteo"]


async def test_destination_without_fallback_reports_not_found():
    from app.tools.geo import Geocoder

    client, _ = nominatim({})
    geocoder = Geocoder(client, "https://nominatim.test", min_interval=0, fallback=False)
    assert await geocoder.destination("Atlantis") is None


async def test_attractions_use_the_same_throttled_client():
    from app.tools.geo import Geocoder

    client, seen = nominatim({"Belem Tower, Lisbon": [{"lat": "38.69", "lon": "-9.21"}]})
    geocoder = Geocoder(client, "https://nominatim.test", min_interval=0)
    assert await geocoder.locate("Belem Tower", "Lisbon") == (38.69, -9.21)
    assert await geocoder.locate("Nowhere", "Lisbon") is None
    assert seen == ["Belem Tower, Lisbon", "Nowhere, Lisbon"]
