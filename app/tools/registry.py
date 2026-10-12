"""LangChain tool wrappers. Built per graph so tests can swap the dependencies underneath."""

import asyncio
import functools
from dataclasses import dataclass
from datetime import date

import httpx
from langchain_core.tools import BaseTool, tool

from app.core.config import Settings
from app.graph.state import Source
from app.tools import budget as budget_lib
from app.tools import collector
from app.tools.collector import ToolFailure
from app.tools.conditions import fetch_conditions
from app.tools.fx import fetch_rate
from app.tools.geo import Geocoder, plan_days
from app.tools.search import SearchProvider, make_provider


def soft_fail(fn):
    """Permanent failures (`retryable=False`) degrade at once; transient ones are re-raised so the
    retry middleware can back off and try again."""

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except ToolFailure as exc:
            if exc.retryable:
                raise
            collector.record_degraded(str(exc))
            return {"error": f"TOOL UNAVAILABLE: {exc}. Continue without it; do not invent data."}

    return wrapper


@dataclass
class Deps:
    """Everything tools need from the outside world. One instance per compiled graph."""

    client: httpx.AsyncClient
    settings: Settings
    search: SearchProvider
    geocoder: Geocoder

    @classmethod
    def from_settings(cls, settings: Settings, client: httpx.AsyncClient) -> "Deps":
        return cls(
            client=client,
            settings=settings,
            search=make_provider(settings, client),
            geocoder=Geocoder(client, settings.nominatim_url),
        )


def build_research_tools(deps: Deps) -> list[BaseTool]:
    @tool
    @soft_fail
    async def web_search(
        query: str, num_results: int | None = None, objective: str | None = None
    ) -> list[dict]:
        """Search the web. Returns title, url, published_date and the passages most relevant to the
        query for each result. Use specific queries (e.g. 'best neighbourhoods to stay in Lisbon
        for first-time visitors', 'Lisbon average cost of a restaurant meal 2026').
        `objective` is the goal of this search: which kind of page should rank first and which
        facts (prices, opening hours, areas) you need from it. Cite results by their exact url."""
        results = await deps.search.search(
            query,
            max(1, min(num_results or deps.settings.search_num_results, 8)),
            objective,
            country=collector.current().search_country,
        )
        sources = collector.current().sources
        for r in results:
            sources.setdefault(
                r.url, Source(title=r.title, url=r.url, published_date=r.published_date)
            )
        return [r.model_dump() for r in results]

    @tool
    @soft_fail
    async def destination_conditions(destination: str, start_date: str, end_date: str) -> dict:
        """Weather for the trip. Dates are YYYY-MM-DD. Returns a daily forecast if the trip starts
        within 14 days, otherwise historical climatology (check the `basis` field)."""
        place = await deps.geocoder.destination(destination)
        if place is None:
            raise ToolFailure(f"destination_conditions: could not geocode '{destination}'")
        summary = await fetch_conditions(
            deps.client, place, date.fromisoformat(start_date), date.fromisoformat(end_date)
        )
        collector.current().conditions = summary
        return summary.model_dump(mode="json", exclude={"daily"}) | {
            "daily_highlights": [d.model_dump(mode="json") for d in summary.daily[:16]]
        }

    @tool
    @soft_fail
    async def exchange_rate(from_currency: str, to_currency: str) -> dict:
        """Current exchange rate between two ISO 4217 currency codes (e.g. USD -> EUR). Use it to
        convert the destination's local prices into the traveller's budget currency. Returns
        `rate` meaning 1 from_currency = rate to_currency."""
        fx = await fetch_rate(deps.client, from_currency, to_currency)
        collector.current().exchange_rate = fx
        return fx.model_dump(mode="json")

    return [web_search, destination_conditions, exchange_rate]


def build_planner_tools(deps: Deps) -> list[BaseTool]:
    @tool
    def allocate_budget(
        total_budget: float,
        travelers: int,
        nights: int,
        cost_tier: str,
        intercity_cost: float = 0,
    ) -> dict:
        """Split the trip budget into category envelopes (lodging, food, activities, transit,
        buffer) and return per-day guidance. cost_tier is 'budget', 'mid' or 'premium'.
        intercity_cost is the chosen round trip for the WHOLE party (per-person price times
        travellers); it is reserved first and the rest is split. Call this first and plan inside
        the envelopes."""
        return budget_lib.allocate(total_budget, travelers, nights, cost_tier, intercity_cost)

    @tool
    def check_budget(envelopes: dict[str, float], planned: dict[str, float]) -> dict:
        """Compare planned spend per category against the envelopes from allocate_budget.
        `planned` maps category (lodging, food, activities, transit, intercity, other) -> party
        total planned so far, including logistics items. Use it before finishing; do your own
        arithmetic nowhere else."""
        return budget_lib.check_remaining(envelopes, planned)

    @tool
    @soft_fail
    async def geo_cluster_and_route(attractions: list[str], city: str, num_days: int) -> dict:
        """Group candidate attractions into geographic day-buckets and order each day to minimise
        travel. Pass attraction names exactly as researched, the city, and the number of days to
        fill. Returns per-group stops, leg distances and travel minutes. Attractions that cannot
        be located come back in `unplaced`; schedule them yourself."""
        names = list(dict.fromkeys(attractions))[:16]
        coords = await asyncio.gather(*(deps.geocoder.locate(n, city) for n in names))
        placed = {n: c for n, c in zip(names, coords, strict=True) if c}
        if not placed:
            raise ToolFailure("geo_cluster_and_route: none of the attractions could be located")
        return {
            "days": plan_days(placed, num_days),
            "unplaced": [n for n in names if n not in placed],
        }

    return [allocate_budget, check_budget, geo_cluster_and_route]
