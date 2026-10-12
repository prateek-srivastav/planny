"""Offline doubles: a scripted chat model and a mock HTTP transport.

The real `create_agent` loops, real tools, real collector and real graph all run. Only the LLM and
the network are replaced.
"""

from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from app.core.config import Settings
from app.graph.state import TravelRequest
from app.tools.geo import Geocoder
from app.tools.registry import Deps
from app.tools.search import SearchResult

Responder = Callable[[list[BaseMessage]], "dict | list[dict]"]


class ScriptedModel(BaseChatModel):
    """Picks a responder by the structured-output tool bound to it (ResearchBody, DraftBody...).

    A responder returns a dict -> the final structured answer, or a list of tool-call dicts
    ({"name", "args"}) -> the agent runs those tools first.
    """

    responders: dict[str, Responder] = Field(default_factory=dict)
    bound: list[str] = Field(default_factory=list)
    calls: list[tuple[str, str]] = Field(default_factory=list)  # (schema, first user prompt)
    fail_times: int = 0  # raise this many times before answering (retry tests)
    error: Any = None  # what to raise (default: ConnectionError)
    failures: list[int] = Field(default_factory=lambda: [0])

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any):
        names = [
            getattr(t, "name", None) or getattr(t, "__name__", None) or t["name"] for t in tools
        ]
        return self.model_copy(update={"bound": names})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if self.failures[0] < self.fail_times:
            self.failures[0] += 1
            raise self.error or ConnectionError("simulated model outage")
        schema = next((n for n in self.bound if n in self.responders), None)
        if schema is None:
            raise AssertionError(f"no responder for tools {self.bound}")
        prompt = next((m.content for m in messages if isinstance(m, HumanMessage)), "")
        out = self.responders[schema](messages)
        if isinstance(out, dict):
            self.calls.append((schema, str(prompt)))
            tool_calls = [
                {"name": schema, "args": out, "id": f"call_{schema}", "type": "tool_call"}
            ]
        else:
            tool_calls = [
                {"name": c["name"], "args": c["args"], "id": f"call_{i}", "type": "tool_call"}
                for i, c in enumerate(out)
            ]
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="", tool_calls=tool_calls))]
        )


# --------------------------------------------------------------------------- canned answers

ATTRACTIONS = [
    ("Belem Tower", "Belem", False, 8),
    ("Jeronimos Monastery", "Belem", True, 12),
    ("Alfama Walk", "Alfama", False, 0),
    ("Sao Jorge Castle", "Alfama", False, 15),
    ("Time Out Market", "Cais do Sodre", True, 0),
    ("Oceanarium", "Parque das Nacoes", True, 25),
]
SEARCH_URL = "https://example.com/lisbon-guide"


def research_answer(messages: list[BaseMessage]) -> dict | list[dict]:
    if not any(isinstance(m, ToolMessage) for m in messages):
        start = _find_dates(messages)
        return [
            {
                "name": "web_search",
                "args": {
                    "query": "best things to do in Lisbon",
                    "objective": "find top attractions and areas",
                },
            },
            {"name": "exchange_rate", "args": {"from_currency": "USD", "to_currency": "EUR"}},
            {
                "name": "destination_conditions",
                "args": {"destination": "Lisbon", "start_date": start[0], "end_date": start[1]},
            },
        ]
    return {
        "destination": "Lisbon",
        "country": "Portugal",
        "cost_tier": "mid",
        "overview": "Hilly coastal capital with great food.",
        "attractions": [
            {
                "name": n,
                "area": area,
                "indoor": indoor,
                "est_cost_per_person": cost,
                "source_url": SEARCH_URL,
            }
            for n, area, indoor, cost in ATTRACTIONS
        ],
        "local_currency": "EUR",
        "entry_requirements": "Indian citizens need a Schengen visa (up to 90 days).",
        "food_notes": "Pasteis de nata.",
        "local_tips": ["Wear flat shoes."],
        "source_urls": [SEARCH_URL, "https://hallucinated.example/nope"],
        "getting_there": []
        if "Travelling from: not stated" in _first_prompt(messages)
        else GETTING_THERE,
    }


GETTING_THERE = [
    {
        "mode": "flight",
        "summary": "DEL -> LIS one stop",
        "duration_hours": 14,
        "cost_per_person_round_trip": 300,
    },
    {"mode": "train", "summary": "not practical", "cost_per_person_round_trip": 900},
]


def _first_prompt(messages: list[BaseMessage]) -> str:
    return next((str(m.content) for m in messages if isinstance(m, HumanMessage)), "")


def _find_dates(messages: list[BaseMessage]) -> tuple[str, str]:
    text = next(m.content for m in messages if isinstance(m, HumanMessage))
    marker = "destination_conditions with "
    seg = text.split(marker)[-1].split(".")[0]
    a, b = seg.split(" to ")
    return a.strip(), b.strip()[:10]


def planner_answer(messages: list[BaseMessage]) -> dict | list[dict]:
    prompt = next(m.content for m in messages if isinstance(m, HumanMessage))
    if not any(isinstance(m, ToolMessage) for m in messages):
        return [
            {
                "name": "allocate_budget",
                "args": {"total_budget": 2000, "travelers": 2, "nights": 3, "cost_tier": "mid"},
            },
            {
                "name": "geo_cluster_and_route",
                "args": {
                    "attractions": [a[0] for a in ATTRACTIONS],
                    "city": "Lisbon",
                    "num_days": 2,
                },
            },
        ]
    revised = "rejected the previous draft" in prompt
    days = []
    for i in range(1, 5):
        days.append(
            {
                "day": i,
                "date": "2030-01-01",  # deliberately wrong: the node must overwrite it
                "theme": f"Day {i}",
                "morning": {"name": f"Morning {i}", "est_cost": 40},
                "afternoon": {"name": f"Afternoon {i}", "est_cost": 60},
                "evening": {"name": f"Dinner {i}", "est_cost": 0},  # meals: food_cost
                "lodging_cost": 150,
                "food_cost": 90,
                "transit_cost": 15,
            }
        )
    return {
        "title": "Revised Lisbon trip" if revised else "Lisbon trip",
        "summary": "A relaxed long weekend.",
        "intercity": (
            {"mode": "flight", "summary": "DEL -> LIS one stop", "cost_per_person_round_trip": 300}
            if "Getting there from" in prompt
            else None
        ),
        "days": days,
        "caveats": [],
    }


def make_model(**overrides: Responder) -> ScriptedModel:
    return ScriptedModel(
        responders={"ResearchBody": research_answer, "DraftBody": planner_answer, **overrides}
    )


def make_classifier(target: str = "planner", steer: dict | None = None) -> ScriptedModel:
    responders: dict[str, Responder] = {
        "RouteChoice": lambda _m: {"target": target, "reason": "test"}
    }
    if steer is not None:
        responders["SteerChoice"] = lambda _m: steer
    return ScriptedModel(responders=responders)


# --------------------------------------------------------------------------- network doubles


class FakeSearch:
    name = "fake"

    def __init__(self, fail: bool = False):
        self.fail, self.queries, self.countries = fail, [], []

    async def search(
        self,
        query: str,
        num_results: int = 5,
        objective: str | None = None,
        country: str | None = None,
    ) -> list[SearchResult]:
        self.queries.append(query)
        self.countries.append(country)
        if self.fail:
            raise ConnectionError("search backend down")
        return [
            SearchResult(
                title="Lisbon guide", url=SEARCH_URL, published_date="2026-05-01", text="..."
            ),
            SearchResult(title="Other", url="https://example.com/other", text="..."),
        ]


def _daily(start: str, end: str, hi: float = 24.0, lo: float = 14.0, rain: float = 0.0) -> dict:
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    days = [(s + timedelta(days=i)).isoformat() for i in range((e - s).days + 1)]
    n = len(days)
    return {
        "daily": {
            "time": days,
            "temperature_2m_max": [hi] * n,
            "temperature_2m_min": [lo] * n,
            "precipitation_sum": [rain] * n,
            "precipitation_probability_max": [10] * n,
        }
    }


def mock_transport(
    *, geocode_empty: bool = False, weather_down: bool = False
) -> httpx.MockTransport:
    coords = {
        "Belem Tower": (38.6916, -9.2160),
        "Jeronimos Monastery": (38.6979, -9.2068),
        "Alfama Walk": (38.7110, -9.1290),
        "Sao Jorge Castle": (38.7139, -9.1334),
        "Time Out Market": (38.7068, -9.1456),
        "Oceanarium": (38.7634, -9.0938),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        host, params = request.url.host, dict(request.url.params)
        if host == "geocoding-api.open-meteo.com":
            if geocode_empty:
                return httpx.Response(200, json={})
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "name": "Lisbon",
                            "latitude": 38.72,
                            "longitude": -9.14,
                            "country": "Portugal",
                        }
                    ]
                },
            )
        if host in ("api.open-meteo.com", "archive-api.open-meteo.com"):
            if weather_down:
                return httpx.Response(503, json={"error": True})
            return httpx.Response(200, json=_daily(params["start_date"], params["end_date"]))
        if host == "api.frankfurter.dev":
            if params["symbols"] == "XXX":
                return httpx.Response(404, json={"message": "not found"})
            rates = {params["symbols"]: 0.9}
            return httpx.Response(
                200, json={"base": params["base"], "date": "2026-10-09", "rates": rates}
            )
        if host == "nominatim.test":
            name = params["q"].split(",")[0]
            hit = coords.get(name)
            return httpx.Response(
                200, json=[{"lat": str(hit[0]), "lon": str(hit[1])}] if hit else []
            )
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def make_deps(*, search: FakeSearch | None = None, **transport_kwargs: bool) -> Deps:
    settings = Settings(
        _env_file=None,
        nominatim_url="https://nominatim.test",
        retry_initial_delay=0.0,
        max_revisions=3,
    )
    client = httpx.AsyncClient(transport=mock_transport(**transport_kwargs))
    geocoder = Geocoder(client, settings.nominatim_url, min_interval=0)
    return Deps(client=client, settings=settings, search=search or FakeSearch(), geocoder=geocoder)


def make_request(**overrides: Any) -> TravelRequest:
    today = date.today()
    data = {
        "destination": "Lisbon",
        "start_date": today + timedelta(days=5),
        "end_date": today + timedelta(days=8),
        "budget_min": 1500,
        "budget_max": 2000,
        "travelers": 2,
        "interests": ["food", "history"],
        "nationality": "India",
        "currency": "USD",
    }
    return TravelRequest(**{**data, **overrides})
