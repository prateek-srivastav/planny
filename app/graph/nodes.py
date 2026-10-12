"""Graph nodes.

`human_review` is the only node that pauses. LangGraph re-runs a node from its first line when it
resumes, so that node holds nothing but the `interrupt()` call and a pure routing decision. Side
effects belong in `finalize`, after the pause.
"""

import logging
import warnings
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from app.core.llm import hint_for
from app.core.markets import notes_for
from app.core.model_pool import ModelPool
from app.graph import quality
from app.graph.agents.planner import build_planner_agent
from app.graph.agents.research import build_research_agent
from app.graph.state import (
    DraftBody,
    FinalPlan,
    ItineraryDraft,
    PlanState,
    ResearchBody,
    ResearchBrief,
    ReviewDecision,
    SteerNote,
    SteerScope,
)
from app.graph.steering import (
    PLANNER_SCOPES,
    RESEARCH_SCOPES,
    SteeringDesk,
    guidance_block,
    listening,
    live_message,
)
from app.tools import budget as budget_lib
from app.tools import collector
from app.tools.registry import Deps, build_planner_tools, build_research_tools

log = logging.getLogger(__name__)

STAGES = {
    "validate": "validating request",
    "research_agent": "researching destination",
    "planner_agent": "drafting itinerary",
    # Still status "running" here: the pause is only recorded a moment after this checkpoint.
    "human_review": "preparing review",
    "apply_patch": "applying edits",
    "route_feedback": "interpreting feedback",
    "finalize": "finalizing",
}


# Extra agent rounds for guidance that arrives after the agent's last model call.
MAX_LATE_STEER_ROUNDS = 2

# The agents run without a checkpointer. LangGraph (1.2.x) crashes if such a loop inherits
# durability="sync" from the parent run, so say explicitly that this one does not persist.
AGENT_DURABILITY = "async"
warnings.filterwarnings("ignore", message="`durability` has no effect", category=UserWarning)


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


class RouteChoice(BaseModel):
    target: Literal["research", "planner"]
    reason: str


def _guard(
    node: Callable[[PlanState, RunnableConfig], Awaitable[Command]],
) -> Callable[[PlanState, RunnableConfig], Awaitable[Command]]:
    """Turn an unexpected failure into an `error` in state instead of a crashed run.

    Status is derived from the checkpoint, so a failure has to be *in* the checkpoint to be seen.
    Guarded nodes route with `Command(goto=...)` and have no static outgoing edge: a static edge
    would still fire after `goto=END`.
    """

    async def wrapper(state: PlanState, config: RunnableConfig):
        try:
            return await node(state, config)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            log.exception("node %s failed", node.__name__)
            message = f"{node.__name__}: {type(exc).__name__}: {exc}{hint_for(exc)}"
            return Command(goto=END, update={"error": message})

    wrapper.__name__ = node.__name__
    return wrapper


def _plan_id(config: RunnableConfig) -> str:
    return config["configurable"]["thread_id"]


def _last_rejection(state: PlanState) -> ReviewDecision | None:
    history = state.get("review_history") or []
    if history and history[-1].action == "reject":
        return history[-1]
    return None


def _market(request) -> str | None:
    """ISO code of the traveller's country, if recognised: localises search results."""
    return request.home_country.iso2 if request.home_country else None


def _request_block(state: PlanState) -> str:
    r = state["request"]
    market_notes = notes_for(_market(r))
    return (
        f"Destination: {r.destination}\n"
        f"Dates: {r.start_date} to {r.end_date} ({r.trip_days} days, {r.nights} nights)\n"
        f"Travellers: {r.travelers}\n"
        f"Budget: {r.budget_min:.0f}-{r.budget_max:.0f} {r.currency} for the whole party "
        f"(treat {r.budget_max:.0f} as the cap)\n"
        f"Traveller nationality (passport): {r.nationality or 'not stated'}\n"
        f"Travelling from: {r.origin or 'not stated (plan and budget the stay only)'}\n"
        f"Interests: {', '.join(r.interests) or 'none stated'}\n"
        f"Notes: {r.notes or 'none'}" + (f"\nMarket notes: {market_notes}" if market_notes else "")
    )


def build_nodes(
    model: BaseChatModel,
    classifier: BaseChatModel,
    deps: Deps,
    force_tool_output: bool = True,
    classifier_kwargs: dict | None = None,
    model_pool: ModelPool | None = None,
    steering: SteeringDesk | None = None,
) -> dict:
    desk = steering or SteeringDesk()
    retry_delay = deps.settings.retry_initial_delay
    research_agent = build_research_agent(
        model,
        build_research_tools(deps),
        retry_delay,
        force_tool_output,
        model_pool,
        deps.settings.max_agent_model_calls,
        deps.settings.max_searches,
    )
    planner_agent = build_planner_agent(
        model,
        build_planner_tools(deps),
        retry_delay,
        force_tool_output,
        model_pool,
        deps.settings.max_agent_model_calls,
    )
    max_revisions = deps.settings.max_revisions

    async def run_agent(agent, prompt: str, plan_id: str, scopes: frozenset[SteerScope]) -> dict:
        """Run `agent`, delivering steering live; give it another round for late guidance."""
        with listening(desk, plan_id, scopes):
            result = await agent.ainvoke(
                {"messages": [("user", prompt)]}, durability=AGENT_DURABILITY
            )
            for _ in range(MAX_LATE_STEER_ROUNDS):
                late = desk.take(plan_id, scopes)
                if not late:
                    break
                result = await agent.ainvoke(
                    {"messages": [*result["messages"], live_message(late)]},
                    durability=AGENT_DURABILITY,
                )
        return result

    def guidance(state: PlanState, plan_id: str, scopes: frozenset[SteerScope]) -> str:
        """Prompt section with all guidance so far, consuming what is waiting for this stage."""
        fresh: list[SteerNote] = desk.take(plan_id, scopes)
        return guidance_block([*(state.get("steering") or []), *fresh])

    # ------------------------------------------------------------------ validate

    async def validate(
        state: PlanState, config: RunnableConfig
    ) -> Command[Literal["research_agent"]]:
        desk.enter(_plan_id(config), "validate")
        request = state["request"]
        try:
            place = await deps.geocoder.destination(request.destination)
        except Exception as exc:  # noqa: BLE001 - a flaky geocoder must not block planning
            return Command(
                goto="research_agent",
                update={"degraded": [f"destination check skipped: {exc}"]},
            )
        if place is None:
            return Command(
                goto=END,
                update={"error": f"Could not find a destination matching '{request.destination}'."},
            )
        return Command(goto="research_agent")

    # ------------------------------------------------------------------ research

    @_guard
    async def research_node(
        state: PlanState, config: RunnableConfig
    ) -> Command[Literal["planner_agent"]]:
        plan_id = _plan_id(config)
        desk.enter(plan_id, "research")
        request = state["request"]
        prompt = _request_block(state) + guidance(state, plan_id, RESEARCH_SCOPES)
        rejection = _last_rejection(state)
        if rejection and state.get("research"):
            prompt += (
                "\n\nA reviewer rejected the previous plan. Their feedback:\n"
                f"{rejection.feedback}\n\nPrevious research overview:\n"
                f"{state['research'].overview}\nPrevious attractions: "
                f"{', '.join(a.name for a in state['research'].attractions)}\n"
                "Research again with this feedback in mind."
            )
        prompt += (
            f"\n\nCall destination_conditions with {request.start_date} to {request.end_date}."
            f" The budget currency is {request.currency}."
        )

        with collector.collecting(search_country=_market(request)) as run:
            result = await run_agent(research_agent, prompt, plan_id, RESEARCH_SCOPES)
        body: ResearchBody = result["structured_response"]

        degraded = list(run.degraded)
        if run.conditions is None and not any("destination_conditions" in d for d in degraded):
            degraded.append("destination_conditions: no weather data was retrieved")
        if not run.sources and not any("web_search" in d for d in degraded):
            degraded.append("web_search: no search results were retrieved")
        brief = ResearchBrief(
            **body.model_dump(),
            sources=[run.sources[u] for u in dict.fromkeys(body.source_urls) if u in run.sources],
            conditions=run.conditions,
            exchange_rate=run.exchange_rate,
        )
        return Command(
            goto="planner_agent",
            update={"research": brief, "degraded": degraded, "steering": desk.commit(plan_id)},
        )

    # ------------------------------------------------------------------ planner

    @_guard
    async def planner_node(
        state: PlanState, config: RunnableConfig
    ) -> Command[Literal["human_review"]]:
        plan_id = _plan_id(config)
        desk.enter(plan_id, "planner")
        request, brief = state["request"], state["research"]
        unavailable = _dedupe(state.get("degraded") or [])
        prompt = (
            _request_block(state)
            + f"\n\nDestination price level: {brief.cost_tier}\nOverview: {brief.overview}\n"
            + "Candidate attractions:\n"
            + "\n".join(
                f"- {a.name} ({a.category}, {'indoor' if a.indoor else 'outdoor'}, "
                f"~{a.est_cost_per_person:.0f}/person, area: {a.area or 'n/a'}, "
                f"url: {a.source_url or 'n/a'}): {a.description}"
                for a in brief.attractions
            )
            + f"\nFood: {brief.food_notes}\nTips: {'; '.join(brief.local_tips)}\n"
        )
        if request.origin:
            prompt += _getting_there_block(request, brief)
        if brief.exchange_rate:
            fx = brief.exchange_rate
            prompt += (
                f"\nExchange rate ({fx.date}): 1 {fx.base} = {fx.rate} {fx.quote}. "
                f"Local prices are in {fx.quote}; every cost you output must be in "
                f"{request.currency}.\n"
            )
        elif brief.local_currency and brief.local_currency != request.currency:
            prompt += (
                f"\nLocal currency is {brief.local_currency} but no exchange rate is available; "
                "say in caveats that converted prices are rough.\n"
            )
        if brief.conditions:
            c = brief.conditions
            prompt += (
                f"\nWeather ({c.basis}): avg high {c.avg_high_c}C, avg low {c.avg_low_c}C, "
                f"rainy-day fraction {c.rainy_day_fraction}. {c.note}\n"
            )
            if c.daily:
                prompt += (
                    "Daily: "
                    + "; ".join(
                        f"{d.date} {d.temp_min_c}-{d.temp_max_c}C rain {d.precipitation_mm}mm"
                        for d in c.daily
                    )
                    + "\n"
                )
        if unavailable:
            prompt += f"\nData that was unavailable (mention in caveats): {unavailable}\n"
        rejection = _last_rejection(state)
        if rejection and state.get("draft"):
            prompt += (
                "\nA reviewer rejected the previous draft. Feedback:\n"
                f"{rejection.feedback}\n\nPrevious draft (revise it):\n"
                f"{state['draft'].model_dump_json(exclude={'budget', 'sources'})}\n"
            )
        prompt += guidance(state, plan_id, PLANNER_SCOPES)
        prompt += f"\nPlan exactly {request.trip_days} days."

        with collector.collecting(search_country=_market(request)) as run:
            result = await run_agent(planner_agent, prompt, plan_id, PLANNER_SCOPES)
        body: DraftBody = result["structured_response"]

        draft = ItineraryDraft(
            **body.model_dump(),
            sources=brief.sources,
            entry_requirements=brief.entry_requirements or None,
            exchange_rate=brief.exchange_rate,
        )
        draft = _normalise_days(draft, state)
        if not request.origin:
            draft.intercity = None
        moved = quality.tidy_logistics(draft)
        dropped = quality.drop_intercity_duplicates(draft, request)
        _refresh(draft, request)
        for note in [*moved, *dropped]:
            log.info("draft cleanup: %s", note)
        steered = desk.commit(plan_id)
        desk.enter(plan_id, "review")
        return Command(
            goto="human_review",
            update={"draft": draft, "degraded": run.degraded, "steering": steered},
        )

    # ------------------------------------------------------------------ human review

    def human_review(
        state: PlanState,
    ) -> Command[Literal["finalize", "apply_patch", "route_feedback"]]:
        raw = interrupt(
            {
                "draft": state["draft"].model_dump(mode="json"),
                "revisions_used": state.get("revision_count", 0),
                "revisions_left": max_revisions - state.get("revision_count", 0),
                "degraded": _dedupe(state.get("degraded") or []),
            },
            response_schema=ReviewDecision,
        )
        decision = ReviewDecision.model_validate(raw)
        update: dict = {"review_history": [decision]}
        if decision.action == "approve":
            return Command(goto="finalize", update=update)
        if decision.action == "modify":
            return Command(goto="apply_patch", update=update)
        used = state.get("revision_count", 0) + 1
        update["revision_count"] = used
        if used >= max_revisions:
            update["needs_human_followup"] = True
            return Command(goto="finalize", update=update)
        return Command(goto="route_feedback", update=update)

    # ------------------------------------------------------------------ apply_patch

    def apply_patch(state: PlanState, config: RunnableConfig) -> Command[Literal["human_review"]]:
        desk.enter(_plan_id(config), "review")
        decision = state["review_history"][-1]
        draft = state["draft"].model_copy(deep=True)
        by_day = {d.day: d for d in draft.days}
        skipped = []
        for patch in decision.patches or []:
            day = by_day.get(patch.day)
            if day is None:
                skipped.append(f"patch ignored: no day {patch.day}")
                continue
            setattr(day, patch.slot, patch.value if patch.op == "replace" else None)
        _refresh(draft, state["request"])
        return Command(goto="human_review", update={"draft": draft, "degraded": skipped})

    # ------------------------------------------------------------------ route_feedback

    async def route_feedback(
        state: PlanState, config: RunnableConfig
    ) -> Command[Literal["research_agent", "planner_agent"]]:
        desk.enter(_plan_id(config), "routing")
        decision = state["review_history"][-1]
        if decision.route_hint:
            target, extra = decision.route_hint, []
        else:
            try:
                router = classifier.with_structured_output(RouteChoice, **(classifier_kwargs or {}))
                choice = await router.ainvoke(
                    "A traveller rejected a draft itinerary. Decide where the problem lies.\n"
                    "- research: the facts or the destination choice are wrong or missing "
                    "(wrong city area, missing attractions, bad prices, wrong interests).\n"
                    "- planner: the facts are fine but the arrangement is wrong (pacing, order, "
                    "budget split, too much travel, too many/few activities).\n"
                    f"Feedback: {decision.feedback}"
                )
                target, extra = choice.target, []
            except Exception as exc:  # noqa: BLE001
                log.warning("feedback classifier failed: %s", exc)
                target = "planner"
                extra = ["feedback routing defaulted to planner (classifier unavailable)"]
        return Command(
            goto="research_agent" if target == "research" else "planner_agent",
            update={"degraded": extra},
        )

    # ------------------------------------------------------------------ finalize

    def finalize(state: PlanState, config: RunnableConfig) -> dict:
        desk.forget(_plan_id(config))
        followup = bool(state.get("needs_human_followup"))
        final = FinalPlan(
            itinerary=state["draft"],
            status="needs_human_followup" if followup else "approved",
            revisions=state.get("revision_count", 0),
            degraded=_dedupe(state.get("degraded") or []),
            finalized_at=datetime.now(UTC),
        )
        log.info("plan finalized: status=%s revisions=%s", final.status, final.revisions)
        return {"final": final}

    return {
        "validate": validate,
        "research_agent": research_node,
        "planner_agent": planner_node,
        "human_review": human_review,
        "apply_patch": apply_patch,
        "route_feedback": route_feedback,
        "finalize": finalize,
    }


MAX_TRANSPORT_OPTIONS = 3


def _getting_there_block(request, brief: ResearchBrief) -> str:
    if not brief.getting_there:
        return (
            f"\nNo researched options for getting there from {request.origin}: estimate the "
            "cheapest sensible round trip and say in caveats that it is an estimate.\n"
        )
    lines = []
    for option in brief.getting_there[:MAX_TRANSPORT_OPTIONS]:
        line = (
            f"- {option.mode}: {option.summary}, "
            f"{option.cost_per_person_round_trip:.0f} {request.currency}"
        )
        if option.duration_hours:
            line += f", ~{option.duration_hours:g}h each way"
        if option.source_url:
            line += f", url: {option.source_url}"
        lines.append(line)
    return (
        f"\nGetting there from {request.origin} (per person, round trip):\n"
        + "\n".join(lines)
        + "\n"
    )


def _refresh(draft: ItineraryDraft, request) -> None:
    """Recompute everything derived from the draft's contents. Call after any change to it."""
    draft.budget = budget_lib.summarize(draft, request)
    draft.quality_issues = quality.find_issues(draft, request)


def _normalise_days(draft: ItineraryDraft, state: PlanState) -> ItineraryDraft:
    """Dates and numbering are facts, not creative output: set them from the request."""
    request = state["request"]
    days = draft.days[: request.trip_days]
    for i, day in enumerate(days):
        day.day = i + 1
        day.date = request.start_date + timedelta(days=i)
    if len(draft.days) != request.trip_days:
        draft.caveats.append(
            f"Planner returned {len(draft.days)} day(s) for a {request.trip_days}-day trip."
        )
    draft.days = days
    return draft
