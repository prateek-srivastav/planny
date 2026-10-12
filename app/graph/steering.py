"""Mid-run steering: guidance the traveller sends while a plan is still being built.

Each note is classified by the earliest stage it affects (its scope), then handled against the
stage the run is in:

* scope after the current stage  -> queued; that stage reads it when it starts
* scope equal to the current one -> live; the running agent sees it before its next model call
* scope before the current stage -> rewind; the run is cancelled and re-run from the checkpoint
  taken just before the scoped stage, with the note
* scope "request"                -> the request itself is edited and the plan restarts

Notes travel through `SteeringDesk`, an in-process inbox, because a running node cannot see writes
made to the checkpoint after it started. A node persists the notes it consumed into
`PlanState.steering` when it finishes, so later runs (rejections, retries) keep honouring them.
Notes that are queued but not yet consumed live only in memory and are lost on restart.
"""

import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from app.graph.state import RequestChanges, SteerChoice, SteerNote, SteerScope, TravelRequest

Stage = Literal["validate", "routing", "research", "planner", "review"]
Action = Literal["queued", "live", "rewind", "restart"]

# Position of each stage in the pipeline. "routing" (classifying a rejection) happens before the
# next agent run, so it ranks with "validate".
STAGE_ORDER: dict[str, int] = {
    "validate": 0,
    "routing": 0,
    "research": 1,
    "planner": 2,
    "review": 3,
}
SCOPE_ORDER: dict[SteerScope, int] = {"request": 0, "research": 1, "planner": 2}
# Graph node that starts each scope's stage.
SCOPE_NODE: dict[SteerScope, str] = {
    "request": "validate",
    "research": "research_agent",
    "planner": "planner_agent",
}
# Which notes each agent consumes. The planner also takes leftovers aimed at research: once
# research has finished it is the only agent left to act on them.
RESEARCH_SCOPES: frozenset[SteerScope] = frozenset({"request", "research"})
PLANNER_SCOPES: frozenset[SteerScope] = frozenset({"request", "research", "planner"})


def new_note(text: str, scope: SteerScope) -> SteerNote:
    return SteerNote(id=uuid.uuid4().hex, text=text.strip(), scope=scope, at=datetime.now(UTC))


def decide(scope: SteerScope, stage: str | None) -> Action:
    """How a note with `scope` is applied to a run currently in `stage` (None: not started)."""
    if scope == "request":
        return "restart"
    current = STAGE_ORDER.get(stage or "validate", 0)
    target = SCOPE_ORDER[scope]
    if target > current:
        return "queued"
    if target == current:
        return "live"
    return "rewind"


@dataclass
class _Inbox:
    stage: str | None = None
    pending: list[SteerNote] = field(default_factory=list)
    # Taken by the running node but not yet persisted to the checkpoint.
    delivered: list[SteerNote] = field(default_factory=list)


class SteeringDesk:
    """Per-plan inbox of steering notes, plus the stage each plan's run is in.

    All methods are synchronous: under asyncio, a route handler that reads the stage and posts a
    note cannot interleave with a node that takes notes.
    """

    def __init__(self) -> None:
        self._boxes: dict[str, _Inbox] = {}

    def _box(self, plan_id: str) -> _Inbox:
        return self._boxes.setdefault(plan_id, _Inbox())

    def enter(self, plan_id: str, stage: Stage) -> None:
        self._box(plan_id).stage = stage

    def stage(self, plan_id: str) -> str | None:
        box = self._boxes.get(plan_id)
        return box.stage if box else None

    def post(self, plan_id: str, note: SteerNote) -> None:
        self._box(plan_id).pending.append(note)

    def pending(self, plan_id: str) -> list[SteerNote]:
        box = self._boxes.get(plan_id)
        return list(box.pending) if box else []

    def take(self, plan_id: str, scopes: Iterable[SteerScope]) -> list[SteerNote]:
        """Remove and return pending notes in `scopes`; they count as delivered."""
        box = self._boxes.get(plan_id)
        if box is None:
            return []
        wanted = set(scopes)
        taken = [n for n in box.pending if n.scope in wanted]
        box.pending = [n for n in box.pending if n.scope not in wanted]
        box.delivered.extend(taken)
        return taken

    def commit(self, plan_id: str) -> list[SteerNote]:
        """Return the delivered notes for the node to persist, and forget them here."""
        box = self._boxes.get(plan_id)
        if box is None:
            return []
        done, box.delivered = box.delivered, []
        return done

    def requeue(self, plan_id: str, notes: Iterable[SteerNote] = ()) -> None:
        """Make delivered notes (and `notes`) pending again, ahead of newer ones.

        Used when a run is cancelled for a rewind: what the cancelled node consumed was never
        persisted, and what earlier nodes persisted may be on the discarded branch.
        """
        box = self._box(plan_id)
        seen = {n.id for n in box.pending}
        back = [n for n in [*notes, *box.delivered] if n.id not in seen]
        unique = list({n.id: n for n in back}.values())
        box.pending = sorted(unique, key=lambda n: n.at) + box.pending
        box.delivered = []

    def forget(self, plan_id: str) -> None:
        self._boxes.pop(plan_id, None)


# --------------------------------------------------------------------------- live delivery


@dataclass
class _Listener:
    desk: SteeringDesk
    plan_id: str
    scopes: frozenset[SteerScope]


_listener: ContextVar[_Listener | None] = ContextVar("steering_listener", default=None)


@contextmanager
def listening(desk: SteeringDesk, plan_id: str, scopes: frozenset[SteerScope]) -> Iterator[None]:
    """Let `SteeringMiddleware` deliver notes in `scopes` to agents run inside this block."""
    token = _listener.set(_Listener(desk, plan_id, scopes))
    try:
        yield
    finally:
        _listener.reset(token)


def take_live() -> list[SteerNote]:
    listener = _listener.get()
    if listener is None:
        return []
    return listener.desk.take(listener.plan_id, listener.scopes)


def live_message(notes: list[SteerNote]) -> HumanMessage:
    lines = "\n".join(f"- {n.text}" for n in notes)
    return HumanMessage(
        "The traveller sent new guidance while you were working:\n"
        f"{lines}\n"
        "It overrides the original request where they conflict. Adjust your work to follow it: "
        "redo only the parts it invalidates and keep the rest."
    )


class SteeringMiddleware(AgentMiddleware):
    """Before each model call, hand the agent any guidance that arrived since the last one."""

    async def abefore_model(self, state, runtime) -> dict | None:
        notes = take_live()
        if not notes:
            return None
        return {"messages": [live_message(notes)]}


def guidance_block(notes: list[SteerNote]) -> str:
    """Prompt section listing guidance already given, oldest first."""
    if not notes:
        return ""
    unique = list({n.id: n for n in notes}.values())
    lines = "\n".join(f"- {n.text}" for n in unique)
    return (
        "\n\nGuidance the traveller gave while this plan was being built (newest last; it "
        f"overrides the request above where they conflict):\n{lines}"
    )


# --------------------------------------------------------------------------- classification

_CLASSIFY_PROMPT = """A traveller sent guidance while an AI was building their trip plan. \
Decide the earliest stage it affects; that stage and everything after it will be redone.

- request: it changes a hard fact of the trip: destination, dates, trip length, budget, number \
of travellers, or the city they travel from. Fill request_changes with only the changed fields \
(ISO dates; budgets in {currency}); leave the rest null.
- research: the facts gathered need to change: different kinds of places, areas, food, \
accommodation style, accessibility or dietary needs, a different way of getting there.
- planner: the facts are fine but the arrangement should change: pacing, order, free time, which \
day does what, splitting the budget differently, fewer or more activities.

If unsure between research and planner, choose research. Set request_changes to null unless the \
scope is request.

Current request: {request}
Guidance: {text}"""


async def classify(
    classifier: BaseChatModel, kwargs: dict, text: str, request: TravelRequest
) -> SteerChoice:
    router = classifier.with_structured_output(SteerChoice, **kwargs)
    summary = (
        f"{request.destination}, {request.start_date} to {request.end_date}, "
        f"{request.travelers} traveller(s), budget {request.budget_min:.0f}-"
        f"{request.budget_max:.0f} {request.currency}, from {request.origin or 'not stated'}, "
        f"interests: {', '.join(request.interests) or 'none'}"
    )
    return await router.ainvoke(
        _CLASSIFY_PROMPT.format(currency=request.currency, request=summary, text=text)
    )


def apply_changes(request: TravelRequest, changes: RequestChanges | None) -> TravelRequest | None:
    """The request with `changes` applied, or None if nothing changes. Raises ValueError if the
    result is not a valid request (e.g. dates in the past)."""
    if changes is None:
        return None
    edits = {k: v for k, v in changes.model_dump().items() if v is not None}
    if not edits:
        return None
    updated = TravelRequest.model_validate({**request.model_dump(), **edits})
    return None if updated == request else updated
