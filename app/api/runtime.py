"""Everything the routes need, built once at startup and kept on `app.state`."""

import logging
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import START
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from app.api.status import derive_status
from app.core.config import Settings
from app.core.runs import RunManager
from app.graph import steering
from app.graph.state import SteerScope, TravelRequest
from app.graph.steering import SCOPE_NODE, Action, SteeringDesk

log = logging.getLogger(__name__)


class SteerError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class SteerOutcome(BaseModel):
    action: Action
    scope: SteerScope
    reason: str
    # Graph node the run was re-started from (rewind and restart only).
    restarted_from: str | None = None
    request: TravelRequest | None = None


@dataclass
class Runtime:
    graph: CompiledStateGraph
    saver: BaseCheckpointSaver
    runs: RunManager
    settings: Settings
    desk: SteeringDesk = field(default_factory=SteeringDesk)
    classifier: BaseChatModel | None = None
    classifier_kwargs: dict = field(default_factory=dict)

    @staticmethod
    def config(plan_id: str) -> dict:
        # thread_id == plan_id: one identifier, no mapping table.
        return {"configurable": {"thread_id": plan_id}}

    def start_run(self, plan_id: str, input_: object, from_config: dict | None = None) -> None:
        """Launch a detached run. The caller must hold the claim from `runs.try_claim`.

        `from_config` (a snapshot's config, with a checkpoint_id) runs from that checkpoint instead
        of the latest one: LangGraph forks the thread there and the new branch becomes current.
        """
        config = self.config(plan_id)

        async def work():
            await self.graph.ainvoke(input_, from_config or config, durability="sync")

        async def on_error(exc: Exception):
            # Last resort for failures outside the guarded nodes (e.g. the checkpointer itself).
            await self.graph.aupdate_state(
                config, {"error": f"{type(exc).__name__}: {exc}"}, as_node="validate"
            )

        self.runs.launch(plan_id, work, on_error)

    async def steer(self, plan_id: str, text: str, scope: SteerScope | None = None) -> SteerOutcome:
        """Apply mid-run guidance to a running plan. See `app.graph.steering` for the rules."""
        config = self.config(plan_id)
        snap = await self.graph.aget_state(config)
        request: TravelRequest = snap.values["request"]

        changes, reason = None, "scope given by the client"
        if scope is None:
            scope, changes, reason = await self._classify(plan_id, text, request)
        new_request = None
        if scope == "request":
            try:
                new_request = steering.apply_changes(request, changes)
            except ValueError as exc:
                raise SteerError(422, f"That change makes the trip request invalid: {exc}") from exc
            if new_request is None:
                scope, reason = "research", f"{reason} (no concrete change to the request found)"

        points = await self._rewind_points(plan_id)

        # Decide and post without awaiting in between, so the run cannot change stage meanwhile.
        if not self.runs.busy(plan_id):
            raise SteerError(409, "The plan is not being built right now.")
        stage = self.desk.stage(plan_id)
        if stage == "review":
            raise SteerError(409, "The draft is ready: request changes from the review instead.")
        action = steering.decide(scope, stage)
        if action == "rewind" and SCOPE_NODE[scope] not in points:
            # The stage did not run in this revision cycle (a rejection skipped it): the running
            # planner reads research-scoped guidance too.
            action = "live"
        self.desk.post(plan_id, steering.new_note(text, scope))
        if action in ("queued", "live"):
            return SteerOutcome(action=action, scope=scope, reason=reason)

        await self.runs.cancel(plan_id)
        if not self.runs.try_claim(plan_id):
            # Another request restarted the run meanwhile; it will pick the note up.
            return SteerOutcome(action="queued", scope=scope, reason=reason)
        try:
            latest = await self.graph.aget_state(config)
            if action == "restart":
                self.desk.requeue(plan_id)
                await self.graph.aupdate_state(
                    config,
                    {"request": new_request, "research": None, "draft": None},
                    as_node=START,
                )
                self.desk.enter(plan_id, "validate")
                restarted_from, from_config = "validate", None
            else:
                restarted_from = SCOPE_NODE[scope]
                from_config = points[restarted_from]
                fork = await self.graph.aget_state(from_config)
                kept = {n.id for n in fork.values.get("steering") or []}
                lost = [n for n in latest.values.get("steering") or [] if n.id not in kept]
                self.desk.requeue(plan_id, lost)
                self.desk.enter(plan_id, scope)
        except BaseException:
            self.runs.release(plan_id)
            raise
        log.info("steer %s: %s from %s (%s)", plan_id, action, restarted_from, reason)
        self.start_run(plan_id, None, from_config=from_config)
        return SteerOutcome(
            action=action,
            scope=scope,
            reason=reason,
            restarted_from=restarted_from,
            request=new_request,
        )

    async def _classify(self, plan_id: str, text: str, request: TravelRequest):
        try:
            if self.classifier is None:
                raise RuntimeError("no classifier configured")
            choice = await steering.classify(self.classifier, self.classifier_kwargs, text, request)
            return choice.scope, choice.request_changes, choice.reason
        except Exception as exc:  # noqa: BLE001
            log.warning("steer classifier failed: %s", exc)
            # Hand it to whichever agent is working; it can act on most guidance.
            scope = "planner" if self.desk.stage(plan_id) == "planner" else "research"
            return scope, None, "classifier unavailable; sent to the agent that is working"

    async def _rewind_points(self, plan_id: str) -> dict[str, dict]:
        """Configs of the checkpoints just before each agent node ran in the current revision
        cycle, walking the current branch back from the latest checkpoint."""
        snap = await self.graph.aget_state(self.config(plan_id))
        reviews = len(snap.values.get("review_history") or [])
        points: dict[str, dict] = {}
        while snap is not None and len(snap.values.get("review_history") or []) == reviews:
            node = snap.next[0] if snap.next else None
            if node in ("research_agent", "planner_agent") and node not in points:
                points[node] = snap.config
            snap = await self.graph.aget_state(snap.parent_config) if snap.parent_config else None
        return points

    async def last_good_checkpoint(self, plan_id: str) -> dict | None:
        """Config of the newest checkpoint that still had work to do and no error: the state just
        before the step that failed. None if there is nothing to retry from."""
        async for snap in self.graph.aget_state_history(self.config(plan_id)):
            if snap.next and not snap.values.get("error"):
                return snap.config
        return None

    async def recover_orphans(self) -> list[str]:
        """Re-invoke plans whose run was cut off (crash, deploy, restart).

        Called at startup, when no run can legitimately be in flight in this process, so any
        thread whose checkpoint says "running" is an orphan. State is checkpointed per super-step
        (`durability="sync"`), so it continues from the last completed step.
        """
        thread_ids: list[str] = []
        async for tup in self.saver.alist(None):
            tid = tup.config["configurable"]["thread_id"]
            if tid not in thread_ids:
                thread_ids.append(tid)
        resumed = []
        for tid in thread_ids:
            snap = await self.graph.aget_state(self.config(tid))
            if derive_status(snap) == "running" and self.runs.try_claim(tid):
                log.info("resuming orphaned plan %s at %s", tid, snap.next)
                self.start_run(tid, None)
                resumed.append(tid)
        return resumed
