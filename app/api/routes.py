import asyncio
import contextlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from langgraph.constants import START
from langgraph.types import Command
from sse_starlette.sse import EventSourceResponse

from app.api.runtime import Runtime, SteerError
from app.api.schemas import (
    FinalPlan,
    PlanCreated,
    PlanView,
    ReviewAccepted,
    ReviewBody,
    ReviewPrompt,
    SteerAccepted,
    SteerBody,
    TravelRequest,
)
from app.api.status import checkpoint_id_of, derive_status, interrupt_of, stage_of
from app.core.countries import COUNTRIES
from app.graph.state import ReviewDecision

router = APIRouter()


def rt_of(request: Request) -> Runtime:
    return request.app.state.rt


def links(plan_id: str) -> dict[str, str]:
    base = f"/plan/{plan_id}"
    return {
        "self": base,
        "review": f"{base}/review",
        "final": f"{base}/final",
        "events": f"{base}/events",
        "history": f"{base}/history",
        "retry": f"{base}/retry",
        "steer": f"{base}/steer",
    }


async def load(rt: Runtime, plan_id: str, overlay: bool = True):
    """Snapshot + status. 404 for an unknown plan."""
    snap = await rt.graph.aget_state(rt.config(plan_id))
    status = derive_status(snap)
    if status == "not_found":
        raise HTTPException(404, f"No plan with id {plan_id}")
    # The one input that isn't in the checkpoint: a resume was accepted a moment ago but its
    # first step hasn't been written yet. Report it as running so clients don't double-submit.
    if (
        overlay
        and status == "awaiting_review"
        and rt.runs.resuming_from(plan_id) == checkpoint_id_of(snap)
    ):
        status = "running"
    return snap, status


@router.get("/", include_in_schema=False)
async def ui():
    """Web client (static HTML, CSS and JS under /static; no build step)."""
    return FileResponse(Path(__file__).resolve().parent.parent / "static" / "index.html")


@router.get("/countries")
async def countries():
    """Countries with a known currency and search market. Unlisted countries are still accepted."""
    return [{"name": c.name, "iso2": c.iso2, "currency": c.currency} for c in COUNTRIES]


@router.get("/healthz")
async def healthz():
    return {"ok": True}


@router.post("/plan", status_code=202, response_model=PlanCreated)
async def create_plan(body: TravelRequest, request: Request):
    """Validate, persist the request, and start planning in the background. Returns immediately.

    Research + planning takes 30-90s, so the HTTP request must not wait for it. Poll
    `GET /plan/{id}` (or follow `/events`) until status is `awaiting_review`.
    """
    rt = rt_of(request)
    plan_id = uuid.uuid4().hex
    config = rt.config(plan_id)
    rt.runs.try_claim(plan_id)
    try:
        # Write the input as the first checkpoint, so GET works the instant this returns.
        await rt.graph.aupdate_state(
            config,
            {"request": body, "revision_count": 0, "review_history": [], "degraded": []},
            as_node=START,
        )
    except Exception:
        rt.runs.release(plan_id)
        raise
    rt.start_run(plan_id, None)  # None: continue from the checkpoint written above
    return PlanCreated(plan_id=plan_id, status="running", links=links(plan_id))


@router.get("/plan/{plan_id}", response_model=PlanView)
async def get_plan(plan_id: str, request: Request):
    rt = rt_of(request)
    snap, status = await load(rt, plan_id)
    values = snap.values
    pending = interrupt_of(snap) if status == "awaiting_review" else None
    return PlanView(
        plan_id=plan_id,
        status=status,
        stage="awaiting review" if pending else stage_of(snap, status),
        checkpoint_id=checkpoint_id_of(snap),
        request=values.get("request"),
        draft=values.get("draft"),
        review=ReviewPrompt(payload=pending.value, response_schema=pending.response_schema)
        if pending
        else None,
        revisions_used=values.get("revision_count", 0),
        degraded=list(dict.fromkeys(values.get("degraded") or [])),
        needs_human_followup=bool(values.get("needs_human_followup")),
        error=values.get("error"),
        steering=[*(values.get("steering") or []), *rt.desk.pending(plan_id)],
        links=links(plan_id),
    )


@router.post("/plan/{plan_id}/review", status_code=202, response_model=ReviewAccepted)
async def review_plan(plan_id: str, body: ReviewBody, request: Request):
    """Approve, reject (with feedback) or modify (with patches). Resumes the paused workflow."""
    rt = rt_of(request)
    if not rt.runs.try_claim(plan_id):
        # The run that just paused the plan may still be winding down (its pause checkpoint is
        # written a few ms before the task ends). Give it a moment before refusing.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(rt.runs.wait(plan_id), timeout=2)
        if not rt.runs.try_claim(plan_id):
            raise HTTPException(409, "A run for this plan is already in progress.")
    try:
        # The claim is held here, so judge the checkpoint itself, without the busy overlay.
        snap, status = await load(rt, plan_id, overlay=False)
        if status != "awaiting_review":
            raise HTTPException(409, f"Plan is '{status}', not awaiting_review.")
        current = checkpoint_id_of(snap)
        if body.checkpoint_id and body.checkpoint_id != current:
            raise HTTPException(
                409, f"Stale checkpoint_id. The plan has moved on; current is {current}."
            )
        decision = ReviewDecision(**body.model_dump(exclude={"checkpoint_id"}))
        if decision.action == "modify":
            days = {d.day for d in snap.values["draft"].days}
            unknown = sorted({p.day for p in decision.patches} - days)
            if unknown:
                raise HTTPException(
                    422, f"Patch refers to day(s) {unknown} that are not in the plan."
                )
    except BaseException:
        rt.runs.release(plan_id)
        raise
    rt.runs.mark_resume(plan_id, current)
    rt.start_run(plan_id, Command(resume=decision.model_dump(mode="json")))
    return ReviewAccepted(plan_id=plan_id, status="running", links=links(plan_id))


@router.post("/plan/{plan_id}/steer", status_code=202, response_model=SteerAccepted)
async def steer_plan(plan_id: str, body: SteerBody, request: Request):
    """Send guidance while the plan is being built, without waiting for the draft.

    The guidance is classified by the earliest stage it affects. Work already done before that
    stage is kept; that stage is redone with the guidance (or receives it live if it is running).
    A change to the trip itself (dates, destination, budget, ...) restarts the plan.
    """
    rt = rt_of(request)
    _, status = await load(rt, plan_id)
    if status != "running":
        raise HTTPException(409, f"Plan is '{status}': guidance can only be sent while it runs.")
    try:
        outcome = await rt.steer(plan_id, body.message, body.scope)
    except SteerError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    return SteerAccepted(plan_id=plan_id, **outcome.model_dump(), links=links(plan_id))


@router.post("/plan/{plan_id}/retry", status_code=202, response_model=ReviewAccepted)
async def retry_plan(plan_id: str, request: Request):
    """Re-run a failed plan from the step that failed (e.g. after fixing an API key).

    Earlier work is kept: if planning failed, research is not repeated. The failed attempt stays
    visible in `/history`.
    """
    rt = rt_of(request)
    if not rt.runs.try_claim(plan_id):
        raise HTTPException(409, "A run for this plan is already in progress.")
    try:
        _, status = await load(rt, plan_id, overlay=False)
        if status != "failed":
            raise HTTPException(409, f"Only failed plans can be retried; this one is '{status}'.")
        resume_from = await rt.last_good_checkpoint(plan_id)
        if resume_from is None:
            raise HTTPException(409, "No earlier step to retry from.")
    except BaseException:
        rt.runs.release(plan_id)
        raise
    rt.start_run(plan_id, None, from_config=resume_from)
    return ReviewAccepted(plan_id=plan_id, status="running", links=links(plan_id))


@router.get("/plan/{plan_id}/final", response_model=FinalPlan)
async def get_final(plan_id: str, request: Request):
    rt = rt_of(request)
    snap, status = await load(rt, plan_id)
    if status != "completed":
        raise HTTPException(
            409,
            {
                "message": "Plan is not finished.",
                "status": status,
                "error": snap.values.get("error"),
            },
        )
    return snap.values["final"]


@router.get("/plan/{plan_id}/history")
async def get_history(plan_id: str, request: Request):
    """The checkpoint timeline, newest first. Each row is a state the plan can be inspected at."""
    rt = rt_of(request)
    await load(rt, plan_id)
    rows = []
    async for snap in rt.graph.aget_state_history(rt.config(plan_id)):
        rows.append(
            {
                "checkpoint_id": checkpoint_id_of(snap),
                "created_at": snap.created_at,
                "step": (snap.metadata or {}).get("step"),
                "source": (snap.metadata or {}).get("source"),
                "next": list(snap.next),
                "revision_count": snap.values.get("revision_count"),
                "has_draft": bool(snap.values.get("draft")),
            }
        )
    return {"plan_id": plan_id, "checkpoints": rows}


@router.get("/plan/{plan_id}/events")
async def plan_events(plan_id: str, request: Request, poll_seconds: float = 1.0):
    """Server-sent progress events. Ends when the plan needs a human, finishes, or fails.

    Implemented by watching the checkpoint, so it works the same for a run started by another
    process and needs no shared event bus.
    """
    rt = rt_of(request)
    await load(rt, plan_id)

    async def stream():
        last = None
        while True:
            if await request.is_disconnected():
                return
            snap, status = await load(rt, plan_id)
            stage = "awaiting review" if status == "awaiting_review" else stage_of(snap, status)
            key = (status, stage, checkpoint_id_of(snap))
            if key != last:
                last = key
                yield {
                    "event": "status",
                    "data": json.dumps(
                        {
                            "plan_id": plan_id,
                            "status": status,
                            "stage": stage,
                            "checkpoint_id": key[2],
                            "at": datetime.now(UTC).isoformat(),
                        }
                    ),
                }
            if status in ("awaiting_review", "completed", "failed"):
                return
            await asyncio.sleep(poll_seconds)

    return EventSourceResponse(stream())
