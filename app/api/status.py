"""Plan status is *derived* from the checkpoint. It is never stored.

A parallel `plans` table with a `status` column would be a second source of truth that can drift
from the real workflow position. The only input that is not in the checkpoint is "is a run task
alive in this process right now", and that is applied as a small overlay by the route handlers.
"""

from typing import Literal

from langgraph.types import StateSnapshot

from app.graph.nodes import STAGES

Status = Literal["not_found", "running", "awaiting_review", "completed", "failed"]


def derive_status(snap: StateSnapshot) -> Status:
    values = snap.values or {}
    if not values and not snap.next:
        return "not_found"
    if values.get("error"):
        return "failed"
    if any(task.interrupts for task in snap.tasks):
        return "awaiting_review"
    if not snap.next:
        return "completed" if values.get("final") else "failed"
    return "running"


def stage_of(snap: StateSnapshot, status: str) -> str:
    if status == "completed":
        return "done"
    if status == "failed":
        return "failed"
    if snap.next:
        return STAGES.get(snap.next[0], snap.next[0])
    return "unknown"


def interrupt_of(snap: StateSnapshot):
    for task in snap.tasks:
        if task.interrupts:
            return task.interrupts[0]
    return None


def checkpoint_id_of(snap: StateSnapshot) -> str | None:
    return (snap.config or {}).get("configurable", {}).get("checkpoint_id")
