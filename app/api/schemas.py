from typing import Any, Literal

from pydantic import BaseModel, Field

from app.graph.state import (
    FinalPlan,
    ItineraryDraft,
    ReviewDecision,
    SteerNote,
    SteerScope,
    TravelRequest,
)

__all__ = ["FinalPlan", "ReviewDecision", "TravelRequest"]

PlanStatus = Literal["running", "awaiting_review", "completed", "failed"]


class PlanCreated(BaseModel):
    plan_id: str
    status: PlanStatus
    links: dict[str, str]


class ReviewPrompt(BaseModel):
    """What the reviewer is being asked, straight from the paused `interrupt()`."""

    payload: dict[str, Any]
    response_schema: dict[str, Any] | None = Field(
        description="JSON Schema for the body of POST /plan/{id}/review"
    )


class PlanView(BaseModel):
    plan_id: str
    status: PlanStatus
    stage: str
    checkpoint_id: str | None
    request: TravelRequest | None
    draft: ItineraryDraft | None
    review: ReviewPrompt | None
    revisions_used: int
    degraded: list[str]
    needs_human_followup: bool
    error: str | None
    steering: list[SteerNote] = Field(
        default_factory=list, description="Guidance sent while planning, oldest first"
    )
    links: dict[str, str]


class ReviewBody(ReviewDecision):
    checkpoint_id: str | None = Field(
        default=None,
        description="Optional. The checkpoint_id from GET /plan/{id}. If it is stale, you get 409.",
    )


class ReviewAccepted(BaseModel):
    plan_id: str
    status: PlanStatus
    links: dict[str, str]


class SteerBody(BaseModel):
    message: str = Field(min_length=2, max_length=500)
    scope: SteerScope | None = Field(
        default=None,
        description="Optional: the earliest stage the guidance affects. Omit to let the "
        "classifier decide",
    )


class SteerAccepted(BaseModel):
    plan_id: str
    action: Literal["queued", "live", "rewind", "restart"] = Field(
        description="queued: read when its stage starts. live: the running agent sees it on its "
        "next step. rewind: re-running from an earlier stage. restart: the request changed"
    )
    scope: SteerScope
    reason: str
    restarted_from: str | None
    request: TravelRequest | None = Field(description="The updated request, after a restart")
    links: dict[str, str]
