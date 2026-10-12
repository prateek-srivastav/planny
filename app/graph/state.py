"""Domain models and graph state.

Everything the workflow produces is a typed Pydantic object, not prose. The HITL payload, the
modify patches, the final render and the tests all key off `ItineraryDraft`.
"""

import operator
from datetime import date, datetime
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.countries import Country
from app.core.countries import resolve as resolve_country

Slot = Literal["morning", "afternoon", "evening"]
SLOTS: tuple[Slot, ...] = ("morning", "afternoon", "evening")
MAX_TRIP_DAYS = 30


# --------------------------------------------------------------------------- request


class TravelRequest(BaseModel):
    destination: str = Field(min_length=2, max_length=120)
    start_date: date
    end_date: date
    budget_min: float = Field(gt=0)
    budget_max: float = Field(gt=0)
    currency: str | None = Field(
        default=None,
        min_length=3,
        max_length=3,
        description="ISO 4217 budget currency. Defaults to the nationality's currency, else USD",
    )
    travelers: int = Field(default=1, ge=1, le=20)
    interests: list[str] = Field(default_factory=list, max_length=15)
    nationality: str | None = Field(
        default=None,
        max_length=60,
        description="Passport country of the traveller (e.g. 'India'); drives entry rules",
    )
    origin: str | None = Field(
        default=None,
        max_length=120,
        description="City the party travels from. When set, the round trip to the destination "
        "is researched, planned and budgeted",
    )
    notes: str | None = Field(default=None, max_length=1000)

    @field_validator("destination")
    @classmethod
    def _strip_destination(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("destination must not be empty")
        return v

    @field_validator("nationality", "origin")
    @classmethod
    def _blank_to_none(cls, v: str | None) -> str | None:
        v = (v or "").strip()
        return v or None

    @field_validator("currency")
    @classmethod
    def _upper_currency(cls, v: str | None) -> str | None:
        return v.upper() if v else v

    @model_validator(mode="after")
    def _check_dates_and_budget(self) -> "TravelRequest":
        if not self.currency:
            home = self.home_country
            self.currency = home.currency if home else "USD"
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        if self.start_date < date.today():
            raise ValueError("start_date must not be in the past")
        if self.trip_days > MAX_TRIP_DAYS:
            raise ValueError(f"trip must be at most {MAX_TRIP_DAYS} days")
        if self.budget_max < self.budget_min:
            raise ValueError("budget_max must be >= budget_min")
        return self

    @property
    def home_country(self) -> Country | None:
        """The traveller's country if we recognise it (drives currency and search market)."""
        return resolve_country(self.nationality)

    @property
    def trip_days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    @property
    def nights(self) -> int:
        return (self.end_date - self.start_date).days


# --------------------------------------------------------------------------- research


class Source(BaseModel):
    title: str
    url: str
    published_date: str | None = None


class Attraction(BaseModel):
    name: str
    description: str = ""
    category: str = "sightseeing"
    area: str | None = Field(default=None, description="Neighbourhood or district")
    indoor: bool = False
    est_cost_per_person: float = Field(default=0, ge=0)
    source_url: str | None = None


class DailyCondition(BaseModel):
    date: date
    temp_max_c: float | None = None
    temp_min_c: float | None = None
    precipitation_mm: float | None = None
    precipitation_probability: float | None = None


class ConditionsSummary(BaseModel):
    """Weather for the trip window. `basis` says whether this is a real forecast or climatology."""

    basis: Literal["forecast", "historical_climatology"]
    location: str
    latitude: float
    longitude: float
    avg_high_c: float | None = None
    avg_low_c: float | None = None
    rainy_day_fraction: float | None = None
    avg_daily_precip_mm: float | None = None
    daily: list[DailyCondition] = Field(default_factory=list)
    note: str = ""


class ExchangeRate(BaseModel):
    """1 `base` = `rate` `quote`. Fetched by a tool, never typed by a model."""

    base: str
    quote: str
    rate: float
    date: str | None = None


TransportMode = Literal["flight", "train", "bus", "car", "ferry", "other"]


class TransportOption(BaseModel):
    """One way of getting from the origin to the destination and back."""

    mode: TransportMode
    summary: str = Field(description="Route and service, e.g. 'DEL -> GOI nonstop, ~2.5h'")
    duration_hours: float | None = Field(default=None, gt=0, description="One way, door to door")
    cost_per_person_round_trip: float = Field(ge=0, description="In the budget currency")
    source_url: str | None = None


class ResearchBody(BaseModel):
    """What the research agent produces (structured output)."""

    destination: str
    country: str | None = None
    cost_tier: Literal["budget", "mid", "premium"] = Field(
        description="Overall price level of the destination for a typical visitor"
    )
    overview: str
    attractions: list[Attraction] = Field(min_length=1)
    food_notes: str = ""
    local_tips: list[str] = Field(default_factory=list)
    local_currency: str | None = Field(
        default=None, description="ISO 4217 code of the destination's currency, e.g. EUR"
    )
    getting_there: list[TransportOption] = Field(
        default_factory=list,
        description="1-3 options from the origin to the destination and back; empty if no origin",
    )
    entry_requirements: str = Field(
        default="",
        description="Passport/visa/health entry rules for the traveller's nationality, from "
        "official sources. Say 'not confirmed' if you could not verify them.",
    )
    source_urls: list[str] = Field(
        default_factory=list,
        description="URLs returned by web_search that support the claims above",
    )


class ResearchBrief(ResearchBody):
    """ResearchBody plus data attached deterministically by the node (not by the LLM)."""

    sources: list[Source] = Field(default_factory=list)
    conditions: ConditionsSummary | None = None
    exchange_rate: ExchangeRate | None = None


# --------------------------------------------------------------------------- itinerary


class Activity(BaseModel):
    name: str
    description: str = ""
    location: str | None = None
    lat: float | None = None
    lon: float | None = None
    category: str = "sightseeing"
    indoor: bool = False
    duration_hours: float = Field(default=2.0, gt=0, le=12)
    est_cost: float = Field(default=0, ge=0, description="Total for the whole party")
    source_url: str | None = None


LogisticsCategory = Literal["lodging", "food", "activities", "transit", "other"]
LOGISTICS_CATEGORIES: tuple[LogisticsCategory, ...] = (
    "lodging",
    "food",
    "activities",
    "transit",
    "other",
)


class LogisticsItem(BaseModel):
    """Necessary but not an activity: check-in/out, transfers, pass pick-up, luggage storage."""

    name: str
    est_cost: float = Field(default=0, ge=0, description="Total for the whole party")
    # A plain string in the schema sent to the model: providers that validate tool arguments
    # server-side reject the whole response over one unexpected value. Normalised below.
    category: str = Field(
        default="other",
        description="Budget category its cost is counted under: lodging, food, activities, "
        "transit or other",
    )

    @field_validator("category")
    @classmethod
    def _known_category(cls, v: str) -> str:
        v = v.strip().lower()
        return v if v in LOGISTICS_CATEGORIES else "other"


class DayPlan(BaseModel):
    day: int = Field(ge=1)
    date: date
    theme: str
    morning: Activity | None = None
    afternoon: Activity | None = None
    evening: Activity | None = None
    lodging_cost: float = Field(default=0, ge=0, description="Total for the party, this night")
    food_cost: float = Field(default=0, ge=0, description="Total for the party, this day")
    transit_cost: float = Field(default=0, ge=0, description="Total for the party, this day")
    logistics: list[LogisticsItem] = Field(default_factory=list)
    weather_note: str | None = None


class BudgetSummary(BaseModel):
    """Computed in Python from the day plans. Never written by a model."""

    currency: str
    cap: float
    floor: float
    total_estimated: float
    remaining: float
    within_budget: bool
    by_category: dict[str, float]


class IntercityTravel(BaseModel):
    """The chosen round trip between origin and destination. Party total is computed in code."""

    mode: TransportMode
    summary: str
    cost_per_person_round_trip: float = Field(ge=0, description="In the budget currency")
    source_url: str | None = None


class _DraftFields(BaseModel):
    """Fields shared by the planner's raw output and the stored draft."""

    title: str
    summary: str
    intercity: IntercityTravel | None = Field(
        default=None, description="Required when the request has an origin, otherwise empty"
    )
    days: list[DayPlan] = Field(min_length=1)
    caveats: list[str] = Field(default_factory=list)


class DraftBody(_DraftFields):
    """What the planner agent produces (structured output)."""

    @model_validator(mode="after")
    def _mostly_filled(self) -> "DraftBody":
        """Reject a near-empty itinerary so the agent is asked to correct it.

        Some models return a schema-valid skeleton with no activities; rejecting it here turns
        that into a correctable error instead of a draft that silently reaches review.
        """
        total = len(self.days) * len(SLOTS)
        filled = sum(getattr(day, slot) is not None for day in self.days for slot in SLOTS)
        if filled * 2 < total:
            raise ValueError(
                f"only {filled} of {total} activity slots are filled; fill morning, afternoon "
                "and evening with real activities (empty slots only on arrival/departure days)"
            )
        return self


class ItineraryDraft(_DraftFields):
    budget: BudgetSummary | None = None
    sources: list[Source] = Field(default_factory=list)
    # Copied from the research brief by the node, not written by the planner model.
    entry_requirements: str | None = None
    exchange_rate: ExchangeRate | None = None
    # Computed by app.graph.quality after every change to the draft.
    quality_issues: list[str] = Field(default_factory=list)


class FinalPlan(BaseModel):
    itinerary: ItineraryDraft
    status: Literal["approved", "needs_human_followup"]
    revisions: int
    degraded: list[str] = Field(default_factory=list)
    finalized_at: datetime


# --------------------------------------------------------------------------- review


class ItineraryPatch(BaseModel):
    """A structured edit applied in plain Python. No LLM involved."""

    op: Literal["replace", "remove"]
    day: int = Field(ge=1)
    slot: Slot
    value: Activity | None = None

    @model_validator(mode="after")
    def _value_required_for_replace(self) -> "ItineraryPatch":
        if self.op == "replace" and self.value is None:
            raise ValueError("value is required for op=replace")
        return self


class ReviewDecision(BaseModel):
    action: Literal["approve", "reject", "modify"]
    feedback: str | None = Field(default=None, description="Required for reject")
    patches: list[ItineraryPatch] | None = Field(default=None, description="Required for modify")
    route_hint: Literal["research", "planner"] | None = Field(
        default=None, description="Optional: send a rejection straight to research or the planner"
    )

    @model_validator(mode="after")
    def _action_requirements(self) -> "ReviewDecision":
        if self.action == "reject" and not (self.feedback and self.feedback.strip()):
            raise ValueError("feedback is required when action=reject")
        if self.action == "modify" and not self.patches:
            raise ValueError("patches are required when action=modify")
        return self


# --------------------------------------------------------------------------- steering

# The earliest stage a piece of mid-run guidance affects. Work from that stage on is redone.
SteerScope = Literal["request", "research", "planner"]


class SteerNote(BaseModel):
    """Guidance the traveller sent while the plan was being built."""

    id: str
    text: str
    scope: SteerScope
    at: datetime


class RequestChanges(BaseModel):
    """Edits to the trip request implied by a steer. Null means unchanged.

    Every field is required-but-nullable so the schema works with strict structured output.
    """

    destination: str | None
    start_date: str | None = Field(description="ISO date")
    end_date: str | None = Field(description="ISO date")
    budget_min: float | None
    budget_max: float | None
    travelers: int | None
    origin: str | None
    interests: list[str] | None


class SteerChoice(BaseModel):
    scope: SteerScope
    reason: str
    request_changes: RequestChanges | None


# --------------------------------------------------------------------------- graph state


class PlanState(TypedDict, total=False):
    request: TravelRequest
    research: ResearchBrief | None
    draft: ItineraryDraft | None
    final: FinalPlan | None
    review_history: Annotated[list[ReviewDecision], operator.add]
    revision_count: int
    # Soft tool failures. Reducer-appended; the plan says what data was unavailable.
    degraded: Annotated[list[str], operator.add]
    # Mid-run guidance that has reached an agent. Reducer-appended; every later agent run sees it.
    steering: Annotated[list[SteerNote], operator.add]
    needs_human_followup: bool
    error: str | None
