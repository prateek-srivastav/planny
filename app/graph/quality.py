"""Deterministic checks applied to every draft before a human sees it.

`tidy_logistics` moves logistics that a model scheduled as activities (check-in, transfers, pass
activation, ...) into the day's `logistics` list, keeping their cost. `find_issues` reports
problems a reviewer should know about; it never changes the draft.
"""

import re

from app.graph.state import (
    SLOTS,
    Activity,
    ItineraryDraft,
    LogisticsCategory,
    LogisticsItem,
    TravelRequest,
)

# Patterns are deliberately narrow: a false positive removes a real activity from the plan.
_TRANSIT = re.compile(
    r"\b(transfers?|airport|pick[- ]?up|drop[- ]?off|departure|arrival)\b", re.IGNORECASE
)
_PASS = re.compile(
    r"\b(card|pass)\b.*\b(activat(e|ion)|collection|pick[- ]?up)\b|"
    r"\b(activat(e|ion)|collect)\b.*\b(card|pass)\b",
    re.IGNORECASE,
)
_OTHER = re.compile(
    r"\b(check[- ]?in|check[- ]?out|luggage|baggage|pack(ing)?|unpack|settle in|"
    r"freshen up|hotel breakfast|breakfast at (the )?hotel|free time|rest at|relax at)\b",
    re.IGNORECASE,
)


def logistics_category(name: str) -> LogisticsCategory | None:
    """The budget category of a logistics item, or None if `name` reads as a real activity."""
    if _PASS.search(name):
        return "activities"
    if _OTHER.search(name):
        return "other"
    if _TRANSIT.search(name):
        return "transit"
    return None


def tidy_logistics(draft: ItineraryDraft) -> list[str]:
    """Move logistics out of activity slots. Returns a note per moved item."""
    moved = []
    for day in draft.days:
        for slot in SLOTS:
            activity: Activity | None = getattr(day, slot)
            if activity is None:
                continue
            category = logistics_category(activity.name)
            if category is None:
                continue
            day.logistics.append(
                LogisticsItem(name=activity.name, est_cost=activity.est_cost, category=category)
            )
            setattr(day, slot, None)
            moved.append(f"day {day.day} {slot}: '{activity.name}' moved to logistics")
    return moved


def _mentions(text: str, phrase: str) -> bool:
    return bool(phrase) and re.search(rf"\b{re.escape(phrase.lower())}\b", text.lower()) is not None


def drop_intercity_duplicates(draft: ItineraryDraft, request: TravelRequest) -> list[str]:
    """Remove logistics items that restate the planned origin<->destination trip.

    The trip is already costed once in `draft.intercity`; a duplicate would count it twice. An
    item is a duplicate if it names the origin, or names the travel mode and the destination.
    """
    if draft.intercity is None or not request.origin:
        return []
    origin = request.origin.split(",")[0].strip()
    destination = request.destination.split(",")[0].strip()
    dropped = []
    for day in draft.days:
        keep = []
        for item in day.logistics:
            duplicate = _mentions(item.name, origin) or (
                _mentions(item.name, draft.intercity.mode) and _mentions(item.name, destination)
            )
            if duplicate:
                dropped.append(f"day {day.day}: '{item.name}' duplicates the intercity trip")
            else:
                keep.append(item)
        day.logistics = keep
    return dropped


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def find_issues(draft: ItineraryDraft, request: TravelRequest) -> list[str]:
    issues = []

    seen: dict[str, int] = {}
    for day in draft.days:
        for slot in SLOTS:
            activity = getattr(day, slot)
            if activity is None:
                continue
            key = _key(activity.name)
            if key in seen:
                issues.append(
                    f"'{activity.name}' is scheduled more than once (day {seen[key]} and "
                    f"day {day.day})"
                )
            else:
                seen[key] = day.day

    last = len(draft.days)
    for day in draft.days:
        if day.day in (1, last):
            continue
        empty = [slot for slot in SLOTS if getattr(day, slot) is None]
        if empty:
            issues.append(f"day {day.day} has no activity in: {', '.join(empty)}")

    for day in draft.days:
        stay = [i.name for i in day.logistics if i.category == "lodging" and i.est_cost > 0]
        if stay and day.lodging_cost > 0:
            issues.append(
                f"day {day.day}: lodging is costed both as the night's lodging and as "
                f"logistics ({', '.join(stay)})"
            )

    nights = [d for d in draft.days if d.day < last]
    unpaid = [
        d.day
        for d in nights
        if d.lodging_cost == 0 and not any(i.category == "lodging" for i in d.logistics)
    ]
    if unpaid:
        issues.append(f"no lodging cost for the night of day {', '.join(map(str, unpaid))}")

    if request.origin and draft.intercity is None:
        issues.append(f"no travel from {request.origin} is planned or budgeted")

    if draft.budget and not draft.budget.within_budget:
        issues.append(
            f"estimated total {draft.budget.total_estimated:,.0f} {draft.budget.currency} is over "
            f"the {draft.budget.cap:,.0f} cap"
        )
    return issues
