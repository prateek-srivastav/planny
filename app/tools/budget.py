"""Deterministic budget arithmetic. Models never compute totals; these functions do."""

from app.graph.state import BudgetSummary, DayPlan, ItineraryDraft, TravelRequest

# Categories that receive a share of the on-the-ground budget, and those every plan reports.
CATEGORIES = ("lodging", "food", "activities", "transit", "buffer")
CORE = ("lodging", "food", "activities", "transit")
# Reserved off the top of the budget before the split. Logistics in category "other" have no
# envelope and are absorbed by the buffer.
INTERCITY = "intercity"

# Share of the on-the-ground budget per category, by destination price level. Rows sum to 1.0.
TIER_SPLITS: dict[str, dict[str, float]] = {
    "budget": {"lodging": 0.35, "food": 0.25, "activities": 0.20, "transit": 0.10, "buffer": 0.10},
    "mid": {"lodging": 0.38, "food": 0.22, "activities": 0.20, "transit": 0.10, "buffer": 0.10},
    "premium": {"lodging": 0.42, "food": 0.22, "activities": 0.18, "transit": 0.08, "buffer": 0.10},
}


def allocate(
    total_budget: float,
    travelers: int,
    nights: int,
    cost_tier: str,
    intercity_cost: float = 0.0,
) -> dict:
    """Split the budget into envelopes after reserving `intercity_cost` (party total)."""
    intercity_cost = max(float(intercity_cost), 0.0)
    ground = max(total_budget - intercity_cost, 0.0)
    split = TIER_SPLITS.get(cost_tier, TIER_SPLITS["mid"])
    envelopes = {cat: round(ground * share, 2) for cat, share in split.items()}
    if intercity_cost:
        envelopes[INTERCITY] = round(intercity_cost, 2)
    days = nights + 1
    result = {
        "total_budget": total_budget,
        "on_the_ground_budget": round(ground, 2),
        "envelopes": envelopes,
        "guidance": {
            "lodging_per_night_party": round(envelopes["lodging"] / max(nights, 1), 2),
            "food_per_day_party": round(envelopes["food"] / days, 2),
            "food_per_day_per_person": round(envelopes["food"] / days / travelers, 2),
            "activities_per_day_party": round(envelopes["activities"] / days, 2),
            "transit_per_day_party": round(envelopes["transit"] / days, 2),
        },
    }
    if intercity_cost >= total_budget:
        result["warning"] = "intercity travel alone uses the whole budget; choose a cheaper option"
    return result


def check_remaining(envelopes: dict[str, float], planned: dict[str, float]) -> dict:
    """Compare planned spend with envelopes. The buffer absorbs spend without an envelope."""
    categories = list(dict.fromkeys([*envelopes, *planned]))
    remaining = {c: round(envelopes.get(c, 0) - planned.get(c, 0), 2) for c in categories}
    over = sorted(c for c in envelopes if c != "buffer" and remaining[c] < 0)
    planned_total = round(sum(v for k, v in planned.items() if k != "buffer"), 2)
    budget_total = round(sum(envelopes.values()), 2)
    return {
        "remaining_by_category": remaining,
        "over_budget_categories": over,
        "planned_total": planned_total,
        "budget_total": budget_total,
        "ok": not over and planned_total <= budget_total,
    }


def _day_totals(day: DayPlan) -> dict[str, float]:
    totals = {
        "lodging": day.lodging_cost,
        "food": day.food_cost,
        "activities": sum(a.est_cost for a in (day.morning, day.afternoon, day.evening) if a),
        "transit": day.transit_cost,
    }
    for item in day.logistics:
        totals[item.category] = totals.get(item.category, 0.0) + item.est_cost
    return totals


def category_totals(draft: ItineraryDraft, travelers: int) -> dict[str, float]:
    """Party totals per category: the core four always, others only when non-zero."""
    totals: dict[str, float] = dict.fromkeys(CORE, 0.0)
    for day in draft.days:
        for category, amount in _day_totals(day).items():
            totals[category] = totals.get(category, 0.0) + amount
    if draft.intercity:
        totals[INTERCITY] = draft.intercity.cost_per_person_round_trip * travelers
    return {k: round(v, 2) for k, v in totals.items() if v or k in CORE}


def summarize(draft: ItineraryDraft, request: TravelRequest) -> BudgetSummary:
    by_cat = category_totals(draft, request.travelers)
    total = round(sum(by_cat.values()), 2)
    return BudgetSummary(
        currency=request.currency,
        cap=request.budget_max,
        floor=request.budget_min,
        total_estimated=total,
        remaining=round(request.budget_max - total, 2),
        within_budget=total <= request.budget_max,
        by_category=by_cat,
    )
