"""Intercity budgeting and the deterministic draft-quality pass."""

from datetime import date

import pytest

from app.graph import quality
from app.graph.state import (
    Activity,
    BudgetSummary,
    DayPlan,
    IntercityTravel,
    ItineraryDraft,
    LogisticsItem,
)
from app.tools import budget
from tests.fakes import make_request


def day(n: int, morning=None, afternoon=None, evening=None, **kw) -> DayPlan:
    kw.setdefault("lodging_cost", 100)

    def act(name):
        return Activity(name=name, est_cost=10) if isinstance(name, str) else name

    return DayPlan(
        day=n,
        date=date(2030, 1, n),
        theme=f"Day {n}",
        morning=act(morning),
        afternoon=act(afternoon),
        evening=act(evening),
        **kw,
    )


def draft(*days: DayPlan, intercity=None) -> ItineraryDraft:
    return ItineraryDraft(title="t", summary="s", days=list(days), intercity=intercity)


# ---------------------------------------------------------------- budget


def test_intercity_is_reserved_before_the_split():
    out = budget.allocate(10_000, travelers=2, nights=3, cost_tier="mid", intercity_cost=4_000)
    env = out["envelopes"]
    assert env["intercity"] == 4_000
    assert out["on_the_ground_budget"] == 6_000
    assert sum(v for k, v in env.items() if k != "intercity") == pytest.approx(6_000, abs=0.05)
    assert "warning" not in out


def test_intercity_that_eats_the_whole_budget_warns():
    out = budget.allocate(1_000, 1, 2, "mid", intercity_cost=1_500)
    assert out["on_the_ground_budget"] == 0 and "warning" in out


def test_no_origin_means_no_intercity_envelope():
    assert "intercity" not in budget.allocate(1_000, 1, 2, "mid")["envelopes"]


def test_check_remaining_covers_intercity_and_lets_buffer_absorb_other():
    env = budget.allocate(10_000, 2, 3, "mid", intercity_cost=4_000)["envelopes"]
    over = budget.check_remaining(env, {"intercity": 4_500})
    assert over["over_budget_categories"] == ["intercity"]
    absorbed = budget.check_remaining(env, {"lodging": 1_000, "other": 200})
    assert absorbed["ok"] and absorbed["over_budget_categories"] == []


def test_totals_include_intercity_per_party_and_logistics_by_category():
    d = draft(
        day(
            1,
            morning="Museum",
            lodging_cost=100,
            logistics=[
                LogisticsItem(name="Airport transfer", est_cost=40, category="transit"),
                LogisticsItem(name="City card", est_cost=60, category="activities"),
                LogisticsItem(name="Luggage storage", est_cost=5, category="other"),
            ],
        ),
        intercity=IntercityTravel(mode="flight", summary="x", cost_per_person_round_trip=300),
    )
    totals = budget.category_totals(d, travelers=2)
    assert totals == {
        "lodging": 100,
        "food": 0,
        "activities": 70,  # 10 museum + 60 city card
        "transit": 40,
        "other": 5,
        "intercity": 600,  # 300 per person x 2
    }


# ---------------------------------------------------------------- logistics


@pytest.mark.parametrize(
    ("name", "category"),
    [
        ("Check-in & Light Breakfast", "other"),
        ("Check-out & Transfer to Airport", "other"),
        ("Lisboa Card (Day 1) Activation", "activities"),
        ("Collect Rail Pass", "activities"),
        ("Arrival in Goa", "transit"),
        ("Airport transfer", "transit"),
        ("Departure", "transit"),
        ("Tram 28 Ride", None),
        ("São Bento Station", None),
        ("Day trip to Sintra", None),
        ("Time Out Market – Vegetarian Dinner", None),
        ("Passport Museum", None),
    ],
)
def test_logistics_category(name, category):
    assert quality.logistics_category(name) == category


def test_tidy_moves_logistics_out_of_slots_and_keeps_their_cost():
    request = make_request(currency="USD")
    pass_activation = Activity(name="Lisboa Card Activation", est_cost=300)
    d = draft(
        day(1, morning="Check-in & Light Breakfast", afternoon="Belem Tower", evening=None),
        day(2, morning=pass_activation, afternoon="Tile Museum", evening="Alfama walk"),
    )
    before = budget.category_totals(d, request.travelers)

    moved = quality.tidy_logistics(d)

    assert len(moved) == 2
    assert d.days[0].morning is None and d.days[0].afternoon.name == "Belem Tower"
    assert [i.name for i in d.days[1].logistics] == ["Lisboa Card Activation"]
    assert d.days[1].logistics[0].category == "activities"
    after = budget.category_totals(d, request.travelers)
    assert sum(after.values()) == sum(before.values())  # no spend lost
    assert after["activities"] == before["activities"] - 10  # check-in moved to "other"
    assert after["other"] == 10


def test_tidy_leaves_real_activities_alone():
    d = draft(day(1, "Belem Tower", "Tile Museum", "Fado dinner at Clube de Fado"))
    assert quality.tidy_logistics(d) == []
    assert d.days[0].evening.name == "Fado dinner at Clube de Fado"


# ---------------------------------------------------------------- issues


def test_issues_flag_repeats_ignoring_case_and_punctuation():
    d = draft(day(1, "Belém Tower"), day(2, "belém tower!"), day(3, "x"))
    issues = quality.find_issues(d, make_request())
    assert any("scheduled more than once (day 1 and day 2)" in i for i in issues)


def test_issues_flag_empty_slots_only_on_middle_days():
    d = draft(day(1, None, "a", None), day(2, "b", None, "c"), day(3, None, None, None))
    issues = quality.find_issues(d, make_request())
    assert issues == ["day 2 has no activity in: afternoon"]


def test_issues_flag_missing_intercity_and_overspend():
    d = draft(day(1, "a", "b", "c"))
    d.budget = BudgetSummary(
        currency="INR",
        cap=1000,
        floor=500,
        total_estimated=1500,
        remaining=-500,
        within_budget=False,
        by_category={},
    )
    issues = quality.find_issues(d, make_request(origin="Delhi"))
    assert "no travel from Delhi is planned or budgeted" in issues
    assert any("over the 1,000 cap" in i for i in issues)
    assert quality.find_issues(draft(day(1, "a", "b", "c")), make_request()) == []


@pytest.mark.parametrize(
    ("given", "stored"),
    [
        ("transit", "transit"),
        ("Lodging", "lodging"),
        (" FOOD ", "food"),
        ("transport", "other"),
        ("", "other"),
    ],
)
def test_logistics_category_is_normalised_not_rejected(given, stored):
    assert LogisticsItem(name="x", category=given).category == stored


def test_logistics_category_is_not_an_enum_in_the_model_schema():
    from app.graph.state import DraftBody

    schema = DraftBody.model_json_schema()["$defs"]["LogisticsItem"]["properties"]["category"]
    assert schema["type"] == "string" and "enum" not in schema


def test_logistics_in_lodging_and_food_count_in_those_categories():
    d = draft(
        day(
            1,
            "a",
            lodging_cost=0,
            logistics=[
                LogisticsItem(name="Early check-in fee", est_cost=50, category="lodging"),
                LogisticsItem(name="Packed lunch", est_cost=20, category="food"),
            ],
        )
    )
    totals = budget.category_totals(d, travelers=2)
    assert totals["lodging"] == 50 and totals["food"] == 20 and "other" not in totals


# ---------------------------------------------------------------- double counting


def flight(cost=7500):
    return IntercityTravel(mode="flight", summary="DEL-GOI", cost_per_person_round_trip=cost)


def test_logistics_that_restate_the_intercity_trip_are_dropped():
    d = draft(
        day(
            1,
            "Calangute Beach",
            logistics=[
                LogisticsItem(
                    name="Delhi-Goa flight (round-trip)", est_cost=15000, category="other"
                ),
                LogisticsItem(name="Flight home from Goa", est_cost=1),
                LogisticsItem(
                    name="Airport transfer (Goa airport -> hotel)", est_cost=800, category="transit"
                ),
            ],
        ),
        intercity=flight(),
    )
    dropped = quality.drop_intercity_duplicates(d, make_request(destination="Goa", origin="Delhi"))
    assert len(dropped) == 2
    assert [i.name for i in d.days[0].logistics] == ["Airport transfer (Goa airport -> hotel)"]


def test_trips_within_the_destination_are_kept():
    d = draft(
        day(
            1,
            "a",
            logistics=[LogisticsItem(name="Train to Palolem", est_cost=100, category="transit")],
        ),
        intercity=IntercityTravel(mode="train", summary="x", cost_per_person_round_trip=1),
    )
    assert (
        quality.drop_intercity_duplicates(d, make_request(destination="Goa", origin="Delhi")) == []
    )
    assert len(d.days[0].logistics) == 1


def test_nothing_is_dropped_without_an_intercity_trip():
    d = draft(day(1, "a", logistics=[LogisticsItem(name="Delhi-Goa flight", est_cost=1)]))
    assert quality.drop_intercity_duplicates(d, make_request(origin="Delhi")) == []


def test_lodging_counted_twice_or_missing_is_flagged():
    stay = LogisticsItem(name="Hotel check-in (3-night stay)", est_cost=7500, category="lodging")
    d = draft(
        day(1, "a", "b", "c", lodging_cost=7500, logistics=[stay]),
        day(2, "d", "e", "f", lodging_cost=0),
        day(3, "g", "h", "i", lodging_cost=0),
        day(4, "j", "k", "l", lodging_cost=0),
    )
    issues = quality.find_issues(d, make_request())
    assert any(i.startswith("day 1: lodging is costed both") for i in issues)
    assert "no lodging cost for the night of day 2, 3" in issues  # the last day has no night
