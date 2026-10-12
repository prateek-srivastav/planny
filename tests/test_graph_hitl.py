"""The three review paths, run through the real graph, agents and tools with a fake LLM."""

import pytest
from langgraph.types import Command

from app.graph.build import build_graph
from app.graph.state import ReviewDecision
from tests.conftest import open_saver
from tests.fakes import (
    SEARCH_URL,
    FakeSearch,
    make_classifier,
    make_deps,
    make_model,
    make_request,
    planner_answer,
    research_answer,
)


def initial(request=None):
    return {
        "request": request or make_request(),
        "revision_count": 0,
        "review_history": [],
        "degraded": [],
    }


async def start(graph, thread="t1", request=None):
    cfg = {"configurable": {"thread_id": thread}}
    out = await graph.ainvoke(initial(request), cfg, durability="sync")
    return cfg, out


async def test_runs_to_interrupt_with_typed_draft(db_path):
    async with await open_saver(db_path) as saver:
        model, deps = make_model(), make_deps()
        graph = build_graph(saver, deps, model, make_classifier())
        cfg, out = await start(graph)

        payload = out["__interrupt__"][0].value
        draft = payload["draft"]
        assert len(draft["days"]) == 4
        # dates were overwritten from the request, not trusted from the model
        assert draft["days"][0]["date"] == str(make_request().start_date)
        # budget arithmetic is Python's: 4 * (150 + 90 + 15) + 4 * (40 + 60) = 1420
        assert draft["budget"]["total_estimated"] == 1420
        assert draft["budget"]["within_budget"] is True
        # only URLs that web_search actually returned survive as sources
        assert [s["url"] for s in draft["sources"]] == [SEARCH_URL]
        state = await graph.aget_state(cfg)
        assert state.values["research"].conditions.basis == "forecast"
        assert state.next == ("human_review",)


async def test_approve_path(db_path):
    async with await open_saver(db_path) as saver:
        graph = build_graph(saver, make_deps(), make_model(), make_classifier())
        cfg, _ = await start(graph)
        out = await graph.ainvoke(Command(resume={"action": "approve"}), cfg, durability="sync")
        final = out["final"]
        assert final.status == "approved"
        assert final.revisions == 0
        assert (await graph.aget_state(cfg)).next == ()


async def test_modify_path_changes_only_patched_slot_and_rechecks_budget(db_path):
    async with await open_saver(db_path) as saver:
        graph = build_graph(saver, make_deps(), make_model(), make_classifier())
        cfg, out = await start(graph)
        before = out["__interrupt__"][0].value["draft"]
        patch = {
            "action": "modify",
            "patches": [
                {
                    "op": "replace",
                    "day": 2,
                    "slot": "afternoon",
                    "value": {"name": "Fado show", "est_cost": 300},
                }
            ],
        }
        out = await graph.ainvoke(Command(resume=patch), cfg, durability="sync")

        # it looped straight back to review: no agent ran again
        after = out["__interrupt__"][0].value["draft"]
        assert after["days"][1]["afternoon"]["name"] == "Fado show"
        for i, (b, a) in enumerate(zip(before["days"], after["days"], strict=True)):
            for slot in ("morning", "afternoon", "evening"):
                if (i, slot) != (1, "afternoon"):
                    assert a[slot] == b[slot]
        assert after["budget"]["total_estimated"] == 1420 - 60 + 300
        assert after["title"] == before["title"]


async def test_modify_remove_slot(db_path):
    async with await open_saver(db_path) as saver:
        graph = build_graph(saver, make_deps(), make_model(), make_classifier())
        cfg, _ = await start(graph)
        out = await graph.ainvoke(
            Command(
                resume={
                    "action": "modify",
                    "patches": [{"op": "remove", "day": 1, "slot": "morning"}],
                }
            ),
            cfg,
            durability="sync",
        )
        draft = out["__interrupt__"][0].value["draft"]
        assert draft["days"][0]["morning"] is None
        assert draft["budget"]["total_estimated"] == 1420 - 40


async def test_reject_routes_to_planner_and_passes_feedback(db_path):
    async with await open_saver(db_path) as saver:
        model = make_model()
        graph = build_graph(saver, make_deps(), model, make_classifier("planner"))
        cfg, _ = await start(graph)
        assert [c[0] for c in model.calls] == ["ResearchBody", "DraftBody"]

        out = await graph.ainvoke(
            Command(resume={"action": "reject", "feedback": "too much walking"}),
            cfg,
            durability="sync",
        )
        # planner re-ran, research did not
        assert [c[0] for c in model.calls] == ["ResearchBody", "DraftBody", "DraftBody"]
        assert "too much walking" in model.calls[-1][1]
        assert out["__interrupt__"][0].value["draft"]["title"] == "Revised Lisbon trip"
        assert (await graph.aget_state(cfg)).values["revision_count"] == 1


async def test_reject_routes_to_research_when_classifier_says_so(db_path):
    async with await open_saver(db_path) as saver:
        model = make_model()
        graph = build_graph(saver, make_deps(), model, make_classifier("research"))
        cfg, _ = await start(graph)
        await graph.ainvoke(
            Command(resume={"action": "reject", "feedback": "wrong part of town"}),
            cfg,
            durability="sync",
        )
        assert [c[0] for c in model.calls] == [
            "ResearchBody",
            "DraftBody",
            "ResearchBody",
            "DraftBody",
        ]
        assert "wrong part of town" in model.calls[2][1]


async def test_route_hint_skips_classifier(db_path):
    async with await open_saver(db_path) as saver:
        model = make_model()
        classifier = make_classifier("planner")
        graph = build_graph(saver, make_deps(), model, classifier)
        cfg, _ = await start(graph)
        await graph.ainvoke(
            Command(resume={"action": "reject", "feedback": "x", "route_hint": "research"}),
            cfg,
            durability="sync",
        )
        assert classifier.calls == []
        assert [c[0] for c in model.calls][-2:] == ["ResearchBody", "DraftBody"]


async def test_classifier_failure_defaults_to_planner_and_is_flagged(db_path):
    async with await open_saver(db_path) as saver:
        broken = make_classifier()
        broken.responders = {}  # AssertionError on use
        graph = build_graph(saver, make_deps(), make_model(), broken)
        cfg, _ = await start(graph)
        out = await graph.ainvoke(
            Command(resume={"action": "reject", "feedback": "meh"}), cfg, durability="sync"
        )
        assert any("defaulted to planner" in d for d in out["__interrupt__"][0].value["degraded"])


async def test_revision_cap_terminates_with_followup_flag(db_path):
    async with await open_saver(db_path) as saver:
        deps = make_deps()  # max_revisions = 3
        graph = build_graph(saver, deps, make_model(), make_classifier())
        cfg, _ = await start(graph)
        reject = Command(resume={"action": "reject", "feedback": "no"})
        out = None
        for _ in range(deps.settings.max_revisions):
            out = await graph.ainvoke(reject, cfg, durability="sync")
        assert "__interrupt__" not in out
        assert out["final"].status == "needs_human_followup"
        assert out["final"].revisions == 3
        assert (await graph.aget_state(cfg)).next == ()


async def test_invalid_review_payload_is_rejected_by_schema():
    with pytest.raises(ValueError):
        ReviewDecision.model_validate({"action": "reject"})  # feedback required
    with pytest.raises(ValueError):
        ReviewDecision.model_validate({"action": "modify"})  # patches required


async def test_unknown_destination_fails_without_running_agents(db_path):
    async with await open_saver(db_path) as saver:
        model = make_model()
        graph = build_graph(saver, make_deps(geocode_empty=True), model, make_classifier())
        cfg, out = await start(graph, request=make_request(destination="Xyzzyland"))
        assert "Could not find a destination" in out["error"]
        assert model.calls == []


async def test_search_and_weather_outages_degrade_instead_of_failing(db_path):
    async with await open_saver(db_path) as saver:
        deps = make_deps(search=FakeSearch(fail=True), weather_down=True)
        graph = build_graph(saver, deps, make_model(), make_classifier())
        cfg, out = await start(graph)

        payload = out["__interrupt__"][0].value
        joined = " | ".join(payload["degraded"])
        assert "web_search" in joined or "search backend down" in joined
        assert "destination_conditions" in joined
        assert payload["draft"]["days"]  # still produced a plan
        assert payload["draft"]["sources"] == []


async def test_nationality_entry_rules_and_exchange_rate_reach_the_draft(db_path):
    async with await open_saver(db_path) as saver:
        model = make_model()
        graph = build_graph(saver, make_deps(), model, make_classifier())
        cfg, out = await start(graph)
        draft = out["__interrupt__"][0].value["draft"]
        # the traveller's nationality was put in front of the research agent
        assert "nationality (passport): India" in model.calls[0][1]
        # entry rules come from the research brief; the rate comes from the tool, not the model
        assert "Schengen visa" in draft["entry_requirements"]
        assert draft["exchange_rate"] == {
            "base": "USD",
            "quote": "EUR",
            "rate": 0.9,
            "date": "2026-10-09",
        }
        # and the planner is told to convert with that rate
        planner_prompt = model.calls[1][1]
        assert "1 USD = 0.9 EUR" in planner_prompt and "must be in USD" in planner_prompt
        state = await graph.aget_state(cfg)
        assert state.values["research"].local_currency == "EUR"


async def test_exhausted_model_rolls_over_to_the_fallback_provider(db_path):
    """Primary model keeps failing (rate limit / daily cap): the chain moves to the next model."""
    async with await open_saver(db_path) as saver:
        primary = make_model()
        primary.fail_times = 10_000  # never recovers
        fallback = make_model()
        graph = build_graph(
            saver, make_deps(), primary, make_classifier(), fallback_models=[fallback]
        )
        cfg, out = await start(graph)
        assert out["__interrupt__"][0].value["draft"]["days"]  # plan completed anyway
        assert [c[0] for c in fallback.calls] == ["ResearchBody", "DraftBody"]
        assert primary.calls == []  # it never produced an answer
        assert primary.failures[0] >= 3  # it did try (and retry) before giving up


async def test_without_fallbacks_a_dead_model_fails_the_plan_cleanly(db_path):
    async with await open_saver(db_path) as saver:
        primary = make_model()
        primary.fail_times = 10_000
        graph = build_graph(saver, make_deps(), primary, make_classifier())
        _, out = await start(graph)
        assert "research_node" in out["error"]


async def test_a_runaway_search_loop_is_capped_and_fails_fast(db_path):
    """A model that never stops searching must not drain the provider quota."""

    def forever(messages):
        return [{"name": "web_search", "args": {"query": "one more thing"}}]

    async with await open_saver(db_path) as saver:
        search = FakeSearch()
        model = make_model(ResearchBody=forever)
        graph = build_graph(saver, make_deps(search=search), model, make_classifier())
        _, out = await start(graph)
        assert "ModelCallLimitExceededError" in out["error"]
        # searches stopped at the cap, and the model was stopped shortly after, not after 100s
        assert len(search.queries) == 6  # settings.max_searches
        assert model.calls == []  # it never produced an answer


async def test_searches_over_the_cap_are_refused_so_the_agent_can_still_finish(db_path):
    """Soft layer: after the cap the agent gets a refusal and can answer with what it has."""

    def eight_searches_then_answer(messages):
        done = sum(1 for m in messages if getattr(m, "type", "") == "tool")
        if done < 8:
            return [{"name": "web_search", "args": {"query": f"q{done}"}}]
        return research_answer(messages)

    async with await open_saver(db_path) as saver:
        search = FakeSearch()
        model = make_model(ResearchBody=eight_searches_then_answer)
        graph = build_graph(saver, make_deps(search=search), model, make_classifier())
        _, out = await start(graph)
        assert len(search.queries) == 6  # the 7th and 8th were refused, not executed
        assert out["__interrupt__"][0].value["draft"]["days"]  # and the plan still completed


async def test_origin_is_researched_planned_and_budgeted(db_path):
    async with await open_saver(db_path) as saver:
        model = make_model()
        graph = build_graph(saver, make_deps(), model, make_classifier())
        cfg, out = await start(graph, request=make_request(origin="Delhi", budget_max=3000))
        research_prompt, planner_prompt = model.calls[0][1], model.calls[1][1]
        assert "Travelling from: Delhi" in research_prompt
        assert "Getting there from Delhi" in planner_prompt and "DEL -> LIS" in planner_prompt

        draft = out["__interrupt__"][0].value["draft"]
        assert draft["intercity"]["mode"] == "flight"
        # 300 per person x 2 travellers, computed in code
        assert draft["budget"]["by_category"]["intercity"] == 600
        assert draft["budget"]["total_estimated"] == 1420 + 600
        assert draft["quality_issues"] == []
        state = await graph.aget_state(cfg)
        assert len(state.values["research"].getting_there) == 2


async def test_without_an_origin_no_intercity_is_kept_or_budgeted(db_path):
    def planner_adds_a_flight_anyway(messages):
        out = planner_answer(messages)
        if isinstance(out, dict):
            out["intercity"] = {"mode": "flight", "summary": "x", "cost_per_person_round_trip": 999}
        return out

    async with await open_saver(db_path) as saver:
        model = make_model(DraftBody=planner_adds_a_flight_anyway)
        graph = build_graph(saver, make_deps(), model, make_classifier())
        _, out = await start(graph)
        draft = out["__interrupt__"][0].value["draft"]
        assert "Travelling from: not stated" in model.calls[0][1]
        assert draft["intercity"] is None
        assert "intercity" not in draft["budget"]["by_category"]


async def test_filler_activities_reach_review_as_logistics(db_path):
    def planner_with_filler(messages):
        out = planner_answer(messages)
        if isinstance(out, dict):
            out["days"][0]["morning"] = {"name": "Check-in & Light Breakfast", "est_cost": 0}
            out["days"][1]["morning"] = {"name": "Lisboa Card Activation", "est_cost": 40}
        return out

    async with await open_saver(db_path) as saver:
        graph = build_graph(
            saver, make_deps(), make_model(DraftBody=planner_with_filler), make_classifier()
        )
        _, out = await start(graph)
        days = out["__interrupt__"][0].value["draft"]["days"]
        assert days[0]["morning"] is None
        assert days[1]["morning"] is None
        assert days[1]["logistics"] == [
            {"name": "Lisboa Card Activation", "est_cost": 40.0, "category": "activities"}
        ]
        issues = out["__interrupt__"][0].value["draft"]["quality_issues"]
        assert issues == ["day 2 has no activity in: morning"]  # flagged for the reviewer


async def test_a_patch_refreshes_quality_issues(db_path):
    async with await open_saver(db_path) as saver:
        graph = build_graph(saver, make_deps(), make_model(), make_classifier())
        cfg, out = await start(graph)
        assert out["__interrupt__"][0].value["draft"]["quality_issues"] == []
        out = await graph.ainvoke(
            Command(
                resume={
                    "action": "modify",
                    "patches": [
                        {
                            "op": "replace",
                            "day": 2,
                            "slot": "morning",
                            "value": {"name": "Morning 1"},
                        }
                    ],
                }
            ),
            cfg,
            durability="sync",
        )
        issues = out["__interrupt__"][0].value["draft"]["quality_issues"]
        assert issues == ["'Morning 1' is scheduled more than once (day 1 and day 2)"]


async def test_an_empty_itinerary_is_sent_back_to_the_model_to_fix(db_path):
    attempts = []

    def empty_then_full(messages):
        out = planner_answer(messages)
        if isinstance(out, dict):
            attempts.append(1)
            if len(attempts) == 1:
                for day in out["days"]:
                    day["morning"] = day["afternoon"] = day["evening"] = None
        return out

    async with await open_saver(db_path) as saver:
        graph = build_graph(
            saver, make_deps(), make_model(DraftBody=empty_then_full), make_classifier()
        )
        _, out = await start(graph)
        days = out["__interrupt__"][0].value["draft"]["days"]
        assert len(attempts) == 2  # rejected once, corrected on the second answer
        assert all(d["morning"] for d in days)


def test_draft_schema_rejects_a_skeleton_but_allows_light_travel_days():
    from datetime import date as d

    from app.graph.state import DraftBody

    def mk(slots_per_day):
        return {
            "title": "t",
            "summary": "s",
            "days": [
                {
                    "day": i + 1,
                    "date": d(2030, 1, i + 1),
                    "theme": "x",
                    **{s: {"name": f"{s} {i}"} for s in ("morning", "afternoon", "evening")[:n]},
                }
                for i, n in enumerate(slots_per_day)
            ],
        }

    with pytest.raises(ValueError, match="0 of 9 activity slots"):
        DraftBody.model_validate(mk([0, 0, 0]))
    DraftBody.model_validate(mk([1, 3, 1]))  # light arrival and departure days are fine
