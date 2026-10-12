import json
import threading
from datetime import date, timedelta

from tests.conftest import running_app
from tests.fakes import make_model, planner_answer, research_answer

TODAY = date.today()
BODY = {
    "destination": "Lisbon",
    "start_date": str(TODAY + timedelta(days=5)),
    "end_date": str(TODAY + timedelta(days=8)),
    "budget_min": 1500,
    "budget_max": 2000,
    "travelers": 2,
    "interests": ["food"],
}


async def create(c, **overrides):
    r = await c.http.post("/plan", json={**BODY, **overrides})
    assert r.status_code == 202, r.text
    return r.json()["plan_id"]


async def test_post_returns_202_without_waiting_for_the_agents(db_path):
    gate = threading.Event()

    def blocked_research(messages):
        gate.wait(10)
        return research_answer(messages)

    model = make_model(ResearchBody=blocked_research)
    async with running_app(db_path, model=model) as c:
        r = await c.http.post("/plan", json=BODY)
        assert r.status_code == 202
        body = r.json()
        assert body["status"] == "running"
        assert body["links"]["review"].endswith("/review")

        # The run is parked inside the research agent, yet GET already works and says so.
        view = (await c.http.get(f"/plan/{body['plan_id']}")).json()
        assert view["status"] == "running"
        assert view["draft"] is None and view["review"] is None

        gate.set()
        await c.wait_status(body["plan_id"], "awaiting_review")


async def test_full_flow_approve(db_path):
    async with running_app(db_path) as c:
        pid = await create(c)
        view = await c.wait_status(pid, "awaiting_review")

        assert view["stage"] == "awaiting review"
        assert view["draft"]["budget"]["within_budget"] is True
        schema = view["review"]["response_schema"]
        assert schema["properties"]["action"]["enum"] == ["approve", "reject", "modify"]
        assert view["review"]["payload"]["revisions_left"] == 3

        # nothing is final yet
        r = await c.http.get(f"/plan/{pid}/final")
        assert r.status_code == 409 and r.json()["detail"]["status"] == "awaiting_review"

        r = await c.http.post(f"/plan/{pid}/review", json={"action": "approve"})
        assert r.status_code == 202 and r.json()["status"] == "running"
        await c.wait_status(pid, "completed")

        final = (await c.http.get(f"/plan/{pid}/final")).json()
        assert final["status"] == "approved"
        assert final["itinerary"]["days"]


async def test_modify_then_approve(db_path):
    async with running_app(db_path) as c:
        pid = await create(c)
        await c.wait_status(pid, "awaiting_review")
        patch = {
            "action": "modify",
            "patches": [
                {
                    "op": "replace",
                    "day": 1,
                    "slot": "evening",
                    "value": {"name": "Fado", "est_cost": 80},
                }
            ],
        }
        assert (await c.http.post(f"/plan/{pid}/review", json=patch)).status_code == 202
        await c.wait_idle(pid)
        view = await c.wait_status(pid, "awaiting_review")  # modify loops back to review
        assert view["draft"]["days"][0]["evening"]["name"] == "Fado"
        assert view["draft"]["budget"]["total_estimated"] == 1500


async def test_reject_loop_and_revision_counter(db_path):
    async with running_app(db_path) as c:
        pid = await create(c)
        await c.wait_status(pid, "awaiting_review")
        r = await c.http.post(
            f"/plan/{pid}/review", json={"action": "reject", "feedback": "less walking"}
        )
        assert r.status_code == 202
        await c.wait_idle(pid)
        view = await c.wait_status(pid, "awaiting_review")
        assert view["revisions_used"] == 1
        assert view["draft"]["title"] == "Revised Lisbon trip"


async def test_review_validation_errors(db_path):
    async with running_app(db_path) as c:
        pid = await create(c)
        await c.wait_status(pid, "awaiting_review")
        # reject needs feedback; modify needs patches
        assert (
            await c.http.post(f"/plan/{pid}/review", json={"action": "reject"})
        ).status_code == 422
        assert (
            await c.http.post(f"/plan/{pid}/review", json={"action": "modify"})
        ).status_code == 422
        # patch for a day the plan doesn't have
        bad = {"action": "modify", "patches": [{"op": "remove", "day": 99, "slot": "morning"}]}
        r = await c.http.post(f"/plan/{pid}/review", json=bad)
        assert r.status_code == 422
        # and the plan is untouched and still reviewable after all those rejections
        assert (await c.http.get(f"/plan/{pid}")).json()["status"] == "awaiting_review"


async def test_review_409_unless_awaiting_review(db_path):
    gate = threading.Event()

    def blocked(messages):
        gate.wait(10)
        return research_answer(messages)

    async with running_app(db_path, model=make_model(ResearchBody=blocked)) as c:
        pid = await create(c)
        r = await c.http.post(f"/plan/{pid}/review", json={"action": "approve"})
        assert r.status_code == 409  # still running
        gate.set()
        await c.wait_status(pid, "awaiting_review")
        assert (
            await c.http.post(f"/plan/{pid}/review", json={"action": "approve"})
        ).status_code == 202
        await c.wait_status(pid, "completed")
        # already finished: a second approval is refused
        r = await c.http.post(f"/plan/{pid}/review", json={"action": "approve"})
        assert r.status_code == 409


async def test_double_submit_resumes_only_once(db_path):
    async with running_app(db_path) as c:
        pid = await create(c)
        await c.wait_status(pid, "awaiting_review")
        first = await c.http.post(f"/plan/{pid}/review", json={"action": "approve"})
        second = await c.http.post(f"/plan/{pid}/review", json={"action": "approve"})
        assert (first.status_code, second.status_code) == (202, 409)
        await c.wait_status(pid, "completed")
        view = (await c.http.get(f"/plan/{pid}")).json()
        assert view["status"] == "completed"
        history = (await c.http.get(f"/plan/{pid}/history")).json()["checkpoints"]
        assert sum(1 for h in history if h["next"] == ["finalize"]) == 1


async def test_stale_checkpoint_id_is_409(db_path):
    async with running_app(db_path) as c:
        pid = await create(c)
        view = await c.wait_status(pid, "awaiting_review")
        stale = view["checkpoint_id"]
        await c.http.post(f"/plan/{pid}/review", json={"action": "reject", "feedback": "again"})
        await c.wait_idle(pid)
        await c.wait_status(pid, "awaiting_review")
        r = await c.http.post(
            f"/plan/{pid}/review", json={"action": "approve", "checkpoint_id": stale}
        )
        assert r.status_code == 409
        fresh = (await c.http.get(f"/plan/{pid}")).json()["checkpoint_id"]
        ok = await c.http.post(
            f"/plan/{pid}/review", json={"action": "approve", "checkpoint_id": fresh}
        )
        assert ok.status_code == 202


async def test_request_validation(db_path):
    async with running_app(db_path) as c:
        cases = [
            {"end_date": BODY["start_date"]},
            {
                "start_date": str(TODAY - timedelta(days=1)),
                "end_date": str(TODAY + timedelta(days=2)),
            },
            {"end_date": str(TODAY + timedelta(days=60))},
            {"budget_min": 3000},
            {"budget_min": 0},
            {"travelers": 0},
            {"destination": "  "},
        ]
        for override in cases:
            r = await c.http.post("/plan", json={**BODY, **override})
            assert r.status_code == 422, override


async def test_unknown_plan_is_404(db_path):
    async with running_app(db_path) as c:
        for path in ("", "/final", "/history", "/events"):
            assert (await c.http.get(f"/plan/nope{path}")).status_code == 404
        assert (
            await c.http.post("/plan/nope/review", json={"action": "approve"})
        ).status_code == 404


async def test_unknown_destination_ends_failed(db_path):
    from tests.fakes import make_deps

    async with running_app(db_path, deps=make_deps(geocode_empty=True)) as c:
        pid = await create(c, destination="Xyzzyland")
        view = await c.wait_status(pid, "failed")
        assert "Could not find a destination" in view["error"]
        r = await c.http.get(f"/plan/{pid}/final")
        assert r.status_code == 409 and r.json()["detail"]["status"] == "failed"


async def test_history_lists_checkpoints(db_path):
    async with running_app(db_path) as c:
        pid = await create(c)
        await c.wait_status(pid, "awaiting_review")
        rows = (await c.http.get(f"/plan/{pid}/history")).json()["checkpoints"]
        assert len(rows) >= 4
        assert all(r["checkpoint_id"] for r in rows)


async def test_sse_streams_progress_and_ends_at_review(db_path):
    async with running_app(db_path) as c:
        pid = await create(c)
        events = []
        async with c.http.stream("GET", f"/plan/{pid}/events?poll_seconds=0.02") as r:
            assert r.status_code == 200
            async for line in r.aiter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:]))
        assert events[-1]["status"] == "awaiting_review"
        # every emitted event is a genuinely new (status, stage, checkpoint) state
        keys = [(e["status"], e["stage"], e["checkpoint_id"]) for e in events]
        assert len(set(keys)) == len(keys)
        # a plan that is still "running" never claims to be awaiting review, and vice versa
        assert all(e["stage"] != "awaiting review" for e in events if e["status"] == "running")
        assert [e["status"] for e in events].count("awaiting_review") == 1


async def test_healthz(db_path):
    async with running_app(db_path) as c:
        assert (await c.http.get("/healthz")).json() == {"ok": True}


async def test_ui_is_served(db_path):
    async with running_app(db_path) as c:
        r = await c.http.get("/")
        assert r.status_code == 200 and "text/html" in r.headers["content-type"]
        assert "Travel Planner" in r.text
        for asset, kind in (("/static/app.css", "text/css"), ("/static/app.js", "javascript")):
            res = await c.http.get(asset)
            assert res.status_code == 200 and kind in res.headers["content-type"], asset
            assert asset in r.text  # referenced by the page


async def test_currency_defaults_from_the_travellers_country(db_path):
    async with running_app(db_path) as c:
        body = {k: v for k, v in BODY.items() if k != "currency"}
        for nationality, expected in (
            ("India", "INR"),
            ("Brazil", "BRL"),
            ("Narnia", "USD"),
            (None, "USD"),
        ):
            r = await c.http.post("/plan", json={**body, "nationality": nationality})
            assert r.status_code == 202, r.text
            view = (await c.http.get(f"/plan/{r.json()['plan_id']}")).json()
            assert view["request"]["currency"] == expected, nationality
        # an explicit currency always wins
        r = await c.http.post("/plan", json={**body, "nationality": "India", "currency": "eur"})
        assert (await c.http.get(f"/plan/{r.json()['plan_id']}")).json()["request"][
            "currency"
        ] == "EUR"


async def test_countries_endpoint_lists_known_markets(db_path):
    async with running_app(db_path) as c:
        rows = (await c.http.get("/countries")).json()
        india = next(r for r in rows if r["iso2"] == "IN")
        assert india == {"name": "India", "iso2": "IN", "currency": "INR"}
        assert len(rows) > 50


async def test_retry_resumes_a_failed_plan_after_the_cause_is_fixed(db_path):
    class UnauthorizedResponseError(Exception):
        pass

    model = make_model()
    model.fail_times = 1  # like a bad key that is then fixed: the first call fails, later ones work
    model.error = UnauthorizedResponseError("Missing Authentication header")
    async with running_app(db_path, model=model) as c:
        pid = await create(c)
        failed = await c.wait_status(pid, "failed")
        assert "Missing Authentication header" in failed["error"]

        r = await c.http.post(f"/plan/{pid}/retry")
        assert r.status_code == 202 and r.json()["status"] == "running"
        view = await c.wait_status(pid, "awaiting_review")
        assert view["error"] is None and view["draft"]["days"]
        # the failed attempt is still in the timeline
        history = (await c.http.get(f"/plan/{pid}/history")).json()["checkpoints"]
        assert len(history) > len([h for h in history if h["next"]])


async def test_retry_after_a_planner_failure_does_not_repeat_research(db_path):
    calls = {"planner": 0}

    def flaky_planner(messages):
        calls["planner"] += 1
        if calls["planner"] <= 5:  # outlasts the 4 automatic retries
            raise ConnectionError("planner model down")
        return planner_answer(messages)

    model = make_model(DraftBody=flaky_planner)
    async with running_app(db_path, model=model) as c:
        pid = await create(c)
        await c.wait_status(pid, "failed")
        assert [k for k, _ in model.calls] == ["ResearchBody"]
        assert (await c.http.post(f"/plan/{pid}/retry")).status_code == 202
        await c.wait_status(pid, "awaiting_review")
        assert [k for k, _ in model.calls] == ["ResearchBody", "DraftBody"]  # research ran once


async def test_retry_is_refused_unless_the_plan_failed(db_path):
    async with running_app(db_path) as c:
        pid = await create(c)
        await c.wait_status(pid, "awaiting_review")
        r = await c.http.post(f"/plan/{pid}/retry")
        assert r.status_code == 409 and "awaiting_review" in r.json()["detail"]
        assert (await c.http.post("/plan/nope/retry")).status_code == 404
