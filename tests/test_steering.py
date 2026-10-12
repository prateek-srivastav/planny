"""Mid-run steering: the decision rule, the inbox, and each way guidance reaches a running plan."""

import asyncio
import threading
from datetime import date, timedelta

import pytest
from langchain_core.messages import BaseMessage, HumanMessage

from app.graph.state import RequestChanges
from app.graph.steering import SteeringDesk, apply_changes, decide, new_note
from tests.conftest import running_app
from tests.fakes import make_classifier, make_model, make_request, planner_answer, research_answer

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


# --------------------------------------------------------------------------- decision rule


@pytest.mark.parametrize(
    ("scope", "stage", "action"),
    [
        ("request", "research", "restart"),
        ("request", None, "restart"),
        ("planner", None, "queued"),
        ("planner", "validate", "queued"),
        ("planner", "research", "queued"),
        ("research", "research", "live"),
        ("planner", "planner", "live"),
        ("research", "planner", "rewind"),
        ("research", "routing", "queued"),
    ],
)
def test_decide(scope, stage, action):
    assert decide(scope, stage) == action


def test_desk_take_commit_and_requeue():
    desk = SteeringDesk()
    a, b = new_note("vegetarian", "research"), new_note("slow mornings", "planner")
    desk.post("p", a)
    desk.post("p", b)

    assert desk.take("p", {"research"}) == [a]
    assert desk.pending("p") == [b]
    # cancelled before persisting: the delivered note goes back, ahead of newer ones
    desk.requeue("p")
    assert desk.pending("p") == [a, b]

    assert desk.take("p", {"research", "planner"}) == [a, b]
    assert desk.commit("p") == [a, b]
    assert desk.commit("p") == []
    assert desk.pending("p") == []


def test_requeue_adds_notes_lost_with_a_discarded_branch_once():
    desk = SteeringDesk()
    lost = new_note("beach areas", "research")
    desk.post("p", lost)
    desk.requeue("p", [lost])
    assert desk.pending("p") == [lost]


def test_apply_changes():
    request = make_request()
    nothing = RequestChanges(
        destination=None,
        start_date=None,
        end_date=None,
        budget_min=None,
        budget_max=None,
        travelers=None,
        origin=None,
        interests=None,
    )
    assert apply_changes(request, None) is None
    assert apply_changes(request, nothing) is None

    porto = apply_changes(request, nothing.model_copy(update={"destination": "Porto"}))
    assert porto.destination == "Porto" and porto.start_date == request.start_date

    past = nothing.model_copy(update={"start_date": "2020-01-01", "end_date": "2020-01-04"})
    with pytest.raises(ValueError, match="past"):
        apply_changes(request, past)


# --------------------------------------------------------------------------- end to end


class Gate:
    """Blocks a responder's first call until released, so a test can steer mid-stage.

    Responders run in a worker thread (the fake model is synchronous), so a threading.Event is the
    right primitive and the event loop stays free to serve the steer request.
    """

    def __init__(self, responder):
        self.responder = responder
        self.entered, self.released = threading.Event(), threading.Event()
        self.calls: list[list[BaseMessage]] = []

    def __call__(self, messages):
        self.calls.append(list(messages))
        if len(self.calls) == 1:
            self.entered.set()
            assert self.released.wait(10), "gate never released"
        return self.responder(messages)

    async def wait_entered(self):
        assert await asyncio.to_thread(self.entered.wait, 10), "stage never started"


def _texts(messages: list[BaseMessage]) -> str:
    return "\n".join(str(m.content) for m in messages if isinstance(m, HumanMessage))


async def create(c):
    r = await c.http.post("/plan", json=BODY)
    assert r.status_code == 202, r.text
    return r.json()["plan_id"]


async def test_planner_guidance_sent_during_research_is_queued_for_the_planner(db_path):
    gate = Gate(research_answer)
    model = make_model(ResearchBody=gate)
    async with running_app(db_path, model=model) as c:
        pid = await create(c)
        await gate.wait_entered()
        r = await c.http.post(
            f"/plan/{pid}/steer", json={"message": "Keep evenings free", "scope": "planner"}
        )
        assert r.status_code == 202, r.text
        assert r.json()["action"] == "queued" and r.json()["restarted_from"] is None
        gate.released.set()

        view = await c.wait_status(pid, "awaiting_review")
        research_prompts = [p for k, p in model.calls if k == "ResearchBody"]
        planner_prompts = [p for k, p in model.calls if k == "DraftBody"]
        assert len(research_prompts) == 1 and "Keep evenings free" not in research_prompts[0]
        assert "Keep evenings free" in planner_prompts[0]
        assert [n["text"] for n in view["steering"]] == ["Keep evenings free"]


async def test_guidance_for_the_running_agent_reaches_it_before_its_next_model_call(db_path):
    gate = Gate(planner_answer)
    model = make_model(DraftBody=gate)
    async with running_app(db_path, model=model) as c:
        pid = await create(c)
        await gate.wait_entered()
        r = await c.http.post(
            f"/plan/{pid}/steer", json={"message": "Make day 2 relaxed", "scope": "planner"}
        )
        assert r.json()["action"] == "live"
        gate.released.set()

        view = await c.wait_status(pid, "awaiting_review")
        assert "Make day 2 relaxed" not in _texts(gate.calls[0])
        assert "new guidance while you were working" in _texts(gate.calls[-1])
        assert "Make day 2 relaxed" in _texts(gate.calls[-1])
        assert [k for k, _ in model.calls] == ["ResearchBody", "DraftBody"]  # no rerun needed
        assert [n["text"] for n in view["steering"]] == ["Make day 2 relaxed"]


async def test_guidance_arriving_after_the_last_model_call_gets_another_round(db_path):
    desk_box: dict = {}

    def planner(messages):
        out = planner_answer(messages)
        if isinstance(out, dict) and not desk_box.get("sent"):
            # Arrives while the final answer is being produced: no model call is left to see it.
            desk_box["sent"] = True
            desk_box["desk"].post(desk_box["pid"], new_note("Swap day 1 and 2", "planner"))
        return out

    model = make_model(DraftBody=planner)
    async with running_app(db_path, model=model) as c:
        desk_box["desk"] = c.rt.desk
        r = await c.http.post("/plan", json=BODY)
        desk_box["pid"] = pid = r.json()["plan_id"]
        view = await c.wait_status(pid, "awaiting_review")
        assert [k for k, _ in model.calls] == ["ResearchBody", "DraftBody", "DraftBody"]
        assert [n["text"] for n in view["steering"]] == ["Swap day 1 and 2"]


async def test_research_guidance_during_planning_rewinds_to_research(db_path):
    gate = Gate(planner_answer)
    model = make_model(DraftBody=gate)
    async with running_app(db_path, model=model) as c:
        pid = await create(c)
        await gate.wait_entered()
        r = await c.http.post(
            f"/plan/{pid}/steer", json={"message": "We are vegetarian", "scope": "research"}
        )
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["action"] == "rewind" and body["restarted_from"] == "research_agent"
        gate.released.set()  # the cancelled planner call finishes into the void

        view = await c.wait_status(pid, "awaiting_review")
        kinds = [k for k, _ in model.calls]
        assert kinds.count("ResearchBody") == 2
        assert kinds[-1] == "DraftBody"
        second_research = [p for k, p in model.calls if k == "ResearchBody"][1]
        assert "We are vegetarian" in second_research
        assert "We are vegetarian" in [p for k, p in model.calls if k == "DraftBody"][-1]
        assert [n["text"] for n in view["steering"]] == ["We are vegetarian"]


async def test_a_change_to_the_trip_restarts_with_the_new_request(db_path):
    gate = Gate(planner_answer)
    model = make_model(DraftBody=gate)
    changes = dict.fromkeys(
        ("start_date", "end_date", "budget_min", "budget_max", "travelers", "origin", "interests")
    )
    classifier = make_classifier(
        steer={
            "scope": "request",
            "reason": "destination changed",
            "request_changes": {**changes, "destination": "Porto"},
        }
    )
    async with running_app(db_path, model=model, classifier=classifier) as c:
        pid = await create(c)
        await gate.wait_entered()
        r = await c.http.post(f"/plan/{pid}/steer", json={"message": "Make it Porto instead"})
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["action"] == "restart" and body["restarted_from"] == "validate"
        assert body["request"]["destination"] == "Porto"
        gate.released.set()

        view = await c.wait_status(pid, "awaiting_review")
        assert view["request"]["destination"] == "Porto"
        research = [p for k, p in model.calls if k == "ResearchBody"]
        assert len(research) == 2 and "Destination: Porto" in research[1]


async def test_classifier_failure_hands_guidance_to_the_working_agent(db_path):
    gate = Gate(planner_answer)
    model = make_model(DraftBody=gate)
    async with running_app(db_path, model=model) as c:  # default classifier: no steer answer
        pid = await create(c)
        await gate.wait_entered()
        r = await c.http.post(f"/plan/{pid}/steer", json={"message": "More street food"})
        body = r.json()
        assert body["action"] == "live" and body["scope"] == "planner"
        assert "classifier unavailable" in body["reason"]
        gate.released.set()
        await c.wait_status(pid, "awaiting_review")


async def test_steer_is_refused_unless_the_plan_is_running(db_path):
    async with running_app(db_path) as c:
        pid = await create(c)
        await c.wait_status(pid, "awaiting_review")
        await c.wait_idle(pid)
        r = await c.http.post(f"/plan/{pid}/steer", json={"message": "faster", "scope": "planner"})
        assert r.status_code == 409 and "awaiting_review" in r.json()["detail"]
        assert (await c.http.post("/plan/nope/steer", json={"message": "hi"})).status_code == 404
        r = await c.http.post(f"/plan/{pid}/steer", json={"message": ""})
        assert r.status_code == 422
