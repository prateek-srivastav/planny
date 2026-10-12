"""The criterion the brief calls core: the workflow must survive the pause, including a restart."""

import os
import threading
import uuid

import pytest

from tests.conftest import running_app
from tests.fakes import make_model, research_answer
from tests.test_api import create


async def test_pause_survives_a_full_restart(db_path):
    # --- process 1: plan until the human gate, then shut everything down
    async with running_app(db_path) as c1:
        pid = await create(c1)
        before = await c1.wait_status(pid, "awaiting_review")

    # --- process 2: brand-new app, graph and model objects, same database file
    async with running_app(db_path) as c2:
        after = (await c2.http.get(f"/plan/{pid}")).json()
        assert after["status"] == "awaiting_review"
        assert after["draft"] == before["draft"]
        assert after["checkpoint_id"] == before["checkpoint_id"]

        r = await c2.http.post(f"/plan/{pid}/review", json={"action": "approve"})
        assert r.status_code == 202
        await c2.wait_status(pid, "completed")
        final = (await c2.http.get(f"/plan/{pid}/final")).json()
        assert final["itinerary"]["title"] == before["draft"]["title"]
        assert final["itinerary"]["days"] == before["draft"]["days"]
        assert final["status"] == "approved"

    # --- process 3: even the finished plan is still readable
    async with running_app(db_path) as c3:
        assert (await c3.http.get(f"/plan/{pid}/final")).status_code == 200


async def test_rejection_counter_and_history_survive_restart(db_path):
    async with running_app(db_path) as c1:
        pid = await create(c1)
        await c1.wait_status(pid, "awaiting_review")
        await c1.http.post(f"/plan/{pid}/review", json={"action": "reject", "feedback": "cheaper"})
        await c1.wait_idle(pid)
        await c1.wait_status(pid, "awaiting_review")

    async with running_app(db_path) as c2:
        view = (await c2.http.get(f"/plan/{pid}")).json()
        assert view["status"] == "awaiting_review"
        assert view["revisions_used"] == 1
        assert view["draft"]["title"] == "Revised Lisbon trip"


async def test_run_cut_off_mid_research_is_resumed_on_startup(db_path):
    gate = threading.Event()

    def stuck(messages):
        gate.wait(15)
        return research_answer(messages)

    # --- process 1 dies while the research agent is still working
    async with running_app(db_path, model=make_model(ResearchBody=stuck)) as c1:
        pid = await create(c1)
        view = (await c1.http.get(f"/plan/{pid}")).json()
        assert view["status"] == "running"
    gate.set()  # release the orphaned worker thread; the cancelled run never saw its answer

    # --- process 2 starts, finds the orphan in the checkpoint store and finishes it
    async with running_app(db_path) as c2:
        view = await c2.wait_status(pid, "awaiting_review")
        assert view["draft"]["days"]


async def test_recovery_can_be_disabled(db_path):
    gate = threading.Event()

    def stuck(messages):
        gate.wait(15)
        return research_answer(messages)

    async with running_app(db_path, model=make_model(ResearchBody=stuck)) as c1:
        pid = await create(c1)
    gate.set()
    async with running_app(db_path, resume_orphans_on_startup=False) as c2:
        assert not c2.rt.runs.busy(pid)
        assert (await c2.http.get(f"/plan/{pid}")).json()["status"] == "running"


async def test_sqlite_is_the_default_store_not_memory(db_path):
    async with running_app(db_path) as c:
        await create(c)
    import os

    assert os.path.getsize(db_path) > 0


@pytest.mark.skipif(not os.getenv("POSTGRES_TEST_URL"), reason="set POSTGRES_TEST_URL to run")
async def test_pause_survives_restart_on_postgres(db_path):
    """Same restart scenario, Postgres backend. Run with POSTGRES_TEST_URL=postgresql://..."""
    url = os.environ["POSTGRES_TEST_URL"]
    async with running_app(db_path, postgres_url=url) as c1:
        pid = await create(c1)
        before = await c1.wait_status(pid, "awaiting_review")
    async with running_app(db_path, postgres_url=url) as c2:
        after = (await c2.http.get(f"/plan/{pid}")).json()
        assert after["status"] == "awaiting_review" and after["draft"] == before["draft"]
        assert (
            await c2.http.post(f"/plan/{pid}/review", json={"action": "approve"})
        ).status_code == 202
        await c2.wait_status(pid, "completed")
    assert uuid.UUID(pid)
