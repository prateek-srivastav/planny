import asyncio
from contextlib import asynccontextmanager

import httpx
import pytest

from app.core.checkpointer import open_checkpointer
from app.core.config import Settings
from app.main import create_app
from tests.fakes import make_classifier, make_deps, make_model


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch):
    """Tests must not see (or spend) real keys, nor pick up a developer's .env or env vars.

    `app.main` loads `.env` into os.environ at import, so clear the environment variable of
    *every* setting (not a hand-picked list) plus the provider keys the SDKs read directly.
    """
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "checkpoints.sqlite")


async def open_saver(path: str):
    """Context manager for a SQLite checkpointer, as the app builds it."""
    return open_checkpointer(Settings(_env_file=None, sqlite_path=path, postgres_url=None))


# --------------------------------------------------------------------------- API helpers


class Client:
    """httpx client bound to an app instance, plus the handles tests need."""

    def __init__(self, http: httpx.AsyncClient, app):
        self.http, self.app = http, app

    @property
    def rt(self):
        return self.app.state.rt

    async def wait_idle(self, plan_id: str):
        await asyncio.wait_for(self.rt.runs.wait(plan_id), timeout=20)

    async def wait_status(self, plan_id: str, status: str, timeout: float = 20):
        async def poll():
            while True:
                r = await self.http.get(f"/plan/{plan_id}")
                if r.json()["status"] == status:
                    return r.json()
                await asyncio.sleep(0.02)

        return await asyncio.wait_for(poll(), timeout)


@asynccontextmanager
async def running_app(db_path: str, *, model=None, classifier=None, deps=None, **settings):
    deps = deps or make_deps()
    s = Settings(
        **{
            "_env_file": None,
            "sqlite_path": db_path,
            "postgres_url": None,
            "retry_initial_delay": 0.0,
            "max_revisions": 3,
            **settings,
        }
    )
    app = create_app(
        s, model=model or make_model(), classifier=classifier or make_classifier(), deps=deps
    )
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            yield Client(http, app)
