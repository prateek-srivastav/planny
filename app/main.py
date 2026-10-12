import logging
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from langchain_core.language_models import BaseChatModel

from app.api.routes import router
from app.api.runtime import Runtime
from app.core.checkpointer import open_checkpointer
from app.core.config import Settings, get_settings
from app.core.llm import describe_keys
from app.core.runs import RunManager
from app.graph.build import build_classifier, build_graph
from app.graph.steering import SteeringDesk
from app.tools.registry import Deps

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(
    settings: Settings | None = None,
    *,
    model: BaseChatModel | None = None,
    classifier: BaseChatModel | None = None,
    deps: Deps | None = None,
) -> FastAPI:
    """App factory. `model`, `classifier` and `deps` exist so tests can run fully offline."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with AsyncExitStack() as stack:
            saver = await stack.enter_async_context(open_checkpointer(settings))
            the_deps = deps
            if the_deps is None:
                client = await stack.enter_async_context(
                    httpx.AsyncClient(
                        timeout=httpx.Timeout(30.0),
                        headers={"User-Agent": settings.http_user_agent},
                    )
                )
                the_deps = Deps.from_settings(settings, client)
            for line in describe_keys(settings):
                (log.warning if ("PROBLEM" in line or "not set" in line) else log.info)(line)
            desk = SteeringDesk()
            graph = build_graph(saver, the_deps, model, classifier, steering=desk)
            steer_classifier, steer_kwargs = build_classifier(the_deps, classifier)
            rt = Runtime(
                graph=graph,
                saver=saver,
                runs=RunManager(),
                settings=settings,
                desk=desk,
                classifier=steer_classifier,
                classifier_kwargs=steer_kwargs,
            )
            app.state.rt = rt
            if settings.resume_orphans_on_startup:
                resumed = await rt.recover_orphans()
                if resumed:
                    log.info("resumed %d orphaned plan(s)", len(resumed))
            try:
                yield
            finally:
                await rt.runs.shutdown()

    app = FastAPI(
        title="AI Travel Planner",
        version="0.1.0",
        description="Multi-agent trip planner with a human approval gate. "
        "LangGraph for the workflow, FastAPI for the control plane.",
        lifespan=lifespan,
    )
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
