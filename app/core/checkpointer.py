"""Checkpointer factory: SQLite (default) or Postgres (when POSTGRES_URL is set).

InMemorySaver is deliberately not offered here: the whole point of the HITL design is that
state survives the pause, including a process restart.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from app.core.config import Settings
from app.graph import state as plan_models

# Pydantic models that live in checkpoints. Listing them explicitly puts deserialization in strict
# mode: a checkpoint cannot instantiate arbitrary classes, only these (plus LangGraph's own).
ALLOWED_MODELS = [
    (plan_models.__name__, name)
    for name, obj in vars(plan_models).items()
    if isinstance(obj, type) and obj.__module__ == plan_models.__name__
]


def make_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MODELS)


@asynccontextmanager
async def open_checkpointer(settings: Settings) -> AsyncIterator[BaseCheckpointSaver]:
    if settings.postgres_url:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async with AsyncPostgresSaver.from_conn_string(
            settings.postgres_url, serde=make_serde()
        ) as saver:
            await saver.setup()
            yield saver
    else:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(settings.sqlite_path) as conn:
            saver = AsyncSqliteSaver(conn, serde=make_serde())
            await saver.setup()
            yield saver
