"""Run the real API with a scripted LLM and mocked network. No keys needed.

    uv run python scripts/offline_server.py   # serves on http://127.0.0.1:8765

Everything except the LLM's words and the network responses is real: the FastAPI app, the graph,
the agents' tool loops, the SQLite checkpointer. The itinerary content is canned.
Use it to poke at the API contract (status transitions, the HITL pause, 409s) for free.
"""

import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from tests.fakes import make_classifier, make_deps, make_model  # noqa: E402

settings = Settings(_env_file=None, sqlite_path="data/offline-demo.sqlite", retry_initial_delay=0)
app = create_app(settings, model=make_model(), classifier=make_classifier(), deps=make_deps())

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
