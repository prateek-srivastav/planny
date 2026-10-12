"""Run-scoped side channel between tools and the node that wraps an agent.

Tools run inside the agent's own subgraph and cannot write to the parent `PlanState`. Instead of
asking the LLM to copy data around (slow, lossy), tools drop what they learn here and the node
reads it after the agent finishes. This is how:

* soft tool failures reach `state["degraded"]`
* real search-result URLs become `ResearchBrief.sources` (the model only names which ones it used)
* the weather summary is attached to the brief verbatim
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from app.graph.state import ConditionsSummary, ExchangeRate, Source


@dataclass
class RunCollector:
    degraded: list[str] = field(default_factory=list)
    sources: dict[str, Source] = field(default_factory=dict)
    conditions: ConditionsSummary | None = None
    exchange_rate: ExchangeRate | None = None
    search_country: str | None = None  # traveller's market, for localising search


_current: ContextVar[RunCollector | None] = ContextVar("run_collector", default=None)


@contextmanager
def collecting(search_country: str | None = None):
    collector = RunCollector(search_country=search_country)
    token = _current.set(collector)
    try:
        yield collector
    finally:
        _current.reset(token)


def current() -> RunCollector:
    """Return the active collector (or a throwaway one if a tool runs outside a node)."""
    return _current.get() or RunCollector()


class ToolFailure(Exception):
    """A tool could not do its job. The message names the tool and is shown to the reviewer.

    `retryable=False` marks errors a retry cannot fix (missing key, exhausted credits, bad
    request). Tools turn those into a soft failure straight away (see `soft_fail` in registry.py)
    instead of letting the retry middleware burn its backoff on them.
    """

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


def record_degraded(message: str) -> None:
    current().degraded.append(message)
