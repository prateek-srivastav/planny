"""Model failover with per-model cooldowns.

Unlike LangChain's stateless `ModelFallbackMiddleware`, `ModelPool` remembers failures: a model
that fails (after the agent's own retries) is benched and skipped until its cooldown expires.
Persistent failures (spent quota, rejected credentials) are benched longer than transient ones.
If every model is benched, all are tried again rather than refusing to run.

One pool is shared by both agents and all plans in the process. Bench state is in memory and
resets on restart.
"""

import logging
import time
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langgraph.errors import GraphBubbleUp

from app.core.llm import is_auth_error, is_quota_exhausted

log = logging.getLogger(__name__)


class ModelPool(AgentMiddleware):
    """Slot 0 is whatever model the agent was built with; slots 1.. are the fallbacks, in order."""

    def __init__(
        self,
        fallbacks: list[BaseChatModel],
        cooldown_seconds: float = 300,
        *,
        quota_cooldown_seconds: float = 3600,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self.fallbacks = list(fallbacks)
        self.cooldown = cooldown_seconds
        self.quota_cooldown = quota_cooldown_seconds
        self._clock = clock
        self._benched_until: dict[int, float] = {}

    def _cooldown_for(self, exc: Exception) -> float:
        """Seconds a model that failed with `exc` is skipped for.

        Spent daily quotas and rejected credentials do not recover within minutes, so they get
        `quota_cooldown`. Everything else, including a request too large for the model (a later,
        smaller request may fit), gets `cooldown`.
        """
        if is_quota_exhausted(exc) or is_auth_error(exc):
            return self.quota_cooldown
        return self.cooldown

    def benched(self) -> list[int]:
        now = self._clock()
        return [slot for slot, until in self._benched_until.items() if until > now]

    async def awrap_model_call(self, request, handler: Callable[..., Awaitable]):
        slots: list[tuple[int, BaseChatModel | None]] = [(0, None)]
        slots += [(i + 1, model) for i, model in enumerate(self.fallbacks)]
        now = self._clock()
        live = [s for s in slots if self._benched_until.get(s[0], 0) <= now] or slots

        last: Exception | None = None
        for slot, model in live:
            try:
                return await handler(request if model is None else request.override(model=model))
            except GraphBubbleUp:
                raise
            except Exception as exc:  # noqa: BLE001 - any model failure means "try the next one"
                last = exc
                cooldown = self._cooldown_for(exc)
                self._benched_until[slot] = self._clock() + cooldown
                log.warning(
                    "model slot %d failed (%s: %s); benched for %ss",
                    slot,
                    type(exc).__name__,
                    str(exc)[:120],
                    cooldown,
                )
        assert last is not None
        raise last
