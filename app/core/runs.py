"""In-process tracking of detached graph runs.

`interrupt()` ends a run, so a plan is only "being worked on" while a task is alive in this
process. This class owns that fact: one active run per plan, tasks kept referenced (asyncio only
holds weak references), cancelled cleanly on shutdown. It is deliberately the *only* in-process
state; everything about the plan itself lives in the checkpointer.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

log = logging.getLogger(__name__)


class RunManager:
    def __init__(self) -> None:
        self._active: set[str] = set()
        self._tasks: dict[str, asyncio.Task] = {}
        self._resuming_from: dict[str, str] = {}

    def busy(self, plan_id: str) -> bool:
        return plan_id in self._active

    def try_claim(self, plan_id: str) -> bool:
        """Reserve the plan for a run. Synchronous, so two racing requests cannot both win."""
        if plan_id in self._active:
            return False
        self._active.add(plan_id)
        return True

    def release(self, plan_id: str) -> None:
        self._active.discard(plan_id)
        self._resuming_from.pop(plan_id, None)

    def mark_resume(self, plan_id: str, checkpoint_id: str | None) -> None:
        """Record the paused checkpoint a review is resuming from (until the run moves past it)."""
        if checkpoint_id:
            self._resuming_from[plan_id] = checkpoint_id

    def resuming_from(self, plan_id: str) -> str | None:
        return self._resuming_from.get(plan_id)

    def launch(
        self,
        plan_id: str,
        work: Callable[[], Awaitable[object]],
        on_error: Callable[[Exception], Awaitable[None]] | None = None,
    ) -> None:
        """Run `work()` detached. The caller holds the claim; it is released when the run ends."""
        self._tasks[plan_id] = asyncio.create_task(self._run(plan_id, work, on_error))

    async def _run(self, plan_id, work, on_error) -> None:
        try:
            await work()
        except asyncio.CancelledError:
            raise  # shutdown: the checkpoint is intact and startup recovery will pick it up
        except Exception as exc:
            log.exception("run for plan %s crashed", plan_id)
            if on_error:
                try:
                    await on_error(exc)
                except Exception:
                    log.exception("could not record failure for plan %s", plan_id)
        finally:
            self.release(plan_id)
            self._tasks.pop(plan_id, None)

    async def wait(self, plan_id: str) -> None:
        task = self._tasks.get(plan_id)
        if task:
            await asyncio.shield(task)

    async def cancel(self, plan_id: str) -> None:
        """Stop the plan's run and wait until it has ended (and released its claim).

        The checkpoint keeps the last completed step: runs use durability="sync".
        """
        task = self._tasks.get(plan_id)
        if task is None or task.done():
            return
        task.cancel()
        await asyncio.wait([task])

    async def shutdown(self) -> None:
        tasks = list(self._tasks.values())
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
