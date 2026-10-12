"""ModelPool: rotate across models, bench the ones that just failed."""

import pytest

from app.core.model_pool import ModelPool


class Req:
    def __init__(self, model="primary"):
        self.model = model

    def override(self, model):
        return Req(model)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make(dead: set[str], fallbacks=("fb1", "fb2"), cooldown=300):
    clock, calls = Clock(), []

    async def handler(req):
        calls.append(req.model)
        if req.model in dead:
            raise ConnectionError(f"{req.model} is rate limited")
        return f"answer from {req.model}"

    return ModelPool(list(fallbacks), cooldown, clock=clock), handler, calls, clock


async def test_healthy_primary_is_used_and_fallbacks_stay_idle():
    pool, handler, calls, _ = make(dead=set())
    assert await pool.awrap_model_call(Req(), handler) == "answer from primary"
    assert calls == ["primary"] and pool.benched() == []


async def test_failed_primary_rolls_over_in_order():
    pool, handler, calls, _ = make(dead={"primary", "fb1"})
    assert await pool.awrap_model_call(Req(), handler) == "answer from fb2"
    assert calls == ["primary", "fb1", "fb2"]


async def test_failed_models_are_benched_so_later_calls_skip_them():
    pool, handler, calls, _ = make(dead={"primary"})
    await pool.awrap_model_call(Req(), handler)
    calls.clear()
    for _ in range(5):  # the next five model calls of the plan
        assert await pool.awrap_model_call(Req(), handler) == "answer from fb1"
    assert calls == ["fb1"] * 5  # the dead primary was never touched again
    assert pool.benched() == [0]


async def test_a_benched_model_is_retried_after_its_cooldown():
    pool, handler, calls, clock = make(dead={"primary"}, cooldown=300)
    await pool.awrap_model_call(Req(), handler)
    clock.t += 299
    calls.clear()
    await pool.awrap_model_call(Req(), handler)
    assert calls == ["fb1"]  # still benched
    clock.t += 2  # past the cooldown, and by now the primary has recovered
    calls.clear()

    async def healthy(req):
        calls.append(req.model)
        return f"answer from {req.model}"

    assert await pool.awrap_model_call(Req(), healthy) == "answer from primary"
    assert calls == ["primary"] and pool.benched() == []


async def test_when_everything_is_benched_it_tries_everything_again():
    pool, handler, calls, _ = make(dead={"primary", "fb1", "fb2"})
    with pytest.raises(ConnectionError):
        await pool.awrap_model_call(Req(), handler)
    calls.clear()
    dead_now = {"primary", "fb1", "fb2"}

    async def recovered(req):
        calls.append(req.model)
        if req.model in dead_now - {"fb2"}:
            raise ConnectionError("still down")
        return f"answer from {req.model}"

    assert await pool.awrap_model_call(Req(), recovered) == "answer from fb2"
    assert calls == ["primary", "fb1", "fb2"]  # nothing was left out because all were benched


async def test_all_fail_raises_the_last_error():
    pool, handler, _, _ = make(dead={"primary", "fb1", "fb2"})
    with pytest.raises(ConnectionError, match="fb2"):
        await pool.awrap_model_call(Req(), handler)


class StatusError(Exception):
    def __init__(self, status_code: int, message: str = ""):
        super().__init__(message or f"HTTP {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (StatusError(429, "Rate limit exceeded: free-models-per-day"), 3600),
        (StatusError(413, "Request too large for model"), 300),  # a smaller request may fit
        (StatusError(401, "User not found."), 3600),
        (StatusError(429, "Rate limit exceeded"), 300),
        (StatusError(503, "provider overloaded"), 300),
        (ConnectionError("reset by peer"), 300),
    ],
)
def test_cooldown_depends_on_whether_the_failure_is_persistent(exc, expected):
    pool = ModelPool([], 300, quota_cooldown_seconds=3600)
    assert pool._cooldown_for(exc) == expected


async def test_a_spent_quota_stays_benched_past_the_transient_cooldown():
    clock, calls = Clock(), []

    async def handler(req):
        calls.append(req.model)
        if req.model == "primary":
            raise StatusError(429, "Rate limit exceeded: free-models-per-day")
        return f"answer from {req.model}"

    pool = ModelPool(["fb1"], 300, quota_cooldown_seconds=3600, clock=clock)
    await pool.awrap_model_call(Req(), handler)
    clock.t += 301  # past the transient cooldown, well inside the quota cooldown
    calls.clear()
    assert await pool.awrap_model_call(Req(), handler) == "answer from fb1"
    assert calls == ["fb1"]
