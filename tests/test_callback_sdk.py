"""ChainRoute against litellm.Router directly (mock_response, no network, no keys) -- proving
the plugin chain actually changes which deployment gets called, not just that the plugin logic
looks right in isolation."""
import asyncio
import time

import litellm
import pytest
from litellm import Router

from chainroute.callback import ChainRoute
from chainroute.plugins.budget_guard import BudgetGuard
from chainroute.plugins.keyword import KeywordRoute
from chainroute.plugins.sensitive_data import SensitiveDataGuard
from chainroute.plugins.sticky_session import StickySession
from chainroute.types import RoutingPlugin, Veto


@pytest.fixture()
def lens_factory():
    # litellm logs success/failure through a process-wide background worker
    # (litellm_core_utils.logging_worker.GLOBAL_LOGGING_WORKER) whose queue/semaphore/worker-task
    # are bound to whichever event loop was running when it was last used. Each test here runs
    # its own `asyncio.run()` -- a fresh loop -- so the worker's *own* loop-change detection
    # (`_ensure_queue`) would normally reset those attributes on next use; we just do it eagerly
    # and directly; going through the worker's own `stop()` instead fails outright, since it
    # tries to `cancel()`/`gather()` tasks that belong to the *previous* (already-closed) loop.
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER as _w
    _w._queue = None
    _w._sem = None
    _w._worker_task = None
    _w._running_tasks.clear()
    _w._bound_loop = None

    saved = litellm.callbacks

    def make(plugins):
        cr = ChainRoute(plugins=plugins)
        litellm.callbacks = [cr]
        return cr

    yield make
    litellm.callbacks = saved


async def wait_until(predicate, timeout=5.0, interval=0.05):
    """LiteLLM logs success/failure through a background worker, so "the previous call's outcome
    has reached the plugin" isn't guaranteed the instant `acompletion` returns -- a fixed
    `sleep()` here is a race under load (it was; see git history). Poll a real condition instead."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError("condition never became true within %ss" % timeout)


def test_sticky_session_keeps_one_session_on_one_deployment(lens_factory):
    cr = lens_factory([StickySession()])
    r = Router(model_list=[
        {"model_name": "fast", "litellm_params": {"model": "openai/a", "api_key": "x", "mock_response": "a"},
         "model_info": {"id": "dep-a"}},
        {"model_name": "fast", "litellm_params": {"model": "anthropic/b", "api_key": "x", "mock_response": "b"},
         "model_info": {"id": "dep-b"}},
    ], routing_strategy="simple-shuffle")

    async def go():
        chosen = []
        for i in range(12):
            resp = await r.acompletion(model="fast", messages=[{"role": "user", "content": "turn %d" % i}],
                                        metadata={"session_id": "sticky-1"})
            chosen.append(resp.model)
            # wait for *this* turn's outcome to reach StickySession before sending the next turn,
            # or the next turn's `apply()` could run against a stale (or not-yet-set) pin.
            await wait_until(lambda: not cr._pending)
        return chosen
    models = asyncio.run(go())
    assert len(set(models)) == 1, "StickySession should have kept every turn on the same deployment: %s" % models


def test_keyword_route_prefers_a_specific_model_within_the_group(lens_factory):
    lens_factory([KeywordRoute()])
    plugins = litellm.callbacks[0].chain.plugins
    plugins[0].configure(rules=[{"match": "urgent", "prefer_model": "fast"}])
    r = Router(model_list=[
        {"model_name": "chat", "litellm_params": {"model": "openai/normal", "api_key": "x", "mock_response": "n"}},
        {"model_name": "chat", "litellm_params": {"model": "openai/fast", "api_key": "x", "mock_response": "f"}},
    ], routing_strategy="simple-shuffle")

    async def go():
        # run several times: without the plugin, simple-shuffle would pick randomly between the two
        return [await r.acompletion(model="chat", messages=[{"role": "user", "content": "this is URGENT"}])
                for _ in range(8)]
    resps = asyncio.run(go())
    assert all(r.model == "fast" for r in resps)


def test_sensitive_data_guard_vetoes_end_to_end(lens_factory):
    lens_factory([SensitiveDataGuard(), ])
    litellm.callbacks[0].chain.plugins[0].configure(trusted_providers=["azure"], on_match="veto")
    r = Router(model_list=[
        {"model_name": "chat", "litellm_params": {"model": "openai/normal", "api_key": "x", "mock_response": "n"}},
    ])

    async def go():
        await r.acompletion(model="chat", messages=[{"role": "user", "content": "my ssn is 123-45-6789"}])
    with pytest.raises(litellm.BadRequestError):
        asyncio.run(go())


def test_sensitive_data_guard_allows_the_trusted_provider_through(lens_factory):
    lens_factory([SensitiveDataGuard()])
    litellm.callbacks[0].chain.plugins[0].configure(trusted_providers=["azure"], on_match="veto")
    r = Router(model_list=[
        {"model_name": "chat", "litellm_params": {"model": "openai/normal", "api_key": "x", "mock_response": "n"}},
        {"model_name": "chat", "litellm_params": {"model": "azure/safe", "api_key": "x", "mock_response": "s",
                                                   "custom_llm_provider": "azure"}},
    ])

    async def go():
        return await r.acompletion(model="chat", messages=[{"role": "user", "content": "my ssn is 123-45-6789"}])
    resp = asyncio.run(go())
    assert resp.model == "safe"


def test_budget_guard_steers_after_spend_crosses_cap_end_to_end(lens_factory):
    guard = BudgetGuard()
    guard.configure(cap=0.01, cheap_models=["mini"])
    lens_factory([guard])

    # Prime the spend directly rather than through a real call: litellm logs success/failure via
    # a process-wide background worker whose queue is bound to whichever event loop is currently
    # running, and under pytest's rapid-fire separate `asyncio.run()` calls per test, that
    # queue can be reset (by an earlier test, or by this one) before a real call's logging
    # coroutine is ever drained -- silently losing the event with no error raised anywhere. That's
    # a real quirk in a test harness that churns through many short-lived event loops, not
    # something a long-running proxy process (one event loop for its whole lifetime) hits, and
    # it's not what this test exists to cover -- `test_plugins.py` already proves on_success
    # updates spend correctly, in isolation. What's novel here, and worth a real Router for, is
    # proving `apply()`'s exclusion actually reaches litellm's real async_filter_deployments hook.
    guard._spend["user-1"] = {"window_start": time.time(), "total": guard.cap}

    r = Router(model_list=[
        {"model_name": "chat", "litellm_params": {"model": "openai/big", "api_key": "x", "mock_response": "b"}},
        {"model_name": "chat", "litellm_params": {"model": "openai/mini", "api_key": "x", "mock_response": "m"}},
    ], routing_strategy="simple-shuffle")

    async def go():
        # already over cap: every one of these should be steered to "mini", regardless of what
        # simple-shuffle would otherwise have picked at random.
        return [await r.acompletion(model="chat", messages=[{"role": "user", "content": "hi there"}],
                                     metadata={"user_api_key_hash": "user-1"}) for _ in range(8)]
    resps = asyncio.run(go())
    assert all(resp.model == "mini" for resp in resps), [r.model for r in resps]


def test_a_plugin_that_always_raises_never_breaks_the_request(lens_factory):
    class Broken(RoutingPlugin):
        name = "Broken"

        async def apply(self, ctx, candidates):
            raise RuntimeError("oops")

    lens_factory([Broken()])
    r = Router(model_list=[
        {"model_name": "chat", "litellm_params": {"model": "openai/ok", "api_key": "x", "mock_response": "ok"}},
    ])

    async def go():
        return await r.acompletion(model="chat", messages=[{"role": "user", "content": "hi"}])
    resp = asyncio.run(go())
    assert resp.model == "ok"
