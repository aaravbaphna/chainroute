"""ChainRoute against litellm.Router directly (mock_response, no network, no keys) -- proving
the plugin chain actually changes which deployment gets called, not just that the plugin logic
looks right in isolation."""
import asyncio
import time

import litellm
import pytest
from litellm import Router

from chainroute.callback import ChainRoute
from chainroute.plugins.bandit import BanditRouter
from chainroute.plugins.budget_guard import BudgetGuard
from chainroute.plugins.keyword import KeywordRoute
from chainroute.plugins.moderation import ModerationGuard
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

    def make(plugins, mode="enforce", default_timeout_s=None):
        kw = {"default_timeout_s": default_timeout_s} if default_timeout_s is not None else {}
        cr = ChainRoute(plugins=plugins, mode=mode, **kw)
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


def test_shadow_mode_lets_a_would_be_veto_through_end_to_end(lens_factory):
    lens_factory([SensitiveDataGuard()], mode="shadow")
    litellm.callbacks[0].chain.plugins[0].configure(trusted_providers=["azure"], on_match="veto")
    r = Router(model_list=[
        {"model_name": "chat", "litellm_params": {"model": "openai/normal", "api_key": "x", "mock_response": "n"}},
    ])

    async def go():
        return await r.acompletion(model="chat", messages=[{"role": "user", "content": "my ssn is 123-45-6789"}])
    # enforce mode raises for this exact request (see test_sensitive_data_guard_vetoes_end_to_end);
    # shadow mode computes the same veto but must not act on it.
    resp = asyncio.run(go())
    assert resp.model == "normal"


def test_shadow_mode_lets_an_exclude_through_end_to_end(lens_factory):
    class ExcludeMini(RoutingPlugin):
        name = "ExcludeMini"

        async def apply(self, ctx, candidates):
            for c in candidates:
                if c.model == "mini":
                    c.exclude("shadow test")

    lens_factory([ExcludeMini()], mode="shadow")
    r = Router(model_list=[
        {"model_name": "chat", "litellm_params": {"model": "openai/mini", "api_key": "x", "mock_response": "m"}},
    ])

    async def go():
        return await r.acompletion(model="chat", messages=[{"role": "user", "content": "hi"}])
    resp = asyncio.run(go())
    assert resp.model == "mini"  # would have been excluded in enforce mode; shadow mode let it through


def test_a_hung_plugin_does_not_block_the_request_end_to_end(lens_factory):
    class Hangs(RoutingPlugin):
        name = "Hangs"

        async def apply(self, ctx, candidates):
            await asyncio.sleep(10)

    lens_factory([Hangs()], default_timeout_s=0.05)
    r = Router(model_list=[
        {"model_name": "chat", "litellm_params": {"model": "openai/ok", "api_key": "x", "mock_response": "ok"}},
    ])

    async def go():
        return await asyncio.wait_for(
            r.acompletion(model="chat", messages=[{"role": "user", "content": "hi"}]), timeout=2.0)
    resp = asyncio.run(go())  # would time out at 2s (the outer wait_for) if the 10s sleep weren't bounded
    assert resp.model == "ok"


def test_moderation_guard_excludes_flagged_content_end_to_end(lens_factory):
    guard = ModerationGuard()
    guard.configure(api_key="test-key", trusted_providers=["azure"], on_match="exclude")

    async def flagged(text):
        return True
    guard._call_api = flagged
    lens_factory([guard])
    r = Router(model_list=[
        {"model_name": "chat", "litellm_params": {"model": "openai/normal", "api_key": "x", "mock_response": "n"}},
        {"model_name": "chat", "litellm_params": {"model": "azure/safe", "api_key": "x", "mock_response": "s",
                                                   "custom_llm_provider": "azure"}},
    ])

    async def go():
        return await r.acompletion(model="chat", messages=[{"role": "user", "content": "bad stuff"}])
    resp = asyncio.run(go())
    assert resp.model == "safe"


def test_moderation_guard_fails_open_end_to_end_when_the_api_errors(lens_factory):
    guard = ModerationGuard()
    guard.configure(api_key="test-key", on_match="veto")

    async def boom(text):
        raise RuntimeError("moderation api is down")
    guard._call_api = boom
    lens_factory([guard])  # ChainRoute's default (enforce) mode -- Chain's own isolation, not the
    # plugin's, is what makes this fail open; see chain.py and the "fails open by default" unit test.
    r = Router(model_list=[
        {"model_name": "chat", "litellm_params": {"model": "openai/ok", "api_key": "x", "mock_response": "ok"}},
    ])

    async def go():
        return await r.acompletion(model="chat", messages=[{"role": "user", "content": "hi"}])
    resp = asyncio.run(go())
    assert resp.model == "ok"


def test_bandit_router_prefers_the_learned_winner_end_to_end(lens_factory):
    # The learning itself (on_success/on_failure updating an arm's running reward) is already
    # covered thoroughly in test_plugins.py, in isolation. Priming it directly here -- rather
    # than warming it up through real failing calls -- sidesteps the same background
    # logging-worker lag documented on the BudgetGuard test above; what's actually novel enough
    # to need a real Router for is proving the *real* async_filter_deployments hook enforces the
    # bandit's exclude-based choice against the deployment that would otherwise still be eligible.
    bandit = BanditRouter()
    bandit.configure(seed=3, epsilon=0.0)  # purely greedy: no exploration once every arm is known
    bandit._arms = {"good": {"n": 5, "mean": 1.0}, "bad": {"n": 5, "mean": 0.0}}
    lens_factory([bandit])
    r = Router(model_list=[
        {"model_name": "chat", "litellm_params": {"model": "openai/good", "api_key": "x", "mock_response": "ok"}},
        {"model_name": "chat", "litellm_params": {"model": "openai/bad", "api_key": "x",
                                                   "mock_response": "litellm.RateLimitError"}},
    ], num_retries=0, routing_strategy="simple-shuffle")

    async def go():
        return [await r.acompletion(model="chat", messages=[{"role": "user", "content": "hi"}])
                for _ in range(8)]
    resps = asyncio.run(go())
    # "bad" would raise RateLimitError if ever selected -- every one of these succeeding at all
    # already proves the bandit's choice (not simple-shuffle's default randomness) is what wins.
    assert all(resp.model == "good" for resp in resps)
