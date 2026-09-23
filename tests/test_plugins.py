import asyncio
import time

import pytest

from chainroute.plugins.budget_guard import BudgetGuard
from chainroute.plugins.canary import WeightedCanary
from chainroute.plugins.keyword import KeywordRoute
from chainroute.plugins.moderation import ModerationGuard
from chainroute.plugins.sensitive_data import SensitiveDataGuard
from chainroute.plugins.sticky_session import StickySession
from chainroute.plugins.tag_affinity import TagAffinity
from chainroute.types import Candidate, RouteContext, Veto


def dep(id_, model="m", provider="openai", tags=None):
    lp = {"model": "%s/%s" % (provider, model)}
    if tags is not None:
        lp["tags"] = tags
    return {"model_info": {"id": id_}, "litellm_params": lp}


def cands(*deps):
    return [Candidate(id=d["model_info"]["id"], model=d["litellm_params"]["model"].split("/", 1)[1],
                       provider=d["litellm_params"]["model"].split("/", 1)[0], deployment=d) for d in deps]


def ctx(**kw):
    base = dict(model_group="g", requested_model="g", messages=[{"role": "user", "content": "hi"}])
    base.update(kw)
    return RouteContext(**base)


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- KeywordRoute
def test_keyword_route_matches_and_pins_within_the_group():
    p = KeywordRoute()
    p.configure(rules=[{"match": "urgent|asap", "prefer_model": "big"}])
    context = ctx(messages=[{"role": "user", "content": "this is URGENT"}])
    candidates = cands(dep("a", model="small"), dep("b", model="big"))
    run(p.apply(context, candidates))
    assert context._pinned is not None and context._pinned.model == "big"


def test_keyword_route_no_match_is_a_noop():
    p = KeywordRoute()
    p.configure(rules=[{"match": "urgent", "prefer_model": "big"}])
    context = ctx(messages=[{"role": "user", "content": "hello"}])
    run(p.apply(context, cands(dep("a", model="small"), dep("b", model="big"))))
    assert context._pinned is None


def test_keyword_route_first_matching_rule_wins():
    p = KeywordRoute()
    p.configure(rules=[{"match": "urgent", "prefer_model": "a"}, {"match": "urgent", "prefer_model": "b"}])
    context = ctx(messages=[{"role": "user", "content": "urgent!"}])
    run(p.apply(context, cands(dep("x", model="a"), dep("y", model="b"))))
    assert context._pinned.model == "a"


def test_keyword_route_skips_a_rule_whose_target_is_not_in_this_group():
    p = KeywordRoute()
    p.configure(rules=[{"match": "urgent", "prefer_model": "not-here"}, {"match": "urgent", "prefer_model": "b"}])
    context = ctx(messages=[{"role": "user", "content": "urgent!"}])
    run(p.apply(context, cands(dep("x", model="a"), dep("y", model="b"))))
    assert context._pinned.model == "b"


# ---------------------------------------------------------------- StickySession
def test_sticky_session_noop_without_a_session_id():
    p = StickySession()
    p.configure()
    c = cands(dep("a"))
    run(p.apply(ctx(session_id=None), c))
    assert not c[0].excluded and c[0].score == 0


def test_sticky_session_pins_after_a_recorded_success():
    p = StickySession()
    p.configure()
    context = ctx(session_id="s1")
    run(p.on_success(context, cands(dep("a"))[0]))
    candidates = cands(dep("a"), dep("b"))
    run(p.apply(context, candidates))
    assert context._pinned is not None and context._pinned.id == "a"


def test_sticky_session_ignores_expired_pins():
    p = StickySession()
    p.configure(ttl_seconds=0.01)
    context = ctx(session_id="s1")
    run(p.on_success(context, cands(dep("a"))[0]))
    time.sleep(0.02)
    candidates = cands(dep("a"), dep("b"))
    run(p.apply(context, candidates))
    assert context._pinned is None


def test_sticky_session_no_op_if_pinned_deployment_gone():
    p = StickySession()
    p.configure()
    context = ctx(session_id="s1")
    run(p.on_success(context, cands(dep("a"))[0]))
    candidates = cands(dep("b"), dep("c"))  # "a" no longer eligible
    run(p.apply(context, candidates))
    assert context._pinned is None


# ---------------------------------------------------------------- TagAffinity
def test_tag_affinity_passthrough_with_no_requested_tags():
    p = TagAffinity()
    p.configure()
    candidates = cands(dep("a", tags=["x"]), dep("b"))
    run(p.apply(ctx(metadata={}), candidates))
    assert not any(c.excluded for c in candidates)


def test_tag_affinity_keeps_only_matching_deployments():
    p = TagAffinity()
    p.configure()
    candidates = cands(dep("a", tags=["team-a"]), dep("b", tags=["team-b"]))
    run(p.apply(ctx(metadata={"tags": ["team-b"]}), candidates))
    assert candidates[0].excluded and not candidates[1].excluded


# ---------------------------------------------------------------- WeightedCanary
def test_canary_zero_percent_always_excludes_the_canary():
    p = WeightedCanary()
    p.configure(canary_model="new", percent=0)
    for session in ("s1", "s2", "s3"):
        candidates = cands(dep("a", model="old"), dep("b", model="new"))
        run(p.apply(ctx(session_id=session), candidates))
        assert candidates[1].excluded and not candidates[0].excluded


def test_canary_hundred_percent_always_keeps_only_the_canary():
    p = WeightedCanary()
    p.configure(canary_model="new", percent=100)
    candidates = cands(dep("a", model="old"), dep("b", model="new"))
    run(p.apply(ctx(session_id="s1"), candidates))
    assert candidates[0].excluded and not candidates[1].excluded


def test_canary_split_is_deterministic_per_session():
    p = WeightedCanary()
    p.configure(canary_model="new", percent=50)
    results = []
    for _ in range(3):
        candidates = cands(dep("a", model="old"), dep("b", model="new"))
        run(p.apply(ctx(session_id="same-session"), candidates))
        results.append(candidates[1].excluded)
    assert len(set(results)) == 1  # same session always lands in the same bucket


# ---------------------------------------------------------------- BudgetGuard
def test_budget_guard_noop_under_cap():
    p = BudgetGuard()
    p.configure(cap=1.0, cheap_models=["mini"])
    candidates = cands(dep("a", model="big"), dep("b", model="mini"))
    run(p.apply(ctx(metadata={"user_api_key_hash": "k"}), candidates))
    assert not any(c.excluded for c in candidates)


def test_budget_guard_steers_to_cheap_once_over_cap():
    p = BudgetGuard()
    p.configure(cap=1.0, cheap_models=["mini"])
    context = ctx(metadata={"user_api_key_hash": "k"})
    run(p.on_success(context, cands(dep("x", model="big"))[0], cost=2.0))
    candidates = cands(dep("a", model="big"), dep("b", model="mini"))
    run(p.apply(context, candidates))
    assert candidates[0].excluded and not candidates[1].excluded


def test_budget_guard_fails_open_if_nothing_cheap_is_eligible():
    p = BudgetGuard()
    p.configure(cap=1.0, cheap_models=["mini"])
    context = ctx(metadata={"user_api_key_hash": "k"})
    run(p.on_success(context, cands(dep("x", model="big"))[0], cost=2.0))
    candidates = cands(dep("a", model="big"), dep("c", model="also-big"))  # no "mini" available
    run(p.apply(context, candidates))
    assert not any(c.excluded for c in candidates)


def test_budget_guard_window_resets_spend():
    p = BudgetGuard()
    p.configure(cap=1.0, window_seconds=0.01, cheap_models=["mini"])
    context = ctx(metadata={"user_api_key_hash": "k"})
    run(p.on_success(context, cands(dep("x", model="big"))[0], cost=2.0))
    time.sleep(0.02)
    candidates = cands(dep("a", model="big"), dep("b", model="mini"))
    run(p.apply(context, candidates))
    assert not any(c.excluded for c in candidates)


# ---------------------------------------------------------------- SensitiveDataGuard
def test_sensitive_guard_passthrough_when_nothing_matches():
    p = SensitiveDataGuard()
    p.configure(trusted_providers=["azure"])
    candidates = cands(dep("a", provider="openai"))
    run(p.apply(ctx(messages=[{"role": "user", "content": "what's the weather"}]), candidates))
    assert not candidates[0].excluded


def test_sensitive_guard_excludes_untrusted_providers_on_match():
    p = SensitiveDataGuard()
    p.configure(trusted_providers=["azure"], on_match="exclude")
    candidates = cands(dep("a", provider="openai"), dep("b", provider="azure"))
    run(p.apply(ctx(messages=[{"role": "user", "content": "my ssn is 123-45-6789"}]), candidates))
    assert candidates[0].excluded and not candidates[1].excluded


def test_sensitive_guard_vetoes_when_nothing_trusted_is_eligible():
    p = SensitiveDataGuard()
    p.configure(trusted_providers=["azure"], on_match="veto")
    candidates = cands(dep("a", provider="openai"))
    with pytest.raises(Veto):
        run(p.apply(ctx(messages=[{"role": "user", "content": "my ssn is 123-45-6789"}]), candidates))


def test_sensitive_guard_rejects_bad_on_match_value():
    with pytest.raises(ValueError):
        SensitiveDataGuard().configure(on_match="nope")


# ---------------------------------------------------------------- ModerationGuard
def _guard(**kw):
    g = ModerationGuard()
    g.configure(api_key="test-key", **kw)
    return g


def _async(value):
    async def f(text):
        return value
    return f


def test_moderation_guard_noop_without_an_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # don't let a real dev/CI env var pass this
    g = ModerationGuard()
    g.configure()
    candidates = cands(dep("a", provider="openai"))
    calls = []
    g._call_api = lambda text: calls.append(text)  # would raise if actually awaited -- must not be called
    run(g.apply(ctx(messages=[{"role": "user", "content": "hello"}]), candidates))
    assert not candidates[0].excluded and calls == []


def test_moderation_guard_passthrough_when_not_flagged():
    g = _guard(trusted_providers=["azure"])
    g._call_api = _async(False)
    candidates = cands(dep("a", provider="openai"))
    run(g.apply(ctx(messages=[{"role": "user", "content": "hello"}]), candidates))
    assert not candidates[0].excluded


def test_moderation_guard_excludes_untrusted_when_flagged():
    g = _guard(trusted_providers=["azure"], on_match="exclude")
    g._call_api = _async(True)
    candidates = cands(dep("a", provider="openai"), dep("b", provider="azure"))
    run(g.apply(ctx(messages=[{"role": "user", "content": "bad stuff"}]), candidates))
    assert candidates[0].excluded and not candidates[1].excluded


def test_moderation_guard_vetoes_when_nothing_trusted_is_eligible():
    g = _guard(trusted_providers=["azure"], on_match="veto")
    g._call_api = _async(True)
    candidates = cands(dep("a", provider="openai"))
    with pytest.raises(Veto):
        run(g.apply(ctx(messages=[{"role": "user", "content": "bad stuff"}]), candidates))


def test_moderation_guard_caches_identical_text():
    g = _guard()
    calls = []

    async def fake_call(text):
        calls.append(text)
        return False
    g._call_api = fake_call
    context = ctx(messages=[{"role": "user", "content": "same message"}])
    run(g.apply(context, cands(dep("a"))))
    run(g.apply(context, cands(dep("a"))))
    assert calls == ["same message"]  # second call was served from cache


def test_moderation_guard_fails_open_by_default():
    g = _guard(on_match="veto")

    async def boom(text):
        raise RuntimeError("api is down")
    g._call_api = boom
    candidates = cands(dep("a"))
    with pytest.raises(RuntimeError):
        # ModerationGuard itself doesn't swallow this -- Chain's own isolation (see chain.py) is
        # what actually makes this fail open in a real chain; this test documents that contract.
        run(g.apply(ctx(messages=[{"role": "user", "content": "hi"}]), candidates))


def test_moderation_guard_fails_closed_when_configured():
    g = _guard(fail_closed=True)

    async def boom(text):
        raise RuntimeError("api is down")
    g._call_api = boom
    with pytest.raises(Veto):
        run(g.apply(ctx(messages=[{"role": "user", "content": "hi"}]), cands(dep("a"))))


def test_moderation_guard_real_call_api_parses_the_apis_response_shape():
    """Exercises the real `_call_api` (not the monkeypatch the other tests use above) against a
    fake httpx client, to prove it parses OpenAI's actual response shape correctly -- both the
    overall `flagged` boolean and the per-category score threshold override."""
    class FakeResponse:
        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self._body

    class FakeClient:
        def __init__(self, body):
            self._body = body

        async def post(self, *a, **kw):
            return FakeResponse(self._body)

    g = _guard(thresholds={"violence": 0.3})
    g._client = FakeClient({"results": [{"flagged": False, "category_scores": {"violence": 0.5}}]})
    assert run(g._call_api("some violent text")) is True  # under no built-in flag, but over our threshold

    g._client = FakeClient({"results": [{"flagged": True, "category_scores": {}}]})
    assert run(g._call_api("flagged text")) is True  # the API's own flag alone is enough

    g._client = FakeClient({"results": [{"flagged": False, "category_scores": {"violence": 0.1}}]})
    assert run(g._call_api("benign text")) is False


def test_moderation_guard_rejects_bad_on_match_value():
    with pytest.raises(ValueError):
        ModerationGuard().configure(api_key="x", on_match="nope")
