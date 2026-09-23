import asyncio
import time

import pytest

from chainroute.plugins.bandit import BanditRouter
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


# ---------------------------------------------------------------- BanditRouter
def winner(candidates):
    """BanditRouter expresses its choice as the unique top-scoring candidate (bias, not
    exclude -- see the module docstring for why), so "who won" means "who has the max score",
    the same thing Chain's own merge rule checks."""
    return max(candidates, key=lambda c: c.score).model


def test_bandit_noop_with_zero_or_one_eligible_candidate():
    p = BanditRouter()
    p.configure(seed=1)
    single = cands(dep("a", model="only"))
    run(p.apply(ctx(), single))
    assert single[0].score == 0


def test_bandit_tries_every_arm_before_exploiting():
    p = BanditRouter()
    p.configure(seed=1)
    seen = set()
    for _ in range(3):
        candidates = cands(dep("a", model="x"), dep("b", model="y"), dep("c", model="z"))
        run(p.apply(ctx(), candidates))
        seen.add(winner(candidates))
    assert seen == {"x", "y", "z"}  # each untried arm gets picked before any repeats


def test_bandit_exploits_the_best_arm_once_warmed_up():
    p = BanditRouter()
    p.configure(seed=1, epsilon=0.0)  # no exploration: purely greedy after warmup
    run(p.on_success(ctx(), Candidate(id="a", model="x", provider="p", deployment={}), cost=0.0))  # reward 1.0
    run(p.on_failure(ctx(), Candidate(id="b", model="y", provider="p", deployment={}), "Error"))    # reward 0.0
    for _ in range(10):
        candidates = cands(dep("a", model="x"), dep("b", model="y"))
        run(p.apply(ctx(), candidates))
        assert winner(candidates) == "x"


def test_bandit_optimize_for_cost_prefers_the_cheaper_arm():
    p = BanditRouter()
    p.configure(seed=1, epsilon=0.0, optimize_for="cost")
    run(p.on_success(ctx(), Candidate(id="a", model="cheap", provider="p", deployment={}), cost=0.01))
    run(p.on_success(ctx(), Candidate(id="b", model="pricey", provider="p", deployment={}), cost=1.00))
    candidates = cands(dep("a", model="cheap"), dep("b", model="pricey"))
    run(p.apply(ctx(), candidates))
    assert winner(candidates) == "cheap"


def test_bandit_failure_counts_against_an_arm_under_cost_optimization():
    p = BanditRouter()
    p.configure(seed=1, epsilon=0.0, optimize_for="cost")
    run(p.on_success(ctx(), Candidate(id="a", model="flaky", provider="p", deployment={}), cost=0.50))
    run(p.on_failure(ctx(), Candidate(id="a", model="flaky", provider="p", deployment={}), "Error"))
    run(p.on_success(ctx(), Candidate(id="b", model="reliable", provider="p", deployment={}), cost=0.50))
    candidates = cands(dep("a", model="flaky"), dep("b", model="reliable"))
    run(p.apply(ctx(), candidates))
    # "flaky" has a failure (reward -1.0) dragging its mean below "reliable"'s single -0.5 success
    assert winner(candidates) == "reliable"


def test_bandit_losers_keep_a_reason_even_though_theyre_not_excluded():
    p = BanditRouter()
    p.configure(seed=1, epsilon=0.0)
    run(p.on_success(ctx(), Candidate(id="a", model="x", provider="p", deployment={}), cost=0.0))
    run(p.on_failure(ctx(), Candidate(id="b", model="y", provider="p", deployment={}), "Error"))
    candidates = cands(dep("a", model="x"), dep("b", model="y"))
    run(p.apply(ctx(), candidates))
    loser = next(c for c in candidates if c.model != winner(candidates))
    assert not loser.excluded and loser.reason is not None and "behind" in loser.reason


def test_bandit_epsilon_explores_a_worse_arm_sometimes():
    p = BanditRouter()
    p.configure(seed=7, epsilon=1.0)  # always explore
    run(p.on_success(ctx(), Candidate(id="a", model="x", provider="p", deployment={}), cost=0.0))
    run(p.on_failure(ctx(), Candidate(id="b", model="y", provider="p", deployment={}), "Error"))
    picks = set()
    for _ in range(20):
        candidates = cands(dep("a", model="x"), dep("b", model="y"))
        run(p.apply(ctx(), candidates))
        picks.add(winner(candidates))
    assert picks == {"x", "y"}  # with epsilon=1.0, both arms get picked over enough tries


def test_bandit_is_deterministic_given_the_same_seed():
    def run_sequence(seed):
        p = BanditRouter()
        p.configure(seed=seed, epsilon=0.5)
        picks = []
        for _ in range(15):
            candidates = cands(dep("a", model="x"), dep("b", model="y"), dep("c", model="z"))
            run(p.apply(ctx(), candidates))
            picks.append(winner(candidates))
            run(p.on_success(ctx(), Candidate(id="?", model=picks[-1], provider="p", deployment={}), cost=0.1))
        return picks
    assert run_sequence(99) == run_sequence(99)


def test_bandit_composes_with_a_pin_from_another_plugin():
    """The reason bias, not exclude (see the module docstring): StickySession's pin from an
    earlier turn must keep winning even once the bandit's own preference has moved on."""
    from chainroute.plugins.sticky_session import StickySession
    bandit = BanditRouter()
    bandit.configure(seed=1, epsilon=0.0)
    bandit._arms = {"x": {"n": 5, "mean": 0.0}, "y": {"n": 5, "mean": 1.0}}  # bandit now prefers "y"
    sticky = StickySession()
    sticky.configure()
    session_ctx = ctx(session_id="s1")
    run(sticky.on_success(session_ctx, cands(dep("a", model="x"))[0]))  # session's first turn used "x"

    for plugins in ([bandit, sticky], [sticky, bandit]):  # composition must not depend on order
        candidates = cands(dep("a", model="x"), dep("b", model="y"))
        turn_ctx = ctx(session_id="s1")
        for plugin in plugins:
            run(plugin.apply(turn_ctx, candidates))
        assert turn_ctx._pinned is not None and turn_ctx._pinned.model == "x"


def test_bandit_rejects_bad_optimize_for_value():
    with pytest.raises(ValueError):
        BanditRouter().configure(optimize_for="latency")
