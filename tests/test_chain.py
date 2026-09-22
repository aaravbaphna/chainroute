import asyncio

import pytest

from chainroute.chain import Chain
from chainroute.types import Candidate, RouteContext, RoutingPlugin, Veto


def dep(id_, model="m", provider="openai"):
    return {"model_info": {"id": id_}, "litellm_params": {"model": "%s/%s" % (provider, model)}}


def ctx(**kw):
    base = dict(model_group="g", requested_model="g", messages=[{"role": "user", "content": "hi"}])
    base.update(kw)
    return RouteContext(**base)


def run(chain, context, deployments):
    return asyncio.run(chain.apply(context, deployments))


class Exclude(RoutingPlugin):
    def __init__(self, target):
        self.target, self.name = target, "Exclude"

    async def apply(self, c, candidates):
        for cand in candidates:
            if cand.id == self.target:
                cand.exclude("test exclusion")


class Bias(RoutingPlugin):
    def __init__(self, target, amount):
        self.target, self.amount, self.name = target, amount, "Bias"

    async def apply(self, c, candidates):
        for cand in candidates:
            if cand.id == self.target:
                cand.bias(self.amount)


class Pin(RoutingPlugin):
    def __init__(self, target):
        self.target, self.name = target, "Pin"

    async def apply(self, c, candidates):
        for cand in candidates:
            if cand.id == self.target:
                c.pin(cand, "test pin")


class Explode(RoutingPlugin):
    name = "Explode"

    async def apply(self, c, candidates):
        raise RuntimeError("boom")


class Rejects(RoutingPlugin):
    name = "Rejects"

    async def apply(self, c, candidates):
        raise Veto("nope")


def test_empty_chain_is_a_byte_for_byte_passthrough():
    deployments = [dep("a"), dep("b")]
    out = run(Chain([]), ctx(), deployments)
    assert out is deployments  # same object, not a rebuilt copy


def test_exclude_removes_a_candidate():
    out = run(Chain([Exclude("a")]), ctx(), [dep("a"), dep("b")])
    assert [d["model_info"]["id"] for d in out] == ["b"]


def test_all_excluded_returns_empty_not_an_error():
    out = run(Chain([Exclude("a"), Exclude("b")]), ctx(), [dep("a"), dep("b")])
    assert out == []


def test_bias_keeps_only_the_top_scorers():
    out = run(Chain([Bias("a", 1.0)]), ctx(), [dep("a"), dep("b"), dep("c")])
    assert [d["model_info"]["id"] for d in out] == ["a"]


def test_bias_ties_keep_all_top_scorers():
    out = run(Chain([Bias("a", 1.0), Bias("b", 1.0)]), ctx(), [dep("a"), dep("b"), dep("c")])
    assert {d["model_info"]["id"] for d in out} == {"a", "b"}


def test_pin_wins_over_bias_and_returns_exactly_one():
    out = run(Chain([Bias("b", 5.0), Pin("a")]), ctx(), [dep("a"), dep("b")])
    assert [d["model_info"]["id"] for d in out] == ["a"]


def test_pin_on_an_excluded_candidate_is_ignored():
    # order matters: Exclude runs after Pin, so the pin must be invalidated at merge time
    out = run(Chain([Pin("a"), Exclude("a")]), ctx(), [dep("a"), dep("b")])
    assert [d["model_info"]["id"] for d in out] == ["b"]


def test_a_broken_plugin_is_isolated_not_fatal():
    out = run(Chain([Explode(), Exclude("a")]), ctx(), [dep("a"), dep("b")])
    assert [d["model_info"]["id"] for d in out] == ["b"]


def test_veto_propagates_out_of_apply():
    with pytest.raises(Veto):
        run(Chain([Rejects()]), ctx(), [dep("a")])


def test_on_success_and_on_failure_reach_every_plugin():
    calls = []

    class Recorder(RoutingPlugin):
        def __init__(self, name):
            self.name = name

        async def on_success(self, c, chosen, cost=0.0):
            calls.append((self.name, "success", chosen.id, cost))

        async def on_failure(self, c, attempted, error_class):
            calls.append((self.name, "failure", attempted.id, error_class))

    chain = Chain([Recorder("one"), Recorder("two")])
    asyncio.run(chain.on_success(ctx(), dep("a"), 0.01))
    asyncio.run(chain.on_failure(ctx(), dep("a"), "RateLimitError"))
    assert calls == [
        ("one", "success", "a", 0.01), ("two", "success", "a", 0.01),
        ("one", "failure", "a", "RateLimitError"), ("two", "failure", "a", "RateLimitError"),
    ]
