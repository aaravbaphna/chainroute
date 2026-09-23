"""BanditRouter: an epsilon-greedy multi-armed bandit across every deployment in a group --
`WeightedCanary`'s fixed split, but adaptive: it shifts traffic toward whichever model is
empirically winning, using nothing but the on_success/on_failure hooks every plugin already has.

Every untried model gets tried at least once; after that, with probability `epsilon` it explores
a random eligible model, and otherwise it exploits the one with the best running average reward
so far. What "winning" means is `optimize_for`:

  - "success" (default): reward is 1.0 on success, 0.0 on failure -- maximize success rate.
  - "cost": reward is -cost on success, a fixed -1.0 penalty on failure -- minimize spend, while
    still treating failing outright as worse than any successful-but-expensive call.

Unlike WeightedCanary, this doesn't try to keep one session on one side of a split -- a bandit's
whole premise is learning from aggregate traffic over time, not per-conversation consistency.
That's why it *biases* its choice (nudges scores) rather than excluding the rest outright: a pin
always wins over a bias at merge time (see chain.py), so `StickySession` anywhere in the same
chain still reliably keeps a session on whatever it first pinned, no matter which way the
bandit's preference drifts afterward -- verified in tests/test_callback_sdk.py, since this is
exactly the kind of cross-plugin interaction that's easy to get backwards and not notice without
a concrete test for it.

    - use: chainroute.plugins.bandit.BanditRouter
      with:
        optimize_for: cost
        epsilon: 0.1     # 10% of (post-warmup) traffic keeps exploring; default 0.1
        seed: 42          # optional: makes exploration reproducible (useful for tests/demos)
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from ..types import Candidate, RouteContext, RoutingPlugin

_FAILURE_REWARD = {"success": 0.0, "cost": -1.0}


class BanditRouter(RoutingPlugin):
    def configure(self, optimize_for: str = "success", epsilon: float = 0.1,
                  seed: Optional[int] = None) -> None:
        if optimize_for not in _FAILURE_REWARD:
            raise ValueError("optimize_for must be one of %s, got %r" % (sorted(_FAILURE_REWARD), optimize_for))
        self.optimize_for = optimize_for
        self.epsilon = epsilon
        self._rng = random.Random(seed)
        self._arms: Dict[str, Dict[str, float]] = {}  # model -> {"n": count, "mean": running average reward}

    def _mean(self, model: str) -> float:
        arm = self._arms.get(model)
        return arm["mean"] if arm else float("-inf")

    def _update(self, model: str, reward: float) -> None:
        arm = self._arms.setdefault(model, {"n": 0, "mean": 0.0})
        arm["n"] += 1
        arm["mean"] += (reward - arm["mean"]) / arm["n"]  # incremental average; no history to store

    async def apply(self, ctx: RouteContext, candidates: List[Candidate]) -> None:
        eligible = [c for c in candidates if not c.excluded]
        if len(eligible) <= 1:
            return  # nothing to choose between; leave it to the rest of the chain / strategy

        unseen = [c for c in eligible if c.model not in self._arms]
        if unseen:
            choice = self._rng.choice(unseen)
            # Mark it seen immediately, not when on_success/on_failure eventually reports the
            # outcome: a request can be slow, dropped, or never awaited by the caller, and
            # without this, the same untried arm could get picked again before its result (if
            # any) ever lands -- defeating "try every arm once" for arms with the misfortune of
            # being slow. on_success/on_failure below still owns the actual reward when it
            # arrives; this only claims the arm as tried.
            self._arms.setdefault(choice.model, {"n": 0, "mean": 0.0})
            reason = "bandit: trying %s for the first time" % choice.model
        elif self._rng.random() < self.epsilon:
            choice = self._rng.choice(eligible)
            reason = "bandit: exploring %s" % choice.model
        else:
            choice = max(eligible, key=lambda c: self._mean(c.model))
            reason = "bandit: %s is winning (mean reward %.3f, n=%d)" % (
                choice.model, self._mean(choice.model), self._arms[choice.model]["n"])

        choice.bias(1.0, reason)
        for c in eligible:
            if c is not choice:
                # 0.0 leaves the score (and so the merge outcome) untouched -- only sets `reason`,
                # so `chainroute simulate` and RouteLens can still show why each arm lost, exactly
                # as if this were an exclude. See the module docstring for why this is bias, not
                # exclude: it's what lets a pin from another plugin (StickySession) still win.
                arm = self._arms.get(c.model)
                stats = ("mean reward %.3f (n=%d)" % (arm["mean"], arm["n"])) if arm else "no data yet"
                c.bias(0.0, "bandit: %s, currently behind %s" % (stats, choice.model))

    async def on_success(self, ctx: RouteContext, chosen: Candidate, cost: float = 0.0) -> None:
        reward = 1.0 if self.optimize_for == "success" else -cost
        self._update(chosen.model, reward)

    async def on_failure(self, ctx: RouteContext, attempted: Candidate, error_class: Optional[str]) -> None:
        self._update(attempted.model, _FAILURE_REWARD[self.optimize_for])
