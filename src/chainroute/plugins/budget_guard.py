"""BudgetGuard: once a key's spend crosses a cap within a rolling window, steer it toward a
cheaper allow-list instead of blocking it outright.

Spend is tracked in memory from each call's actual cost, keyed by a metadata field (the proxy's
hashed API key by default, so it's per-caller without ever seeing the raw key). It fails open by
design: if none of the currently eligible deployments are on `cheap_models`, BudgetGuard does
nothing rather than leave the caller with zero options — a budget policy should degrade service,
not break it.

    - use: chainroute.plugins.budget_guard.BudgetGuard
      with:
        cap: 5.00                          # dollars
        window_seconds: 86400              # rolling 24h (default)
        cheap_models: [gpt-4o-mini, claude-haiku-4-5]
        key_field: user_api_key_hash       # a field in request metadata (default shown)

Spend is process-local: on a multi-instance proxy, each instance enforces its own cap
independently rather than sharing a global counter (no shared store is assumed).
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import List

from ..types import Candidate, RouteContext, RoutingPlugin


class BudgetGuard(RoutingPlugin):
    def configure(self, cap: float = 10.0, window_seconds: float = 86400, cheap_models: List[str] = None,
                  key_field: str = "user_api_key_hash", max_keys: int = 50_000) -> None:
        self.cap = cap
        self.window = window_seconds
        self.cheap_models = set(cheap_models or [])
        self.key_field = key_field
        self.max_keys = max_keys
        self._spend: "OrderedDict[str, dict]" = OrderedDict()

    def _current_spend(self, key: str) -> float:
        entry = self._spend.get(key)
        if entry is None or time.time() - entry["window_start"] > self.window:
            return 0.0
        return entry["total"]

    async def apply(self, ctx: RouteContext, candidates: List[Candidate]) -> None:
        key = ctx.metadata.get(self.key_field)
        if not key or not self.cheap_models:
            return
        if self._current_spend(str(key)) < self.cap:
            return
        cheap = [c for c in candidates if c.model in self.cheap_models and not c.excluded]
        if not cheap:
            return  # fail open: nothing cheap is eligible, so don't cut off service entirely
        for c in candidates:
            if c.model not in self.cheap_models:
                c.exclude("over budget ($%.2f cap): steered to a cheaper model" % self.cap)

    async def on_success(self, ctx: RouteContext, chosen: Candidate, cost: float = 0.0) -> None:
        key = ctx.metadata.get(self.key_field)
        if not key or cost <= 0:
            return
        key = str(key)
        now = time.time()
        entry = self._spend.get(key)
        if entry is None or now - entry["window_start"] > self.window:
            entry = {"window_start": now, "total": 0.0}
        entry["total"] += cost
        self._spend[key] = entry
        self._spend.move_to_end(key)
        while len(self._spend) > self.max_keys:
            self._spend.popitem(last=False)
