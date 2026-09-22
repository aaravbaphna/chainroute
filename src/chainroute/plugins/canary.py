"""WeightedCanary: send a fixed percentage of traffic to a candidate model, deterministically.

Hashing (not `random()`) means the same session lands on the same side of the split on every
turn, and the split is reproducible across proxy restarts — useful for trialling a new model on,
say, 10% of traffic without it flapping mid-conversation or changing every deploy.

    - use: chainroute.plugins.canary.WeightedCanary
      with:
        canary_model: gpt-4o-mini-2024-11   # matched against Candidate.model
        percent: 10                          # 0-100
        # split key: ctx.session_id if the caller sent one, else falls back to the request id
        # (metadata.chainroute_request_id, stamped by ChainRoute itself) so it's still stable
        # for retries/fallbacks within one request, just not across a caller's whole conversation.
"""
from __future__ import annotations

import hashlib
from typing import List, Optional

from ..types import Candidate, RouteContext, RoutingPlugin


class WeightedCanary(RoutingPlugin):
    def configure(self, canary_model: str = "", percent: float = 0.0) -> None:
        self.canary_model = canary_model
        self.percent = max(0.0, min(100.0, percent))

    def _bucket(self, key: str) -> float:
        digest = hashlib.sha256(key.encode()).hexdigest()[:8]
        return (int(digest, 16) % 10_000) / 100.0  # 0.00-99.99

    async def apply(self, ctx: RouteContext, candidates: List[Candidate]) -> None:
        canary = [c for c in candidates if c.model == self.canary_model]
        if not canary:
            return
        key = ctx.session_id or ctx.metadata.get("chainroute_request_id") or ctx.requested_model
        in_canary_group = self._bucket(str(key)) < self.percent
        group = "canary" if in_canary_group else "control"
        for c in candidates:
            if (c.model == self.canary_model) != in_canary_group:
                c.exclude("canary split: this request is in the %s group" % group)
