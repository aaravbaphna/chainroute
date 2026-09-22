"""StickySession: keep every turn of one conversation on the deployment that handled its first
turn, instead of `simple-shuffle`/`latency-based-routing` picking a new one each time.

This only acts when the caller sends an explicit session id (an `x-litellm-session-id` header,
or `metadata.session_id` / `metadata.conversation_id`) — with no session id, every request is
its own session and StickySession is a no-op. That's deliberate: guessing at a session id would
mean silently pinning unrelated requests together, which is a routing *decision*, not just a
display grouping (contrast RouteLens's dashboard, which infers sessions for display but never
feeds that guess back into routing).

If the pinned deployment is no longer eligible (cooldown, removed from config, excluded by a
plugin earlier in the chain), StickySession simply lets the rest of the chain decide and adopts
whatever wins as the new pin — it never vetoes on the pinned deployment's behalf.

    - use: chainroute.plugins.sticky_session.StickySession
      with:
        ttl_seconds: 3600   # forget a session's pin after an hour of inactivity (default: 6h)
        max_sessions: 50000 # LRU cap so long-running proxies don't grow this unbounded
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import List

from ..types import Candidate, RouteContext, RoutingPlugin


class StickySession(RoutingPlugin):
    def configure(self, ttl_seconds: float = 6 * 3600, max_sessions: int = 50_000) -> None:
        self.ttl = ttl_seconds
        self.max_sessions = max_sessions
        self._pins: "OrderedDict[str, dict]" = OrderedDict()

    async def apply(self, ctx: RouteContext, candidates: List[Candidate]) -> None:
        if not ctx.session_id:
            return
        entry = self._pins.get(ctx.session_id)
        if entry is None or time.time() - entry["ts"] > self.ttl:
            return
        for c in candidates:
            if c.id == entry["deployment_id"]:
                ctx.pin(c, "sticky session: turn 1 used %s" % c.model)
                return

    async def on_success(self, ctx: RouteContext, chosen: Candidate, cost: float = 0.0) -> None:
        if not ctx.session_id:
            return
        self._pins[ctx.session_id] = {"deployment_id": chosen.id, "ts": time.time()}
        self._pins.move_to_end(ctx.session_id)
        while len(self._pins) > self.max_sessions:
            self._pins.popitem(last=False)
