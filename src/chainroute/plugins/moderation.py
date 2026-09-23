"""ModerationGuard: check the latest message against OpenAI's moderation endpoint before it
reaches a model, and restrict (or reject) the request if it's flagged.

This is chainroute's first plugin that calls a real external API rather than just running a
local check, which is exactly why it needs nothing extra for reliability: `Chain` already
bounds every plugin hook with a timeout and isolates a plugin that raises (see the shadow-mode
/ timeouts PR), so a slow or unreachable moderation endpoint just gets skipped for that request
-- the plugin below has no bespoke retry or timeout logic of its own. The one thing worth
choosing deliberately is what "skipped" should mean for a *compliance* plugin specifically: see
`fail_closed` below, since letting a request through unmoderated is not always the safe default.

    - use: chainroute.plugins.moderation.ModerationGuard
      with:
        # api_key defaults to $OPENAI_API_KEY. Moderation calls are free of charge as of writing.
        trusted_providers: [azure]   # only used when on_match: exclude
        on_match: exclude            # or "veto"
        thresholds:                  # optional: flag *below* the API's own built-in threshold
          violence: 0.3
        fail_closed: false           # true = veto (rather than let through) if the API call itself fails
      timeout_s: 1.5                 # chain-level setting (see Chain) -- moderation calls are usually fast
"""
from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from ..types import Candidate, RouteContext, RoutingPlugin, Veto

MODERATIONS_URL = "https://api.openai.com/v1/moderations"


class ModerationGuard(RoutingPlugin):
    def configure(self, api_key: Optional[str] = None, model: str = "omni-moderation-latest",
                  trusted_providers: Optional[List[str]] = None, on_match: str = "exclude",
                  thresholds: Optional[Dict[str, float]] = None, fail_closed: bool = False,
                  cache_size: int = 1000, base_url: str = MODERATIONS_URL) -> None:
        if on_match not in ("exclude", "veto"):
            raise ValueError("on_match must be 'exclude' or 'veto', got %r" % on_match)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.trusted = set(trusted_providers or [])
        self.on_match = on_match
        self.thresholds = thresholds or {}
        self.fail_closed = fail_closed
        self.base_url = base_url
        self._cache: "OrderedDict[str, bool]" = OrderedDict()
        self._cache_size = cache_size
        self._client: Optional[Any] = None  # an httpx.AsyncClient, created lazily (see _call_api)
        if not self.api_key:
            import logging
            logging.getLogger("chainroute").warning(
                "ModerationGuard: no api_key given and OPENAI_API_KEY is unset -- this plugin will do nothing")

    async def _flagged(self, text: str) -> bool:
        """Cached: identical text (a repeated or resent message) doesn't re-call the API."""
        key = hashlib.sha256(text.encode()).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        result = await self._call_api(text)
        self._cache[key] = result
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return result

    async def _call_api(self, text: str) -> bool:
        """Split out from `_flagged` so tests can monkeypatch this one method and never touch
        the network -- see tests/test_plugins.py and test_callback_sdk.py.

        Reuses one client across calls (connection pooling) rather than opening a fresh
        connection per request; it's created lazily, on the event loop that's actually running a
        request, rather than in `configure()` (construction time may have no running loop, and
        `configure()` itself can run twice -- see RoutingPlugin.__init__)."""
        import httpx
        if self._client is None:
            self._client = httpx.AsyncClient()
        resp = await self._client.post(
            self.base_url, timeout=None,  # Chain's own timeout already bounds this call
            headers={"Authorization": "Bearer %s" % self.api_key},
            json={"input": text, "model": self.model})
        resp.raise_for_status()
        result = resp.json()["results"][0]
        if result.get("flagged"):
            return True
        scores = result.get("category_scores") or {}
        return any(scores.get(cat, 0.0) >= threshold for cat, threshold in self.thresholds.items())

    async def apply(self, ctx: RouteContext, candidates: List[Candidate]) -> None:
        if not self.api_key:
            return
        text = ctx.last_user_text()
        if not text:
            return
        try:
            flagged = await self._flagged(text)
        except Exception:
            if self.fail_closed:
                raise Veto("moderation check failed and fail_closed is set")
            raise  # let Chain's own isolation log it and skip this plugin for the request (fail open)
        if not flagged:
            return
        for c in candidates:
            if c.provider not in self.trusted:
                c.exclude("flagged by moderation; only %s are trusted" % (sorted(self.trusted) or "no providers"))
        if self.on_match == "veto" and not any(not c.excluded for c in candidates):
            raise Veto("request was flagged by moderation, and no trusted deployment is eligible")
