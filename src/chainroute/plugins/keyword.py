"""KeywordRoute: prefer a specific deployment within the current group when the request looks
like it needs it, by regex on the last user message.

This pins *within* the model group already resolved for the request — put the deployments you
want to choose between under one `model_name` (they're probably already there, if you're using
one of them as a fallback for the other):

    model_list:
      - model_name: chat
        litellm_params: {model: openai/gpt-4o-mini, ...}
      - model_name: chat
        litellm_params: {model: openai/gpt-4o, ...}

    - use: chainroute.plugins.keyword.KeywordRoute
      with:
        rules:
          - match: "urgent|production is down|sev1"
            prefer_model: gpt-4o
          # first matching rule whose target is actually in this group wins; with no match
          # (or a match whose target isn't part of this group), the rest of the chain and
          # litellm's own routing_strategy decide as usual.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..types import Candidate, RouteContext, RoutingPlugin


class KeywordRoute(RoutingPlugin):
    def configure(self, rules: Optional[List[Dict[str, Any]]] = None, case_sensitive: bool = False) -> None:
        flags = 0 if case_sensitive else re.IGNORECASE
        self.rules = [(re.compile(r["match"], flags), r["prefer_model"]) for r in (rules or [])]

    async def apply(self, ctx: RouteContext, candidates: List[Candidate]) -> None:
        text = ctx.last_user_text()
        if not text:
            return
        by_model = {c.model: c for c in candidates if not c.excluded}
        for pattern, prefer_model in self.rules:
            if pattern.search(text) and prefer_model in by_model:
                ctx.pin(by_model[prefer_model], "keyword match: prefers %s" % prefer_model)
                return
