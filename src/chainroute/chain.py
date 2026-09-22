"""Runs a list of plugins against one request and turns their verdicts into the deployment
list ChainRoute hands back to LiteLLM.

The merge rule, in order:

  1. If any plugin pinned a candidate (and that candidate wasn't also excluded), that candidate
     alone is returned — litellm's routing_strategy has nothing left to choose between.
  2. Otherwise, excluded candidates are dropped.
  3. Otherwise, if any candidate's score changed, only the highest-scoring survivors are kept
     (ties included) — litellm's routing_strategy still picks among *those*.
  4. If nothing above applied, the candidates come back exactly as they went in: chainroute
     with an empty or fully-neutral chain is a no-op, byte for byte.

A plugin that raises is logged and skipped for the rest of *that request* — one broken plugin
degrades to "did nothing" for that call, never to "took the proxy down." `Veto` is the one
exception that's allowed through, since raising it is how a plugin says the request must stop.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .types import Candidate, RouteContext, RoutingPlugin, Veto

log = logging.getLogger("chainroute")


def _candidate_from_deployment(d: Dict[str, Any]) -> Candidate:
    lp = d.get("litellm_params") or {}
    model = lp.get("model") or ""
    provider = lp.get("custom_llm_provider") or (model.split("/", 1)[0] if "/" in model else "unknown")
    short = model[len(provider) + 1:] if model.startswith(provider + "/") else model
    return Candidate(id=(d.get("model_info") or {}).get("id", ""), model=short, provider=provider, deployment=d)


class Chain:
    def __init__(self, plugins: List[RoutingPlugin]):
        self.plugins = plugins

    async def apply(self, ctx: RouteContext, deployments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates = [_candidate_from_deployment(d) for d in deployments]
        by_id = {c.id: c for c in candidates}

        for plugin in self.plugins:
            try:
                await plugin.apply(ctx, candidates)
            except Veto:
                raise
            except Exception as e:
                log.warning("chainroute: %s.apply raised %r, skipping it for this request", plugin.name, e)

        pinned = ctx._pinned
        if pinned is not None and pinned.id in by_id and not by_id[pinned.id].excluded:
            log.info("chainroute: pinned to %s (%s)", pinned.model, ctx._pin_reason)
            return [by_id[pinned.id].deployment]

        survivors = [c for c in candidates if not c.excluded]
        if not survivors:
            return []  # litellm's own "no healthy deployment" error covers this clearly

        if any(c.score != 0 for c in survivors):
            top = max(c.score for c in survivors)
            survivors = [c for c in survivors if c.score == top]

        if len(survivors) == len(candidates) and all(c.score == 0 for c in candidates):
            return deployments  # untouched: return the original objects, not rebuilt copies
        return [c.deployment for c in survivors]

    async def on_success(self, ctx: RouteContext, chosen_deployment: Dict[str, Any], cost: float = 0.0) -> None:
        chosen = _candidate_from_deployment(chosen_deployment)
        for plugin in self.plugins:
            try:
                await plugin.on_success(ctx, chosen, cost)
            except Exception as e:
                log.warning("chainroute: %s.on_success raised %r", plugin.name, e)

    async def on_failure(self, ctx: RouteContext, attempted_deployment: Dict[str, Any],
                          error_class: Optional[str]) -> None:
        attempted = _candidate_from_deployment(attempted_deployment)
        for plugin in self.plugins:
            try:
                await plugin.on_failure(ctx, attempted, error_class)
            except Exception as e:
                log.warning("chainroute: %s.on_failure raised %r", plugin.name, e)
