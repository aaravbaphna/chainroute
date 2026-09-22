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

A plugin that raises -- or runs past its timeout -- is logged and skipped for the rest of *that
request*: one broken or slow plugin degrades to "did nothing" for that call, never to "took the
proxy down" or "added unbounded latency to every request." `Veto` is the one exception that's
allowed through in enforce mode, since raising it is how a plugin says the request must stop.

In shadow mode (`Chain(plugins, mode="shadow")`), every plugin still runs and its verdict is
still computed and logged -- but the *original*, untouched deployment list is what actually goes
back to litellm, and a would-be `Veto` is logged rather than raised. Use this to see what a new
plugin or chain would have done against real traffic before it can affect any real request.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from .types import Candidate, RouteContext, RoutingPlugin, Veto

log = logging.getLogger("chainroute")

DEFAULT_TIMEOUT_S = 2.0  # generous enough for a plugin that calls out to a real API (e.g. moderation)


def _candidate_from_deployment(d: Dict[str, Any]) -> Candidate:
    lp = d.get("litellm_params") or {}
    model = lp.get("model") or ""
    provider = lp.get("custom_llm_provider") or (model.split("/", 1)[0] if "/" in model else "unknown")
    short = model[len(provider) + 1:] if model.startswith(provider + "/") else model
    return Candidate(id=(d.get("model_info") or {}).get("id", ""), model=short, provider=provider, deployment=d)


def _timeout_for(plugin: RoutingPlugin, default: float) -> float:
    """Set by the loader from a chain.yaml entry's `timeout_s:`, if given; otherwise the chain's default."""
    return getattr(plugin, "_chainroute_timeout_s", None) or default


class Chain:
    def __init__(self, plugins: List[RoutingPlugin], mode: str = "enforce", default_timeout_s: float = DEFAULT_TIMEOUT_S):
        if mode not in ("enforce", "shadow"):
            raise ValueError("mode must be 'enforce' or 'shadow', got %r" % mode)
        self.plugins = plugins
        self.mode = mode
        self.default_timeout_s = default_timeout_s

    async def _run_hook(self, plugin: RoutingPlugin, hook_name: str, *args: Any) -> bool:
        """Runs one plugin hook with its timeout. Returns False (and logs) if it raised or timed
        out, so the caller can just skip it and move on -- except Veto, which always propagates
        to `apply()`, where enforce/shadow mode decides whether it's actually raised further."""
        timeout = _timeout_for(plugin, self.default_timeout_s)
        try:
            await asyncio.wait_for(getattr(plugin, hook_name)(*args), timeout=timeout)
            return True
        except Veto:
            raise
        except asyncio.TimeoutError:
            log.warning("chainroute: %s.%s exceeded its %.2fs timeout, skipping it for this request",
                        plugin.name, hook_name, timeout)
            return False
        except Exception as e:
            log.warning("chainroute: %s.%s raised %r, skipping it for this request", plugin.name, hook_name, e)
            return False

    async def _decide(self, ctx: RouteContext, deployments: List[Dict[str, Any]]
                       ) -> Tuple[List[Dict[str, Any]], str, Optional[Veto]]:
        """Runs every plugin and computes the merge result, whether or not it's the one that
        actually gets enforced. Returns (result, description, veto-if-any)."""
        candidates = [_candidate_from_deployment(d) for d in deployments]
        by_id = {c.id: c for c in candidates}
        veto: Optional[Veto] = None

        for plugin in self.plugins:
            try:
                await self._run_hook(plugin, "apply", ctx, candidates)
            except Veto as v:
                veto = v
                break  # a vetoing plugin's own verdict is final; later plugins don't get a say
        if veto is not None:
            return [], "would veto: %s" % veto if self.mode == "shadow" else "vetoed: %s" % veto, veto

        pinned = ctx._pinned
        if pinned is not None and pinned.id in by_id and not by_id[pinned.id].excluded:
            return [by_id[pinned.id].deployment], "pin %s (%s)" % (pinned.model, ctx._pin_reason), None

        survivors = [c for c in candidates if not c.excluded]
        if not survivors:
            return [], "excluded every candidate", None

        if any(c.score != 0 for c in survivors):
            top = max(c.score for c in survivors)
            survivors = [c for c in survivors if c.score == top]

        if len(survivors) == len(candidates) and all(c.score == 0 for c in candidates):
            return deployments, "no opinion (passthrough)", None  # untouched: the original objects, not copies
        return [c.deployment for c in survivors], "narrowed to %d/%d" % (len(survivors), len(candidates)), None

    async def apply(self, ctx: RouteContext, deployments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result, desc, veto = await self._decide(ctx, deployments)
        if self.mode == "shadow":
            if desc != "no opinion (passthrough)":
                # WARNING, not INFO: shadow mode's entire point is watching what it would have
                # done, and a real proxy's default logging config filters INFO out -- the same
                # reason the loader's own "loaded N plugin(s)" startup line is a warning too.
                log.warning("chainroute[shadow]: %s -- not enforced", desc)
            return deployments
        if veto is not None:
            raise veto
        if desc not in ("no opinion (passthrough)",):
            log.info("chainroute: %s", desc)
        return result

    async def on_success(self, ctx: RouteContext, chosen_deployment: Dict[str, Any], cost: float = 0.0) -> None:
        chosen = _candidate_from_deployment(chosen_deployment)
        for plugin in self.plugins:
            await self._run_hook(plugin, "on_success", ctx, chosen, cost)

    async def on_failure(self, ctx: RouteContext, attempted_deployment: Dict[str, Any],
                          error_class: Optional[str]) -> None:
        attempted = _candidate_from_deployment(attempted_deployment)
        for plugin in self.plugins:
            await self._run_hook(plugin, "on_failure", ctx, attempted, error_class)
