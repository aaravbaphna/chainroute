"""The LiteLLM CustomLogger that runs a Chain against every request.

Install with `chainroute install --config config.yaml --patch`, which adds:

    litellm_settings:
      callbacks: chainroute_callback.instance

ChainRoute never touches a request except through what the chain's plugins decide (see
chain.py's merge rule) — with an empty chain.yaml, it is a byte-for-byte no-op.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import litellm
from litellm.integrations.custom_logger import CustomLogger

from .chain import Chain
from .loader import ChainConfigError, default_chain_path, load_chain
from .types import RouteContext, Veto

log = logging.getLogger("chainroute")


class ChainRoute(CustomLogger):
    def __init__(self, chain_path: Optional[str] = None, plugins: Optional[list] = None) -> None:
        super().__init__()
        if plugins is not None:
            self.chain = Chain(plugins)  # SDK users: pass instances directly, skip the YAML
        else:
            self.chain_path = chain_path or default_chain_path()
            try:
                self.chain = Chain(load_chain(self.chain_path))
                log.warning("chainroute: loaded %d plugin(s) from %s", len(self.chain.plugins), self.chain_path)
            except ChainConfigError as e:
                # Fail loudly at startup (this is a config typo, the kind you want to catch in
                # CI), but never take the proxy down for it: fall back to an empty, no-op chain.
                log.error("chainroute: %s -- routing continues WITHOUT any plugins", e)
                self.chain = Chain([])
        self._pending: List[Dict[str, Any]] = []  # decisions awaiting their outcome, oldest first

    # ------------------------------------------------------------------ request-id stamping
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):  # type: ignore[override]
        try:
            key = "litellm_metadata" if isinstance(data.get("litellm_metadata"), dict) else "metadata"
            if not isinstance(data.get(key), dict):
                data[key] = {}
            data[key].setdefault("chainroute_request_id", uuid.uuid4().hex)
        except Exception as e:
            log.debug("chainroute pre_call hook error: %r", e)
        return None

    # ------------------------------------------------------------------ apply (filter/score/pin)
    async def async_filter_deployments(self, model, healthy_deployments, messages, request_kwargs=None,
                                        parent_otel_span=None):  # type: ignore[override]
        if not self.chain.plugins or not healthy_deployments:
            return healthy_deployments
        rk = request_kwargs or {}
        ctx = self._build_context(model, rk, messages)
        try:
            result = await self.chain.apply(ctx, healthy_deployments)
        except Veto as v:
            raise litellm.BadRequestError(message="Rejected by chainroute: %s" % v, model=model,
                                           llm_provider="") from None
        self._remember(ctx, rk, result)
        return result

    # ------------------------------------------------------------------ outcome hooks
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            entry = self._pop(kwargs)
            if entry is not None:
                sl = kwargs.get("standard_logging_object") or {}
                await self.chain.on_success(entry["ctx"], _deployment_for(entry, sl), float(sl.get("response_cost") or 0.0))
        except Exception as e:
            log.debug("chainroute success hook error: %r", e)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            entry = self._pop(kwargs)
            if entry is not None:
                sl = kwargs.get("standard_logging_object") or {}
                err = (sl.get("error_information") or {}).get("error_class")
                await self.chain.on_failure(entry["ctx"], _deployment_for(entry, sl), err)
        except Exception as e:
            log.debug("chainroute failure hook error: %r", e)

    async def async_post_call_success_deployment_hook(self, request_data, response, call_type):
        """Fires inline, before the response returns -- unlike the log events above, which are
        background tasks. Used only to close out SDK-direct `Router` calls that share a trace id
        across requests (the proxy always has its own stamped id from async_pre_call_hook, so
        this is a no-op there); see RouteLens's callback.py for the fuller version of this same
        problem, if you want the long explanation."""
        try:
            trace = str((request_data or {}).get("litellm_trace_id"))
            for entry in self._pending:
                if entry.get("filter_trace") == trace and not entry["ctx"].scratch.get("chainroute_stamped"):
                    entry["done"] = True
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ internals
    def _build_context(self, model: str, rk: Dict[str, Any], messages) -> RouteContext:
        md = rk.get("metadata") or {}
        lmd = rk.get("litellm_metadata") or {}
        headers = {str(k).lower(): v for k, v in (md.get("headers") or {}).items()}
        session_id = (rk.get("litellm_session_id") or md.get("session_id") or md.get("conversation_id")
                      or headers.get("x-litellm-session-id") or headers.get("x-session-id")
                      or headers.get("x-conversation-id"))
        body = ((rk.get("proxy_server_request") or {}).get("body") or {})
        return RouteContext(
            model_group=model, requested_model=body.get("model") or model, messages=messages,
            metadata=md, request_kwargs=rk, session_id=str(session_id) if session_id else None,
            attempt_no=int(rk.get("fallback_depth") or 0),
        )

    def _remember(self, ctx: RouteContext, rk: Dict[str, Any], result: List[Dict[str, Any]]) -> None:
        stamped = (rk.get("metadata") or {}).get("chainroute_request_id") or (rk.get("litellm_metadata") or {}).get(
            "chainroute_request_id")
        ctx.scratch["chainroute_stamped"] = bool(stamped)
        now = time.time()
        self._pending = [e for e in self._pending if now - e["ts"] < 600][-4999:]
        self._pending.append({
            "ts": now, "ctx": ctx, "done": False,
            "call_id": rk.get("litellm_call_id"),
            "logging_id": id(rk["litellm_logging_obj"]) if rk.get("litellm_logging_obj") is not None else None,
            "filter_trace": rk.get("litellm_trace_id"),
            "cand_ids": {(d.get("model_info") or {}).get("id") for d in result},
            "group": ctx.model_group,
        })

    def _pop(self, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sl = kwargs.get("standard_logging_object") or {}
        call_id = kwargs.get("litellm_call_id")
        logging_id = id(kwargs.get("litellm_logging_obj")) if kwargs.get("litellm_logging_obj") is not None else None
        model_id = sl.get("model_id") or (sl.get("hidden_params") or {}).get("model_id")
        best, best_score = None, 0
        for e in self._pending:
            score = 0
            if call_id and e["call_id"] == call_id:
                score = 4
            elif logging_id is not None and e["logging_id"] == logging_id:
                score = 3
            elif sl.get("trace_id") and e["filter_trace"] == sl.get("trace_id"):
                score = 2
            elif e["group"] == sl.get("model_group") and model_id in e["cand_ids"]:
                score = 1
            if score > best_score:
                best, best_score = e, score
        if best is not None:
            self._pending.remove(best)
        return best


def _deployment_for(entry: Dict[str, Any], sl: Dict[str, Any]) -> Dict[str, Any]:
    model_id = sl.get("model_id") or (sl.get("hidden_params") or {}).get("model_id") or ""
    provider = sl.get("custom_llm_provider") or ""
    model = sl.get("model") or ""
    return {"model_info": {"id": model_id},
            "litellm_params": {"model": ("%s/%s" % (provider, model)) if provider else model,
                                "custom_llm_provider": provider}}
