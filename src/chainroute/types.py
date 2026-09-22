"""The plugin contract.

A plugin sees the request (`RouteContext`) and the deployments already eligible for it within
its model group (`Candidate` objects — one per deployment configured under that `model_name`),
and can do exactly four things to influence the outcome:

  - exclude a candidate (`candidate.exclude("reason")`)         -> filter
  - nudge a candidate's score (`candidate.bias(amount, "why")`) -> prefer
  - pin the whole request to one candidate (`ctx.pin(candidate, "why")`) -> force
  - raise `Veto("why")`                                          -> reject

A plugin that does none of these is a no-op for that request, and ChainRoute falls through to
whatever routing_strategy is already configured in LiteLLM. Nothing here can violate that:
chainroute never invents deployments litellm didn't already consider eligible, and it only
arbitrates *within* one model group — it can't send a request to a different `model_name`
(LiteLLM has no public hook for that outside its own built-in semantic/complexity auto-routers,
which is exactly the gap chainroute exists to fill for everything else). If you want a plugin
to prefer a stronger or cheaper model for some requests, put those deployments under the same
`model_name` and pin between them — see KeywordRoute.
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class Veto(Exception):
    """Raise from `apply()` or `pre_route()` to reject the request outright.

    Surfaces to the caller as a `litellm.BadRequestError` with this message.
    Use it for "this request must not be served," not for "I'd rather not
    handle this" — the latter is `candidate.exclude(...)`.
    """


@dataclass
class Candidate:
    """One deployment litellm considers eligible for this request."""

    id: str
    model: str
    provider: str
    deployment: Dict[str, Any]  # the raw litellm deployment dict, for anything not exposed above
    score: float = 0.0
    excluded: bool = False
    reason: Optional[str] = None  # set by whichever plugin last excluded or biased this candidate
    tags: List[str] = field(default_factory=list)

    def exclude(self, reason: str) -> None:
        self.excluded = True
        self.reason = reason

    def bias(self, amount: float, reason: Optional[str] = None) -> None:
        """Positive favors this candidate over its siblings; negative disfavors it.
        Only matters relative to other candidates' scores in the same request."""
        self.score += amount
        if reason:
            self.reason = reason


@dataclass
class RouteContext:
    """Everything a plugin might need to decide, gathered once per request."""

    model_group: str  # the model name/group this request will route within (may change via pre_route)
    requested_model: str  # what the caller originally asked for, before any pre_route rewrite
    messages: Optional[List[Dict[str, Any]]]
    input: Any = None  # for embedding/moderation-style calls that pass `input` instead of `messages`
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_kwargs: Dict[str, Any] = field(default_factory=dict)  # raw litellm kwargs, for advanced plugins
    session_id: Optional[str] = None  # only set if the caller sent an explicit session/trace id
    attempt_no: int = 0  # 0 on the first try; >0 means a previous attempt in this request failed
    previous_failure: Optional[Dict[str, Any]] = None  # {"error_class", "error_code", "model_group"}
    scratch: Dict[str, Any] = field(default_factory=dict)  # plugins share state across one request's chain here

    _pinned: Optional["Candidate"] = field(default=None, repr=False)
    _pin_reason: Optional[str] = field(default=None, repr=False)

    def pin(self, candidate: Candidate, reason: str) -> None:
        """Force this request onto `candidate`, skipping the rest of the chain's scoring and
        LiteLLM's own routing_strategy. The last plugin in the chain to call pin() wins, so put
        anything that should be overridable *before* anything that should be final."""
        self._pinned = candidate
        self._pin_reason = reason

    def last_user_text(self) -> str:
        """Convenience: the text of the most recent `role: user` message, flattened to a string."""
        for m in reversed(self.messages or []):
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    return c
                if isinstance(c, list):
                    return " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
        return ""


class RoutingPlugin(ABC):
    """Subclass this. Every method is optional to override — the defaults are no-ops — so a
    plugin only needs to implement the hook(s) it actually uses.

    A plugin instance is long-lived (one instance handles every request), so store per-request
    state on `ctx.scratch` or an explicit dict keyed by `ctx.session_id`/request id, never on
    `self`, unless you've deliberately made it thread/async-safe shared state (see StickySession
    for the pattern: an LRU dict guarded by nothing but simple, GIL-safe dict ops).
    """

    #: Shown in explanations and `chainroute list`. Defaults to the class name.
    name: Optional[str] = None

    def __init__(self) -> None:
        # Plugins keep their config in attributes `configure()` sets, so a bare `StickySession()`
        # (the SDK path -- no chain.yaml, no `with:` block) still needs defaults applied once.
        # `configure()` re-running later with real kwargs (the chain.yaml path) is fine: every
        # built-in only assigns fresh attributes, so calling it twice just re-initializes.
        self.configure()

    def configure(self, **kwargs: Any) -> None:
        """Called once, right after construction, with whatever `with:` block your chain.yaml
        entry has under this plugin. Default stores nothing; override to validate/save config."""

    async def apply(self, ctx: RouteContext, candidates: List[Candidate]) -> None:
        """Mutate `candidates` in place — exclude some, bias others, or `ctx.pin(...)` one."""

    async def on_success(self, ctx: RouteContext, chosen: Candidate, cost: float = 0.0) -> None:
        """Runs after a deployment actually succeeds, with the one that was used and what it
        cost. This is the only place a plugin reliably learns the outcome — `apply()` runs
        before anyone decides, so it can't know the winner or the price yet."""

    async def on_failure(self, ctx: RouteContext, attempted: Candidate, error_class: Optional[str]) -> None:
        """Runs after a deployment attempt fails (before any fallback/retry is chosen)."""
