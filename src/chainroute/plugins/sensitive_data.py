"""SensitiveDataGuard: if the latest message looks like it contains sensitive data, restrict
the request to a trusted set of providers (e.g. your own VPC-hosted deployments) instead of
whatever the caller or the load-balancing strategy would otherwise have picked.

This is a pattern check, not a data-loss-prevention product — treat the default patterns as a
starting point, tune them for what your own traffic actually looks like, and use `on_match:
"veto"` (the default) if serving the request to an untrusted provider is not an acceptable
fallback.

    - use: chainroute.plugins.sensitive_data.SensitiveDataGuard
      with:
        patterns: ["\\bssn\\b", "\\bpassport\\s*(no|number)\\b", "\\b\\d{3}-\\d{2}-\\d{4}\\b"]
        trusted_providers: [azure, bedrock]   # your own VPC/on-prem-backed deployments
        on_match: veto                         # or "exclude": narrow candidates, don't reject
"""
from __future__ import annotations

import re
from typing import List, Optional

from ..types import Candidate, RouteContext, RoutingPlugin, Veto

_DEFAULT_PATTERNS = [
    r"\bssn\b", r"\bsocial security\b", r"\bpassport\s*(no|number)?\b",
    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN-shaped
    r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b",  # card-shaped
]


class SensitiveDataGuard(RoutingPlugin):
    def configure(self, patterns: Optional[List[str]] = None, trusted_providers: Optional[List[str]] = None,
                  on_match: str = "veto") -> None:
        self.patterns = [re.compile(p, re.IGNORECASE) for p in (patterns or _DEFAULT_PATTERNS)]
        self.trusted = set(trusted_providers or [])
        if on_match not in ("veto", "exclude"):
            raise ValueError("on_match must be 'veto' or 'exclude', got %r" % on_match)
        self.on_match = on_match

    async def apply(self, ctx: RouteContext, candidates: List[Candidate]) -> None:
        text = ctx.last_user_text()
        if not text or not any(p.search(text) for p in self.patterns):
            return
        for c in candidates:
            if c.provider not in self.trusted:
                c.exclude("looks like sensitive data; only %s are trusted" % (sorted(self.trusted) or "no providers"))
        if self.on_match == "veto" and not any(not c.excluded for c in candidates):
            raise Veto("request looks like it contains sensitive data, and no trusted deployment is eligible")
