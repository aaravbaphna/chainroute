"""TagAffinity: only route to deployments tagged for this request.

Reads a tag list off the request (`metadata.tags` by default — the same field LiteLLM's own
`tag_based_routing` strategy uses) and keeps only the deployments whose `litellm_params.tags`
overlaps with it. A request with no tags is passed through untouched — this narrows routing for
*tagged* requests, it doesn't require every request to be tagged.

    model_list:
      - model_name: chat
        litellm_params:
          model: openai/gpt-4o-mini
          tags: ["team-a"]
      - model_name: chat
        litellm_params:
          model: anthropic/claude-haiku-4-5
          tags: ["team-b"]

    - use: chainroute.plugins.tag_affinity.TagAffinity
      # a request tagged "team-b" (metadata: {"tags": ["team-b"]}) only ever sees the second deployment.
"""
from __future__ import annotations

from typing import List

from ..types import Candidate, RouteContext, RoutingPlugin


class TagAffinity(RoutingPlugin):
    def configure(self, metadata_field: str = "tags") -> None:
        self.field = metadata_field

    async def apply(self, ctx: RouteContext, candidates: List[Candidate]) -> None:
        wanted = set(ctx.metadata.get(self.field) or [])
        if not wanted:
            return
        for c in candidates:
            have = set((c.deployment.get("litellm_params") or {}).get("tags") or [])
            if not (wanted & have):
                c.exclude("missing tag(s): wanted one of %s" % sorted(wanted))
