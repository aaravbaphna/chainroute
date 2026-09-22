"""Your own chainroute plugins. Reference these from chain.yaml as `use: plugins.<ClassName>`
-- this file just needs to sit next to chain.yaml, no packaging or PYTHONPATH required.
"""
import datetime

from chainroute import RouteContext, RoutingPlugin


class BusinessHours(RoutingPlugin):
    """Outside business hours, prefer a cheaper deployment already in this same group. A
    minimal from-scratch example -- see chainroute.plugins for more (StickySession,
    KeywordRoute, BudgetGuard, ...)."""

    def configure(self, cheap_model: str = "gpt-4o-mini", start_hour: int = 9, end_hour: int = 18) -> None:
        self.cheap_model = cheap_model
        self.start_hour, self.end_hour = start_hour, end_hour

    async def apply(self, ctx: RouteContext, candidates: list) -> None:
        hour = datetime.datetime.now().hour
        if self.start_hour <= hour < self.end_hour:
            return  # business hours: no preference, let the rest of the chain / strategy decide
        for c in candidates:
            if c.model == self.cheap_model and not c.excluded:
                ctx.pin(c, "outside business hours: prefers %s" % self.cheap_model)
                return
