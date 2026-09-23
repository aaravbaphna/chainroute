"""Built-in plugins. Each is small enough to read in a minute and use as a template for your
own — see the README's "writing a plugin" section for a from-scratch walkthrough."""
from .budget_guard import BudgetGuard
from .canary import WeightedCanary
from .keyword import KeywordRoute
from .moderation import ModerationGuard
from .sensitive_data import SensitiveDataGuard
from .sticky_session import StickySession
from .tag_affinity import TagAffinity

BUILTINS = {
    "chainroute.plugins.keyword.KeywordRoute": KeywordRoute,
    "chainroute.plugins.sticky_session.StickySession": StickySession,
    "chainroute.plugins.tag_affinity.TagAffinity": TagAffinity,
    "chainroute.plugins.canary.WeightedCanary": WeightedCanary,
    "chainroute.plugins.budget_guard.BudgetGuard": BudgetGuard,
    "chainroute.plugins.sensitive_data.SensitiveDataGuard": SensitiveDataGuard,
    "chainroute.plugins.moderation.ModerationGuard": ModerationGuard,
}

__all__ = ["BUILTINS", "BudgetGuard", "WeightedCanary", "KeywordRoute", "ModerationGuard",
           "SensitiveDataGuard", "StickySession", "TagAffinity"]
