"""ChainRoute: a plugin chain for LiteLLM's routing decisions.

    litellm_settings:
      callbacks: chainroute_callback.instance

See types.py for the plugin contract (`RoutingPlugin`), or the README for a walkthrough.
"""
from __future__ import annotations

from typing import Any

__version__ = "0.1.0"
__all__ = ["ChainRoute", "RoutingPlugin", "RouteContext", "Candidate", "Veto", "instance", "__version__"]


def __getattr__(name: str) -> Any:
    global _instance
    if name == "ChainRoute":
        from .callback import ChainRoute
        return ChainRoute
    if name in ("RoutingPlugin", "RouteContext", "Candidate", "Veto"):
        from . import types
        return getattr(types, name)
    if name == "instance":
        if _instance is None:
            from .callback import ChainRoute
            _instance = ChainRoute()
        return _instance
    raise AttributeError(name)


_instance = None
