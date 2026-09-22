"""Turns a chain.yaml into a list of configured `RoutingPlugin` instances.

    mode: enforce   # or "shadow": compute and log every plugin's verdict, but enforce none of
                    # them -- see chain.py. CHAINROUTE_MODE overrides this at deploy time.
    plugins:
      - use: chainroute.plugins.sticky_session.StickySession
      - use: chainroute.plugins.keyword.KeywordRoute
        timeout_s: 0.5   # optional; overrides the chain's default timeout for this plugin only
        with:
          rules:
            - match: "urgent|asap"
              route_to_group: priority-pool
      - use: plugins.MyCustomPlugin   # a local plugins.py next to chain.yaml, no packaging needed
        with:
          threshold: 0.7
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from pathlib import Path
from typing import List, Optional

import yaml

from .types import RoutingPlugin

log = logging.getLogger("chainroute")


class ChainConfigError(Exception):
    """Raised for a chain.yaml problem a human needs to fix (bad path, wrong type, ...)."""


def _resolve(dotted: str, config_dir: Optional[Path]):
    """`pkg.module.ClassName` -> the class object. Tries a real import first (installed
    packages, or anything already on sys.path); if that fails and a chain.yaml directory was
    given, tries `<config_dir>/<module_path>.py` next — the same trick litellm's own proxy uses
    so a plain file next to your config works with no `pip install -e .` and no PYTHONPATH."""
    if "." not in dotted:
        raise ChainConfigError("'%s' is not <module>.<ClassName> (missing a dot)" % dotted)
    module_name, class_name = dotted.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        if config_dir is None:
            raise ChainConfigError("could not import '%s': no such module" % module_name) from None
        file_path = config_dir / (module_name.replace(".", "/") + ".py")
        if not file_path.exists():
            raise ChainConfigError(
                "could not import '%s', and no file at %s either" % (module_name, file_path)) from None
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ChainConfigError("could not load %s as a module" % file_path) from None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    try:
        obj = getattr(module, class_name)
    except AttributeError:
        raise ChainConfigError("'%s' has no attribute '%s'" % (module_name, class_name)) from None
    return obj


def load_chain(path: str) -> List[RoutingPlugin]:
    """Reads a chain.yaml (or .json — yaml.safe_load handles both) and returns configured plugin
    instances in the order they'll run. Raises ChainConfigError on anything a human should fix;
    never raises for reasons a user can't act on."""
    p = Path(path)
    if not p.exists():
        raise ChainConfigError(
            "chain file not found: %s (set CHAINROUTE_CHAIN, or run `chainroute init`)" % p)
    try:
        doc = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as e:
        raise ChainConfigError("%s is not valid YAML: %s" % (p, e)) from e
    entries = doc.get("plugins")
    if not isinstance(entries, list):
        raise ChainConfigError("%s needs a top-level `plugins:` list" % p)

    plugins: List[RoutingPlugin] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or "use" not in entry:
            raise ChainConfigError("plugins[%d] needs a `use: <module>.<ClassName>` key" % i)
        cls = _resolve(str(entry["use"]), p.parent)
        if not (isinstance(cls, type) and issubclass(cls, RoutingPlugin)):
            raise ChainConfigError("%s is not a RoutingPlugin subclass" % entry["use"])
        instance = cls()
        instance.configure(**(entry.get("with") or {}))
        if instance.name is None:
            instance.name = cls.__name__
        if "timeout_s" in entry:
            instance._chainroute_timeout_s = float(entry["timeout_s"])  # read by Chain; overrides the chain default
        plugins.append(instance)
    return plugins


def read_chain_mode(path: str, default: str = "enforce") -> str:
    """Reads chain.yaml's top-level `mode:` (enforce|shadow), separately from `load_chain` so a
    YAML problem here can't also block loading the plugins themselves. Forgiving by design: any
    problem (missing file, bad YAML, unset key) just falls back to `default`, since a chain that
    fails to load will already have logged that error loudly elsewhere."""
    try:
        doc = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return default
    mode = doc.get("mode", default)
    return mode if mode in ("enforce", "shadow") else default


def default_chain_path() -> str:
    return os.environ.get("CHAINROUTE_CHAIN", "chain.yaml")
