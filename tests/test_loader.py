import pytest

from chainroute.loader import ChainConfigError, load_chain


def write(tmp_path, name, content):
    f = tmp_path / name
    f.write_text(content)
    return f


def test_loads_a_builtin_by_dotted_path(tmp_path):
    f = write(tmp_path, "chain.yaml", """
plugins:
  - use: chainroute.plugins.sticky_session.StickySession
    with:
      ttl_seconds: 42
""")
    [plugin] = load_chain(str(f))
    assert plugin.name == "StickySession" and plugin.ttl == 42


def test_loads_a_local_file_next_to_the_chain_with_no_packaging(tmp_path):
    write(tmp_path, "myplugins.py", """
from chainroute import RoutingPlugin

class Local(RoutingPlugin):
    def configure(self, greeting="hi"):
        self.greeting = greeting
""")
    f = write(tmp_path, "chain.yaml", """
plugins:
  - use: myplugins.Local
    with:
      greeting: yo
""")
    [plugin] = load_chain(str(f))
    assert plugin.greeting == "yo"


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(ChainConfigError, match="not found"):
        load_chain(str(tmp_path / "nope.yaml"))


def test_missing_use_key_is_a_clear_error(tmp_path):
    f = write(tmp_path, "chain.yaml", "plugins:\n  - with: {}\n")
    with pytest.raises(ChainConfigError, match="use"):
        load_chain(str(f))


def test_not_a_plugin_subclass_is_rejected(tmp_path):
    write(tmp_path, "notaplugin.py", "class NotAPlugin:\n    pass\n")
    f = write(tmp_path, "chain.yaml", "plugins:\n  - use: notaplugin.NotAPlugin\n")
    with pytest.raises(ChainConfigError, match="not a RoutingPlugin"):
        load_chain(str(f))


def test_bad_yaml_is_a_clear_error(tmp_path):
    f = write(tmp_path, "chain.yaml", "plugins: [this is: not: valid")
    with pytest.raises(ChainConfigError, match="not valid YAML"):
        load_chain(str(f))


def test_no_plugins_key_is_a_clear_error(tmp_path):
    f = write(tmp_path, "chain.yaml", "foo: bar\n")
    with pytest.raises(ChainConfigError, match="plugins"):
        load_chain(str(f))


def test_plugins_run_in_the_order_listed(tmp_path):
    f = write(tmp_path, "chain.yaml", """
plugins:
  - use: chainroute.plugins.keyword.KeywordRoute
  - use: chainroute.plugins.sticky_session.StickySession
  - use: chainroute.plugins.tag_affinity.TagAffinity
""")
    names = [p.name for p in load_chain(str(f))]
    assert names == ["KeywordRoute", "StickySession", "TagAffinity"]
