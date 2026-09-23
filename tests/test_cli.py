import json

import pytest

from chainroute.__main__ import _deployments_for_group, _read_prompts, _simulate


def write(path, content):
    path.write_text(content)
    return path


def test_deployments_for_group_filters_and_synthesizes_ids(tmp_path):
    doc = {"model_list": [
        {"model_name": "chat", "litellm_params": {"model": "openai/a"}},
        {"model_name": "chat", "litellm_params": {"model": "openai/b"}, "model_info": {"id": "given"}},
        {"model_name": "other", "litellm_params": {"model": "openai/c"}},
    ]}
    deps = _deployments_for_group(doc, "chat")
    assert [d["litellm_params"]["model"] for d in deps] == ["openai/a", "openai/b"]
    assert deps[0]["model_info"]["id"] == "sim-0"
    assert deps[1]["model_info"]["id"] == "given"  # a real id, if the config set one, is kept


def test_deployments_for_group_empty_when_no_match():
    assert _deployments_for_group({"model_list": []}, "chat") == []


def test_read_prompts_accepts_a_plain_string_line(tmp_path):
    f = write(tmp_path / "p.jsonl", '"hello there"\n')
    [prompt] = list(_read_prompts(str(f)))
    assert prompt == {"messages": [{"role": "user", "content": "hello there"}]}


def test_read_prompts_accepts_a_full_object_line(tmp_path):
    f = write(tmp_path / "p.jsonl",
              json.dumps({"messages": [{"role": "user", "content": "hi"}], "session_id": "s1"}) + "\n")
    [prompt] = list(_read_prompts(str(f)))
    assert prompt["session_id"] == "s1"


def test_read_prompts_skips_blank_lines(tmp_path):
    f = write(tmp_path / "p.jsonl", '"a"\n\n   \n"b"\n')
    prompts = list(_read_prompts(str(f)))
    assert len(prompts) == 2


def test_read_prompts_rejects_bad_json(tmp_path):
    f = write(tmp_path / "p.jsonl", "not json{\n")
    with pytest.raises(SystemExit):
        list(_read_prompts(str(f)))


def test_read_prompts_rejects_an_object_with_no_messages_key(tmp_path):
    f = write(tmp_path / "p.jsonl", '{"foo": "bar"}\n')
    with pytest.raises(SystemExit):
        list(_read_prompts(str(f)))


@pytest.fixture()
def fixture_files(tmp_path):
    config = write(tmp_path / "config.yaml", """
model_list:
  - model_name: chat
    litellm_params: {model: openai/mini}
  - model_name: chat
    litellm_params: {model: azure/safe, custom_llm_provider: azure}
""")
    chain = write(tmp_path / "chain.yaml", """
plugins:
  - use: chainroute.plugins.sensitive_data.SensitiveDataGuard
    with:
      trusted_providers: [azure]
      on_match: exclude
""")
    prompts = write(tmp_path / "prompts.jsonl", '"hello"\n"my ssn is 123-45-6789"\n')
    return str(config), str(chain), str(prompts)


def test_simulate_prints_a_readable_report(fixture_files, capsys):
    config, chain, prompts = fixture_files
    _simulate(config, "chat", chain, prompts, as_json=False)
    out = capsys.readouterr().out
    assert "no opinion (passthrough)" in out
    assert "EXCLUDED - looks like sensitive data" in out
    assert "openai/mini" in out and "azure/safe" in out


def test_simulate_json_mode_emits_one_parseable_object_per_line(fixture_files, capsys):
    config, chain, prompts = fixture_files
    _simulate(config, "chat", chain, prompts, as_json=True)
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l]
    assert len(lines) == 2
    assert lines[0]["decision"] == "no opinion (passthrough)"
    assert any(c["excluded"] for c in lines[1]["candidates"])


def test_simulate_exits_clearly_for_an_unknown_group(fixture_files):
    config, chain, prompts = fixture_files
    with pytest.raises(SystemExit):
        _simulate(config, "no-such-group", chain, prompts, as_json=False)
