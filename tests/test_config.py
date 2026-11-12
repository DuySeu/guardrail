from __future__ import annotations

import asyncio

import pytest

from core import Action, GuardrailConfigError, load_guardrail
from tests.fakes import FakeChatClient, FakeEmbedder

JUDGE = "  - {type: llm_judge, model: m, categories: [x], threshold: 0.5}\n"


def _write(tmp_path, body):
    path = tmp_path / "config.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_rejects_unknown_top_level_key(tmp_path):
    with pytest.raises(GuardrailConfigError, match="unknown top-level"):
        load_guardrail(_write(tmp_path, "inputs: []\n"))


def test_rejects_unregistered_type(tmp_path):
    with pytest.raises(GuardrailConfigError, match="unknown guard type"):
        load_guardrail(_write(tmp_path, "input:\n  - {type: nope}\n"))


def test_rejects_entry_without_type(tmp_path):
    with pytest.raises(GuardrailConfigError, match="'type'"):
        load_guardrail(_write(tmp_path, "input:\n  - {limit: 3}\n"))


def test_rejects_duplicate_names_in_stage(tmp_path):
    body = "input:\n  - {type: max_length, limit: 3}\n  - {type: max_length, limit: 5}\n"
    with pytest.raises(GuardrailConfigError, match="duplicate"):
        load_guardrail(_write(tmp_path, body))


def test_allows_same_name_in_different_stages(tmp_path):
    body = "input:\n  - {type: max_length, limit: 3}\noutput:\n  - {type: max_length, limit: 3}\n"
    load_guardrail(_write(tmp_path, body))


def test_rejects_missing_referenced_file(tmp_path):
    with pytest.raises(GuardrailConfigError, match="file not found"):
        load_guardrail(_write(tmp_path, "input:\n  - {type: keyword, file: nope.txt}\n"))


def test_rejects_missing_api_key_when_client_not_passed(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(GuardrailConfigError, match="OPENROUTER_API_KEY"):
        load_guardrail(_write(tmp_path, f"input:\n{JUDGE}"))


def test_rejects_missing_api_key_when_only_other_client_passed(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (tmp_path / "ex.txt").write_text("x\n", encoding="utf-8")
    body = f"input:\n{JUDGE}  - {{type: embedding, mode: deny, examples: ex.txt, model: m, threshold: 0.5}}\n"
    with pytest.raises(GuardrailConfigError, match="OPENROUTER_API_KEY"):
        load_guardrail(_write(tmp_path, body), chat_client=FakeChatClient([]))


def test_accepts_missing_api_key_when_needed_clients_are_passed(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    load_guardrail(_write(tmp_path, f"input:\n{JUDGE}"), chat_client=FakeChatClient([]))


def test_rule_only_config_needs_no_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    load_guardrail(_write(tmp_path, "input:\n  - {type: max_length, limit: 3}\n"))


def test_resolves_paths_relative_to_yaml_file(tmp_path):
    sub = tmp_path / "conf"
    sub.mkdir()
    (sub / "deny.txt").write_text("cấm\n", encoding="utf-8")
    guardrail = load_guardrail(_write(sub, "input:\n  - {type: keyword, file: deny.txt}\n"))
    decision = asyncio.run(guardrail.check_input("từ cấm"))
    assert decision.action is Action.BLOCK


def test_empty_config_allows_everything(tmp_path):
    guardrail = load_guardrail(_write(tmp_path, ""), embedder=FakeEmbedder({}))
    assert asyncio.run(guardrail.check_input("x")).action is Action.ALLOW
