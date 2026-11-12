from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import core  # noqa: F401  registers the built-in guards
from core.registry import GuardDeps, build_guard
from core.types import Action, GuardContext, GuardrailConfigError, Stage

CTX = GuardContext(stage=Stage.INPUT)


def _check(type_name, params, text, base_dir=Path(".")):
    guard = build_guard(type_name, params, GuardDeps(base_dir=base_dir))
    return asyncio.run(guard.check(text, CTX))


@pytest.mark.parametrize(
    ("preset", "hit", "miss"),
    [
        ("email", "liên hệ a.b@example.com nhé", "a at example dot com"),
        ("phone_vn", "gọi 0912 345 678 nhé", "mã đơn 12345"),
        ("phone_vn", "số +84912345678", "số 091234567"),
        ("credit_card", "thẻ 4111 1111 1111 1111", "năm 2026"),
        ("api_key", "key=sk-abcdefghijklmnop1234", "sk-short"),
        ("api_key", "AKIAABCDEFGHIJKLMNOP", "AKIA123"),
    ],
)
def test_preset_matches_hit_and_ignores_miss(preset, hit, miss):
    assert _check("regex", {"presets": [preset]}, hit).action is Action.BLOCK
    assert _check("regex", {"presets": [preset]}, miss).action is Action.ALLOW


def test_regex_redact_replaces_every_match():
    result = _check("regex", {"presets": ["email"], "action": "redact"}, "a@b.co và c@d.vn")
    assert result.action is Action.REDACT
    assert result.text == "[EMAIL] và [EMAIL]"


def test_regex_redact_uses_custom_replacement_literally():
    params = {"patterns": {"id": r"\d{12}"}, "action": "redact", "replacement": r"\1***"}
    result = _check("regex", params, "cccd 012345678901")
    assert result.text == r"cccd \1***"


def test_regex_block_returns_original_text():
    result = _check("regex", {"presets": ["email"]}, "mail a@b.co")
    assert result.action is Action.BLOCK
    assert result.text == "mail a@b.co"
    assert result.score == 1.0


def test_keyword_does_not_match_inside_a_word():
    result = _check("keyword", {"words": ["hack"]}, "hackathon tuần sau")
    assert result.action is Action.ALLOW
    assert result.score == 0.0


def test_keyword_matches_vietnamese_phrase_case_insensitively():
    result = _check("keyword", {"words": ["chế tạo bom"]}, "Cách CHẾ TẠO BOM tại nhà")
    assert result.action is Action.BLOCK


def test_keyword_reads_words_from_file(tmp_path):
    (tmp_path / "deny.txt").write_text("cấm\n\n  từ khoá  \n", encoding="utf-8")
    result = _check("keyword", {"file": "deny.txt"}, "đây là từ khoá", base_dir=tmp_path)
    assert result.action is Action.BLOCK


def test_max_length_allows_text_at_limit():
    assert _check("max_length", {"limit": 3}, "abc").action is Action.ALLOW


def test_max_length_blocks_text_over_limit():
    assert _check("max_length", {"limit": 3}, "abcd").action is Action.BLOCK


def test_empty_text_is_allowed():
    assert _check("regex", {"presets": ["email"]}, "").action is Action.ALLOW
    assert _check("keyword", {"words": ["x"]}, "").action is Action.ALLOW


def test_category_defaults_to_guard_name():
    result = _check("regex", {"name": "pii", "presets": ["email"]}, "a@b.co")
    assert result.category == "pii"


def test_invalid_regex_raises_config_error():
    with pytest.raises(GuardrailConfigError, match="invalid regex"):
        _check("regex", {"patterns": {"bad": "("}}, "x")


def test_unknown_preset_raises_config_error():
    with pytest.raises(GuardrailConfigError):
        _check("regex", {"presets": ["nope"]}, "x")


def test_missing_keyword_file_raises_config_error(tmp_path):
    with pytest.raises(GuardrailConfigError, match="file not found"):
        _check("keyword", {"file": "missing.txt"}, "x", base_dir=tmp_path)
