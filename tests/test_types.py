from __future__ import annotations

from core.types import Action, GuardResult, decide


def _result(guard: str, action: Action) -> GuardResult:
    return GuardResult(guard=guard, action=action, text="t", score=1.0)


def test_decide_returns_block_when_any_result_blocks():
    decision = decide("t", [_result("a", Action.REDACT), _result("b", Action.BLOCK)])
    assert decision.action is Action.BLOCK


def test_decide_returns_redact_when_only_redacts():
    decision = decide("t", [_result("a", Action.ALLOW), _result("b", Action.REDACT)])
    assert decision.action is Action.REDACT
    assert decision.blocked_by is None


def test_decide_returns_allow_for_empty_results():
    decision = decide("t", [])
    assert decision.action is Action.ALLOW
    assert decision.text == "t"


def test_decide_sets_blocked_by_to_blocking_guard():
    decision = decide("t", [_result("a", Action.ALLOW), _result("b", Action.BLOCK)])
    assert decision.blocked_by == "b"
