from __future__ import annotations

import asyncio
from typing import ClassVar

from core import (
    Action,
    CommonParams,
    GuardContext,
    GuardDeps,
    GuardResult,
    load_guardrail,
    register_guard,
)
from tests.fakes import FakeEmbedder


class StepParams(CommonParams):
    mode: str = "allow"
    sleep_s: float = 0.0


# Test guard: appends its name to the text, then allows, redacts, blocks, raises or sleeps
@register_guard("test_step")
class StepGuard:
    Params = StepParams
    requires: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, params: StepParams, deps: GuardDeps) -> None:
        self.params = params
        self.name = params.name

    async def check(self, text: str, context: GuardContext) -> GuardResult:
        if self.params.mode == "raise":
            raise RuntimeError("boom")
        await asyncio.sleep(self.params.sleep_s)
        action = {"redact": Action.REDACT, "block": Action.BLOCK}.get(self.params.mode, Action.ALLOW)
        new_text = f"{text}>{self.name}" if action is Action.REDACT else text
        return GuardResult(guard=self.name, action=action, text=new_text, score=0.5)


def _guardrail(tmp_path, *steps):
    lines = "".join(f"  - {{type: test_step, name: {name}, {extra}}}\n" for name, extra in steps)
    path = tmp_path / "config.yaml"
    path.write_text(f"input:\n{lines}", encoding="utf-8")
    return load_guardrail(path)


def _check(guardrail, text="t"):
    return asyncio.run(guardrail.check_input(text))


def test_runs_guards_in_declared_order(tmp_path):
    decision = _check(_guardrail(tmp_path, ("a", "mode: allow"), ("b", "mode: allow")))
    assert [r.guard for r in decision.results] == ["a", "b"]


def test_redacted_text_is_passed_to_next_guard(tmp_path):
    decision = _check(_guardrail(tmp_path, ("a", "mode: redact"), ("b", "mode: redact")))
    assert decision.action is Action.REDACT
    assert decision.text == "t>a>b"


def test_stops_after_first_block(tmp_path):
    decision = _check(_guardrail(tmp_path, ("a", "mode: block"), ("b", "mode: allow")))
    assert decision.action is Action.BLOCK
    assert decision.blocked_by == "a"
    assert [r.guard for r in decision.results] == ["a"]


def test_block_text_keeps_earlier_redactions(tmp_path):
    decision = _check(_guardrail(tmp_path, ("a", "mode: redact"), ("b", "mode: block")))
    assert decision.text == "t>a"


def test_error_with_on_error_block_stops_pipeline(tmp_path):
    decision = _check(_guardrail(tmp_path, ("a", "mode: raise"), ("b", "mode: allow")))
    assert decision.action is Action.BLOCK
    assert decision.blocked_by == "a"
    assert decision.results[0].error == "RuntimeError: boom"


def test_error_with_on_error_allow_continues_with_unchanged_text(tmp_path):
    guardrail = _guardrail(
        tmp_path, ("a", "mode: raise, on_error: allow"), ("b", "mode: redact")
    )
    decision = _check(guardrail)
    assert decision.action is Action.REDACT
    assert decision.text == "t>b"


def test_timeout_is_treated_as_error(tmp_path):
    decision = _check(_guardrail(tmp_path, ("a", "sleep_s: 1, timeout_s: 0.01")))
    assert decision.action is Action.BLOCK
    assert decision.results[0].error.startswith("TimeoutError")


def test_error_result_has_no_score(tmp_path):
    decision = _check(_guardrail(tmp_path, ("a", "mode: raise")))
    assert decision.results[0].score is None
    assert decision.results[0].category is None


def test_latency_is_recorded(tmp_path):
    decision = _check(_guardrail(tmp_path, ("a", "sleep_s: 0.02")))
    assert decision.results[0].latency_ms >= 20


def test_missing_stage_returns_allow_with_unchanged_text(tmp_path):
    guardrail = _guardrail(tmp_path, ("a", "mode: block"))
    decision = asyncio.run(guardrail.check_output("answer", user_input="q"))
    assert decision.action is Action.ALLOW
    assert decision.text == "answer"
    assert decision.results == ()


def test_only_returns_guardrail_with_same_guard_instance(tmp_path):
    guardrail = _guardrail(tmp_path, ("a", "mode: block"), ("b", "mode: redact"))
    single = guardrail.only("input", "b")
    assert single.guards("input") == (guardrail.guards("input")[1],)
    assert _check(single).action is Action.REDACT


def test_warmup_embeds_examples_before_first_check(tmp_path):
    (tmp_path / "ex.txt").write_text("bad\n", encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text(
        "input:\n  - {type: embedding, mode: deny, examples: ex.txt, model: m, threshold: 0.9}\n",
        encoding="utf-8",
    )
    embedder = FakeEmbedder({"bad": [1.0, 0.0], "ok": [0.0, 1.0]})
    guardrail = load_guardrail(path, embedder=embedder)
    asyncio.run(guardrail.warmup())
    assert embedder.calls == 1
    assert _check(guardrail, "ok").action is Action.ALLOW
    assert embedder.calls == 2
