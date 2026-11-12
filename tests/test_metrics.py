from __future__ import annotations

import pytest

from core import Action, GuardResult
from eval.dataset import Sample
from eval.metrics import (
    Record,
    classification,
    latency,
    redact_leak_rate,
    summarize_pipeline,
    sweep,
)


def _record(expected, action, *, category="c", score=0.5, error=None, text="t", leak=(), ms=1.0):
    sample = Sample(
        id=f"{expected}-{action}-{score}-{category}-{text}",
        stage="input",
        text=text,
        expected=expected,
        category=category,
        must_not_contain=leak,
    )
    result = GuardResult(
        guard="g", action=Action(action), text=text, score=score, error=error, latency_ms=ms
    )
    return Record.from_result(sample, result)


def test_precision_recall_f1_match_hand_computed_values():
    # tp=2, fp=1, fn=1, tn=1
    records = [
        _record("block", "block", score=0.1),
        _record("redact", "redact", score=0.2),
        _record("allow", "block", score=0.3),
        _record("block", "allow", score=0.4),
        _record("allow", "allow", score=0.5),
    ]
    m = classification(records)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(0.5)


def test_metrics_are_none_when_no_positive_samples():
    m = classification([_record("allow", "allow")])
    assert m["recall"] is None
    assert m["precision"] is None
    assert m["f1"] is None
    assert m["fpr"] == 0.0


def test_metrics_are_none_when_no_negative_samples():
    m = classification([_record("block", "block")])
    assert m["fpr"] is None
    assert m["recall"] == 1.0


def test_f1_is_zero_when_nothing_is_caught():
    m = classification([_record("block", "allow"), _record("allow", "block", score=0.9)])
    assert m["f1"] == 0.0


def test_action_accuracy_counts_block_on_redact_sample_as_mismatch():
    m = classification([_record("redact", "block"), _record("block", "block", score=0.9)])
    assert m["recall"] == 1.0
    assert m["action_accuracy"] == 0.5


def test_leak_rate_counts_samples_with_remaining_string():
    records = [
        _record("redact", "redact", text="mail [EMAIL]", leak=("a@b.co",)),
        _record("redact", "allow", text="mail a@b.co", leak=("a@b.co",)),
    ]
    assert redact_leak_rate(records) == 0.5


def test_leak_rate_is_none_without_redact_samples():
    assert redact_leak_rate([_record("block", "block")]) is None


def test_sweep_excludes_errored_records_and_reports_count():
    records = [
        _record("block", "block", score=0.9),
        _record("allow", "allow", score=0.1),
        _record("block", "block", score=None, error="TimeoutError"),
    ]
    result = sweep(records)
    assert result["excluded_errors"] == 1
    at_half = next(p for p in result["points"] if p["threshold"] == 0.5)
    assert (at_half["precision"], at_half["recall"], at_half["fpr"]) == (1.0, 1.0, 0.0)


def test_sweep_at_threshold_boundary_counts_equal_score_as_positive():
    result = sweep([_record("block", "allow", score=0.3), _record("allow", "allow", score=0.0)])
    at_point_three = next(p for p in result["points"] if p["threshold"] == 0.3)
    assert at_point_three["recall"] == 1.0


def test_sweep_with_categories_drops_other_positives():
    records = [
        _record("block", "allow", category="injection", score=0.9),
        _record("block", "allow", category="toxic", score=0.0),
        _record("allow", "allow", category="benign", score=0.0),
    ]
    at_half = next(p for p in sweep(records, {"injection"})["points"] if p["threshold"] == 0.5)
    assert at_half["recall"] == 1.0


def test_summary_excluding_errors_drops_errored_samples():
    records = [
        _record("allow", "block", score=None, error="HTTP 429"),
        _record("allow", "allow", score=0.1),
    ]
    summary = summarize_pipeline(
        [Record(r.sample, r.action, r.text, "g" if r.action is Action.BLOCK else None, r.results)
         for r in records],
        network_guards={"input/g"},
    )
    assert summary["overall"]["fp"] == 1
    assert summary["overall"]["errors"] == 1
    assert summary["overall"]["network_calls"] == 2
    assert summary["overall_excluding_errors"]["fp"] == 0
    assert summary["by_guard"] == {"input/g": {"tp": 0, "fp": 1}}


def test_p95_of_single_record_equals_its_latency():
    stats = latency([_record("allow", "allow", ms=12.5)])
    assert stats["total"] == {"p50": 12.5, "p95": 12.5}
    assert stats["by_guard"]["input/g"]["p95"] == 12.5
