from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core import Action, Decision, GuardResult

from eval.dataset import Sample

THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(1, 20))


@dataclass(frozen=True)
class Record:
    sample: Sample
    action: Action
    text: str
    blocked_by: str | None
    results: tuple[GuardResult, ...]

    @classmethod
    def from_decision(cls, sample: Sample, decision: Decision) -> Record:
        return cls(sample, decision.action, decision.text, decision.blocked_by, decision.results)

    @classmethod
    def from_result(cls, sample: Sample, result: GuardResult) -> Record:
        blocked_by = result.guard if result.action is Action.BLOCK else None
        return cls(sample, result.action, result.text, blocked_by, (result,))

    @property
    def has_error(self) -> bool:
        return any(r.error for r in self.results)


# Confusion matrix and rates; positive means expected or predicted is not allow
def classification(records: Sequence[Record]) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for record in records:
        actual = record.sample.expected is not Action.ALLOW
        predicted = record.action is not Action.ALLOW
        tp += actual and predicted
        fp += predicted and not actual
        fn += actual and not predicted
        tn += not actual and not predicted
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    if precision is None or recall is None:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    matched = sum(r.action is r.sample.expected for r in records)
    return {
        "n": len(records),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": _ratio(fp, fp + tn),
        "action_accuracy": _ratio(matched, len(records)),
    }


def by_category(records: Sequence[Record]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[record.sample.category].append(record)
    return {category: classification(group) for category, group in sorted(groups.items())}


# Share of redact samples whose final text still contains a forbidden string
def redact_leak_rate(records: Sequence[Record]) -> float | None:
    checked = [
        r for r in records if r.sample.expected is Action.REDACT and r.sample.must_not_contain
    ]
    leaked = sum(any(s in r.text for s in r.sample.must_not_contain) for r in checked)
    return _ratio(leaked, len(checked))


# p50/p95 latency of the whole run per record and per guard
def latency(records: Sequence[Record]) -> dict[str, Any]:
    per_guard: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for result in record.results:
            per_guard[_key(record, result)].append(result.latency_ms)
    totals = [sum(r.latency_ms for r in record.results) for record in records]
    return {
        "total": _percentiles(totals),
        "by_guard": {key: _percentiles(values) for key, values in sorted(per_guard.items())},
    }


def error_count(records: Sequence[Record]) -> int:
    return sum(r.has_error for r in records)


# How many records each guard blocked, split into correct and wrong blocks
def blocked_by_guard(records: Sequence[Record]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0})
    for record in records:
        if record.blocked_by:
            key = f"{record.sample.stage}/{record.blocked_by}"
            counts[key]["tp" if record.sample.expected is not Action.ALLOW else "fp"] += 1
    return dict(sorted(counts.items()))


# Precision/recall at each threshold from recorded scores of single-guard records.
# With positive_categories, positives from other categories are left out entirely.
def sweep(
    records: Sequence[Record], positive_categories: Collection[str] | None = None
) -> dict[str, Any]:
    scored = [r for r in records if r.results[0].score is not None]
    if positive_categories:
        scored = [
            r
            for r in scored
            if r.sample.expected is Action.ALLOW or r.sample.category in positive_categories
        ]
    points = []
    for threshold in THRESHOLDS:
        thresholded = [
            _with_action(r, Action.BLOCK if r.results[0].score >= threshold else Action.ALLOW)
            for r in scored
        ]
        metrics = classification(thresholded)
        rates = {k: metrics[k] for k in ("precision", "recall", "f1", "fpr")}
        points.append({"threshold": threshold, **rates})
    errored = sum(r.results[0].score is None for r in records)
    return {"excluded_errors": errored, "points": points}


# summary.json for a pipeline run
def summarize_pipeline(
    records: Sequence[Record], network_guards: Collection[str]
) -> dict[str, Any]:
    return {
        "mode": "pipeline",
        "overall": _overall(records, network_guards),
        "overall_excluding_errors": classification([r for r in records if not r.has_error]),
        "by_category": by_category(records),
        "by_guard": blocked_by_guard(records),
    }


# summary.json for an isolate run; threshold sweep only for network guards (continuous scores)
def summarize_isolate(
    records_by_guard: Mapping[str, Sequence[Record]],
    network_guards: Collection[str],
    positive_categories: Collection[str] | None = None,
) -> dict[str, Any]:
    guards = {}
    for key, records in records_by_guard.items():
        entry = {
            "overall": _overall(records, network_guards),
            "overall_excluding_errors": classification([r for r in records if not r.has_error]),
            "by_category": by_category(records),
        }
        if key in network_guards:
            entry["sweep"] = sweep(records, positive_categories)
        guards[key] = entry
    return {"mode": "isolate", "guards": guards}


def _overall(records: Sequence[Record], network_guards: Collection[str]) -> dict[str, Any]:
    network_calls = sum(_key(r, res) in network_guards for r in records for res in r.results)
    return {
        **classification(records),
        "redact_leak_rate": redact_leak_rate(records),
        "errors": error_count(records),
        "network_calls": network_calls,
        "latency_ms": latency(records),
    }


def _key(record: Record, result: GuardResult) -> str:
    return f"{record.sample.stage}/{result.guard}"


def _with_action(record: Record, action: Action) -> Record:
    return Record(record.sample, action, record.text, record.blocked_by, record.results)


# None when the denominator is zero, so empty classes never read as 0%
def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


# Nearest-rank percentiles; works for a single value
def _percentiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None}
    ordered = sorted(values)
    ranks = {p: ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)] for p in (50, 95)}
    return {"p50": ranks[50], "p95": ranks[95]}
