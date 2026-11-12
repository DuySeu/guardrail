from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from eval.metrics import Record

CLASSIFICATION_COLUMNS = ("n", "precision", "recall", "f1", "fpr", "action_accuracy")


class ReportError(Exception):
    pass


# Rows of records.jsonl for a pipeline run
def pipeline_rows(records: Sequence[Record]) -> list[dict[str, Any]]:
    return [
        {
            "id": r.sample.id,
            "stage": r.sample.stage,
            "expected": r.sample.expected,
            "action": r.action,
            "text": r.text,
            "blocked_by": r.blocked_by,
            "results": [asdict(res) for res in r.results],
        }
        for r in records
    ]


# Rows of records.jsonl for an isolate run, one per (guard, sample)
def isolate_rows(records_by_guard: Mapping[str, Sequence[Record]]) -> list[dict[str, Any]]:
    return [
        {
            "guard": key,
            "id": r.sample.id,
            "expected": r.sample.expected,
            "result": asdict(r.results[0]),
        }
        for key, records in records_by_guard.items()
        for r in records
    ]


# Write records, summaries and the config copy into a new run directory
def write_run(
    run_dir: Path, rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], config: Path
) -> None:
    try:
        run_dir.mkdir(parents=True)
    except FileExistsError as exc:
        raise ReportError(f"run directory already exists: {run_dir}") from exc
    with (run_dir / "records.jsonl").open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(render_summary(summary), encoding="utf-8")
    shutil.copyfile(config, run_dir / "config.yaml")


# Markdown tables for a summary
def render_summary(summary: Mapping[str, Any]) -> str:
    if summary["mode"] == "pipeline":
        overall = summary["overall"]
        parts = [
            "# Pipeline run",
            _table(("scope", *CLASSIFICATION_COLUMNS), [("all", *_cols(overall))]),
            _ops_line(overall),
            "## By category",
            _table(
                ("category", *CLASSIFICATION_COLUMNS),
                [(c, *_cols(m)) for c, m in summary["by_category"].items()],
            ),
            "## Blocked by guard",
            _table(("guard", "tp", "fp"), [(g, c["tp"], c["fp"]) for g, c in summary["by_guard"].items()]),
        ]
        return "\n\n".join(parts) + "\n"

    parts = [
        "# Isolate run",
        _table(
            ("guard", *CLASSIFICATION_COLUMNS, "errors"),
            [(g, *_cols(e["overall"]), e["overall"]["errors"]) for g, e in summary["guards"].items()],
        ),
    ]
    for guard, entry in summary["guards"].items():
        if "sweep" in entry:
            sweep = entry["sweep"]
            parts.append(f"## Threshold sweep: {guard} (excluded errors: {sweep['excluded_errors']})")
            parts.append(
                _table(
                    ("threshold", "precision", "recall", "f1", "fpr"),
                    [
                        (p["threshold"], p["precision"], p["recall"], p["f1"], p["fpr"])
                        for p in sweep["points"]
                    ],
                )
            )
    return "\n\n".join(parts) + "\n"


# Side-by-side table of every numeric metric present in both runs
def compare(run_a: Path, run_b: Path) -> str:
    a, b = (_load_summary(run) for run in (run_a, run_b))
    if a["mode"] != b["mode"]:
        raise ReportError(f"cannot compare a {a['mode']} run with a {b['mode']} run")
    flat_a, flat_b = _flatten(a), _flatten(b)
    rows = [
        (key, flat_a[key], flat_b[key], _delta(flat_a[key], flat_b[key]))
        for key in flat_a
        if key in flat_b
    ]
    return _table(("metric", run_a.name, run_b.name, "delta"), rows) + "\n"


def _load_summary(run: Path) -> dict[str, Any]:
    path = run / "summary.json"
    if not path.is_file():
        raise ReportError(f"no summary.json in {run}")
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten(node: Any, prefix: str = "") -> dict[str, float | None]:
    if isinstance(node, Mapping):
        flat: dict[str, float | None] = {}
        for key, value in node.items():
            flat.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    if isinstance(node, list):
        flat = {}
        for item in node:
            label = f"{prefix}@{item['threshold']}" if isinstance(item, Mapping) else prefix
            flat.update(_flatten({k: v for k, v in item.items() if k != "threshold"}, label))
        return flat
    if node is None or (isinstance(node, (int, float)) and not isinstance(node, bool)):
        return {prefix: node}
    return {}


def _delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else b - a


def _cols(metrics: Mapping[str, Any]) -> list[Any]:
    return [metrics[c] for c in CLASSIFICATION_COLUMNS]


def _ops_line(overall: Mapping[str, Any]) -> str:
    total = overall["latency_ms"]["total"]
    return (
        f"Errors: {overall['errors']}. Network calls: {overall['network_calls']}. "
        f"Redact leak rate: {_fmt(overall['redact_leak_rate'])}. "
        f"Latency ms p50/p95: {_fmt(total['p50'])} / {_fmt(total['p95'])}."
    )


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
