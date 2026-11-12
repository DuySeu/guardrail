from __future__ import annotations

import asyncio
import json
from typing import ClassVar

import pytest

from core import Action, CommonParams, GuardContext, GuardDeps, GuardResult, register_guard
from core import load_guardrail
from eval.dataset import Sample
from eval.metrics import summarize_isolate, summarize_pipeline
from eval.report import ReportError, compare, isolate_rows, pipeline_rows, write_run
from eval.runner import run_isolate, run_pipeline


class WordParams(CommonParams):
    word: str
    delay_s: float = 0.0


# Test guard: blocks text containing its word, tracks peak concurrency
@register_guard("test_word")
class WordGuard:
    Params = WordParams
    requires: ClassVar[frozenset[str]] = frozenset()
    active = 0
    peak = 0

    def __init__(self, params: WordParams, deps: GuardDeps) -> None:
        self.params = params
        self.name = params.name

    async def check(self, text: str, context: GuardContext) -> GuardResult:
        WordGuard.active += 1
        WordGuard.peak = max(WordGuard.peak, WordGuard.active)
        await asyncio.sleep(self.params.delay_s)
        WordGuard.active -= 1
        hit = self.params.word in text
        action = Action.BLOCK if hit else Action.ALLOW
        return GuardResult(guard=self.name, action=action, text=text, score=float(hit))


def _sample(id_, text, expected, stage="input"):
    return Sample(id=id_, stage=stage, text=text, expected=expected, category="c")


SAMPLES = [
    _sample("1", "alpha", "block"),
    _sample("2", "beta", "block"),
    _sample("3", "alpha beta", "block"),
    _sample("4", "gamma", "allow"),
    _sample("5", "beta", "allow", stage="output"),
]


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "input:\n"
        "  - {type: test_word, name: a, word: alpha}\n"
        "  - {type: test_word, name: b, word: beta}\n",
        encoding="utf-8",
    )
    return path


def test_pipeline_run_returns_one_record_per_sample_in_order(config):
    records = asyncio.run(run_pipeline(load_guardrail(config), SAMPLES, 2))
    assert [r.sample.id for r in records] == ["1", "2", "3", "4", "5"]
    assert records[2].blocked_by == "a"


def test_samples_run_against_their_own_stage(config):
    records = asyncio.run(run_pipeline(load_guardrail(config), SAMPLES, 2))
    assert records[4].action is Action.ALLOW
    assert records[4].results == ()


def test_isolate_run_keys_records_by_stage_and_name(config):
    runs = asyncio.run(run_isolate(load_guardrail(config), SAMPLES, 2))
    assert sorted(runs) == ["input/a", "input/b"]
    assert len(runs["input/a"]) == 4


def test_isolate_run_reveals_guard_hidden_by_short_circuit(config):
    runs = asyncio.run(run_isolate(load_guardrail(config), SAMPLES, 2))
    sample_3 = next(r for r in runs["input/b"] if r.sample.id == "3")
    assert sample_3.action is Action.BLOCK


def test_concurrency_limit_is_respected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("input:\n  - {type: test_word, word: x, delay_s: 0.01}\n", encoding="utf-8")
    WordGuard.peak = 0
    samples = [_sample(str(i), "t", "allow") for i in range(10)]
    asyncio.run(run_pipeline(load_guardrail(path), samples, 3))
    assert WordGuard.peak == 3


def _pipeline_run(config, run_dir):
    records = asyncio.run(run_pipeline(load_guardrail(config), SAMPLES, 2))
    write_run(run_dir, pipeline_rows(records), summarize_pipeline(records, set()), config)


def test_write_run_creates_four_files(config, tmp_path):
    run_dir = tmp_path / "runs" / "r1"
    _pipeline_run(config, run_dir)
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "config.yaml",
        "records.jsonl",
        "summary.json",
        "summary.md",
    ]
    first = json.loads((run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["action"] == "block"
    assert first["results"][0]["guard"] == "a"


def test_write_run_refuses_existing_directory(config, tmp_path):
    run_dir = tmp_path / "r1"
    _pipeline_run(config, run_dir)
    with pytest.raises(ReportError, match="already exists"):
        _pipeline_run(config, run_dir)


def test_isolate_rows_have_one_line_per_guard_and_sample(config, tmp_path):
    runs = asyncio.run(run_isolate(load_guardrail(config), SAMPLES, 2))
    rows = isolate_rows(runs)
    assert len(rows) == 8
    assert rows[0]["guard"] == "input/a"
    summary = summarize_isolate(runs, {"input/b"})
    assert "sweep" in summary["guards"]["input/b"]
    assert "sweep" not in summary["guards"]["input/a"]
    write_run(tmp_path / "iso", rows, summary, config)
    assert "Threshold sweep: input/b" in (tmp_path / "iso" / "summary.md").read_text(encoding="utf-8")


def test_compare_reports_metric_deltas(config, tmp_path):
    _pipeline_run(config, tmp_path / "a")
    config.write_text("input:\n  - {type: test_word, name: a, word: alpha}\n", encoding="utf-8")
    _pipeline_run(config, tmp_path / "b")
    table = compare(tmp_path / "a", tmp_path / "b")
    recall_row = next(line for line in table.splitlines() if line.startswith("| overall.recall |"))
    assert recall_row == "| overall.recall | 1.000 | 0.667 | -0.333 |"


def test_compare_rejects_different_modes(config, tmp_path):
    _pipeline_run(config, tmp_path / "a")
    runs = asyncio.run(run_isolate(load_guardrail(config), SAMPLES, 2))
    write_run(tmp_path / "b", isolate_rows(runs), summarize_isolate(runs, set()), config)
    with pytest.raises(ReportError, match="cannot compare"):
        compare(tmp_path / "a", tmp_path / "b")
