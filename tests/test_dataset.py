from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import Action, Stage
from eval.dataset import DatasetError, load_dataset

VALID = {"id": "a", "stage": "input", "text": "t", "expected": "allow", "category": "benign"}


def _write(tmp_path, *lines):
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_loads_valid_file(tmp_path):
    row = {**VALID, "id": "b", "expected": "redact", "must_not_contain": ["x"]}
    samples = load_dataset(_write(tmp_path, json.dumps(VALID), json.dumps(row)))
    assert [s.id for s in samples] == ["a", "b"]
    assert samples[0].stage is Stage.INPUT
    assert samples[1].expected is Action.REDACT
    assert samples[1].must_not_contain == ("x",)


def test_reports_line_number_for_invalid_json(tmp_path):
    with pytest.raises(DatasetError, match=r"d\.jsonl:2"):
        load_dataset(_write(tmp_path, json.dumps(VALID), "{not json"))


def test_rejects_missing_required_field(tmp_path):
    row = {k: v for k, v in VALID.items() if k != "category"}
    with pytest.raises(DatasetError, match="category"):
        load_dataset(_write(tmp_path, json.dumps(row)))


def test_rejects_unknown_expected_value(tmp_path):
    with pytest.raises(DatasetError, match=":1"):
        load_dataset(_write(tmp_path, json.dumps({**VALID, "expected": "deny"})))


def test_rejects_unknown_field(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(_write(tmp_path, json.dumps({**VALID, "label": 1})))


def test_rejects_duplicate_id(tmp_path):
    with pytest.raises(DatasetError, match="duplicate id 'a'"):
        load_dataset(_write(tmp_path, json.dumps(VALID), json.dumps(VALID)))


def test_skips_blank_lines(tmp_path):
    assert len(load_dataset(_write(tmp_path, "", json.dumps(VALID), "   "))) == 1


def test_missing_file_raises_dataset_error(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "nope.jsonl")


def test_bundled_sample_dataset_is_valid():
    assert len(load_dataset(Path("datasets/sample.jsonl"))) == 20
