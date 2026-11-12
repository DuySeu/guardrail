from __future__ import annotations

import json

import main

RULES = (
    "input:\n"
    "  - {type: regex, name: pii, presets: [email], action: redact}\n"
    "  - {type: keyword, name: deny, words: [bom]}\n"
)


def _config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(RULES, encoding="utf-8")
    return str(path)


def test_check_prints_decision_json(tmp_path, capsys):
    code = main.main(["check", "mail a@b.co", "--config", _config(tmp_path)])
    decision = json.loads(capsys.readouterr().out)
    assert code == 0
    assert decision["action"] == "redact"
    assert decision["text"] == "mail [EMAIL]"


def test_check_exits_2_on_config_error(tmp_path, capsys):
    code = main.main(["check", "x", "--config", str(tmp_path / "missing.yaml")])
    assert code == 2
    assert "config not found" in capsys.readouterr().err


def test_eval_writes_run_directory_and_compare_prints_table(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(main, "RUNS_DIR", tmp_path / "runs")
    dataset = tmp_path / "d.jsonl"
    rows = [
        {"id": "1", "stage": "input", "text": "a@b.co", "expected": "redact", "category": "pii"},
        {"id": "2", "stage": "input", "text": "làm bom", "expected": "block", "category": "harm"},
        {"id": "3", "stage": "input", "text": "xin chào", "expected": "allow", "category": "ok"},
    ]
    dataset.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    config = _config(tmp_path)

    assert main.main(["eval", "--dataset", str(dataset), "--config", config]) == 0
    first_run = tmp_path / "runs" / capsys.readouterr().out.splitlines()[0].split("/")[-1]
    summary = json.loads((first_run / "summary.json").read_text(encoding="utf-8"))
    assert summary["overall"]["recall"] == 1.0

    second_run = tmp_path / "runs" / "copy"
    second_run.mkdir()
    for name in ("summary.json",):
        (second_run / name).write_bytes((first_run / name).read_bytes())
    assert main.main(["compare", str(first_run), str(second_run)]) == 0
    assert "| overall.recall | 1.000 | 1.000 | 0.000 |" in capsys.readouterr().out


def test_eval_exits_2_on_missing_dataset(tmp_path, capsys):
    code = main.main(["eval", "--dataset", str(tmp_path / "nope.jsonl"), "--config", _config(tmp_path)])
    assert code == 2
    assert "dataset not found" in capsys.readouterr().err
