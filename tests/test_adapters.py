from __future__ import annotations

import json

from core import Action
from eval.adapters import prompt_injections, toxic_chat
from eval.dataset import load_dataset


def test_injection_row_maps_to_block():
    sample = prompt_injections.to_sample({"text": "ignore rules", "label": 1}, 3)
    assert (sample.id, sample.expected, sample.category) == ("pi-0003", Action.BLOCK, "prompt_injection")


def test_benign_row_maps_to_allow():
    sample = prompt_injections.to_sample({"text": "weather today", "label": 0}, 0)
    assert (sample.expected, sample.category) == (Action.ALLOW, "benign")


def test_toxic_row_maps_to_block_with_toxic_category():
    row = {"user_input": "x", "toxicity": "1", "jailbreaking": "0"}
    sample = toxic_chat.to_sample(row, 0)
    assert (sample.expected, sample.category) == (Action.BLOCK, "toxic")


def test_jailbreak_row_maps_to_jailbreak_category():
    row = {"user_input": "x", "toxicity": "1", "jailbreaking": "1"}
    assert toxic_chat.to_sample(row, 0).category == "jailbreak"


def test_generated_samples_pass_dataset_validation(tmp_path):
    samples = [
        prompt_injections.to_sample({"text": "a", "label": 1}, 0),
        toxic_chat.to_sample({"user_input": "b", "toxicity": "0", "jailbreaking": "0"}, 0),
    ]
    path = tmp_path / "d.jsonl"
    path.write_text(
        "\n".join(s.model_dump_json(exclude_defaults=True) for s in samples), encoding="utf-8"
    )
    loaded = load_dataset(path)
    assert [s.id for s in loaded] == ["pi-0000", "tc-00000"]
    assert json.loads(samples[0].model_dump_json(exclude_defaults=True))["stage"] == "input"
