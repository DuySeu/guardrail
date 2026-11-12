from __future__ import annotations

import csv
import io
from collections.abc import Mapping

import httpx

from eval.dataset import Sample

# lmsys/toxic-chat, CC-BY-NC-4.0: internal benchmarking only, output stays in datasets/public/
CSV_URL = (
    "https://huggingface.co/datasets/lmsys/toxic-chat/resolve/main/"
    "data/0124/toxic-chat_annotation_test.csv"
)


def to_sample(row: Mapping[str, str], index: int) -> Sample:
    if row["jailbreaking"] == "1":
        category = "jailbreak"
    elif row["toxicity"] == "1":
        category = "toxic"
    else:
        category = "benign"
    return Sample(
        id=f"tc-{index:05d}",
        stage="input",
        text=row["user_input"],
        expected="allow" if category == "benign" else "block",
        category=category,
    )


def fetch() -> list[Sample]:
    response = httpx.get(CSV_URL, timeout=60, follow_redirects=True)
    response.raise_for_status()
    rows = csv.DictReader(io.StringIO(response.text))
    return [to_sample(row, index) for index, row in enumerate(rows)]
