from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from eval.dataset import Sample

# deepset/prompt-injections, Apache-2.0; label 1 = injection, 0 = benign (checked against rows)
ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "deepset/prompt-injections"
PAGE = 100


def to_sample(row: Mapping[str, Any], index: int) -> Sample:
    injection = row["label"] == 1
    return Sample(
        id=f"pi-{index:04d}",
        stage="input",
        text=row["text"],
        expected="block" if injection else "allow",
        category="prompt_injection" if injection else "benign",
    )


# Page through the test split via the Hugging Face datasets-server JSON API
def fetch(split: str = "test") -> list[Sample]:
    rows: list[Mapping[str, Any]] = []
    with httpx.Client(timeout=30) as http:
        while True:
            params = {"dataset": DATASET, "config": "default", "split": split}
            response = http.get(ROWS_URL, params={**params, "offset": len(rows), "length": PAGE})
            response.raise_for_status()
            body = response.json()
            rows += [item["row"] for item in body["rows"]]
            if not body["rows"] or len(rows) >= body["num_rows_total"]:
                break
    return [to_sample(row, index) for index, row in enumerate(rows)]
