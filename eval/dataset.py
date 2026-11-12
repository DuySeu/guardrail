from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from core import Action, Stage


class Sample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    stage: Stage
    text: str
    expected: Action
    category: str
    user_input: str | None = None
    must_not_contain: tuple[str, ...] = ()


class DatasetError(Exception):
    pass


# Read and validate a JSONL dataset, failing on the first bad line
def load_dataset(path: Path) -> list[Sample]:
    if not path.is_file():
        raise DatasetError(f"dataset not found: {path}")
    samples: list[Sample] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                sample = Sample.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise DatasetError(f"{path}:{number}: {exc}") from exc
            if sample.id in seen:
                raise DatasetError(f"{path}:{number}: duplicate id '{sample.id}'")
            seen.add(sample.id)
            samples.append(sample)
    return samples
