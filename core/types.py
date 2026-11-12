from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum


class Stage(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


class Action(StrEnum):
    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"


@dataclass(frozen=True)
class GuardContext:
    stage: Stage
    user_input: str | None = None


@dataclass(frozen=True)
class GuardResult:
    guard: str
    action: Action
    text: str
    score: float | None
    category: str | None = None
    reason: str | None = None
    latency_ms: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class Decision:
    action: Action
    text: str
    results: tuple[GuardResult, ...]
    blocked_by: str | None


class GuardrailConfigError(Exception):
    pass


# Merge guard results into one decision: any block wins, then any redact, else allow
def decide(text: str, results: Sequence[GuardResult]) -> Decision:
    blocked = next((r for r in results if r.action is Action.BLOCK), None)
    if blocked:
        action = Action.BLOCK
    elif any(r.action is Action.REDACT for r in results):
        action = Action.REDACT
    else:
        action = Action.ALLOW
    return Decision(
        action=action,
        text=text,
        results=tuple(results),
        blocked_by=blocked.guard if blocked else None,
    )
