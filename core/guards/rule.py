from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from typing import ClassVar, Literal

from pydantic import Field, model_validator

from ..registry import CommonParams, GuardDeps, register_guard
from ..types import Action, GuardContext, GuardrailConfigError, GuardResult

PRESETS: Mapping[str, str] = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "phone_vn": r"(?<!\d)(?:\+84|0)(?:[\s.-]?\d){9}(?!\d)",
    "credit_card": r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)",
    "api_key": r"\b(?:sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36})\b",
}


class MaxLengthParams(CommonParams):
    limit: int = Field(gt=0)


class RegexParams(CommonParams):
    presets: list[str] = []
    patterns: dict[str, str] = {}
    action: Literal["block", "redact"] = "block"
    replacement: str | None = None

    @model_validator(mode="after")
    def _check_patterns(self) -> RegexParams:
        unknown = set(self.presets) - PRESETS.keys()
        if unknown:
            raise ValueError(f"unknown presets: {sorted(unknown)}")
        if not self.presets and not self.patterns:
            raise ValueError("needs at least one of presets or patterns")
        return self


class KeywordParams(CommonParams):
    words: list[str] = []
    file: str | None = None
    case_sensitive: bool = False
    action: Literal["block", "redact"] = "block"
    replacement: str | None = None

    @model_validator(mode="after")
    def _check_source(self) -> KeywordParams:
        if not self.words and not self.file:
            raise ValueError("needs words or file")
        return self


@register_guard("max_length")
class MaxLengthGuard:
    Params = MaxLengthParams
    requires: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, params: MaxLengthParams, deps: GuardDeps) -> None:
        self.params = params
        self.name = params.name

    # Block text longer than the limit
    async def check(self, text: str, context: GuardContext) -> GuardResult:
        too_long = len(text) > self.params.limit
        return GuardResult(
            guard=self.name,
            action=Action.BLOCK if too_long else Action.ALLOW,
            text=text,
            score=1.0 if too_long else 0.0,
            category=self.params.category or self.name,
            reason=f"{len(text)} > {self.params.limit} chars" if too_long else None,
        )


# Shared match-then-block-or-redact logic for regex and keyword guards
class _PatternGuard:
    requires: ClassVar[frozenset[str]] = frozenset()
    params: RegexParams | KeywordParams
    name: str
    compiled: list[tuple[str, re.Pattern[str]]]

    async def check(self, text: str, context: GuardContext) -> GuardResult:
        # Find which named patterns match
        hits = [label for label, pattern in self.compiled if pattern.search(text)]
        result = GuardResult(
            guard=self.name,
            action=Action.ALLOW,
            text=text,
            score=1.0 if hits else 0.0,
            category=self.params.category or self.name,
        )
        if not hits:
            return result

        # Block keeps the text, redact replaces every match
        if self.params.action == "block":
            return replace(result, action=Action.BLOCK, reason=f"matched {hits}")
        redacted = text
        for label, pattern in self.compiled:
            replacement = self.params.replacement or f"[{label.upper()}]"
            redacted = pattern.sub(lambda _, r=replacement: r, redacted)
        return replace(result, action=Action.REDACT, text=redacted, reason=f"redacted {hits}")


@register_guard("regex")
class RegexGuard(_PatternGuard):
    Params = RegexParams

    def __init__(self, params: RegexParams, deps: GuardDeps) -> None:
        self.params = params
        self.name = params.name
        sources = {**{p: PRESETS[p] for p in params.presets}, **params.patterns}
        try:
            self.compiled = [(label, re.compile(src)) for label, src in sources.items()]
        except re.error as exc:
            raise GuardrailConfigError(f"guard '{self.name}': invalid regex: {exc}") from exc


@register_guard("keyword")
class KeywordGuard(_PatternGuard):
    Params = KeywordParams

    def __init__(self, params: KeywordParams, deps: GuardDeps) -> None:
        self.params = params
        self.name = params.name
        words = [*params.words, *(deps.read_lines(params.file) if params.file else [])]
        if not words:
            raise GuardrailConfigError(f"guard '{self.name}': keyword list is empty")

        # One alternation bounded by Unicode word boundaries, longest first
        alternation = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
        flags = 0 if params.case_sensitive else re.IGNORECASE
        self.compiled = [(self.name, re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", flags))]

