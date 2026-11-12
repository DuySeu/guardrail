from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from core.registry import CommonParams, GuardDeps, build_guard, register_guard
from core.types import Action, GuardContext, GuardrailConfigError, GuardResult

DEPS = GuardDeps(base_dir=Path("."))


class EchoParams(CommonParams):
    word: str


@register_guard("test_echo")
class EchoGuard:
    Params = EchoParams
    requires: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, params: EchoParams, deps: GuardDeps) -> None:
        self.params = params
        self.name = params.name
        self.deps = deps

    async def check(self, text: str, context: GuardContext) -> GuardResult:
        return GuardResult(guard=self.name, action=Action.ALLOW, text=text, score=0.0)


def test_build_guard_raises_for_unknown_type():
    with pytest.raises(GuardrailConfigError, match="unknown guard type"):
        build_guard("nope", {}, DEPS)


def test_build_guard_raises_for_invalid_params():
    with pytest.raises(GuardrailConfigError, match="test_echo"):
        build_guard("test_echo", {"word": 3}, DEPS)


def test_build_guard_raises_for_unknown_param():
    with pytest.raises(GuardrailConfigError):
        build_guard("test_echo", {"word": "x", "typo": 1}, DEPS)


def test_registered_guard_is_built_with_params_and_deps():
    guard = build_guard("test_echo", {"word": "x", "name": "echo1"}, DEPS)
    assert guard.name == "echo1"
    assert guard.params.word == "x"
    assert guard.deps is DEPS


def test_name_defaults_to_type():
    guard = build_guard("test_echo", {"word": "x"}, DEPS)
    assert guard.name == "test_echo"
