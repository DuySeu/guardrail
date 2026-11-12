from __future__ import annotations

import asyncio

import pytest

import core  # noqa: F401  registers the built-in guards
from core.registry import GuardDeps, build_guard
from core.types import Action, GuardContext, GuardrailConfigError, Stage
from tests.fakes import FakeEmbedder

CTX = GuardContext(stage=Stage.INPUT)
VECTORS = {
    "ignore instructions": [1.0, 0.0],
    "giờ mở cửa": [0.0, 1.0],
    "attack": [0.9, 0.1],
    "hours": [0.1, 0.9],
    "opposite": [-1.0, 0.0],
}


def _guard(tmp_path, mode, threshold, examples="ignore instructions\n", embedder=None):
    (tmp_path / "ex.txt").write_text(examples, encoding="utf-8")
    params = {"mode": mode, "examples": "ex.txt", "model": "m", "threshold": threshold}
    deps = GuardDeps(base_dir=tmp_path, embedder=embedder or FakeEmbedder(VECTORS))
    return build_guard("embedding", params, deps)


def test_deny_blocks_when_similarity_reaches_threshold(tmp_path):
    result = asyncio.run(_guard(tmp_path, "deny", 1.0).check("ignore instructions", CTX))
    assert result.action is Action.BLOCK
    assert result.score == pytest.approx(1.0)


def test_deny_allows_below_threshold(tmp_path):
    result = asyncio.run(_guard(tmp_path, "deny", 0.995).check("attack", CTX))
    assert result.action is Action.ALLOW


def test_allow_mode_blocks_off_topic_text(tmp_path):
    guard = _guard(tmp_path, "allow", 0.5, examples="giờ mở cửa\n")
    result = asyncio.run(guard.check("attack", CTX))
    assert result.action is Action.BLOCK


def test_allow_mode_allows_on_topic_text(tmp_path):
    guard = _guard(tmp_path, "allow", 0.5, examples="giờ mở cửa\n")
    result = asyncio.run(guard.check("hours", CTX))
    assert result.action is Action.ALLOW


def test_negative_cosine_is_clamped_to_zero(tmp_path):
    result = asyncio.run(_guard(tmp_path, "deny", 0.5).check("opposite", CTX))
    assert result.score == 0.0


def test_score_is_returned_when_allowed(tmp_path):
    result = asyncio.run(_guard(tmp_path, "deny", 0.995).check("attack", CTX))
    assert 0.9 < result.score < 1.0


def test_examples_are_embedded_once_across_concurrent_checks(tmp_path):
    embedder = FakeEmbedder(VECTORS)
    guard = _guard(tmp_path, "deny", 0.9, embedder=embedder)

    async def run():
        await asyncio.gather(*(guard.check("attack", CTX) for _ in range(5)))

    asyncio.run(run())
    # one call for the examples, one per checked text
    assert embedder.calls == 6


def test_empty_examples_file_raises_config_error(tmp_path):
    with pytest.raises(GuardrailConfigError, match="empty"):
        _guard(tmp_path, "deny", 0.5, examples="\n\n")


def test_threshold_is_required(tmp_path):
    (tmp_path / "ex.txt").write_text("x\n", encoding="utf-8")
    with pytest.raises(GuardrailConfigError):
        build_guard(
            "embedding",
            {"mode": "deny", "examples": "ex.txt", "model": "m"},
            GuardDeps(base_dir=tmp_path, embedder=FakeEmbedder({})),
        )
