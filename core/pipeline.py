from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .providers import ChatClient, Embedder, OpenRouterClient, openrouter_from_env
from .registry import Guard, GuardDeps, build_guard, guard_class
from .types import Action, Decision, GuardContext, GuardResult, Stage, decide

NETWORK_TIMEOUT_S = 10.0

logger = logging.getLogger(__name__)


# Ordered input and output guard pipelines built from one config
class Guardrail:
    def __init__(
        self,
        guards: Mapping[Stage, Sequence[Guard]],
        owned_clients: Sequence[OpenRouterClient] = (),
    ) -> None:
        self._guards = {stage: tuple(guards.get(stage, ())) for stage in Stage}
        self._owned_clients = tuple(owned_clients)

    async def check_input(self, text: str) -> Decision:
        return await self._run(text, GuardContext(stage=Stage.INPUT))

    async def check_output(self, text: str, user_input: str | None = None) -> Decision:
        return await self._run(text, GuardContext(stage=Stage.OUTPUT, user_input=user_input))

    # Prepare lazy resources (example embeddings) before the first request
    async def warmup(self) -> None:
        for guards in self._guards.values():
            for guard in guards:
                if hasattr(guard, "warmup"):
                    await guard.warmup()

    def guards(self, stage: Stage) -> tuple[Guard, ...]:
        return self._guards[stage]

    # A guardrail holding only one guard, sharing the same instance
    def only(self, stage: Stage, name: str) -> Guardrail:
        guard = next((g for g in self._guards[stage] if g.name == name), None)
        if guard is None:
            raise KeyError(f"no guard '{name}' in '{stage}'")
        return Guardrail({stage: [guard]})

    # Close HTTP clients this guardrail created
    async def aclose(self) -> None:
        for client in self._owned_clients:
            await client.aclose()

    async def _run(self, text: str, context: GuardContext) -> Decision:
        results: list[GuardResult] = []
        current = text
        for guard in self._guards[context.stage]:
            result = await _run_guard(guard, current, context)
            results.append(result)
            current = result.text
            if result.action is Action.BLOCK:
                break
        return decide(current, results)


# Load a guardrail from YAML; clients not passed in default to OpenRouter from the environment
def load_guardrail(
    path: str | Path,
    chat_client: ChatClient | None = None,
    embedder: Embedder | None = None,
) -> Guardrail:
    path = Path(path)
    stages = load_config(path)

    # Create an OpenRouter client only for the client kinds a guard needs and the caller omitted
    needed = set().union(*(guard_class(e["type"]).requires for es in stages.values() for e in es))
    missing = (needed & {"chat"} and chat_client is None) or (
        needed & {"embedder"} and embedder is None
    )
    owned = [openrouter_from_env(os.environ)] if missing else []
    if owned:
        chat_client = chat_client or owned[0]
        embedder = embedder or owned[0]

    deps = GuardDeps(base_dir=path.parent, chat_client=chat_client, embedder=embedder)
    guards = {
        stage: [build_guard(e["type"], e, deps) for e in entries] for stage, entries in stages.items()
    }
    return Guardrail(guards, owned)


# Run one guard with its timeout, turning any failure into a result per on_error
async def _run_guard(guard: Guard, text: str, context: GuardContext) -> GuardResult:
    timeout = guard.params.timeout_s or (NETWORK_TIMEOUT_S if guard.requires else None)
    start = time.perf_counter()
    try:
        result = await asyncio.wait_for(guard.check(text, context), timeout)
    # Any guard failure must fail closed or open per config, so catch everything but cancellation
    except Exception as exc:  # noqa: BLE001
        logger.warning("guard %s failed: %s", guard.name, type(exc).__name__)
        action = Action.BLOCK if guard.params.on_error == "block" else Action.ALLOW
        result = GuardResult(
            guard=guard.name,
            action=action,
            text=text,
            score=None,
            error=f"{type(exc).__name__}: {exc}",
        )
    return replace(result, latency_ms=(time.perf_counter() - start) * 1000)
