from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence

from core import Decision, Guardrail, Stage

from eval.dataset import Sample
from eval.metrics import Record


# Run every sample through the full pipeline of its stage
async def run_pipeline(
    guardrail: Guardrail, samples: Sequence[Sample], concurrency: int
) -> list[Record]:
    await guardrail.warmup()
    limit = asyncio.Semaphore(concurrency)

    async def one(sample: Sample) -> Record:
        decision = await _limited(limit, lambda: _check(guardrail, sample))
        return Record.from_decision(sample, decision)

    return list(await asyncio.gather(*(one(s) for s in samples)))


# Run each guard alone over every sample of its stage, keyed by "<stage>/<name>"
async def run_isolate(
    guardrail: Guardrail, samples: Sequence[Sample], concurrency: int
) -> dict[str, list[Record]]:
    await guardrail.warmup()
    limit = asyncio.Semaphore(concurrency)
    runs: dict[str, list[Record]] = {}
    for stage in Stage:
        stage_samples = [s for s in samples if s.stage is stage]
        for guard in guardrail.guards(stage):
            single = guardrail.only(stage, guard.name)

            async def one(sample: Sample, single: Guardrail = single) -> Record:
                decision = await _limited(limit, lambda: _check(single, sample))
                return Record.from_result(sample, decision.results[0])

            runs[f"{stage}/{guard.name}"] = list(
                await asyncio.gather(*(one(s) for s in stage_samples))
            )
    return runs


def _check(guardrail: Guardrail, sample: Sample) -> Awaitable[Decision]:
    if sample.stage is Stage.INPUT:
        return guardrail.check_input(sample.text)
    return guardrail.check_output(sample.text, user_input=sample.user_input)


async def _limited(limit: asyncio.Semaphore, call: Callable[[], Awaitable[Decision]]) -> Decision:
    async with limit:
        return await call()
