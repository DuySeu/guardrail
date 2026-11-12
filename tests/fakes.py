from __future__ import annotations

from collections.abc import Mapping, Sequence


# Chat client that returns canned responses in order and records what it was sent
class FakeChatClient:
    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    async def complete(
        self, model: str, messages: Sequence[dict[str, str]], temperature: float = 0.0
    ) -> str:
        self.calls.append(list(messages))
        return self.responses.pop(0)


# Embedder that looks vectors up by text and counts calls
class FakeEmbedder:
    def __init__(self, vectors: Mapping[str, list[float]]) -> None:
        self.vectors = dict(vectors)
        self.calls = 0

    async def embed(self, model: str, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [self.vectors[t] for t in texts]
