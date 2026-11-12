from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import httpx

from .constants import (
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_BASE_URL,
    OPENROUTER_BASE_URL_ENV,
)
from .types import GuardrailConfigError

HTTP_TIMEOUT_S = 30.0


class ChatClient(Protocol):
    async def complete(
        self, model: str, messages: Sequence[dict[str, str]], temperature: float = 0.0
    ) -> str: ...


class Embedder(Protocol):
    async def embed(self, model: str, texts: Sequence[str]) -> list[list[float]]: ...


class ProviderError(Exception):
    pass


# OpenRouter chat and embedding client over one HTTP connection pool
class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = OPENROUTER_BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=HTTP_TIMEOUT_S,
            transport=transport,
        )

    # POST /chat/completions - return the first choice's text
    async def complete(
        self, model: str, messages: Sequence[dict[str, str]], temperature: float = 0.0
    ) -> str:
        body = await self._post(
            "/chat/completions",
            {"model": model, "messages": list(messages), "temperature": temperature},
        )
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                "chat response has no choices[0].message.content"
            ) from exc
        if not isinstance(content, str):
            raise ProviderError("chat response content is not text")
        return content

    # POST /embeddings - return vectors in input order
    # https://openrouter.ai/docs/api/api-reference/embeddings/submit-an-embedding-request.md
    async def embed(self, model: str, texts: Sequence[str]) -> list[list[float]]:
        body = await self._post("/embeddings", {"model": model, "input": list(texts)})
        try:
            items = sorted(body["data"], key=lambda item: item["index"])
            vectors = [item["embedding"] for item in items]
        except (KeyError, TypeError) as exc:
            raise ProviderError("embedding response has no data[].embedding") from exc
        if len(vectors) != len(texts):
            raise ProviderError(f"expected {len(texts)} embeddings, got {len(vectors)}")
        return vectors

    async def aclose(self) -> None:
        await self.http.aclose()

    async def _post(self, path: str, payload: dict[str, Any]) -> Any:
        try:
            response = await self.http.post(path, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"{path} returned HTTP {exc.response.status_code}"
            ) from exc
        except ValueError as exc:
            raise ProviderError(f"{path} returned invalid JSON") from exc


# Build the OpenRouter client from environment variables
def openrouter_from_env(environ: Mapping[str, str]) -> OpenRouterClient:
    api_key = environ.get(OPENROUTER_API_KEY_ENV)
    if not api_key:
        raise GuardrailConfigError(f"{OPENROUTER_API_KEY_ENV} is not set")
    return OpenRouterClient(
        api_key, environ.get(OPENROUTER_BASE_URL_ENV) or OPENROUTER_BASE_URL
    )
