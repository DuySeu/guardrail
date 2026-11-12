from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from core.providers import OpenRouterClient, ProviderError


def _client(handler):
    return OpenRouterClient("test-key", "https://example.test/api/v1", httpx.MockTransport(handler))


def _capture(response_body, status=200):
    seen = {}

    def handler(request):
        seen["request"] = request
        return httpx.Response(status, json=response_body)

    return seen, handler


def test_chat_sends_model_messages_and_temperature():
    seen, handler = _capture({"choices": [{"message": {"content": "hi"}}]})
    asyncio.run(_client(handler).complete("m", [{"role": "user", "content": "x"}]))
    request = seen["request"]
    assert request.url.path == "/api/v1/chat/completions"
    assert json.loads(request.content) == {
        "model": "m",
        "messages": [{"role": "user", "content": "x"}],
        "temperature": 0.0,
    }


def test_chat_returns_first_choice_content():
    _, handler = _capture({"choices": [{"message": {"content": "hi"}}]})
    assert asyncio.run(_client(handler).complete("m", [])) == "hi"


def test_api_key_is_sent_as_bearer_header():
    seen, handler = _capture({"choices": [{"message": {"content": "hi"}}]})
    asyncio.run(_client(handler).complete("m", []))
    assert seen["request"].headers["Authorization"] == "Bearer test-key"


def test_embed_returns_vectors_in_input_order():
    body = {"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]}
    seen, handler = _capture(body)
    vectors = asyncio.run(_client(handler).embed("e", ["a", "b"]))
    assert vectors == [[1.0], [2.0]]
    assert seen["request"].url.path == "/api/v1/embeddings"
    assert json.loads(seen["request"].content) == {"model": "e", "input": ["a", "b"]}


def test_http_error_raises_provider_error():
    _, handler = _capture({"error": "rate limited"}, status=429)
    with pytest.raises(ProviderError, match="429"):
        asyncio.run(_client(handler).complete("m", []))


def test_malformed_response_raises_provider_error():
    _, handler = _capture({"choices": []})
    with pytest.raises(ProviderError):
        asyncio.run(_client(handler).complete("m", []))


def test_null_content_raises_provider_error():
    _, handler = _capture({"choices": [{"message": {"content": None}}]})
    with pytest.raises(ProviderError):
        asyncio.run(_client(handler).complete("m", []))


def test_embedding_count_mismatch_raises_provider_error():
    _, handler = _capture({"data": [{"index": 0, "embedding": [1.0]}]})
    with pytest.raises(ProviderError, match="expected 2"):
        asyncio.run(_client(handler).embed("e", ["a", "b"]))
