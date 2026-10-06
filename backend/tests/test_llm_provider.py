import json
from collections.abc import Iterator
from typing import TypeVar

import httpx
import pytest

_HTTPX_CLIENT = httpx.Client
from pydantic import BaseModel

from app.config.settings import Settings
from app.llm.exceptions import LLMRateLimitError
from app.llm.factory import create_llm_provider, llm_configuration_status
from app.llm.provider import LLMProvider
from app.llm.providers.openai_compatible import OpenAICompatibleProvider
from app.llm.observing_provider import ObservingLLMProvider
from app.llm.resilient import ResilientLLMProvider
from app.llm.retry import execute_with_retry
from app.llm.types import LLMMessage, LLMRequest, LLMResponse, LLMStreamChunk

T = TypeVar("T", bound=BaseModel)


class _MockProvider(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def default_model(self) -> str:
        return "mock-model"

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            content=f"echo:{request.messages[-1].content}",
            model=self.default_model,
            provider=self.provider_name,
        )

    def generate_structured(self, request: LLMRequest, response_model: type[T]) -> T:
        self.calls += 1
        payload = {"answer": request.messages[-1].content, "confidence": 0.9}
        return response_model.model_validate(payload)

    def stream_generate(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        text = f"stream:{request.messages[-1].content}"
        yield LLMStreamChunk(content_delta=text[:3], done=False)
        yield LLMStreamChunk(content_delta=text[3:], done=True)


class StructuredAnswer(BaseModel):
    answer: str
    confidence: float


def test_mock_provider_generate_structured_and_stream():
    provider = _MockProvider()
    response = provider.generate(
        LLMRequest(messages=[LLMMessage(role="user", content="hello")]),
    )
    assert response.content == "echo:hello"

    structured = provider.generate_structured(
        LLMRequest(messages=[LLMMessage(role="user", content="hybrid retrieval")]),
        StructuredAnswer,
    )
    assert structured.answer == "hybrid retrieval"
    assert structured.confidence == 0.9

    chunks = list(
        provider.stream_generate(
            LLMRequest(messages=[LLMMessage(role="user", content="abc")]),
        )
    )
    assert "".join(chunk.content_delta for chunk in chunks) == "stream:abc"
    assert chunks[-1].done is True


def test_resilient_provider_retries_rate_limits():
    attempts = {"count": 0}

    class _FlakyProvider(_MockProvider):
        def generate(self, request: LLMRequest) -> LLMResponse:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise LLMRateLimitError("rate limited", retry_after_seconds=0)
            return super().generate(request)

    resilient = ResilientLLMProvider(
        _FlakyProvider(),
        max_retries=3,
        base_backoff_seconds=0,
    )
    response = resilient.generate(LLMRequest(messages=[LLMMessage(role="user", content="ok")]))
    assert response.content == "echo:ok"
    assert attempts["count"] == 3


def test_execute_with_retry_stops_after_max_retries():
    calls = {"count": 0}

    def _fail() -> None:
        calls["count"] += 1
        raise LLMRateLimitError("rate limited")

    with pytest.raises(LLMRateLimitError):
        execute_with_retry(_fail, max_retries=2, base_backoff_seconds=0)
    assert calls["count"] == 3


def test_omnirouter_openai_compatible_generate_with_mock_transport(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = {
            "choices": [{"message": {"content": "provider response"}}],
            "model": "google/gemini-2.0-flash-001",
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return _HTTPX_CLIENT(*args, **kwargs)

    monkeypatch.setattr("app.llm.providers.openai_compatible.httpx.Client", _client)
    provider = OpenAICompatibleProvider(
        provider_name="omnirouter",
        base_url="https://example-omnirouter.test/v1",
        api_key="test-key",
        default_model="google/gemini-2.0-flash-001",
        timeout_seconds=5,
    )
    response = provider.generate(
        LLMRequest(messages=[LLMMessage(role="user", content="ping")]),
    )
    assert response.content == "provider response"
    assert response.provider == "omnirouter"


def test_factory_configuration_status_without_credentials(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "omnirouter")
    monkeypatch.delenv("OMNIROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OMNIROUTER_BASE_URL", raising=False)

    settings = Settings()
    status = llm_configuration_status(settings)
    assert status["provider"] == "omnirouter"
    assert status["ready"] is False
    assert "OMNIROUTER_API_KEY" in str(status["detail"])


def test_factory_builds_openrouter_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-from-env")
    monkeypatch.setenv("OPENROUTER_MODEL", "openrouter/free")

    settings = Settings()
    provider = create_llm_provider(settings)
    assert isinstance(provider, ObservingLLMProvider)
    assert isinstance(provider._inner, ResilientLLMProvider)
    assert provider.provider_name == "openrouter"
    assert provider.default_model == "openrouter/free"


def test_factory_builds_omnirouter_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "omnirouter")
    monkeypatch.setenv("OMNIROUTER_API_KEY", "secret-from-env")
    monkeypatch.setenv("OMNIROUTER_BASE_URL", "https://example-omnirouter.test/v1")
    monkeypatch.setenv("OMNIROUTER_MODEL", "google/gemini-2.0-flash-001")

    settings = Settings()
    provider = create_llm_provider(settings)
    assert isinstance(provider, ObservingLLMProvider)
    assert isinstance(provider._inner, ResilientLLMProvider)
    assert provider.provider_name == "omnirouter"
    assert provider.default_model == "google/gemini-2.0-flash-001"


def test_openai_compatible_streaming_parses_sse(monkeypatch):
    events = [
        'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        "data: [DONE]",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert body["stream"] is True
        return httpx.Response(200, text="\n".join(events))

    transport = httpx.MockTransport(handler)

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return _HTTPX_CLIENT(*args, **kwargs)

    monkeypatch.setattr("app.llm.providers.openai_compatible.httpx.Client", _client)
    provider = OpenAICompatibleProvider(
        provider_name="omnirouter",
        base_url="https://example-omnirouter.test/v1",
        api_key="test-key",
        default_model="model",
        timeout_seconds=5,
    )
    chunks = list(
        provider.stream_generate(
            LLMRequest(messages=[LLMMessage(role="user", content="hi")]),
        )
    )
    assert "".join(chunk.content_delta for chunk in chunks if not chunk.done) == "Hello"
    assert chunks[-1].done is True
