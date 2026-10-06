from collections.abc import Iterator
from typing import TypeVar, cast

from pydantic import BaseModel

from app.llm.provider import LLMProvider
from app.llm.retry import execute_with_retry
from app.llm.types import LLMRequest, LLMResponse, LLMStreamChunk

T = TypeVar("T", bound=BaseModel)


class ResilientLLMProvider(LLMProvider):
    """Retry and timeout wrapper that keeps agents decoupled from transport details."""

    def __init__(
        self,
        inner: LLMProvider,
        *,
        max_retries: int,
        base_backoff_seconds: float,
    ) -> None:
        self._inner = inner
        self._max_retries = max_retries
        self._base_backoff_seconds = base_backoff_seconds

    @property
    def provider_name(self) -> str:
        return self._inner.provider_name

    @property
    def default_model(self) -> str:
        return self._inner.default_model

    def generate(self, request: LLMRequest) -> LLMResponse:
        return cast(
            LLMResponse,
            execute_with_retry(
                lambda: self._inner.generate(request),
                max_retries=self._max_retries,
                base_backoff_seconds=self._base_backoff_seconds,
            ),
        )

    def generate_structured(self, request: LLMRequest, response_model: type[T]) -> T:
        return cast(
            T,
            execute_with_retry(
                lambda: self._inner.generate_structured(request, response_model),
                max_retries=self._max_retries,
                base_backoff_seconds=self._base_backoff_seconds,
            ),
        )

    def stream_generate(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        # Streaming is not retried as a whole; provider-level transient errors still surface.
        return self._inner.stream_generate(request)
