from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TypeVar

from pydantic import BaseModel

from app.llm.types import LLMRequest, LLMResponse, LLMStreamChunk

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Provider-agnostic LLM interface for agents and services."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def default_model(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    def generate_structured(self, request: LLMRequest, response_model: type[T]) -> T:
        raise NotImplementedError

    def stream_generate(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        """Optional streaming; providers that do not support streaming should override."""
        response = self.generate(request)
        yield LLMStreamChunk(content_delta=response.content, done=True)
