import time
from typing import TypeVar

from pydantic import BaseModel

from app.llm.provider import LLMProvider
from app.llm.types import LLMRequest, LLMResponse, LLMStreamChunk
from app.observability.context import get_current_research_trace
from app.observability.service import get_observability_service

T = TypeVar("T", bound=BaseModel)


class ObservingLLMProvider(LLMProvider):
    """Wraps an LLM provider with Langfuse generation tracing when enabled."""

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner

    @property
    def provider_name(self) -> str:
        return self._inner.provider_name

    @property
    def default_model(self) -> str:
        return self._inner.default_model

    def generate(self, request: LLMRequest) -> LLMResponse:
        obs = get_observability_service()
        agent_name = self._resolve_agent_name(request)
        started = time.perf_counter()
        try:
            response = self._inner.generate(request)
        except Exception as exc:
            obs.record_llm_call(
                agent_name=agent_name,
                request=request,
                response=None,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
                metadata=self._llm_metadata(request, provider=self.provider_name),
            )
            raise
        obs.record_llm_call(
            agent_name=agent_name,
            request=request,
            response=response,
            latency_ms=(time.perf_counter() - started) * 1000,
            metadata=self._llm_metadata(request, provider=response.provider, model=response.model),
        )
        return response

    def generate_structured(self, request: LLMRequest, response_model: type[T]) -> T:
        obs = get_observability_service()
        agent_name = self._resolve_agent_name(request)
        started = time.perf_counter()
        try:
            result = self._inner.generate_structured(request, response_model)
        except Exception as exc:
            obs.record_llm_call(
                agent_name=agent_name,
                request=request,
                response=None,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
                metadata={**self._llm_metadata(request), "structured": response_model.__name__},
            )
            raise
        obs.record_llm_call(
            agent_name=agent_name,
            request=request,
            response=LLMResponse(
                content=result.model_dump_json(),
                model=request.model or self.default_model,
                provider=self.provider_name,
            ),
            latency_ms=(time.perf_counter() - started) * 1000,
            metadata={**self._llm_metadata(request), "structured": response_model.__name__},
        )
        return result

    def _resolve_agent_name(self, request: LLMRequest) -> str:
        if request.observability and request.observability.agent_name:
            return request.observability.agent_name
        return f"llm:{self.provider_name}"

    def _llm_metadata(self, request: LLMRequest, *, provider: str | None = None, model: str | None = None) -> dict:
        meta: dict = {"provider": provider or self.provider_name, "model": model or request.model or self.default_model}
        if request.observability:
            if request.observability.research_id:
                meta["research_id"] = request.observability.research_id
            if request.observability.source:
                meta["source"] = request.observability.source
            if request.observability.retrieval_method:
                meta["retrieval_method"] = request.observability.retrieval_method
        trace = get_current_research_trace()
        if trace and trace.research_id and "research_id" not in meta:
            meta["research_id"] = trace.research_id
        return meta

    def stream_generate(self, request: LLMRequest):
        chunks: list[str] = []
        started = time.perf_counter()
        agent_name = self._resolve_agent_name(request)
        obs = get_observability_service()
        try:
            for chunk in self._inner.stream_generate(request):
                chunks.append(chunk.content_delta)
                yield chunk
        except Exception as exc:
            obs.record_llm_call(
                agent_name=agent_name,
                request=request,
                response=None,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
                metadata={"provider": self.provider_name, "streaming": True},
            )
            raise
        obs.record_llm_call(
            agent_name=agent_name,
            request=request,
            response=LLMResponse(
                content="".join(chunks),
                model=request.model or self.default_model,
                provider=self.provider_name,
            ),
            latency_ms=(time.perf_counter() - started) * 1000,
            metadata={"provider": self.provider_name, "streaming": True},
        )
