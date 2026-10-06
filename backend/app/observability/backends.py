import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any

from app.llm.types import LLMRequest, LLMResponse, LLMUsage
from app.observability.types import AgentStepHandle, ResearchTraceHandle

logger = logging.getLogger(__name__)


class ObservabilityBackend(ABC):
    @property
    @abstractmethod
    def enabled(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def start_research_trace(self, *, question: str, metadata: dict[str, Any]) -> ResearchTraceHandle:
        raise NotImplementedError

    @abstractmethod
    def end_research_trace(
        self,
        handle: ResearchTraceHandle,
        *,
        research_id: str | None,
        output: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def fail_research_trace(self, handle: ResearchTraceHandle, *, error: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def start_agent_step(
        self,
        handle: ResearchTraceHandle,
        *,
        agent_name: str,
        input_payload: dict[str, Any] | None,
        metadata: dict[str, Any],
    ) -> AgentStepHandle:
        raise NotImplementedError

    @abstractmethod
    def end_agent_step(
        self,
        handle: ResearchTraceHandle,
        step: AgentStepHandle,
        *,
        output_payload: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
        latency_ms: float | None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def fail_agent_step(self, handle: ResearchTraceHandle, step: AgentStepHandle, *, error: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def record_llm_generation(
        self,
        handle: ResearchTraceHandle,
        step: AgentStepHandle | None,
        *,
        agent_name: str,
        request: LLMRequest,
        response: LLMResponse | None,
        latency_ms: float,
        error: str | None,
        metadata: dict[str, Any],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def flush(self) -> None:
        raise NotImplementedError


class NoOpObservabilityBackend(ObservabilityBackend):
    @property
    def enabled(self) -> bool:
        return False

    def start_research_trace(self, *, question: str, metadata: dict[str, Any]) -> ResearchTraceHandle:
        return ResearchTraceHandle(trace_id=None, user_question=question, research_id=metadata.get("research_id"))

    def end_research_trace(
        self,
        handle: ResearchTraceHandle,
        *,
        research_id: str | None,
        output: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        return None

    def fail_research_trace(self, handle: ResearchTraceHandle, *, error: str) -> None:
        return None

    def start_agent_step(
        self,
        handle: ResearchTraceHandle,
        *,
        agent_name: str,
        input_payload: dict[str, Any] | None,
        metadata: dict[str, Any],
    ) -> AgentStepHandle:
        return AgentStepHandle(agent_name=agent_name, metadata=metadata)

    def end_agent_step(
        self,
        handle: ResearchTraceHandle,
        step: AgentStepHandle,
        *,
        output_payload: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
        latency_ms: float | None,
    ) -> None:
        return None

    def fail_agent_step(self, handle: ResearchTraceHandle, step: AgentStepHandle, *, error: str) -> None:
        return None

    def record_llm_generation(
        self,
        handle: ResearchTraceHandle,
        step: AgentStepHandle | None,
        *,
        agent_name: str,
        request: LLMRequest,
        response: LLMResponse | None,
        latency_ms: float,
        error: str | None,
        metadata: dict[str, Any],
    ) -> None:
        return None

    def flush(self) -> None:
        return None


class LangfuseObservabilityBackend(ObservabilityBackend):
    def __init__(self, public_key: str, secret_key: str, host: str | None) -> None:
        from langfuse import Langfuse

        self._client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        self._active_steps: dict[str, AgentStepHandle] = {}

    @property
    def enabled(self) -> bool:
        return True

    def _trace_key(self, handle: ResearchTraceHandle) -> str:
        return handle.trace_id or "unknown"

    def start_research_trace(self, *, question: str, metadata: dict[str, Any]) -> ResearchTraceHandle:
        trace_id = str(uuid.uuid4())
        trace = self._client.trace(
            id=trace_id,
            name="research_workflow",
            input={"question": question},
            metadata=metadata,
        )
        return ResearchTraceHandle(
            trace_id=trace_id,
            user_question=question,
            research_id=metadata.get("research_id"),
            _root=trace,
        )

    def end_research_trace(
        self,
        handle: ResearchTraceHandle,
        *,
        research_id: str | None,
        output: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        if handle._root is None:
            return
        merged = {**(metadata or {})}
        if research_id:
            merged["research_id"] = research_id
        handle._root.update(output=output, metadata=merged)

    def fail_research_trace(self, handle: ResearchTraceHandle, *, error: str) -> None:
        if handle._root is None:
            return
        handle._root.update(level="ERROR", status_message=error, metadata={"error": error})

    def start_agent_step(
        self,
        handle: ResearchTraceHandle,
        *,
        agent_name: str,
        input_payload: dict[str, Any] | None,
        metadata: dict[str, Any],
    ) -> AgentStepHandle:
        if handle._root is None:
            return AgentStepHandle(agent_name=agent_name, metadata=metadata)
        span = handle._root.span(
            name=agent_name,
            input=input_payload,
            metadata={**metadata, "agent_name": agent_name},
        )
        step = AgentStepHandle(agent_name=agent_name, span=span, metadata=metadata)
        self._active_steps[f"{self._trace_key(handle)}:{agent_name}:{time.time_ns()}"] = step
        return step

    def end_agent_step(
        self,
        handle: ResearchTraceHandle,
        step: AgentStepHandle,
        *,
        output_payload: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
        latency_ms: float | None,
    ) -> None:
        if step.span is None:
            return
        meta = {**(metadata or {}), **step.metadata}
        if latency_ms is not None:
            meta["latency_ms"] = round(latency_ms, 3)
        step.span.end(output=output_payload, metadata=meta)

    def fail_agent_step(self, handle: ResearchTraceHandle, step: AgentStepHandle, *, error: str) -> None:
        if step.span is None:
            return
        step.span.end(level="ERROR", status_message=error, metadata={**step.metadata, "error": error})

    @staticmethod
    def _usage_payload(usage: LLMUsage | None) -> dict[str, Any] | None:
        if usage is None:
            return None
        payload: dict[str, Any] = {"unit": "TOKENS"}
        if usage.prompt_tokens is not None:
            payload["input"] = usage.prompt_tokens
        if usage.completion_tokens is not None:
            payload["output"] = usage.completion_tokens
        if usage.total_tokens is not None:
            payload["total"] = usage.total_tokens
        return payload

    def record_llm_generation(
        self,
        handle: ResearchTraceHandle,
        step: AgentStepHandle | None,
        *,
        agent_name: str,
        request: LLMRequest,
        response: LLMResponse | None,
        latency_ms: float,
        error: str | None,
        metadata: dict[str, Any],
    ) -> None:
        parent = step.span if step and step.span is not None else handle._root
        if parent is None:
            return
        generation = parent.generation(
            name=f"llm:{agent_name}",
            model=(response.model if response else request.model) or metadata.get("model"),
            input=[message.model_dump() for message in request.messages],
            metadata={
                **metadata,
                "agent_name": agent_name,
                "research_id": handle.research_id,
                "latency_ms": round(latency_ms, 3),
                "provider": response.provider if response else metadata.get("provider"),
            },
        )
        if error:
            generation.end(level="ERROR", status_message=error)
            return
        generation.end(
            output=response.content if response else None,
            usage=self._usage_payload(response.usage if response else None),
        )

    def flush(self) -> None:
        self._client.flush()


def create_observability_backend(
    *,
    enabled: bool,
    public_key: str | None,
    secret_key: str | None,
    host: str | None,
) -> ObservabilityBackend:
    if not enabled or not public_key or not secret_key:
        return NoOpObservabilityBackend()
    try:
        return LangfuseObservabilityBackend(public_key=public_key, secret_key=secret_key, host=host)
    except Exception as exc:
        logger.warning("Langfuse initialization failed; observability disabled: %s", exc)
        return NoOpObservabilityBackend()
