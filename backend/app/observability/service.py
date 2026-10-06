import logging
import time
from functools import lru_cache
from typing import Any

from app.config.settings import Settings, get_settings
from app.llm.types import LLMRequest, LLMResponse
from app.observability.backends import ObservabilityBackend, create_observability_backend
from app.observability.context import get_current_research_trace, reset_current_research_trace, set_current_research_trace
from app.observability.types import AgentStepHandle, ResearchTraceHandle

logger = logging.getLogger(__name__)

_NODE_AGENT_MAP = {
    "load_history_context": "planner",
    "planner": "planner",
    "sub_questions": "planner",
    "source_search": "source_search",
    "evidence_retrieval": "retrieval",
    "hybrid_retrieval": "retrieval",
    "reranking": "reranking",
    "evidence_collection": "retrieval",
    "critic": "critic",
    "additional_research": "source_search",
    "synthesizer": "synthesis",
}


class ObservabilityService:
    def __init__(self, backend: ObservabilityBackend) -> None:
        self._backend = backend

    @property
    def enabled(self) -> bool:
        return self._backend.enabled

    def start_research_trace(self, *, question: str, research_id: str | None = None) -> ResearchTraceHandle:
        handle = self._backend.start_research_trace(
            question=question,
            metadata={"research_id": research_id, "user_question": question},
        )
        handle.research_id = research_id
        return handle

    def bind_research_trace(self, handle: ResearchTraceHandle):
        return set_current_research_trace(handle)

    def unbind_research_trace(self, token) -> None:
        reset_current_research_trace(token)

    def complete_research_trace(
        self,
        handle: ResearchTraceHandle,
        *,
        research_id: str | None,
        report_summary: str | None = None,
    ) -> None:
        handle.research_id = research_id or handle.research_id
        self._backend.end_research_trace(
            handle,
            research_id=research_id,
            output={"research_id": research_id, "executive_summary": report_summary},
            metadata={"research_id": research_id},
        )
        self._backend.flush()

    def fail_research_trace(self, handle: ResearchTraceHandle, *, error: str) -> None:
        self._backend.fail_research_trace(handle, error=error)
        self._backend.flush()

    def agent_name_for_node(self, node_name: str) -> str:
        return _NODE_AGENT_MAP.get(node_name, node_name)

    def run_agent_step(
        self,
        handle: ResearchTraceHandle | None,
        *,
        agent_name: str,
        input_payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "_AgentStepContext | _NullStepContext":
        if handle is None:
            handle = get_current_research_trace()
        if handle is None:
            return _NullStepContext()
        return _AgentStepContext(self, handle, agent_name, input_payload, metadata or {})

    def record_llm_call(
        self,
        *,
        agent_name: str,
        request: LLMRequest,
        response: LLMResponse | None,
        latency_ms: float,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
        step: AgentStepHandle | None = None,
    ) -> None:
        handle = get_current_research_trace()
        if handle is None:
            return
        meta = {**(metadata or {})}
        if response:
            meta["source"] = response.provider
        self._backend.record_llm_generation(
            handle,
            step,
            agent_name=agent_name,
            request=request,
            response=response,
            latency_ms=latency_ms,
            error=error,
            metadata=meta,
        )


class _NullStepContext:
    def set_output(self, payload: dict[str, Any]) -> None:
        return None

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _AgentStepContext:
    def __init__(
        self,
        service: ObservabilityService,
        handle: ResearchTraceHandle,
        agent_name: str,
        input_payload: dict[str, Any] | None,
        metadata: dict[str, Any],
    ) -> None:
        self._service = service
        self._handle = handle
        self._agent_name = agent_name
        self._input_payload = input_payload
        self._metadata = metadata
        self._step: AgentStepHandle | None = None
        self._started = time.perf_counter()
        self._output_payload: dict[str, Any] | None = None

    def set_output(self, payload: dict[str, Any]) -> None:
        self._output_payload = payload

    def __enter__(self) -> AgentStepHandle | None:
        self._step = self._service._backend.start_agent_step(
            self._handle,
            agent_name=self._agent_name,
            input_payload=self._input_payload,
            metadata=self._metadata,
        )
        return self._step

    def __exit__(self, exc_type, exc, tb):
        latency_ms = (time.perf_counter() - self._started) * 1000
        if self._step is None:
            return False
        if exc is not None:
            self._service._backend.fail_agent_step(self._handle, self._step, error=str(exc))
            return False
        output = self._output_payload or {"status": "ok"}
        merged_meta = {**self._metadata, **(output if isinstance(output, dict) else {})}
        self._service._backend.end_agent_step(
            self._handle,
            self._step,
            output_payload=output,
            metadata=merged_meta,
            latency_ms=latency_ms,
        )
        return False


def create_observability_service(settings: Settings | None = None) -> ObservabilityService:
    cfg = settings or get_settings()
    enabled = bool(
        cfg.langfuse_enabled and cfg.langfuse_public_key and cfg.langfuse_secret_key,
    )
    backend = create_observability_backend(
        enabled=enabled,
        public_key=cfg.langfuse_public_key,
        secret_key=cfg.langfuse_secret_key,
        host=cfg.langfuse_host,
    )
    return ObservabilityService(backend)


@lru_cache
def get_observability_service() -> ObservabilityService:
    return create_observability_service(get_settings())
