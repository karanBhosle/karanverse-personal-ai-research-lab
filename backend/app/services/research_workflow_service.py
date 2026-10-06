import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from app.agents.research_workflow.deps import ResearchWorkflowDeps
from app.agents.research_workflow.graph import compile_research_workflow
from app.config.settings import Settings, get_settings
from app.llm.factory import create_llm_provider
from app.models.research_history import ResearchHistoryContextSnippet, ResearchWorkflowResponse
from app.models.research_report import ResearchReport
from app.models.research_state import NodeTraceEntry, ResearchState
from app.observability.service import get_observability_service
from app.observability.types import ResearchTraceHandle
from app.services.research_history_service import create_research_history_service
from app.retrieval.hybrid_retriever import get_hybrid_retriever
from app.retrieval.reranker import create_cross_encoder_reranker
from app.sources.registry import SourceRegistry, get_source_registry

logger = logging.getLogger(__name__)


class ResearchWorkflowService:
    """End-to-end LangGraph research workflow."""

    def __init__(self, deps: ResearchWorkflowDeps) -> None:
        self._deps = deps
        self._graph = compile_research_workflow(deps)

    @contextmanager
    def _research_trace(self, question: str):
        """Bind a Langfuse research trace for the duration of a workflow run."""
        obs = get_observability_service()
        research_id = str(uuid.uuid4())
        handle = obs.start_research_trace(question=question.strip(), research_id=research_id)
        token = obs.bind_research_trace(handle)
        try:
            yield handle, research_id
        except Exception as exc:
            obs.fail_research_trace(handle, error=str(exc))
            raise
        finally:
            obs.unbind_research_trace(token)

    def _complete_trace(
        self,
        handle: ResearchTraceHandle,
        *,
        research_id: str,
        report: ResearchReport,
    ) -> None:
        summary = (report.executive_summary or "")[:500]
        get_observability_service().complete_research_trace(
            handle,
            research_id=research_id,
            report_summary=summary or None,
        )

    def _invoke_graph(
        self,
        question: str,
        *,
        evidence_top_k: int,
        max_iterations: int,
        use_research_history: bool = True,
        history_top_k: int | None = None,
        allow_private_knowledge: bool = False,
    ) -> dict:
        payload: dict = {
            "question": question.strip(),
            "evidence_top_k": evidence_top_k,
            "iteration": 1,
            "max_iterations": max_iterations,
            "use_research_history": use_research_history,
            "allow_private_knowledge": allow_private_knowledge,
            "node_traces": [],
        }
        if history_top_k is not None:
            payload["history_top_k"] = history_top_k
        return self._graph.invoke(payload)

    def run(
        self,
        question: str,
        *,
        evidence_top_k: int | None = None,
        max_iterations: int | None = None,
        use_research_history: bool = True,
        history_top_k: int | None = None,
        allow_private_knowledge: bool = False,
    ) -> ResearchWorkflowResponse:
        settings = self._deps.settings
        top_k = evidence_top_k or settings.research_agent_evidence_top_k
        iterations = max_iterations or settings.research_max_iterations
        with self._research_trace(question) as (trace_handle, research_id):
            result = self._invoke_graph(
                question,
                evidence_top_k=top_k,
                max_iterations=iterations,
                use_research_history=use_research_history,
                history_top_k=history_top_k,
                allow_private_knowledge=allow_private_knowledge,
            )
            report = result.get("report")
            if report is None:
                raise RuntimeError("Research workflow did not produce a report")
            plan = result.get("plan")
            if plan is None:
                raise RuntimeError("Research workflow did not produce a plan")
            traces: list[NodeTraceEntry] = result.get("node_traces") or []
            snippets = [
                ResearchHistoryContextSnippet.model_validate(raw)
                for raw in (result.get("history_context_snippets") or [])
            ]
            record = self._deps.history_service.save_run(
                question=question.strip(),
                plan=plan,
                bundle=result.get("bundle"),
                critique=result.get("critique"),
                report=report,
                research_id=research_id,
            )
            self._complete_trace(trace_handle, research_id=record.research_id, report=report)
        logger.info(
            "Research workflow finished research_id=%s question=%r iterations=%d traces=%d",
            record.research_id,
            question[:80],
            result.get("iteration", 1),
            len(traces),
        )
        return ResearchWorkflowResponse(
            research_id=record.research_id,
            report=report,
            history_context_used=snippets,
        )

    def stream_run(
        self,
        question: str,
        *,
        evidence_top_k: int | None = None,
        max_iterations: int | None = None,
        use_research_history: bool = True,
        history_top_k: int | None = None,
        allow_private_knowledge: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Yield progress events, then a terminal complete or error event."""
        settings = self._deps.settings
        top_k = evidence_top_k or settings.research_agent_evidence_top_k
        iterations = max_iterations or settings.research_max_iterations
        payload: dict[str, Any] = {
            "question": question.strip(),
            "evidence_top_k": top_k,
            "iteration": 1,
            "max_iterations": iterations,
            "use_research_history": use_research_history,
            "allow_private_knowledge": allow_private_knowledge,
            "node_traces": [],
        }
        if history_top_k is not None:
            payload["history_top_k"] = history_top_k

        seen_traces: set[tuple[str, int, str]] = set()
        final_state: dict[str, Any] = dict(payload)

        try:
            with self._research_trace(question) as (trace_handle, research_id):
                for chunk in self._graph.stream(payload, stream_mode="updates"):
                    for node_name, update in chunk.items():
                        if not isinstance(update, dict):
                            continue
                        for trace in update.get("node_traces") or []:
                            key = (trace.node, trace.iteration, trace.message)
                            if key in seen_traces:
                                continue
                            seen_traces.add(key)
                            yield {
                                "type": "progress",
                                "node": trace.node,
                                "iteration": trace.iteration,
                                "message": trace.message,
                            }
                        final_state.update({k: v for k, v in update.items() if k != "node_traces"})
                        if update.get("node_traces"):
                            final_state["node_traces"] = [
                                *(final_state.get("node_traces") or []),
                                *update["node_traces"],
                            ]

                report = final_state.get("report")
                plan = final_state.get("plan")
                if report is None or plan is None:
                    get_observability_service().fail_research_trace(
                        trace_handle,
                        error="Research workflow did not produce a report",
                    )
                    yield {"type": "error", "message": "Research workflow did not produce a report"}
                    return

                snippets = [
                    ResearchHistoryContextSnippet.model_validate(raw)
                    for raw in (final_state.get("history_context_snippets") or [])
                ]
                record = self._deps.history_service.save_run(
                    question=question.strip(),
                    plan=plan,
                    bundle=final_state.get("bundle"),
                    critique=final_state.get("critique"),
                    report=report,
                    research_id=research_id,
                )
                self._complete_trace(trace_handle, research_id=record.research_id, report=report)
                response = ResearchWorkflowResponse(
                    research_id=record.research_id,
                    report=report,
                    history_context_used=snippets,
                )
                yield {
                    "type": "complete",
                    "research_id": record.research_id,
                    "result": response.model_dump(mode="json"),
                }
        except Exception as exc:
            logger.exception("Research workflow stream failed")
            yield {"type": "error", "message": str(exc)}

    def run_with_state(
        self,
        question: str,
        *,
        evidence_top_k: int | None = None,
        max_iterations: int | None = None,
        use_research_history: bool = True,
        history_top_k: int | None = None,
        persist_history: bool = True,
    ) -> ResearchState:
        settings = self._deps.settings
        top_k = evidence_top_k or settings.research_agent_evidence_top_k
        iterations = max_iterations or settings.research_max_iterations
        result = self._invoke_graph(
            question,
            evidence_top_k=top_k,
            max_iterations=iterations,
            use_research_history=use_research_history,
            history_top_k=history_top_k,
        )
        report = result.get("report")
        if report is None:
            raise RuntimeError("Research workflow did not produce a report")
        if persist_history and result.get("plan") is not None:
            self._deps.history_service.save_run(
                question=question.strip(),
                plan=result["plan"],
                bundle=result.get("bundle"),
                critique=result.get("critique"),
                report=report,
            )
        return ResearchState(
            question=question.strip(),
            iteration=int(result.get("iteration") or 1),
            max_iterations=iterations,
            plan=result.get("plan"),
            bundle=result.get("bundle"),
            critique=result.get("critique"),
            report=report,
            traces=result.get("node_traces") or [],
        )


def create_research_workflow_service(
    settings: Settings,
    *,
    source_registry: SourceRegistry | None = None,
    hybrid_retriever=None,
    reranker=None,
    llm=None,
    history_service=None,
) -> ResearchWorkflowService:
    deps = ResearchWorkflowDeps(
        settings=settings,
        llm=llm or create_llm_provider(settings),
        source_registry=source_registry or get_source_registry(),
        hybrid_retriever=hybrid_retriever or get_hybrid_retriever(),
        reranker=reranker or create_cross_encoder_reranker(settings.reranker_model),
        history_service=history_service or create_research_history_service(settings),
    )
    return ResearchWorkflowService(deps)


@lru_cache
def get_research_workflow_service() -> ResearchWorkflowService:
    return create_research_workflow_service(get_settings())
