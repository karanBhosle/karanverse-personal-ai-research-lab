import logging
from collections.abc import Callable
from typing import Any

from app.agents.research_workflow.deps import ResearchWorkflowDeps
from app.agents.research_workflow.state import ResearchWorkflowState
from app.models.research_state import NodeTraceEntry
from app.observability.context import get_current_research_trace
from app.observability.service import get_observability_service

logger = logging.getLogger(__name__)

_RETRIEVAL_METHOD_BY_NODE = {
    "evidence_retrieval": "hybrid_document_pool",
    "hybrid_retrieval": "hybrid_rrf",
    "reranking": "hybrid_rerank",
    "evidence_collection": "hybrid_rerank_bundle",
}


def _step_metadata(node_name: str, state: ResearchWorkflowState, result: dict[str, Any]) -> dict[str, Any]:
    obs = get_observability_service()
    agent_name = obs.agent_name_for_node(node_name)
    metadata: dict[str, Any] = {
        "agent_name": agent_name,
        "node": node_name,
        "iteration": int(state.get("iteration") or 1),
    }
    trace = get_current_research_trace()
    if trace and trace.research_id:
        metadata["research_id"] = trace.research_id
    retrieval_method = _RETRIEVAL_METHOD_BY_NODE.get(node_name)
    if retrieval_method:
        metadata["retrieval_method"] = retrieval_method
    if node_name == "source_search":
        papers = result.get("papers") or state.get("papers") or []
        metadata["source"] = sorted({paper.source for paper in papers})
        metadata["paper_count"] = len(papers)
    if node_name in {"evidence_collection", "evidence_retrieval", "hybrid_retrieval"}:
        bundle = result.get("bundle")
        if bundle is not None:
            metadata["evidence_count"] = len(getattr(bundle, "evidence", []) or [])
    return metadata


def trace_node(
    node_name: str,
    fn: Callable[[ResearchWorkflowState, ResearchWorkflowDeps], dict[str, Any]],
) -> Callable[[ResearchWorkflowState, ResearchWorkflowDeps], dict[str, Any]]:
    def wrapped(state: ResearchWorkflowState, deps: ResearchWorkflowDeps) -> dict[str, Any]:
        iteration = int(state.get("iteration") or 0)
        logger.info("Research workflow node=%s iteration=%d start", node_name, iteration)
        obs = get_observability_service()
        handle = get_current_research_trace()
        agent_name = obs.agent_name_for_node(node_name)
        message = f"{node_name} completed"
        step_ctx = obs.run_agent_step(
            handle,
            agent_name=agent_name,
            input_payload={"question": state.get("question"), "node": node_name, "iteration": iteration},
            metadata={"agent_name": agent_name, "node": node_name},
        )
        with step_ctx:
            result = fn(state, deps)
            message = result.pop("_trace_message", message)
            step_meta = _step_metadata(node_name, state, result)
            step_meta["message"] = message
            step_ctx.set_output(step_meta)
        trace = NodeTraceEntry(node=node_name, iteration=iteration, message=message)
        logger.info("Research workflow node=%s iteration=%d end message=%s", node_name, iteration, message)
        existing_traces = result.get("node_traces", [])
        result["node_traces"] = [*existing_traces, trace]
        return result

    return wrapped
