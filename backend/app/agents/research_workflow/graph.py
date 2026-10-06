from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.agents.research_workflow.deps import ResearchWorkflowDeps
from app.agents.research_workflow.nodes import (
    run_additional_research,
    run_critic_node,
    run_evidence_collection_node,
    run_evidence_retrieval,
    run_hybrid_retrieval_node,
    run_load_history_context,
    run_planner,
    run_reranking_node,
    run_source_search,
    run_sub_questions_node,
    run_synthesizer_node,
)
from app.agents.research_workflow.state import ResearchWorkflowState
from app.agents.research_workflow.tracing import trace_node


def _route_after_critic(state: ResearchWorkflowState) -> Literal["additional_research", "synthesizer"]:
    critique = state["critique"]
    if critique.evidence_sufficient:
        return "synthesizer"
    iteration = int(state.get("iteration") or 1)
    max_iterations = int(state.get("max_iterations") or 1)
    if iteration >= max_iterations:
        return "synthesizer"
    return "additional_research"


def compile_research_workflow(deps: ResearchWorkflowDeps):
    graph = StateGraph(ResearchWorkflowState)

    graph.add_node(
        "load_history_context",
        lambda state: trace_node("load_history_context", run_load_history_context)(state, deps),
    )
    graph.add_node("planner", lambda state: trace_node("planner", run_planner)(state, deps))
    graph.add_node("sub_questions", lambda state: trace_node("sub_questions", run_sub_questions_node)(state, deps))
    graph.add_node("source_search", lambda state: trace_node("source_search", run_source_search)(state, deps))
    graph.add_node("evidence_retrieval", lambda state: trace_node("evidence_retrieval", run_evidence_retrieval)(state, deps))
    graph.add_node("hybrid_retrieval", lambda state: trace_node("hybrid_retrieval", run_hybrid_retrieval_node)(state, deps))
    graph.add_node("reranking", lambda state: trace_node("reranking", run_reranking_node)(state, deps))
    graph.add_node(
        "evidence_collection",
        lambda state: trace_node("evidence_collection", run_evidence_collection_node)(state, deps),
    )
    graph.add_node("critic", lambda state: trace_node("critic", run_critic_node)(state, deps))
    graph.add_node(
        "additional_research",
        lambda state: trace_node("additional_research", run_additional_research)(state, deps),
    )
    graph.add_node("synthesizer", lambda state: trace_node("synthesizer", run_synthesizer_node)(state, deps))

    graph.add_edge(START, "load_history_context")
    graph.add_edge("load_history_context", "planner")
    graph.add_edge("planner", "sub_questions")
    graph.add_edge("sub_questions", "source_search")
    graph.add_edge("source_search", "evidence_retrieval")
    graph.add_edge("evidence_retrieval", "hybrid_retrieval")
    graph.add_edge("hybrid_retrieval", "reranking")
    graph.add_edge("reranking", "evidence_collection")
    graph.add_edge("evidence_collection", "critic")
    graph.add_conditional_edges("critic", _route_after_critic, ["additional_research", "synthesizer"])
    graph.add_edge("additional_research", "source_search")
    graph.add_edge("synthesizer", END)

    return graph.compile()
