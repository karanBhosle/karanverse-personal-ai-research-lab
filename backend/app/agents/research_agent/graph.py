from langgraph.graph import END, START, StateGraph

from app.agents.research_agent.deps import ResearchAgentDeps
from app.agents.research_agent.nodes import (
    run_document_retrieval,
    run_evidence_collection,
    run_hybrid_retrieval,
    run_literature_search,
    run_reranking,
    run_research_planner,
    run_sub_question_generation,
)
from app.agents.research_agent.state import ResearchAgentState


def compile_research_agent(deps: ResearchAgentDeps):
    graph = StateGraph(ResearchAgentState)

    graph.add_node("research_planner", lambda state: run_research_planner(state, deps))
    graph.add_node("sub_question_generation", lambda state: run_sub_question_generation(state, deps))
    graph.add_node("literature_search", lambda state: run_literature_search(state, deps))
    graph.add_node("document_retrieval", lambda state: run_document_retrieval(state, deps))
    graph.add_node("hybrid_retrieval", lambda state: run_hybrid_retrieval(state, deps))
    graph.add_node("reranking", lambda state: run_reranking(state, deps))
    graph.add_node("evidence_collection", lambda state: run_evidence_collection(state, deps))

    graph.add_edge(START, "research_planner")
    graph.add_edge("research_planner", "sub_question_generation")
    graph.add_edge("sub_question_generation", "literature_search")
    graph.add_edge("literature_search", "document_retrieval")
    graph.add_edge("document_retrieval", "hybrid_retrieval")
    graph.add_edge("hybrid_retrieval", "reranking")
    graph.add_edge("reranking", "evidence_collection")
    graph.add_edge("evidence_collection", END)

    return graph.compile()
