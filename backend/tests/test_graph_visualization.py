from datetime import UTC, datetime

from app.config.settings import Settings
from app.models.research_evidence import ResearchEvidenceBundle
from app.models.research_plan import ResearchPlan
from app.models.research_report import ResearchReport
from app.models.graph_viz import GraphQueryRequest
from app.services.graph_materializer import materialize_graph
from app.services.graph_visualization_service import create_graph_visualization_service
from app.services.research_history_service import create_research_history_service
from app.knowledge_graph.backends import DisabledKnowledgeGraphBackend
from app.knowledge_graph.service import KnowledgeGraphService


def _seed_history(tmp_path):
    history = create_research_history_service(
        Settings(data_processed_dir=tmp_path, research_history_dir=tmp_path / "history")
    )
    plan = ResearchPlan(
        original_question="How is Graph RAG connected to my projects?",
        research_objective="Map Graph RAG to portfolio projects.",
        sub_questions=["What is Graph RAG?", "How does hybrid retrieval relate?"],
        search_queries=["graph RAG retrieval", "knowledge graph RAG"],
        required_sources=["openalex"],
        expected_evidence=["surveys"],
        research_strategy="collect",
    )
    history.save_run(
        question=plan.original_question,
        plan=plan,
        bundle=ResearchEvidenceBundle(question=plan.original_question, plan=plan, papers=[], evidence=[], citations=[]),
        critique=None,
        report=ResearchReport(
            question=plan.original_question,
            executive_summary="Graph RAG connects structured knowledge graphs with retrieval-augmented generation.",
        ),
        research_id="run-graph-1",
    )
    return history


def test_materialize_graph_creates_concepts_and_question(tmp_path):
    history = _seed_history(tmp_path)
    graph = materialize_graph(history.list_records(), processed_dir=tmp_path)
    assert graph.nodes
    assert any(node.label.value == "ResearchQuestion" for node in graph.nodes.values())
    assert any("graph" in node.title.lower() for node in graph.nodes.values() if node.label.value == "Concept")


def test_graph_query_finds_graph_rag_subgraph(tmp_path):
    history = _seed_history(tmp_path)
    service = create_graph_visualization_service(
        Settings(data_processed_dir=tmp_path, research_history_dir=tmp_path / "history"),
        history_service=history,
        kg_service=KnowledgeGraphService(DisabledKnowledgeGraphBackend()),
    )
    result = service.query(GraphQueryRequest(query="Graph RAG projects", expand_depth=2, limit=50))
    assert result.nodes
    assert any("Graph RAG" in node.title or "graph" in node.title.lower() for node in result.nodes)


def test_graph_node_detail_for_research_question(tmp_path):
    history = _seed_history(tmp_path)
    service = create_graph_visualization_service(
        Settings(data_processed_dir=tmp_path, research_history_dir=tmp_path / "history"),
        history_service=history,
        kg_service=KnowledgeGraphService(DisabledKnowledgeGraphBackend()),
    )
    graph = service.query(GraphQueryRequest(query="Graph RAG", limit=20))
    rq = next(node for node in graph.nodes if node.label.value == "ResearchQuestion")
    detail = service.get_node_detail(rq.id)
    assert detail is not None
    assert detail.research_history
    assert detail.research_history[0].research_id == "run-graph-1"
