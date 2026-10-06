import pytest
from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.knowledge_graph.backends import DisabledKnowledgeGraphBackend, InMemoryKnowledgeGraphBackend
from app.knowledge_graph.service import KnowledgeGraphService, create_knowledge_graph_service
from app.knowledge_graph.types import GraphEntityLabel, GraphRelationshipType


@pytest.fixture
def memory_service() -> KnowledgeGraphService:
    return KnowledgeGraphService(InMemoryKnowledgeGraphBackend())


def test_disabled_backend_is_safe():
    service = KnowledgeGraphService(DisabledKnowledgeGraphBackend())
    assert not service.enabled
    view = service.add_entity(GraphEntityLabel.CONCEPT, "c1", {"name": "RAG"})
    assert view is not None
    assert service.find_related_concepts("c1") == []
    assert service.find_supporting_evidence("q1") == []
    assert service.find_related_projects("c1", GraphEntityLabel.CONCEPT) == []


def test_in_memory_graph_queries(memory_service: KnowledgeGraphService):
    service = memory_service
    service.add_entity(GraphEntityLabel.CONCEPT, "c-rag", {"name": "RAG"})
    service.add_entity(GraphEntityLabel.CONCEPT, "c-agent", {"name": "Agents"})
    service.add_entity(GraphEntityLabel.RESEARCH_QUESTION, "q1", {"text": "Agentic RAG?"})
    service.add_entity(GraphEntityLabel.EVIDENCE, "e1", {"excerpt": "Hybrid retrieval in agents"})
    service.add_entity(GraphEntityLabel.PROJECT, "p1", {"name": "karanVerse Lab"})
    service.add_entity(GraphEntityLabel.RESEARCH_PAPER, "paper-1", {"title": "Agentic RAG"})
    service.add_entity(GraphEntityLabel.AUTHOR, "a1", {"name": "Ada"})

    service.add_relationship(
        GraphRelationshipType.RELATED_TO,
        GraphEntityLabel.CONCEPT,
        "c-rag",
        GraphEntityLabel.CONCEPT,
        "c-agent",
    )
    service.add_relationship(
        GraphRelationshipType.SUPPORTS,
        GraphEntityLabel.EVIDENCE,
        "e1",
        GraphEntityLabel.RESEARCH_QUESTION,
        "q1",
    )
    service.add_relationship(
        GraphRelationshipType.USED_IN_PROJECT,
        GraphEntityLabel.RESEARCH_PAPER,
        "paper-1",
        GraphEntityLabel.PROJECT,
        "p1",
    )
    service.add_relationship(
        GraphRelationshipType.AUTHORED_BY,
        GraphEntityLabel.RESEARCH_PAPER,
        "paper-1",
        GraphEntityLabel.AUTHOR,
        "a1",
    )

    related = service.find_related_concepts("c-rag", limit=5)
    assert any(item.entity_id == "c-agent" for item in related)

    evidence = service.find_supporting_evidence("q1", limit=5)
    assert len(evidence) == 1
    assert evidence[0].entity_id == "e1"

    projects = service.find_related_projects("paper-1", GraphEntityLabel.RESEARCH_PAPER, limit=5)
    assert any(item.entity_id == "p1" for item in projects)


def test_create_service_defaults_to_disabled_without_config(tmp_path):
    settings = Settings(data_processed_dir=tmp_path, neo4j_enabled=False)
    service = create_knowledge_graph_service(settings)
    assert not service.enabled


def test_health_reports_neo4j_flags(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NEO4J_ENABLED", "false")
    from app.config.settings import get_settings
    from app.knowledge_graph.service import get_knowledge_graph_service
    from app.main import create_app

    get_settings.cache_clear()
    get_knowledge_graph_service.cache_clear()
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["neo4j_enabled"] is False
    assert body["neo4j_connected"] is False
