from pathlib import Path

import pytest

from app.config.settings import Settings
from app.models.chunking import ChunkingConfig
from app.models.knowledge_source import DocumentKind, KnowledgeScope
from app.services.ingestion_service import IngestionService
from app.services.portfolio_import_service import import_portfolio_seed


@pytest.fixture
def ingestion_service(tmp_path: Path) -> IngestionService:
    processed = tmp_path / "processed"
    raw = tmp_path / "raw"
    processed.mkdir()
    raw.mkdir()
    return IngestionService(
        processed_dir=processed,
        raw_dir=raw,
        chunking=ChunkingConfig(chunk_size=300, chunk_overlap=40, min_chunk_size=20),
    )


def test_ingest_text_document_tags_scope(ingestion_service: IngestionService):
    document = ingestion_service.ingest_text_document(
        title="Graph RAG notes",
        text="# Graph RAG\n\nI learned that graph traversal complements vector search.",
        filename="graph_rag.md",
        knowledge_scope=KnowledgeScope.PERSONAL_KNOWLEDGE,
        document_kind=DocumentKind.TECHNICAL_NOTE,
        tags=["graph-rag"],
    )
    assert document.knowledge_scope == KnowledgeScope.PERSONAL_KNOWLEDGE
    assert document.chunks
    assert document.chunks[0].knowledge_scope == KnowledgeScope.PERSONAL_KNOWLEDGE


def test_import_portfolio_seed(ingestion_service: IngestionService):
    root = Path(__file__).resolve().parents[2]
    seed = root / "data" / "personal" / "portfolio_seed.json"
    ids = import_portfolio_seed(ingestion_service, seed_path=seed)
    assert len(ids) >= 3
    chunks_path = ingestion_service.processed_dir / ids[0] / "chunks.json"
    assert chunks_path.is_file()


def test_knowledge_search_endpoint(tmp_path, ingestion_service: IngestionService):
    ingestion_service.ingest_text_document(
        title="Hybrid retrieval project",
        text="# Project\n\nThis project uses hybrid retrieval with BM25 and dense vectors.",
        filename="project.md",
        knowledge_scope=KnowledgeScope.PROJECT,
        document_kind=DocumentKind.PROJECT_DOCUMENTATION,
        project_id="karanverse-research-lab",
        tags=["hybrid retrieval"],
    )
    ingestion_service.bm25_retriever = None  # not used
    from app.config.settings import get_settings
    from app.main import create_app
    from app.retrieval.bm25_retriever import BM25Retriever, get_bm25_retriever
    from app.retrieval.hybrid_retriever import HybridRetriever, get_hybrid_retriever
    index_dir = tmp_path / "bm25"
    retriever = BM25Retriever(index_dir=index_dir, processed_dir=ingestion_service.processed_dir)
    retriever.build_from_processed()

    class _EmptyVectorRetriever:
        def retrieve(self, query: str, top_k: int | None = None, *, knowledge_scopes=None):
            return []

    get_settings.cache_clear()
    get_bm25_retriever.cache_clear()
    get_hybrid_retriever.cache_clear()

    app = create_app()
    hybrid = HybridRetriever(
        bm25_retriever=retriever,
        vector_retriever=_EmptyVectorRetriever(),
    )
    app.dependency_overrides[get_hybrid_retriever] = lambda: hybrid

    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post(
        "/knowledge/search",
        json={"query": "Which of my projects use hybrid retrieval?", "top_k": 5},
    )
    assert response.status_code == 200
    body = response.json()
    assert "PROJECT" in body["knowledge_scopes"]
    assert body["results"]
