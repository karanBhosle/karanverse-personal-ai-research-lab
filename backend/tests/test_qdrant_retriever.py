import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.llm.embedding import EmbeddingService
from app.main import create_app
from app.models.document import Chunk, SourceType
from app.retrieval.qdrant_retriever import QdrantRetriever
from app.retrieval.qdrant_store import QdrantVectorStore
from app.services.ingestion_service import IngestionService
from app.services.vector_index_service import VectorIndexService


class FixedEmbeddingService(EmbeddingService):
    """Deterministic embeddings for unit tests."""

    def __init__(self, vectors_by_text: dict[str, list[float]], dim: int) -> None:
        self._vectors_by_text = vectors_by_text
        self._dim = dim
        self._name = "test-fixed-embedder"

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors_by_text[text] for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vectors_by_text[query]


def _chunk(text: str, *, document_id: str = "doc-1") -> Chunk:
    return Chunk(
        chunk_id=str(uuid.uuid4()),
        document_id=document_id,
        filename="paper.pdf",
        page_number=2,
        section="Methods",
        text=text,
        source_type=SourceType.PDF,
        ingestion_timestamp=datetime(2026, 6, 1, tzinfo=UTC),
    )


@pytest.fixture
def vector_stack() -> tuple[QdrantVectorStore, QdrantRetriever, FixedEmbeddingService]:
    dim = 3
    text_a = "alpha vector chunk about neural retrieval"
    text_b = "beta vector chunk about graph databases"
    vectors = {
        text_a: [1.0, 0.0, 0.0],
        text_b: [0.0, 1.0, 0.0],
        "neural retrieval": [0.95, 0.05, 0.0],
    }
    embedding = FixedEmbeddingService(vectors, dim=dim)
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(
        client=client,
        collection_name="test_chunks",
        embedding_service=embedding,
        vector_dimension=dim,
    )
    retriever = QdrantRetriever(store=store, embedding_service=embedding, default_top_k=5)
    return store, retriever, embedding


def test_qdrant_indexing_and_retrieval(vector_stack):
    store, retriever, embedding = vector_stack
    chunk_a = _chunk("alpha vector chunk about neural retrieval")
    chunk_b = _chunk("beta vector chunk about graph databases")
    indexed = store.upsert_chunks([chunk_a, chunk_b])
    assert indexed == 2

    results = retriever.retrieve("neural retrieval", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk_id == chunk_a.chunk_id
    assert results[0].document_id == chunk_a.document_id
    assert results[0].score > results[-1].score or len(results) == 1


def test_qdrant_metadata_preservation(vector_stack):
    store, retriever, _ = vector_stack
    chunk = _chunk("alpha vector chunk about neural retrieval")
    store.upsert_chunks([chunk])
    result = retriever.retrieve("neural retrieval", top_k=1)[0]

    assert result.metadata["filename"] == "paper.pdf"
    assert result.metadata["page_number"] == 2
    assert result.metadata["section"] == "Methods"
    assert result.metadata["source_type"] == "pdf"
    assert result.metadata["ingestion_timestamp"] == chunk.ingestion_timestamp.isoformat()


def test_ingestion_auto_indexes_into_qdrant(vector_stack, tmp_path):
    from unittest.mock import patch

    from app.models.document import Document

    store, _, _ = vector_stack
    processed = tmp_path / "processed"
    processed.mkdir()
    vector_indexer = VectorIndexService(store=store, enabled=True)
    service = IngestionService(
        processed_dir=processed,
        vector_indexer=vector_indexer,
    )

    chunk = _chunk("alpha vector chunk about neural retrieval")
    document = Document(
        document_id=chunk.document_id,
        title="Test",
        filename="paper.pdf",
        source_type=SourceType.PDF,
        markdown="# Test",
        blocks=[],
        ingestion_timestamp=chunk.ingestion_timestamp,
    )

    with (
        patch.object(service, "extract_content", return_value=document),
        patch.object(service, "chunk_document", return_value=[chunk]),
    ):
        service.ingest_document(tmp_path / "dummy.pdf", source_filename="paper.pdf")

    count = store.client.count(collection_name=store.collection_name).count
    assert count == 1


def test_vector_search_api(vector_stack, monkeypatch):
    store, _, embedding = vector_stack
    chunk = _chunk("alpha vector chunk about neural retrieval")
    store.upsert_chunks([chunk])

    from app.config.settings import get_settings
    from app.retrieval.qdrant_retriever import get_qdrant_retriever

    get_settings.cache_clear()
    get_qdrant_retriever.cache_clear()
    monkeypatch.setenv("QDRANT_INDEXING_ENABLED", "true")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", store.collection_name)
    monkeypatch.setenv("VECTOR_DIMENSION", "3")
    monkeypatch.setenv("EMBEDDING_MODEL", embedding.model_name)

    def _retriever_override():
        return QdrantRetriever(store=store, embedding_service=embedding, default_top_k=5)

    client = TestClient(create_app())
    client.app.dependency_overrides[get_qdrant_retriever] = _retriever_override
    response = client.post(
        "/search/vector",
        json={"query": "neural retrieval", "top_k": 3},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results"]
    assert body["vector_dimension"] == 3
    assert body["collection_name"] == store.collection_name
