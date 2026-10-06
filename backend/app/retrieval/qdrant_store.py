import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.llm.embedding import EmbeddingService
from app.models.document import Chunk
from app.models.knowledge_source import KnowledgeScope
from app.retrieval.provenance import chunk_payload

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Qdrant collection management and chunk vector upsert/search."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding_service: EmbeddingService,
        vector_dimension: int,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.embedding_service = embedding_service
        self.vector_dimension = vector_dimension

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            info = self.client.get_collection(self.collection_name)
            existing_size = info.config.params.vectors.size  # type: ignore[union-attr]
            if existing_size != self.vector_dimension:
                raise ValueError(
                    f"Qdrant collection {self.collection_name} has dimension {existing_size}, "
                    f"expected {self.vector_dimension}"
                )
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=self.vector_dimension,
                distance=qmodels.Distance.COSINE,
            ),
        )
        logger.info(
            "Created Qdrant collection %s (dim=%d)",
            self.collection_name,
            self.vector_dimension,
        )

    def upsert_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        self.ensure_collection()
        texts = [chunk.text for chunk in chunks]
        vectors = self.embedding_service.embed_documents(texts)
        points = [
            qmodels.PointStruct(
                id=_point_id(chunk.chunk_id),
                vector=vector,
                payload=chunk_payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        logger.info("Upserted %d vectors into %s", len(points), self.collection_name)
        return len(points)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        *,
        knowledge_scopes: set[KnowledgeScope] | None = None,
    ) -> list[qmodels.ScoredPoint]:
        self.ensure_collection()
        if top_k < 1:
            return []
        query_filter = None
        if knowledge_scopes and len(knowledge_scopes) < len(KnowledgeScope):
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="knowledge_scope",
                        match=qmodels.MatchAny(any=[scope.value for scope in knowledge_scopes]),
                    )
                ]
            )
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return list(response.points)


def _point_id(chunk_id: str) -> str:
    return str(uuid.UUID(chunk_id))


def scored_point_to_search_fields(point: qmodels.ScoredPoint) -> dict[str, Any]:
    payload = point.payload or {}
    return {
        "chunk_id": str(payload.get("chunk_id", "")),
        "document_id": str(payload.get("document_id", "")),
        "text": str(payload.get("text", "")),
        "score": float(point.score),
        "metadata": {
            "filename": payload.get("filename"),
            "page_number": payload.get("page_number"),
            "section": payload.get("section"),
            "source_type": payload.get("source_type"),
            "knowledge_scope": payload.get("knowledge_scope"),
            "document_kind": payload.get("document_kind"),
            "project_id": payload.get("project_id"),
            "tags": payload.get("tags") or [],
            "ingestion_timestamp": payload.get("ingestion_timestamp"),
        },
    }
