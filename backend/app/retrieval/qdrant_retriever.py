import logging
from functools import lru_cache

from qdrant_client import QdrantClient

from app.config.settings import get_settings
from app.llm.embedding import EmbeddingService, create_embedding_service
from app.models.knowledge_source import KnowledgeScope
from app.models.search import SearchResult
from app.retrieval.qdrant_store import QdrantVectorStore, scored_point_to_search_fields

logger = logging.getLogger(__name__)


class QdrantRetriever:
    """Dense vector retrieval over Qdrant with query embedding."""

    def __init__(
        self,
        store: QdrantVectorStore,
        embedding_service: EmbeddingService,
        default_top_k: int = 10,
    ) -> None:
        self.store = store
        self.embedding_service = embedding_service
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        *,
        knowledge_scopes: set[KnowledgeScope] | None = None,
    ) -> list[SearchResult]:
        limit = top_k or self.default_top_k
        query_vector = self.embedding_service.embed_query(query)
        hits = self.store.search(
            query_vector=query_vector,
            top_k=limit,
            knowledge_scopes=knowledge_scopes,
        )
        results: list[SearchResult] = []
        for hit in hits:
            fields = scored_point_to_search_fields(hit)
            if not fields["chunk_id"] or not fields["text"]:
                continue
            results.append(
                SearchResult(
                    chunk_id=fields["chunk_id"],
                    document_id=fields["document_id"],
                    text=fields["text"],
                    score=fields["score"],
                    metadata=fields["metadata"],
                )
            )
        logger.info("Vector search query=%r hits=%d", query, len(results))
        return results


def create_qdrant_client(url: str, api_key: str | None = None) -> QdrantClient:
    if url == ":memory:":
        return QdrantClient(location=":memory:")
    if api_key:
        return QdrantClient(url=url, api_key=api_key)
    return QdrantClient(url=url)


@lru_cache
def get_qdrant_retriever() -> QdrantRetriever:
    settings = get_settings()
    embedding = create_embedding_service(
        model_name=settings.embedding_model,
        vector_dimension=settings.vector_dimension,
    )
    client = create_qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
    store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding_service=embedding,
        vector_dimension=settings.vector_dimension,
    )
    return QdrantRetriever(
        store=store,
        embedding_service=embedding,
        default_top_k=settings.vector_search_top_k,
    )
