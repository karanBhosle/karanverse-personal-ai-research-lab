import logging
from functools import lru_cache

from app.config.settings import get_settings
from app.models.knowledge_source import KnowledgeScope
from app.models.search import HybridSearchResult
from app.retrieval.bm25_retriever import BM25Retriever, get_bm25_retriever
from app.retrieval.qdrant_retriever import QdrantRetriever, get_qdrant_retriever
from app.retrieval.rrf import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class HybridRetriever:
    """BM25 + dense vector retrieval fused with weighted RRF."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: QdrantRetriever,
        *,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
        candidate_pool: int = 50,
    ) -> None:
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.candidate_pool = candidate_pool

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        candidate_count: int | None = None,
        knowledge_scopes: set[KnowledgeScope] | None = None,
    ) -> list[HybridSearchResult]:
        fusion_top_k = candidate_count or top_k
        pool = max(fusion_top_k, self.candidate_pool)

        self.bm25_retriever.ensure_index()
        bm25_hits = self.bm25_retriever.retrieve(query, top_k=pool, knowledge_scopes=knowledge_scopes)
        try:
            vector_hits = self.vector_retriever.retrieve(
                query,
                top_k=pool,
                knowledge_scopes=knowledge_scopes,
            )
        except Exception as exc:
            logger.warning("Vector retrieval unavailable (Qdrant down?); using BM25 only: %s", exc)
            vector_hits = []

        fused = reciprocal_rank_fusion(
            bm25_hits,
            vector_hits,
            top_k=fusion_top_k,
            rrf_k=self.rrf_k,
            bm25_weight=self.bm25_weight,
            vector_weight=self.vector_weight,
        )
        logger.info(
            "Hybrid search query=%r bm25=%d vector=%d fused=%d",
            query,
            len(bm25_hits),
            len(vector_hits),
            len(fused),
        )
        return fused


@lru_cache
def get_hybrid_retriever() -> HybridRetriever:
    settings = get_settings()
    return HybridRetriever(
        bm25_retriever=get_bm25_retriever(),
        vector_retriever=get_qdrant_retriever(),
        rrf_k=settings.rrf_k,
        bm25_weight=settings.hybrid_bm25_weight,
        vector_weight=settings.hybrid_vector_weight,
        candidate_pool=settings.hybrid_candidate_pool,
    )
