import logging
from functools import lru_cache

from app.config.settings import get_settings
from app.models.knowledge_source import KnowledgeScope
from app.models.search import RerankedSearchResult
from app.retrieval.hybrid_retriever import HybridRetriever, get_hybrid_retriever
from app.retrieval.reranker import Reranker, create_cross_encoder_reranker

logger = logging.getLogger(__name__)


class RerankingHybridRetriever:
    """
    Hybrid retrieval (BM25 + dense + RRF) followed by cross-encoder reranking.

    Pipeline: BM25 → Dense → RRF (top candidate_count) → Cross-Encoder → top final_top_k
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        *,
        candidate_count: int = 30,
        final_top_k: int = 10,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.candidate_count = candidate_count
        self.final_top_k = final_top_k

    def retrieve(
        self,
        query: str,
        *,
        final_top_k: int | None = None,
        candidate_count: int | None = None,
        knowledge_scopes: set[KnowledgeScope] | None = None,
    ) -> list[RerankedSearchResult]:
        candidates_n = candidate_count or self.candidate_count
        output_k = final_top_k or self.final_top_k

        hybrid_candidates = self.hybrid_retriever.retrieve(
            query,
            top_k=candidates_n,
            candidate_count=candidates_n,
            knowledge_scopes=knowledge_scopes,
        )
        reranked = self.reranker.rerank(query, hybrid_candidates, top_k=output_k)
        logger.info(
            "Reranking hybrid pipeline query=%r candidates=%d final=%d",
            query,
            len(hybrid_candidates),
            len(reranked),
        )
        return reranked


@lru_cache
def get_reranking_hybrid_retriever() -> RerankingHybridRetriever:
    settings = get_settings()
    hybrid = get_hybrid_retriever()
    reranker = create_cross_encoder_reranker(settings.reranker_model)
    return RerankingHybridRetriever(
        hybrid_retriever=hybrid,
        reranker=reranker,
        candidate_count=settings.rerank_candidate_count,
        final_top_k=settings.rerank_final_top_k,
    )
