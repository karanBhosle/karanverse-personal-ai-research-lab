import logging
from abc import ABC, abstractmethod

from app.models.search import HybridSearchResult, RerankedSearchResult

logger = logging.getLogger(__name__)


class Reranker(ABC):
    """Rerank retrieval candidates by query-document relevance."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[HybridSearchResult],
        *,
        top_k: int,
    ) -> list[RerankedSearchResult]:
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    """sentence-transformers CrossEncoder reranker."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder reranker: %s", self._model_name)
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[HybridSearchResult],
        *,
        top_k: int,
    ) -> list[RerankedSearchResult]:
        if not candidates or top_k < 1:
            return []

        pairs = [[query, candidate.text] for candidate in candidates]
        raw_scores = self._get_model().predict(pairs, show_progress_bar=False)
        scored = list(zip(candidates, [float(score) for score in raw_scores], strict=True))
        scored.sort(key=lambda item: item[1], reverse=True)

        results: list[RerankedSearchResult] = []
        for candidate, rerank_score in scored[:top_k]:
            results.append(
                RerankedSearchResult(
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    text=candidate.text,
                    hybrid_score=candidate.final_score,
                    bm25_score=candidate.bm25_score,
                    vector_score=candidate.vector_score,
                    rank_contribution=candidate.rank_contribution,
                    metadata=dict(candidate.metadata),
                    rerank_score=rerank_score,
                    final_score=rerank_score,
                )
            )
        logger.info("Reranked %d candidates to top %d", len(candidates), len(results))
        return results


def create_cross_encoder_reranker(model_name: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name=model_name)
