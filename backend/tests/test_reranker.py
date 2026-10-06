from unittest.mock import MagicMock

from app.models.search import HybridSearchResult, RankContribution
from app.retrieval.reranker import CrossEncoderReranker, Reranker
from app.retrieval.reranking_hybrid_retriever import RerankingHybridRetriever


class _StubReranker(Reranker):
    def __init__(self) -> None:
        self._name = "stub-reranker"

    @property
    def model_name(self) -> str:
        return self._name

    def rerank(self, query: str, candidates: list[HybridSearchResult], *, top_k: int):
        ordered = sorted(candidates, key=lambda c: len(c.text), reverse=True)
        from app.models.search import RerankedSearchResult

        results = []
        for idx, candidate in enumerate(ordered[:top_k]):
            score = float(100 - idx)
            results.append(
                RerankedSearchResult(
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    text=candidate.text,
                    hybrid_score=candidate.final_score,
                    bm25_score=candidate.bm25_score,
                    vector_score=candidate.vector_score,
                    rank_contribution=candidate.rank_contribution,
                    metadata=candidate.metadata,
                    rerank_score=score,
                    final_score=score,
                )
            )
        return results


def _hybrid_hit(chunk_id: str, text: str, score: float) -> HybridSearchResult:
    return HybridSearchResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
        final_score=score,
        bm25_score=1.0,
        vector_score=0.5,
        rank_contribution=RankContribution(bm25_rank=1, vector_rank=2, bm25_rrf=0.01, vector_rrf=0.009),
        metadata={"filename": "paper.pdf", "source_type": "pdf"},
    )


def test_cross_encoder_reranker_orders_by_model_scores():
    reranker = CrossEncoderReranker("stub-model")
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.2, 0.9, 0.5]
    reranker._model = mock_model

    candidates = [
        _hybrid_hit("a", "short", 0.03),
        _hybrid_hit("b", "much longer candidate text", 0.02),
        _hybrid_hit("c", "medium length", 0.025),
    ]
    results = reranker.rerank("query", candidates, top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "b"
    assert results[0].rerank_score == 0.9
    assert results[0].final_score == 0.9
    assert results[0].hybrid_score == 0.02
    assert results[0].metadata["filename"] == "paper.pdf"
    mock_model.predict.assert_called_once()


def test_reranking_hybrid_pipeline_uses_candidate_pool_and_final_top_k():
    hybrid = MagicMock()
    hybrid.retrieve.return_value = [
        _hybrid_hit("a", "alpha " * 5, 0.04),
        _hybrid_hit("b", "beta " * 20, 0.03),
        _hybrid_hit("c", "gamma " * 10, 0.02),
    ]
    pipeline = RerankingHybridRetriever(
        hybrid_retriever=hybrid,
        reranker=_StubReranker(),
        candidate_count=3,
        final_top_k=2,
    )

    results = pipeline.retrieve("q", final_top_k=2, candidate_count=3)
    hybrid.retrieve.assert_called_once_with("q", top_k=3, candidate_count=3, knowledge_scopes=None)
    assert len(results) == 2
    assert results[0].chunk_id == "b"
    assert results[0].rerank_score == 100.0
    assert results[0].bm25_score == 1.0
    assert results[0].rank_contribution.bm25_rank == 1
