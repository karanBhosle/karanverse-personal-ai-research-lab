import pytest

from app.models.search import SearchResult
from app.retrieval.rrf import _rrf_contribution, reciprocal_rank_fusion


def _hit(chunk_id: str, score: float, text: str = "text") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=text,
        score=score,
        metadata={"filename": f"{chunk_id}.pdf", "source_type": "pdf"},
    )


def test_rrf_contribution_formula():
    assert _rrf_contribution(1, 60, 1.0) == pytest.approx(1 / 61)
    assert _rrf_contribution(3, 60, 2.0) == pytest.approx(2 / 63)


def test_rrf_prefers_documents_present_in_both_lists():
    bm25 = [_hit("a", 5.0), _hit("shared", 4.0), _hit("b", 1.0)]
    vector = [_hit("shared", 0.92), _hit("c", 0.8), _hit("a", 0.5)]
    k = 60

    fused = reciprocal_rank_fusion(bm25, vector, top_k=3, rrf_k=k)
    assert fused[0].chunk_id == "shared"
    expected_shared = _rrf_contribution(2, k, 1.0) + _rrf_contribution(1, k, 1.0)
    assert fused[0].final_score == pytest.approx(expected_shared)
    assert fused[0].rank_contribution.bm25_rank == 2
    assert fused[0].rank_contribution.vector_rank == 1
    assert fused[0].rank_contribution.bm25_rrf == pytest.approx(_rrf_contribution(2, k, 1.0))
    assert fused[0].rank_contribution.vector_rrf == pytest.approx(_rrf_contribution(1, k, 1.0))


def test_rrf_respects_retriever_weights():
    bm25 = [_hit("only-bm25", 3.0)]
    vector = [_hit("only-vector", 0.9)]
    k = 60

    fused = reciprocal_rank_fusion(
        bm25,
        vector,
        top_k=2,
        rrf_k=k,
        bm25_weight=2.0,
        vector_weight=1.0,
    )
    assert fused[0].chunk_id == "only-bm25"
    assert fused[0].final_score == pytest.approx(_rrf_contribution(1, k, 2.0))
    assert fused[1].chunk_id == "only-vector"


def test_rrf_preserves_scores_and_metadata():
    bm25 = [_hit("x", 1.23, text="bm25 text")]
    vector = [_hit("x", 0.88, text="vector text")]

    fused = reciprocal_rank_fusion(bm25, vector, top_k=1, rrf_k=60)
    result = fused[0]
    assert result.bm25_score == 1.23
    assert result.vector_score == 0.88
    assert result.metadata["filename"] == "x.pdf"
    assert result.text in {"bm25 text", "vector text"}
