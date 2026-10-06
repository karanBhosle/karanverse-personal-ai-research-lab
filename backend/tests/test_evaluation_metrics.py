from app.evaluation.answer_metrics import evaluate_answer, faithfulness_score, GeneratedAnswerEval
from app.evaluation.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_recall_precision_mrr_ndcg():
    ranked = ["a", "b", "c", "d"]
    relevant = {"b", "d", "z"}
    assert recall_at_k(ranked, relevant, 2) == 1 / 3
    assert precision_at_k(ranked, relevant, 2) == 0.5
    assert mean_reciprocal_rank(ranked, relevant) == 0.5
    assert ndcg_at_k(ranked, relevant, 4) > 0.0


def test_answer_quality_metrics():
    generated = GeneratedAnswerEval(
        answer="Hybrid retrieval combines BM25 and dense vectors using RRF.",
        cited_chunk_ids=["chunk-1"],
    )
    metrics = evaluate_answer(
        generated,
        reference_answer="Hybrid retrieval fuses BM25 and dense retrieval with reciprocal rank fusion.",
        expected_evidence_chunk_ids={"chunk-1", "chunk-2"},
        evidence_text_by_chunk={
            "chunk-1": "BM25 and dense vectors are fused with reciprocal rank fusion.",
        },
    )
    assert metrics.faithfulness > 0.0
    assert metrics.answer_relevance > 0.0
    assert metrics.citation_correctness == 1.0
    assert metrics.citation_completeness == 0.5


def test_faithfulness_requires_evidence_support():
    score = faithfulness_score(
        "This claim is unsupported by any source.",
        ["Completely unrelated content about sourdough baking."],
    )
    assert score == 0.0
