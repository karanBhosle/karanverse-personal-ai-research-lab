import json
from pathlib import Path
from unittest.mock import MagicMock

from app.evaluation.dataset import EvaluationDataset, EvaluationItem
from app.evaluation.runner import EvaluationRunConfig, RetrievalEvaluationRunner
from app.evaluation.strategies import RetrievalStrategyName
from app.models.search import HybridSearchResult, RankContribution, RerankedSearchResult, SearchResult


def _dataset(tmp_path: Path) -> Path:
    data = EvaluationDataset(
        name="mock",
        items=[
            EvaluationItem(
                id="q1",
                question="hybrid retrieval",
                expected_evidence=["c-relevant"],
                reference_answer="Hybrid retrieval uses RRF.",
                retrieval_sub_queries=["bm25 dense fusion"],
            )
        ],
    )
    path = tmp_path / "dataset.json"
    data.save(path)
    return path


def _mock_backend():
    backend = MagicMock()
    backend.candidate_count = 10
    backend.hybrid_top_k = 5
    backend.vector_backend = "mock"
    backend.bm25.chunk_count = 3

    backend.bm25.retrieve.return_value = [
        SearchResult(chunk_id="c-bm25", document_id="d1", text="bm25", score=1.0, metadata={})
    ]
    backend.dense.retrieve.return_value = [
        SearchResult(chunk_id="c-dense", document_id="d1", text="dense", score=0.9, metadata={})
    ]
    hybrid_hit = HybridSearchResult(
        chunk_id="c-relevant",
        document_id="d1",
        text="hybrid retrieval rrf",
        final_score=0.8,
        bm25_score=1.0,
        vector_score=0.9,
        rank_contribution=RankContribution(bm25_rank=1, vector_rank=1),
        metadata={},
    )
    backend.hybrid.retrieve.return_value = [hybrid_hit]
    reranked = RerankedSearchResult(
        chunk_id="c-relevant",
        document_id="d1",
        text="hybrid retrieval rrf",
        hybrid_score=0.8,
        bm25_score=1.0,
        vector_score=0.9,
        rank_contribution=RankContribution(bm25_rank=1, vector_rank=1),
        metadata={},
        rerank_score=0.95,
        final_score=0.95,
    )
    backend.hybrid_rerank.retrieve.return_value = [reranked]
    backend.reranker.rerank.return_value = [reranked]
    return backend


def test_runner_writes_reports(tmp_path, monkeypatch):
    dataset_path = _dataset(tmp_path)
    output_dir = tmp_path / "reports"
    mock_backend = _mock_backend()

    monkeypatch.setattr(
        "app.evaluation.runner.build_retrieval_backend",
        lambda *args, **kwargs: mock_backend,
    )

    config = EvaluationRunConfig(
        dataset_path=dataset_path,
        output_dir=output_dir,
        k_values=[1, 3],
        max_k=3,
        strategies=[
            RetrievalStrategyName.BM25,
            RetrievalStrategyName.HYBRID_RRF,
            RetrievalStrategyName.HYBRID_RERANK,
            RetrievalStrategyName.AGENTIC,
        ],
        include_reranker=True,
        seed=7,
    )
    result = RetrievalEvaluationRunner(config).run()

    assert (output_dir / "report.json").is_file()
    assert (output_dir / "report.md").is_file()
    assert (output_dir / "config.json").is_file()
    payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    hybrid_row = next(row for row in payload["strategies"] if row["strategy"] == "hybrid_rrf")
    assert hybrid_row["mrr_mean"] == 1.0
    assert hybrid_row["recall_at_k"]["1"] == 1.0
    assert result.config["seed"] == 7
