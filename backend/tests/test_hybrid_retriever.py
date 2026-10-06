from unittest.mock import MagicMock

from app.models.search import SearchResult
from app.retrieval.hybrid_retriever import HybridRetriever


def test_hybrid_retriever_executes_both_paths_and_fuses():
    bm25 = MagicMock()
    vector = MagicMock()
    bm25.ensure_index = MagicMock()
    bm25.retrieve.return_value = [
        SearchResult(
            chunk_id="c1",
            document_id="d1",
            text="bm25 hit",
            score=2.0,
            metadata={"source_type": "pdf"},
        )
    ]
    vector.retrieve.return_value = [
        SearchResult(
            chunk_id="c2",
            document_id="d2",
            text="vector hit",
            score=0.9,
            metadata={"source_type": "pdf"},
        ),
        SearchResult(
            chunk_id="c1",
            document_id="d1",
            text="bm25 hit",
            score=0.95,
            metadata={"source_type": "pdf"},
        ),
    ]

    hybrid = HybridRetriever(bm25, vector, candidate_pool=10, rrf_k=60)
    results = hybrid.retrieve("query", top_k=2)

    bm25.retrieve.assert_called_once_with("query", top_k=10, knowledge_scopes=None)
    vector.retrieve.assert_called_once_with("query", top_k=10, knowledge_scopes=None)
    assert len(results) == 2
    assert results[0].chunk_id == "c1"
    assert results[0].bm25_score == 2.0
    assert results[0].vector_score == 0.95
    assert results[0].final_score > results[1].final_score
