import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from app.agents.research_agent.nodes import _dedupe_hybrid, _dedupe_reranked
from app.evaluation.dataset import EvaluationItem
from app.models.knowledge_source import KnowledgeScope
from app.models.search import HybridSearchResult, RerankedSearchResult, SearchResult
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.qdrant_retriever import QdrantRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.reranking_hybrid_retriever import RerankingHybridRetriever


class RetrievalStrategyName(str, Enum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_RERANK = "hybrid_rerank"
    AGENTIC = "agentic"


@dataclass(frozen=True, slots=True)
class RetrievalRunResult:
    chunk_ids: list[str]
    latency_ms: float
    hits: list[SearchResult | HybridSearchResult | RerankedSearchResult]


class RetrievalBackend(Protocol):
    bm25: BM25Retriever
    dense: QdrantRetriever
    hybrid: HybridRetriever
    hybrid_rerank: RerankingHybridRetriever | None
    reranker: Reranker | None
    candidate_count: int
    hybrid_top_k: int


def _chunk_ids_from_hits(hits: list) -> list[str]:
    return [hit.chunk_id for hit in hits]


def retrieve_bm25(backend: RetrievalBackend, query: str, top_k: int) -> RetrievalRunResult:
    t0 = time.perf_counter()
    hits = backend.bm25.retrieve(query, top_k=top_k)
    latency = (time.perf_counter() - t0) * 1000
    return RetrievalRunResult(chunk_ids=_chunk_ids_from_hits(hits), latency_ms=latency, hits=hits)


def retrieve_dense(backend: RetrievalBackend, query: str, top_k: int) -> RetrievalRunResult:
    t0 = time.perf_counter()
    hits = backend.dense.retrieve(query, top_k=top_k)
    latency = (time.perf_counter() - t0) * 1000
    return RetrievalRunResult(chunk_ids=_chunk_ids_from_hits(hits), latency_ms=latency, hits=hits)


def retrieve_hybrid_rrf(backend: RetrievalBackend, query: str, top_k: int) -> RetrievalRunResult:
    t0 = time.perf_counter()
    hits = backend.hybrid.retrieve(
        query,
        top_k=top_k,
        candidate_count=max(top_k, backend.candidate_count),
    )
    latency = (time.perf_counter() - t0) * 1000
    return RetrievalRunResult(chunk_ids=_chunk_ids_from_hits(hits), latency_ms=latency, hits=hits)


def retrieve_hybrid_rerank(backend: RetrievalBackend, query: str, top_k: int) -> RetrievalRunResult:
    if backend.hybrid_rerank is None:
        raise RuntimeError("Hybrid rerank pipeline is not configured for this evaluation run.")
    t0 = time.perf_counter()
    hits = backend.hybrid_rerank.retrieve(
        query,
        final_top_k=top_k,
        candidate_count=max(top_k, backend.candidate_count),
    )
    latency = (time.perf_counter() - t0) * 1000
    return RetrievalRunResult(chunk_ids=_chunk_ids_from_hits(hits), latency_ms=latency, hits=hits)


def retrieve_agentic(
    backend: RetrievalBackend,
    item: EvaluationItem,
    top_k: int,
    *,
    knowledge_scopes: set[KnowledgeScope] | None = None,
) -> RetrievalRunResult:
    """Multi-query retrieval (sub-questions) with hybrid + rerank merge, mirroring the research agent."""
    sub_queries = [q.strip() for q in item.retrieval_sub_queries if q.strip()]
    if not sub_queries:
        sub_queries = [item.question.strip()]

    t0 = time.perf_counter()
    all_hybrid: list[HybridSearchResult] = []
    for sub_q in sub_queries:
        all_hybrid.extend(
            backend.hybrid.retrieve(
                sub_q,
                top_k=backend.hybrid_top_k,
                candidate_count=max(backend.hybrid_top_k, backend.candidate_count),
                knowledge_scopes=knowledge_scopes,
            )
        )
    candidates = _dedupe_hybrid(all_hybrid)

    if backend.reranker is not None and candidates:
        reranked_map: list[RerankedSearchResult] = []
        for sub_q in sub_queries:
            reranked_map.extend(backend.reranker.rerank(sub_q, candidates, top_k=top_k))
        hits = _dedupe_reranked(reranked_map)[:top_k]
    else:
        hits = candidates[:top_k]

    latency = (time.perf_counter() - t0) * 1000
    return RetrievalRunResult(chunk_ids=_chunk_ids_from_hits(hits), latency_ms=latency, hits=hits)


STRATEGY_RUNNERS: dict[RetrievalStrategyName, Callable[..., RetrievalRunResult]] = {
    RetrievalStrategyName.BM25: retrieve_bm25,
    RetrievalStrategyName.DENSE: retrieve_dense,
    RetrievalStrategyName.HYBRID_RRF: retrieve_hybrid_rrf,
    RetrievalStrategyName.HYBRID_RERANK: retrieve_hybrid_rerank,
}
