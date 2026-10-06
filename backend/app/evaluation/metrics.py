import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalMetricsAtK:
    k: int
    recall: float
    precision: float
    ndcg: float


@dataclass(frozen=True, slots=True)
class QueryRetrievalMetrics:
    query_id: str
    mrr: float
    metrics_at_k: list[RetrievalMetricsAtK]
    latency_ms: float
    retrieved_chunk_ids: list[str]


def _relevant_set(relevant_chunk_ids: set[str], ranked_chunk_ids: list[str]) -> set[str]:
    return {chunk_id for chunk_id in ranked_chunk_ids if chunk_id in relevant_chunk_ids}


def recall_at_k(ranked_chunk_ids: list[str], relevant_chunk_ids: set[str], k: int) -> float:
    if not relevant_chunk_ids:
        return 0.0
    hits = len(_relevant_set(relevant_chunk_ids, ranked_chunk_ids[:k]))
    return hits / len(relevant_chunk_ids)


def precision_at_k(ranked_chunk_ids: list[str], relevant_chunk_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = len(_relevant_set(relevant_chunk_ids, ranked_chunk_ids[:k]))
    return hits / k


def mean_reciprocal_rank(ranked_chunk_ids: list[str], relevant_chunk_ids: set[str]) -> float:
    for rank, chunk_id in enumerate(ranked_chunk_ids, start=1):
        if chunk_id in relevant_chunk_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_chunk_ids: list[str], relevant_chunk_ids: set[str], k: int) -> float:
    if not relevant_chunk_ids:
        return 0.0
    k = min(k, len(ranked_chunk_ids)) if ranked_chunk_ids else k
    dcg = 0.0
    for index, chunk_id in enumerate(ranked_chunk_ids[:k], start=1):
        rel = 1.0 if chunk_id in relevant_chunk_ids else 0.0
        if rel:
            dcg += rel / math.log2(index + 1)
    ideal_hits = min(len(relevant_chunk_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def aggregate_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
