from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class StrategyAggregateMetrics(BaseModel):
    strategy: str
    query_count: int
    mrr_mean: float
    latency_ms_mean: float
    latency_ms_p95: float
    recall_at_k: dict[int, float] = Field(default_factory=dict)
    precision_at_k: dict[int, float] = Field(default_factory=dict)
    ndcg_at_k: dict[int, float] = Field(default_factory=dict)
    answer_faithfulness_mean: float | None = None
    answer_relevance_mean: float | None = None
    citation_correctness_mean: float | None = None
    citation_completeness_mean: float | None = None


class EvaluationRunResult(BaseModel):
    dataset_name: str
    dataset_hash: str
    generated_at: datetime
    config: dict[str, Any]
    strategies: list[StrategyAggregateMetrics]
    per_query: dict[str, dict[str, Any]]
