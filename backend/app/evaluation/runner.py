import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config.settings import Settings, get_settings
from app.evaluation.answer_metrics import AnswerQualityMetrics, GeneratedAnswerEval, evaluate_answer
from app.evaluation.dataset import EvaluationDataset, EvaluationItem
from app.evaluation.infrastructure import build_retrieval_backend
from app.evaluation.metrics import (
    QueryRetrievalMetrics,
    RetrievalMetricsAtK,
    aggregate_mean,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from app.evaluation.report import write_evaluation_reports
from app.evaluation.results import EvaluationRunResult, StrategyAggregateMetrics
from app.evaluation.strategies import (
    RetrievalStrategyName,
    STRATEGY_RUNNERS,
    retrieve_agentic,
)
from app.llm.factory import create_llm_provider
from app.llm.provider import LLMProvider
from app.llm.types import LLMMessage, LLMRequest
from app.models.rag import GroundedAnswerLLMOutput
from app.services.citation_service import CitationService
from app.services.rag_context import build_evidence_context, build_grounded_system_prompt, build_grounded_user_prompt


@dataclass
class EvaluationRunConfig:
    dataset_path: Path
    output_dir: Path
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    max_k: int = 10
    strategies: list[RetrievalStrategyName] | None = None
    use_remote_qdrant: bool = False
    include_reranker: bool = True
    evaluate_answers: bool = False
    seed: int = 42
    settings: Settings | None = None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def _relevant_chunks(item: EvaluationItem) -> set[str]:
    return set(item.expected_evidence)


def _serialize_query_metrics(metrics: QueryRetrievalMetrics) -> dict:
    return {
        "query_id": metrics.query_id,
        "mrr": metrics.mrr,
        "latency_ms": metrics.latency_ms,
        "retrieved_chunk_ids": metrics.retrieved_chunk_ids,
        "metrics_at_k": [asdict(item) for item in metrics.metrics_at_k],
    }


class RetrievalEvaluationRunner:
    def __init__(
        self,
        config: EvaluationRunConfig,
        *,
        llm: LLMProvider | None = None,
    ) -> None:
        self.config = config
        self.settings = config.settings or get_settings()
        self.llm = llm

    def run(self) -> EvaluationRunResult:
        dataset = EvaluationDataset.load(self.config.dataset_path)
        strategies = self.config.strategies or list(RetrievalStrategyName)
        max_k = max(self.config.max_k, max(self.config.k_values))
        backend = build_retrieval_backend(
            self.settings,
            use_remote_qdrant=self.config.use_remote_qdrant,
            include_reranker=self.config.include_reranker,
            candidate_count=max(self.settings.rerank_candidate_count, max_k),
        )

        per_query: dict[str, dict[str, Any]] = {}
        strategy_metrics: dict[str, StrategyAggregateMetrics] = {}

        for strategy in strategies:
            mrr_values: list[float] = []
            latency_values: list[float] = []
            recall_buckets: dict[int, list[float]] = {k: [] for k in self.config.k_values}
            precision_buckets: dict[int, list[float]] = {k: [] for k in self.config.k_values}
            ndcg_buckets: dict[int, list[float]] = {k: [] for k in self.config.k_values}
            faithfulness_values: list[float] = []
            relevance_values: list[float] = []
            correctness_values: list[float] = []
            completeness_values: list[float] = []

            for item in dataset.items:
                relevant = _relevant_chunks(item)
                if strategy == RetrievalStrategyName.AGENTIC:
                    run = retrieve_agentic(backend, item, top_k=max_k)
                else:
                    runner = STRATEGY_RUNNERS[strategy]
                    run = runner(backend, item.question, top_k=max_k)

                mrr = mean_reciprocal_rank(run.chunk_ids, relevant)
                mrr_values.append(mrr)
                latency_values.append(run.latency_ms)

                metrics_at_k: list[RetrievalMetricsAtK] = []
                for k in self.config.k_values:
                    kk = min(k, max_k)
                    recall = recall_at_k(run.chunk_ids, relevant, kk)
                    precision = precision_at_k(run.chunk_ids, relevant, kk)
                    ndcg = ndcg_at_k(run.chunk_ids, relevant, kk)
                    recall_buckets[k].append(recall)
                    precision_buckets[k].append(precision)
                    ndcg_buckets[k].append(ndcg)
                    metrics_at_k.append(
                        RetrievalMetricsAtK(k=kk, recall=recall, precision=precision, ndcg=ndcg)
                    )

                query_metrics = QueryRetrievalMetrics(
                    query_id=item.id,
                    mrr=mrr,
                    metrics_at_k=metrics_at_k,
                    latency_ms=run.latency_ms,
                    retrieved_chunk_ids=run.chunk_ids,
                )
                per_query.setdefault(item.id, {})[strategy.value] = _serialize_query_metrics(query_metrics)

                if self.config.evaluate_answers and item.reference_answer:
                    if strategy in {
                        RetrievalStrategyName.HYBRID_RERANK,
                        RetrievalStrategyName.AGENTIC,
                    }:
                        answer_eval = self._evaluate_generated_answer(item, run)
                        if answer_eval is not None:
                            faithfulness_values.append(answer_eval.faithfulness)
                            relevance_values.append(answer_eval.answer_relevance)
                            correctness_values.append(answer_eval.citation_correctness)
                            completeness_values.append(answer_eval.citation_completeness)
                            per_query[item.id][strategy.value + "_answer"] = answer_eval.model_dump()

            aggregate = StrategyAggregateMetrics(
                strategy=strategy.value,
                query_count=len(dataset.items),
                mrr_mean=round(aggregate_mean(mrr_values), 4),
                latency_ms_mean=round(aggregate_mean(latency_values), 3),
                latency_ms_p95=round(_percentile(latency_values, 95), 3),
                recall_at_k={k: round(aggregate_mean(recall_buckets[k]), 4) for k in self.config.k_values},
                precision_at_k={k: round(aggregate_mean(precision_buckets[k]), 4) for k in self.config.k_values},
                ndcg_at_k={k: round(aggregate_mean(ndcg_buckets[k]), 4) for k in self.config.k_values},
                answer_faithfulness_mean=round(aggregate_mean(faithfulness_values), 4)
                if faithfulness_values
                else None,
                answer_relevance_mean=round(aggregate_mean(relevance_values), 4) if relevance_values else None,
                citation_correctness_mean=round(aggregate_mean(correctness_values), 4)
                if correctness_values
                else None,
                citation_completeness_mean=round(aggregate_mean(completeness_values), 4)
                if completeness_values
                else None,
            )
            strategy_metrics[strategy.value] = aggregate

        run_config = {
            "dataset_path": str(self.config.dataset_path.resolve()),
            "dataset_hash": dataset.content_hash(),
            "seed": self.config.seed,
            "k_values": self.config.k_values,
            "max_k": max_k,
            "strategies": [s.value for s in strategies],
            "use_remote_qdrant": self.config.use_remote_qdrant,
            "include_reranker": self.config.include_reranker,
            "evaluate_answers": self.config.evaluate_answers,
            "vector_backend": backend.vector_backend,
            "bm25_index_chunks": backend.bm25.chunk_count,
            "embedding_model": self.settings.embedding_model,
            "reranker_model": self.settings.reranker_model if self.config.include_reranker else None,
            "rrf_k": self.settings.rrf_k,
            "hybrid_bm25_weight": self.settings.hybrid_bm25_weight,
            "hybrid_vector_weight": self.settings.hybrid_vector_weight,
        }

        result = EvaluationRunResult(
            dataset_name=dataset.name,
            dataset_hash=dataset.content_hash(),
            generated_at=datetime.now(UTC),
            config=run_config,
            strategies=list(strategy_metrics.values()),
            per_query=per_query,
        )

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        write_evaluation_reports(result, self.config.output_dir)
        (self.config.output_dir / "config.json").write_text(
            json.dumps(run_config, indent=2),
            encoding="utf-8",
        )
        return result

    def _evaluate_generated_answer(self, item: EvaluationItem, run) -> AnswerQualityMetrics | None:
        reranked_hits = [hit for hit in run.hits if hasattr(hit, "rerank_score")]
        if not reranked_hits:
            return None
        llm = self.llm or create_llm_provider(self.settings)
        citation_service = CitationService(processed_dir=self.settings.data_processed_dir)
        citations = citation_service.register_reranked_results(reranked_hits[: self.config.max_k])
        evidence_items = [citation_service.get_evidence(c.evidence_id) for c in citations]
        evidence_by_id = {evidence.evidence_id: evidence for evidence in evidence_items}
        if not citations:
            return None

        context = build_evidence_context(citations, evidence_by_id)
        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=build_grounded_system_prompt()),
                LLMMessage(role="user", content=build_grounded_user_prompt(item.question, context)),
            ],
            temperature=0.1,
        )
        llm_output = llm.generate_structured(request, GroundedAnswerLLMOutput)
        cited_chunk_ids = [
            evidence_by_id[citations[num - 1].evidence_id].chunk_id
            for num in llm_output.citation_numbers
            if 1 <= num <= len(citations)
        ]
        evidence_text_by_chunk = {evidence.chunk_id: evidence.text for evidence in evidence_items}
        return evaluate_answer(
            GeneratedAnswerEval(answer=llm_output.answer, cited_chunk_ids=cited_chunk_ids),
            reference_answer=item.reference_answer,
            expected_evidence_chunk_ids=set(item.expected_evidence),
            evidence_text_by_chunk=evidence_text_by_chunk,
        )
