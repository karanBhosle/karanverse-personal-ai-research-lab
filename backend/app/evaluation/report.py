import json
from pathlib import Path

from app.evaluation.results import EvaluationRunResult, StrategyAggregateMetrics


def _rank_strategies_by_ndcg(strategies: list[StrategyAggregateMetrics], k: int) -> list[StrategyAggregateMetrics]:
    return sorted(
        strategies,
        key=lambda item: item.ndcg_at_k.get(k, 0.0),
        reverse=True,
    )


def render_markdown_report(result: EvaluationRunResult) -> str:
    primary_k = max(result.config.get("k_values", [10]))
    ranked = _rank_strategies_by_ndcg(result.strategies, primary_k)
    lines = [
        "# Retrieval evaluation report",
        "",
        f"- **Dataset:** {result.dataset_name} (`{result.dataset_hash}`)",
        f"- **Generated:** {result.generated_at.isoformat()}",
        f"- **Vector backend:** `{result.config.get('vector_backend')}`",
        f"- **BM25 chunks:** {result.config.get('bm25_index_chunks')}",
        f"- **Seed:** {result.config.get('seed')}",
        "",
        f"## Leaderboard (nDCG@{primary_k})",
        "",
        "| Rank | Strategy | MRR | nDCG@K | Recall@K | Precision@K | Latency ms (mean) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(ranked, start=1):
        lines.append(
            f"| {index} | `{row.strategy}` | {row.mrr_mean:.4f} | "
            f"{row.ndcg_at_k.get(primary_k, 0.0):.4f} | "
            f"{row.recall_at_k.get(primary_k, 0.0):.4f} | "
            f"{row.precision_at_k.get(primary_k, 0.0):.4f} | "
            f"{row.latency_ms_mean:.2f} |"
        )

    lines.extend(["", "## Metrics by strategy", ""])
    for row in result.strategies:
        lines.append(f"### `{row.strategy}`")
        lines.append(f"- MRR: **{row.mrr_mean:.4f}**")
        lines.append(f"- Latency mean / p95: **{row.latency_ms_mean:.2f}** / **{row.latency_ms_p95:.2f}** ms")
        for k in sorted(row.recall_at_k):
            lines.append(
                f"- @{k}: recall={row.recall_at_k[k]:.4f}, "
                f"precision={row.precision_at_k[k]:.4f}, ndcg={row.ndcg_at_k[k]:.4f}"
            )
        if row.answer_faithfulness_mean is not None:
            lines.append(
                f"- Answers: faithfulness={row.answer_faithfulness_mean:.4f}, "
                f"relevance={row.answer_relevance_mean:.4f}, "
                f"citation correctness={row.citation_correctness_mean:.4f}, "
                f"completeness={row.citation_completeness_mean:.4f}"
            )
        lines.append("")

    lines.extend(
        [
            "## Reproducibility",
            "",
            "Re-run with the same dataset and `config.json` from this output directory. "
            "Metrics depend on the processed corpus, embedding model, and reranker weights "
            "captured in `config.json` and `report.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_evaluation_reports(result: EvaluationRunResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(render_markdown_report(result), encoding="utf-8")
