#!/usr/bin/env python3
"""
Run reproducible retrieval evaluation across BM25, dense, hybrid RRF, hybrid+rerank, and agentic retrieval.

Usage (from repo root):
  cd backend && python ../scripts/run_retrieval_eval.py \\
    --dataset ../data/evaluation/sample_dataset.json \\
    --output ../data/evaluation/reports/latest \\
    --k 1 3 5 10 \\
    --seed 42

Optional answer-quality metrics (requires configured LLM):
  cd backend && python ../scripts/run_retrieval_eval.py \\
    --dataset ../data/evaluation/sample_dataset.json \\
    --output ../data/evaluation/reports/with_answers \\
    --evaluate-answers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.evaluation.runner import EvaluationRunConfig, RetrievalEvaluationRunner
from app.evaluation.strategies import RetrievalStrategyName


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval strategies on a labeled dataset")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/sample_dataset.json"),
        help="Path to evaluation dataset JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/reports/latest"),
        help="Output directory for report.json, report.md, and config.json",
    )
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10], help="K values for @K metrics")
    parser.add_argument("--max-k", type=int, default=10, help="Maximum retrieved depth per query")
    parser.add_argument("--seed", type=int, default=42, help="Recorded in config for reproducibility")
    parser.add_argument(
        "--strategy",
        action="append",
        dest="strategies",
        choices=[s.value for s in RetrievalStrategyName],
        help="Limit to specific strategies (repeatable)",
    )
    parser.add_argument("--use-remote-qdrant", action="store_true")
    parser.add_argument("--skip-reranker", action="store_true", help="Skip hybrid+rerank and agentic rerank stage")
    parser.add_argument(
        "--evaluate-answers",
        action="store_true",
        help="Generate grounded answers for hybrid_rerank and agentic (requires LLM)",
    )
    args = parser.parse_args()

    dataset_path = args.dataset
    if not dataset_path.is_file():
        # Allow running from backend/ with ../data paths
        alt = Path.cwd().parent / args.dataset
        if alt.is_file():
            dataset_path = alt

    if not dataset_path.is_file():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    strategies = None
    if args.strategies:
        strategies = [RetrievalStrategyName(value) for value in args.strategies]

    config = EvaluationRunConfig(
        dataset_path=dataset_path.resolve(),
        output_dir=args.output.resolve(),
        k_values=args.k,
        max_k=args.max_k,
        strategies=strategies,
        use_remote_qdrant=args.use_remote_qdrant,
        include_reranker=not args.skip_reranker,
        evaluate_answers=args.evaluate_answers,
        seed=args.seed,
    )
    result = RetrievalEvaluationRunner(config).run()
    print(f"Wrote reports to {config.output_dir}")
    print(f"Dataset hash: {result.dataset_hash}")
    if result.strategies:
        best = max(result.strategies, key=lambda row: row.ndcg_at_k.get(max(args.k), 0.0))
        print(f"Best nDCG@{max(args.k)}: {best.strategy} ({best.ndcg_at_k.get(max(args.k), 0.0):.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
