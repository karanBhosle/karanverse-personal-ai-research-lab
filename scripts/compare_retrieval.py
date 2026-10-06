#!/usr/bin/env python3
"""
Legacy qualitative retrieval comparison (per-query hit lists).

For labeled metrics (Recall@K, MRR, nDCG, latency, leaderboard reports), use:
  python ../scripts/run_retrieval_eval.py --dataset ../data/evaluation/sample_dataset.json

Usage (from repo root):
  cd backend && python ../scripts/compare_retrieval.py --output ../data/research/retrieval_comparison.md
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.config.paths import get_processed_dir
from app.config.settings import get_settings
from app.llm.embedding import create_embedding_service
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.chunk_store import load_chunks_from_processed
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import create_cross_encoder_reranker
from app.retrieval.reranking_hybrid_retriever import RerankingHybridRetriever
from app.retrieval.qdrant_retriever import QdrantRetriever, create_qdrant_client, get_qdrant_retriever
from app.retrieval.qdrant_store import QdrantVectorStore

DEFAULT_QUERIES = [
    "bm25 hybrid retrieval",
    "dense vectors reciprocal rank fusion",
    "Hybrid Retrieval Notes",
    "page numbers section metadata",
    "research intelligence platform",
]


def _build_vector_retriever(use_remote_qdrant: bool) -> QdrantRetriever:
    settings = get_settings()
    if use_remote_qdrant:
        return get_qdrant_retriever()

    embedding = create_embedding_service(
        model_name=settings.embedding_model,
        vector_dimension=settings.vector_dimension,
    )
    client = create_qdrant_client(":memory:")
    store = QdrantVectorStore(
        client=client,
        collection_name=f"{settings.qdrant_collection}_eval",
        embedding_service=embedding,
        vector_dimension=settings.vector_dimension,
    )
    chunks = load_chunks_from_processed(settings.data_processed_dir)
    if chunks:
        store.upsert_chunks(chunks)
    return QdrantRetriever(
        store=store,
        embedding_service=embedding,
        default_top_k=settings.vector_search_top_k,
    )


def _format_hits(results, score_attr: str = "score") -> str:
    if not results:
        return "_no results_"
    lines = []
    for idx, hit in enumerate(results, start=1):
        score = getattr(hit, score_attr)
        preview = hit.text.replace("\n", " ")[:80]
        lines.append(f"{idx}. `{hit.chunk_id[:8]}…` score={score:.4f} — {preview}")
    return "\n".join(lines)


def run_comparison(
    queries: list[str],
    top_k: int,
    use_remote_qdrant: bool,
    *,
    include_reranker: bool,
) -> tuple[str, dict]:
    settings = get_settings()
    bm25 = BM25Retriever(settings.bm25_index_dir, settings.data_processed_dir)
    bm25.ensure_index()
    vector = _build_vector_retriever(use_remote_qdrant)
    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        vector_retriever=vector,
        rrf_k=settings.rrf_k,
        bm25_weight=settings.hybrid_bm25_weight,
        vector_weight=settings.hybrid_vector_weight,
        candidate_pool=max(settings.hybrid_candidate_pool, settings.rerank_candidate_count),
    )
    rerank_pipeline: RerankingHybridRetriever | None = None
    if include_reranker:
        rerank_pipeline = RerankingHybridRetriever(
            hybrid_retriever=hybrid,
            reranker=create_cross_encoder_reranker(settings.reranker_model),
            candidate_count=max(top_k, settings.rerank_candidate_count),
            final_top_k=top_k,
        )

    latencies: dict[str, list[float]] = {
        "bm25": [],
        "vector": [],
        "hybrid": [],
        "hybrid_rerank": [],
    }
    sections: list[str] = []

    for query in queries:
        sections.append(f"## Query: `{query}`\n")

        t0 = time.perf_counter()
        bm25_hits = bm25.retrieve(query, top_k=top_k)
        latencies["bm25"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        vector_hits = vector.retrieve(query, top_k=top_k)
        latencies["vector"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        hybrid_hits = hybrid.retrieve(query, top_k=top_k)
        latencies["hybrid"].append((time.perf_counter() - t0) * 1000)

        rerank_hits = []
        if rerank_pipeline is not None:
            t0 = time.perf_counter()
            rerank_hits = rerank_pipeline.retrieve(
                query,
                final_top_k=top_k,
                candidate_count=max(top_k, settings.rerank_candidate_count),
            )
            latencies["hybrid_rerank"].append((time.perf_counter() - t0) * 1000)

        sections.append("### BM25\n" + _format_hits(bm25_hits) + "\n")
        sections.append("### Dense vector\n" + _format_hits(vector_hits) + "\n")
        sections.append("### Hybrid (RRF)\n" + _format_hits(hybrid_hits, "final_score") + "\n")
        if rerank_pipeline is not None:
            sections.append(
                "### Hybrid + Reranker\n" + _format_hits(rerank_hits, "rerank_score") + "\n"
            )
            if rerank_hits:
                top_r = rerank_hits[0]
                sections.append(
                    f"Top reranked: rerank_score={top_r.rerank_score:.4f}, "
                    f"hybrid_score={top_r.hybrid_score:.5f}\n"
                )

        if hybrid_hits:
            top = hybrid_hits[0]
            sections.append(
                "Top hybrid provenance: "
                f"bm25_rank={top.rank_contribution.bm25_rank}, "
                f"vector_rank={top.rank_contribution.vector_rank}, "
                f"bm25_rrf={top.rank_contribution.bm25_rrf:.5f}, "
                f"vector_rrf={top.rank_contribution.vector_rrf:.5f}\n"
            )

    summary = {
        "queries": len(queries),
        "top_k": top_k,
        "bm25_index_chunks": bm25.chunk_count,
        "vector_backend": "remote_qdrant" if use_remote_qdrant else "in_memory_from_processed",
        "latency_ms_mean": {
            k: statistics.mean(v) if v else 0.0 for k, v in latencies.items() if v or k != "hybrid_rerank"
        },
        "reranker_model": settings.reranker_model if include_reranker else None,
        "rerank_candidate_count": settings.rerank_candidate_count if include_reranker else None,
    }
    return "\n".join(sections), summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare retrieval modes (BM25 / vector / hybrid)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("data/research/retrieval_comparison.md"))
    parser.add_argument("--query", action="append", dest="queries", help="Repeatable query override")
    parser.add_argument(
        "--use-remote-qdrant",
        action="store_true",
        help="Use QDRANT_URL instead of an ephemeral in-memory index built from processed chunks",
    )
    parser.add_argument(
        "--skip-reranker",
        action="store_true",
        help="Skip hybrid+cross-encoder mode (avoids loading the reranker model)",
    )
    args = parser.parse_args()

    queries = args.queries or DEFAULT_QUERIES
    body, summary = run_comparison(
        queries,
        top_k=args.top_k,
        use_remote_qdrant=args.use_remote_qdrant,
        include_reranker=not args.skip_reranker,
    )

    processed = get_processed_dir()
    report = "\n".join(
        [
            "# Retrieval comparison report",
            "",
            f"- Generated: {datetime.now(UTC).isoformat()}",
            f"- Processed corpus: `{processed}`",
            f"- Vector backend: `{summary['vector_backend']}`",
            f"- Queries: {summary['queries']}",
            f"- top_k: {summary['top_k']}",
            f"- BM25 index chunks: {summary['bm25_index_chunks']}",
            "",
            "## Latency (mean ms)",
            "",
            f"- BM25: {summary['latency_ms_mean']['bm25']:.3f}",
            f"- Dense: {summary['latency_ms_mean']['vector']:.3f}",
            f"- Hybrid: {summary['latency_ms_mean']['hybrid']:.3f}",
            *(
                [f"- Hybrid + Reranker: {summary['latency_ms_mean'].get('hybrid_rerank', 0.0):.3f}"]
                if summary.get("reranker_model")
                else []
            ),
            "",
            "## Notes",
            "",
            "- Hybrid uses weighted RRF over BM25 and dense ranks (`RRF_K`, `HYBRID_*_WEIGHT`).",
            "- RRF scores are rank-based; raw `bm25_score` / `vector_score` are preserved on hybrid hits.",
            *(
                [
                    f"- Reranker: `{summary['reranker_model']}` "
                    f"(candidates={summary['rerank_candidate_count']}).",
                ]
                if summary.get("reranker_model")
                else []
            ),
            "",
            "---",
            "",
            body,
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote report to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
