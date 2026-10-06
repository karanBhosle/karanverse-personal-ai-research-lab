#!/usr/bin/env python3
"""Helper to discover chunk_id labels for evaluation datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.config.settings import get_settings
from app.evaluation.infrastructure import build_retrieval_backend


def main() -> int:
    parser = argparse.ArgumentParser(description="List top retrieved chunk IDs for labeling eval datasets")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--use-remote-qdrant", action="store_true")
    parser.add_argument("--skip-reranker", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    backend = build_retrieval_backend(
        settings,
        use_remote_qdrant=args.use_remote_qdrant,
        include_reranker=not args.skip_reranker,
    )
    hits = backend.hybrid.retrieve(args.query, top_k=args.top_k)
    for index, hit in enumerate(hits, start=1):
        filename = (hit.metadata or {}).get("filename", "")
        preview = hit.text.replace("\n", " ")[:100]
        print(f"{index}. chunk_id={hit.chunk_id} document_id={hit.document_id} filename={filename}")
        print(f"   {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
