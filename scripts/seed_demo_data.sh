#!/usr/bin/env bash
# Seed local demo data for README screenshots (portfolio → processed chunks).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export QDRANT_INDEXING_ENABLED="${QDRANT_INDEXING_ENABLED:-false}"

python - <<'PY'
from app.config.settings import get_settings
from app.models.chunking import ChunkingConfig
from app.services.ingestion_service import IngestionService
from app.services.portfolio_import_service import import_portfolio_seed

get_settings.cache_clear()
s = get_settings()
ing = IngestionService(
    processed_dir=s.data_processed_dir,
    raw_dir=s.data_raw_dir,
    chunking=ChunkingConfig(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        min_chunk_size=s.chunk_min_size,
    ),
    vector_indexer=None,
)
ids = import_portfolio_seed(ing)
print(f"Imported {len(ids)} portfolio documents into {s.data_processed_dir}")
PY

if curl -sf -o /dev/null http://127.0.0.1:8000/health 2>/dev/null; then
  echo "API is up — you can also run: curl -X POST http://localhost:8000/knowledge/import/portfolio"
fi
