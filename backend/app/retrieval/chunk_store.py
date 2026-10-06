import json
import logging
from pathlib import Path

from app.models.document import Chunk

logger = logging.getLogger(__name__)


def load_chunks_from_processed(processed_dir: Path) -> list[Chunk]:
    """Load all chunk records from data/processed/<document_id>/chunks.json."""
    if not processed_dir.exists():
        return []

    chunks: list[Chunk] = []
    for doc_dir in sorted(processed_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        chunks_path = doc_dir / "chunks.json"
        if not chunks_path.is_file():
            continue
        try:
            raw = json.loads(chunks_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Skipping invalid chunks file: %s", chunks_path)
            continue
        for item in raw:
            chunks.append(Chunk.model_validate(item))
    return chunks
