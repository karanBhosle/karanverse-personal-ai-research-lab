import json
import logging
from pathlib import Path

from app.models.document import Chunk
from app.security.identifiers import validate_safe_identifier
from app.security.paths import resolve_path_under

logger = logging.getLogger(__name__)


class ChunkResolutionError(Exception):
    """Raised when a chunk cannot be resolved to on-disk processed storage."""


def load_chunk_from_processed(processed_dir: Path, document_id: str, chunk_id: str) -> Chunk:
    """Load a single chunk from data/processed/<document_id>/chunks.json."""
    validate_safe_identifier(document_id, field="document_id")
    validate_safe_identifier(chunk_id, field="chunk_id")
    chunks_path = resolve_path_under(processed_dir, document_id, "chunks.json")
    if not chunks_path.is_file():
        raise ChunkResolutionError(f"Missing chunks file: {chunks_path}")

    try:
        raw = json.loads(chunks_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ChunkResolutionError(f"Invalid chunks JSON at {chunks_path}") from exc

    for item in raw:
        if item.get("chunk_id") == chunk_id:
            return Chunk.model_validate(item)

    raise ChunkResolutionError(
        f"chunk_id {chunk_id} not found in document {document_id} ({chunks_path})"
    )


def load_document_title(processed_dir: Path, document_id: str) -> str | None:
    validate_safe_identifier(document_id, field="document_id")
    document_path = resolve_path_under(processed_dir, document_id, "document.json")
    if not document_path.is_file():
        return None
    try:
        payload = json.loads(document_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    title = payload.get("title")
    return str(title) if title else None
