import logging

from app.models.document import Chunk
from app.retrieval.qdrant_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class VectorIndexService:
    """Indexes document chunks into Qdrant."""

    def __init__(self, store: QdrantVectorStore, enabled: bool = True) -> None:
        self.store = store
        self.enabled = enabled

    def index_chunks(self, chunks: list[Chunk]) -> int:
        if not self.enabled:
            logger.debug("Vector indexing disabled; skipping %d chunks", len(chunks))
            return 0
        try:
            return self.store.upsert_chunks(chunks)
        except Exception:
            logger.exception("Failed to index chunks into Qdrant")
            raise
