import hashlib
import heapq
import json
import logging
import pickle
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25L

from app.config.settings import get_settings
from app.models.document import Chunk
from app.models.knowledge_source import KnowledgeScope
from app.models.search import SearchResult
from app.retrieval.knowledge_scope import chunk_matches_scopes
from app.retrieval.chunk_store import load_chunks_from_processed
from app.retrieval.provenance import chunk_metadata as _chunk_metadata
from app.retrieval.tokenizer import tokenize

logger = logging.getLogger(__name__)

INDEX_FILENAME = "bm25_index.pkl"
META_FILENAME = "meta.json"


def processed_corpus_fingerprint(processed_dir: Path) -> str:
    parts: list[str] = []
    if processed_dir.exists():
        for chunks_path in sorted(processed_dir.glob("*/chunks.json")):
            stat = chunks_path.stat()
            parts.append(f"{chunks_path.parent.name}:{stat.st_mtime_ns}:{stat.st_size}")
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest


class BM25Retriever:
    """Persistent BM25 retriever over ingested document chunks."""

    def __init__(self, index_dir: Path, processed_dir: Path) -> None:
        self.index_dir = index_dir
        self.processed_dir = processed_dir
        self._bm25: BM25L | None = None
        self._chunks: list[Chunk] = []

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def index_path(self) -> Path:
        return self.index_dir / INDEX_FILENAME

    def meta_path(self) -> Path:
        return self.index_dir / META_FILENAME

    def is_index_on_disk(self) -> bool:
        return self.index_path().is_file() and self.meta_path().is_file()

    def load(self) -> None:
        if not self.is_index_on_disk():
            raise FileNotFoundError(f"BM25 index not found under {self.index_dir}")
        with self.index_path().open("rb") as handle:
            payload = pickle.load(handle)
        self._bm25 = payload["bm25"]
        self._chunks = [Chunk.model_validate(item) for item in payload["chunks"]]
        logger.info("Loaded BM25 index (%d chunks) from %s", len(self._chunks), self.index_dir)

    def build_from_processed(self) -> int:
        chunks = load_chunks_from_processed(self.processed_dir)
        return self.build_from_chunks(chunks)

    def build_from_chunks(self, chunks: list[Chunk]) -> int:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if not chunks:
            self._bm25 = None
            self._chunks = []
            self._persist(fingerprint=processed_corpus_fingerprint(self.processed_dir))
            return 0

        tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
        # BM25L behaves better than Okapi on very small corpora (typical during early indexing).
        self._bm25 = BM25L(tokenized_corpus)
        self._chunks = chunks
        self._persist(fingerprint=processed_corpus_fingerprint(self.processed_dir))
        logger.info("Built BM25 index with %d chunks at %s", len(chunks), self.index_dir)
        return len(chunks)

    def _persist(self, fingerprint: str) -> None:
        with self.index_path().open("wb") as handle:
            pickle.dump(
                {
                    "bm25": self._bm25,
                    "chunks": [chunk.model_dump(mode="json") for chunk in self._chunks],
                },
                handle,
            )
        meta = {
            "built_at": datetime.now(UTC).isoformat(),
            "chunk_count": len(self._chunks),
            "corpus_fingerprint": fingerprint,
            "algorithm": "BM25L",
        }
        self.meta_path().write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def ensure_index(self, *, force_rebuild: bool = False) -> None:
        current_fp = processed_corpus_fingerprint(self.processed_dir)
        if force_rebuild:
            self.build_from_processed()
            return

        if self.is_index_on_disk():
            meta = json.loads(self.meta_path().read_text(encoding="utf-8"))
            if meta.get("corpus_fingerprint") == current_fp:
                if self._bm25 is None:
                    self.load()
                return

        self.build_from_processed()

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        knowledge_scopes: set[KnowledgeScope] | None = None,
    ) -> list[SearchResult]:
        if top_k < 1:
            return []
        self.ensure_index()
        if self._bm25 is None or not self._chunks:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        pool = top_k * 5 if knowledge_scopes else top_k
        ranked = heapq.nlargest(pool, enumerate(scores), key=lambda pair: pair[1])

        results: list[SearchResult] = []
        for idx, score in ranked:
            if score == 0:
                continue
            chunk = self._chunks[idx]
            metadata = _chunk_metadata(chunk)
            if knowledge_scopes and not chunk_matches_scopes(metadata, knowledge_scopes):
                continue
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=float(score),
                    metadata=metadata,
                )
            )
            if len(results) >= top_k:
                break
        return results


@lru_cache
def get_bm25_retriever() -> BM25Retriever:
    settings = get_settings()
    return BM25Retriever(
        index_dir=settings.bm25_index_dir,
        processed_dir=settings.data_processed_dir,
    )


def create_bm25_retriever(
    index_dir: Path,
    processed_dir: Path,
) -> BM25Retriever:
    return BM25Retriever(index_dir=index_dir, processed_dir=processed_dir)
