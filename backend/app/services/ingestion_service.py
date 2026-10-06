import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

from docling.document_converter import DocumentConverter

from app.ingestion.chunker import chunk_document
from app.ingestion.docling_extractor import extract_pdf_with_docling
from app.ingestion.text_extractor import build_document_from_text
from app.models.chunking import ChunkingConfig
from app.models.document import Chunk, Document
from app.models.knowledge_source import DocumentKind, KnowledgeScope
from app.services.vector_index_service import VectorIndexService

logger = logging.getLogger(__name__)


class IngestionService:
    """PDF ingestion: Docling extraction, chunking, and persistence under data/processed."""

    def __init__(
        self,
        processed_dir: Path,
        raw_dir: Path | None = None,
        chunking: ChunkingConfig | None = None,
        converter: DocumentConverter | None = None,
        vector_indexer: VectorIndexService | None = None,
    ) -> None:
        self.processed_dir = processed_dir
        self.raw_dir = raw_dir
        self.chunking = chunking or ChunkingConfig()
        self._converter = converter
        self.vector_indexer = vector_indexer
        self._path: Path | None = None
        self._source_filename: str | None = None
        self._document: Document | None = None
        self._chunks: list[Chunk] = []

    def ingest_document(
        self,
        path: Path,
        *,
        source_filename: str | None = None,
        knowledge_scope: KnowledgeScope = KnowledgeScope.PUBLIC_RESEARCH,
        document_kind: DocumentKind = DocumentKind.PDF,
        project_id: str | None = None,
        tags: list[str] | None = None,
    ) -> Document:
        self._path = Path(path).resolve()
        self._source_filename = source_filename
        self._knowledge_scope = knowledge_scope
        self._document_kind = document_kind
        self._project_id = project_id
        self._tags = tags
        document = self.extract_content()
        self._chunks = self.chunk_document(document)
        document.chunks = self._chunks
        self.save_processed_document(document, self._chunks)
        if self.vector_indexer and self._chunks:
            self.vector_indexer.index_chunks(self._chunks)
        self._document = document
        return document

    def ingest_text_document(
        self,
        *,
        title: str,
        text: str,
        filename: str,
        knowledge_scope: KnowledgeScope,
        document_kind: DocumentKind,
        project_id: str | None = None,
        tags: list[str] | None = None,
    ) -> Document:
        document = build_document_from_text(
            title=title,
            filename=filename,
            text=text,
            knowledge_scope=knowledge_scope,
            document_kind=document_kind,
            project_id=project_id,
            tags=tags,
        )
        self._path = None
        self._chunks = self.chunk_document(document)
        document.chunks = self._chunks
        self.save_processed_document(document, self._chunks)
        if self.vector_indexer and self._chunks:
            self.vector_indexer.index_chunks(self._chunks)
        self._document = document
        return document

    def extract_content(self) -> Document:
        if self._path is None:
            raise RuntimeError("Call ingest_document(path) or set _path before extract_content()")
        document = extract_pdf_with_docling(
            self._path,
            source_filename=self._source_filename,
            converter=self._converter,
            ingestion_timestamp=datetime.now(UTC),
        )
        document.knowledge_scope = getattr(self, "_knowledge_scope", KnowledgeScope.PUBLIC_RESEARCH)
        document.document_kind = getattr(self, "_document_kind", DocumentKind.PDF)
        document.project_id = getattr(self, "_project_id", None)
        document.tags = list(getattr(self, "_tags", None) or [])
        return document

    def chunk_document(self, document: Document) -> list[Chunk]:
        return chunk_document(document, self.chunking)

    def save_processed_document(self, document: Document, chunks: list[Chunk]) -> Path:
        out_dir = self.processed_dir / document.document_id
        out_dir.mkdir(parents=True, exist_ok=True)

        document_path = out_dir / "document.json"
        chunks_path = out_dir / "chunks.json"
        markdown_path = out_dir / "content.md"

        document_path.write_text(
            document.model_dump_json(indent=2),
            encoding="utf-8",
        )
        chunks_path.write_text(
            json.dumps([c.model_dump(mode="json") for c in chunks], indent=2),
            encoding="utf-8",
        )
        markdown_path.write_text(document.markdown, encoding="utf-8")

        if self.raw_dir and self._path and self._path.exists():
            raw_target = self.raw_dir / f"{document.document_id}_{document.filename}"
            if self._path != raw_target:
                shutil.copy2(self._path, raw_target)

        logger.info(
            "Saved processed document %s (%d chunks) to %s",
            document.document_id,
            len(chunks),
            out_dir,
        )
        return out_dir
