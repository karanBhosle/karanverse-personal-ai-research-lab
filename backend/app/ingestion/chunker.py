import uuid
from datetime import datetime

from app.models.chunking import ChunkingConfig
from app.models.document import Chunk, Document, DocumentBlock, SourceType


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        pieces.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def chunk_document(document: Document, config: ChunkingConfig) -> list[Chunk]:
    """Build chunks from document blocks with configurable size and overlap."""
    chunks: list[Chunk] = []
    buffer = ""
    buffer_page: int | None = None
    buffer_section: str | None = None

    def flush_buffer() -> None:
        nonlocal buffer, buffer_page, buffer_section
        text = buffer.strip()
        if len(text) < config.min_chunk_size:
            buffer = ""
            return
        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document.document_id,
                filename=document.filename,
                page_number=buffer_page,
                section=buffer_section,
                text=text,
                source_type=document.source_type,
                knowledge_scope=document.knowledge_scope,
                document_kind=document.document_kind,
                project_id=document.project_id,
                tags=list(document.tags),
                ingestion_timestamp=document.ingestion_timestamp,
            )
        )
        if config.chunk_overlap > 0 and len(text) > config.chunk_overlap:
            buffer = text[-config.chunk_overlap :]
        else:
            buffer = ""
        buffer_page = buffer_page if buffer else None
        buffer_section = buffer_section if buffer else None

    def append_block(block: DocumentBlock) -> None:
        nonlocal buffer, buffer_page, buffer_section
        piece = block.text.strip()
        if not piece:
            return

        if buffer_page is None:
            buffer_page = block.page_number
        if buffer_section is None:
            buffer_section = block.section

        candidate = f"{buffer}\n\n{piece}".strip() if buffer else piece
        if len(candidate) <= config.chunk_size:
            buffer = candidate
            return

        if buffer:
            flush_buffer()
            buffer_page = block.page_number
            buffer_section = block.section

        if len(piece) > config.chunk_size:
            for segment in _split_long_text(piece, config.chunk_size, config.chunk_overlap):
                buffer = segment
                buffer_page = block.page_number
                buffer_section = block.section
                flush_buffer()
        else:
            buffer = piece

    for block in document.blocks:
        append_block(block)

    if buffer.strip():
        flush_buffer()

    return chunks
