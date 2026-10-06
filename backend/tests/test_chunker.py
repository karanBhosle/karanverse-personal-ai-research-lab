from datetime import UTC, datetime

from app.ingestion.chunker import chunk_document
from app.models.chunking import ChunkingConfig
from app.models.document import BlockType, Document, DocumentBlock, SourceType


def test_chunk_document_preserves_provenance_fields():
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    document = Document(
        document_id="doc-1",
        title="Test",
        filename="test.pdf",
        source_type=SourceType.PDF,
        markdown="# Test",
        ingestion_timestamp=ts,
        blocks=[
            DocumentBlock(
                block_id="b1",
                block_type=BlockType.HEADING,
                text="Introduction",
                page_number=1,
                section="Introduction",
            ),
            DocumentBlock(
                block_id="b2",
                block_type=BlockType.PARAGRAPH,
                text="First paragraph with enough characters to pass the minimum chunk size threshold.",
                page_number=1,
                section="Introduction",
            ),
        ],
    )
    config = ChunkingConfig(chunk_size=500, chunk_overlap=0, min_chunk_size=20)
    chunks = chunk_document(document, config)

    assert len(chunks) >= 1
    chunk = chunks[0]
    assert chunk.document_id == "doc-1"
    assert chunk.filename == "test.pdf"
    assert chunk.page_number == 1
    assert chunk.section == "Introduction"
    assert chunk.source_type == SourceType.PDF
    assert chunk.ingestion_timestamp == ts
    assert chunk.chunk_id
