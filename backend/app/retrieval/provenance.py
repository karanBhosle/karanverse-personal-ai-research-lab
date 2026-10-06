from app.models.document import Chunk


def chunk_metadata(chunk: Chunk) -> dict:
    return {
        "filename": chunk.filename,
        "page_number": chunk.page_number,
        "section": chunk.section,
        "source_type": chunk.source_type.value,
        "knowledge_scope": chunk.knowledge_scope.value,
        "document_kind": chunk.document_kind.value,
        "project_id": chunk.project_id,
        "tags": list(chunk.tags),
        "ingestion_timestamp": chunk.ingestion_timestamp.isoformat(),
    }


def chunk_payload(chunk: Chunk) -> dict:
    """Qdrant payload: provenance plus text for retrieval display."""
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "text": chunk.text,
        **chunk_metadata(chunk),
    }
