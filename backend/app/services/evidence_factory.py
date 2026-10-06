import uuid
from pathlib import Path

from app.models.evidence import Evidence, RetrievalMethod, SourceMetadata
from app.models.search import HybridSearchResult, RerankedSearchResult, SearchResult
from app.services.evidence_resolver import load_document_title


def _source_metadata_from_dict(metadata: dict, processed_dir: Path | None, document_id: str) -> SourceMetadata:
    filename = str(metadata.get("filename") or "")
    return SourceMetadata(
        filename=filename,
        source_type=str(metadata.get("source_type") or "unknown"),
        knowledge_scope=metadata.get("knowledge_scope"),
        document_kind=metadata.get("document_kind"),
        project_id=metadata.get("project_id"),
        ingestion_timestamp=metadata.get("ingestion_timestamp"),
        processed_path=str(processed_dir / document_id) if processed_dir else None,
    )


def _resolve_title(metadata: dict, document_id: str, processed_dir: Path | None) -> str:
    if processed_dir:
        title = load_document_title(processed_dir, document_id)
        if title:
            return title
    filename = metadata.get("filename")
    if filename:
        return Path(str(filename)).stem
    return document_id


def evidence_from_search_result(
    result: SearchResult,
    *,
    method: RetrievalMethod,
    processed_dir: Path | None = None,
    title: str | None = None,
) -> Evidence:
    metadata = result.metadata or {}
    doc_title = title or _resolve_title(metadata, result.document_id, processed_dir)
    source_name = str(metadata.get("filename") or doc_title)
    return Evidence(
        evidence_id=str(uuid.uuid4()),
        document_id=result.document_id,
        chunk_id=result.chunk_id,
        source=source_name,
        title=doc_title,
        page=metadata.get("page_number"),
        section=metadata.get("section"),
        text=result.text,
        retrieval_score=result.score,
        retrieval_method=method,
        source_metadata=_source_metadata_from_dict(metadata, processed_dir, result.document_id),
    )


def evidence_from_hybrid_result(
    result: HybridSearchResult,
    *,
    processed_dir: Path | None = None,
    title: str | None = None,
) -> Evidence:
    metadata = result.metadata or {}
    doc_title = title or _resolve_title(metadata, result.document_id, processed_dir)
    source_name = str(metadata.get("filename") or doc_title)
    return Evidence(
        evidence_id=str(uuid.uuid4()),
        document_id=result.document_id,
        chunk_id=result.chunk_id,
        source=source_name,
        title=doc_title,
        page=metadata.get("page_number"),
        section=metadata.get("section"),
        text=result.text,
        retrieval_score=result.final_score,
        retrieval_method=RetrievalMethod.HYBRID,
        source_metadata=_source_metadata_from_dict(metadata, processed_dir, result.document_id),
    )


def evidence_from_reranked_result(
    result: RerankedSearchResult,
    *,
    processed_dir: Path | None = None,
    title: str | None = None,
) -> Evidence:
    metadata = result.metadata or {}
    doc_title = title or _resolve_title(metadata, result.document_id, processed_dir)
    source_name = str(metadata.get("filename") or doc_title)
    return Evidence(
        evidence_id=str(uuid.uuid4()),
        document_id=result.document_id,
        chunk_id=result.chunk_id,
        source=source_name,
        title=doc_title,
        page=metadata.get("page_number"),
        section=metadata.get("section"),
        text=result.text,
        retrieval_score=result.rerank_score,
        retrieval_method=RetrievalMethod.HYBRID_RERANK,
        source_metadata=_source_metadata_from_dict(metadata, processed_dir, result.document_id),
    )
