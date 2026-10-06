import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.models.document import BlockType, Document, DocumentBlock, SourceType
from app.models.knowledge_source import DocumentKind, KnowledgeScope


def _infer_source_type(kind: DocumentKind) -> SourceType:
    if kind == DocumentKind.MARKDOWN:
        return SourceType.MARKDOWN
    if kind == DocumentKind.RESEARCH_REPORT:
        return SourceType.RESEARCH_REPORT
    return SourceType.TEXT


def build_document_from_text(
    *,
    title: str,
    filename: str,
    text: str,
    knowledge_scope: KnowledgeScope,
    document_kind: DocumentKind,
    project_id: str | None = None,
    tags: list[str] | None = None,
    ingestion_timestamp: datetime | None = None,
) -> Document:
    ts = ingestion_timestamp or datetime.now(UTC)
    blocks: list[DocumentBlock] = []
    current_section: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        block_type = BlockType.PARAGRAPH
        section = current_section
        if stripped.startswith("#"):
            block_type = BlockType.HEADING
            current_section = stripped.lstrip("#").strip()
            section = current_section
        blocks.append(
            DocumentBlock(
                block_id=str(uuid.uuid4()),
                block_type=block_type,
                text=stripped,
                page_number=None,
                section=section,
                heading_level=len(stripped) - len(stripped.lstrip("#")) if block_type == BlockType.HEADING else None,
            )
        )

    return Document(
        document_id=str(uuid.uuid4()),
        title=title,
        filename=filename,
        source_type=_infer_source_type(document_kind),
        knowledge_scope=knowledge_scope,
        document_kind=document_kind,
        project_id=project_id,
        tags=list(tags or []),
        markdown=text,
        blocks=blocks,
        ingestion_timestamp=ts,
        page_count=None,
    )


def build_document_from_path(
    path: Path,
    *,
    knowledge_scope: KnowledgeScope,
    document_kind: DocumentKind,
    project_id: str | None = None,
    tags: list[str] | None = None,
) -> Document:
    text = path.read_text(encoding="utf-8")
    title = path.stem.replace("_", " ").replace("-", " ").title()
    return build_document_from_text(
        title=title,
        filename=path.name,
        text=text,
        knowledge_scope=knowledge_scope,
        document_kind=document_kind,
        project_id=project_id,
        tags=tags,
    )
