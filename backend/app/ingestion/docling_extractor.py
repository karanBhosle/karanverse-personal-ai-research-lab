import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import DoclingDocument

from app.config import get_settings
from app.models.document import BlockType, Document, DocumentBlock, SourceType

logger = logging.getLogger(__name__)

_HEADING_TYPE_NAMES = frozenset({"SectionHeaderItem", "TitleItem"})


def _page_number(item: object) -> int | None:
    prov = getattr(item, "prov", None)
    if prov:
        return prov[0].page_no
    return None


def _item_text(item: object) -> str:
    text = getattr(item, "text", None)
    if text:
        return str(text).strip()
    export_md = getattr(item, "export_to_markdown", None)
    if callable(export_md):
        return export_md().strip()
    return ""


def _block_type_for_item(item: object) -> BlockType:
    type_name = type(item).__name__
    if type_name == "TableItem":
        return BlockType.TABLE
    if type_name in _HEADING_TYPE_NAMES:
        return BlockType.HEADING
    label = getattr(item, "label", None)
    if label is not None and str(label).lower() in {"section_header", "title"}:
        return BlockType.HEADING
    return BlockType.PARAGRAPH


def _heading_level(item: object) -> int | None:
    level = getattr(item, "level", None)
    if level is not None:
        return int(level)
    return None


def _infer_title(blocks: list[DocumentBlock], filename: str, markdown: str) -> str:
    for block in blocks:
        if block.block_type == BlockType.HEADING and block.text:
            return block.text
    first_line = next((line.lstrip("#").strip() for line in markdown.splitlines() if line.strip()), "")
    if first_line:
        return first_line
    return Path(filename).stem


def _page_count(doc: DoclingDocument) -> int | None:
    pages = getattr(doc, "pages", None)
    if pages is not None:
        return len(pages)
    return None


def extract_pdf_with_docling(
    path: Path,
    *,
    source_filename: str | None = None,
    converter: DocumentConverter | None = None,
    ingestion_timestamp: datetime | None = None,
) -> Document:
    """Convert a PDF file into a normalized Document using Docling."""
    path = path.resolve()
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF ingestion is supported; got {path.suffix}")

    converter = converter or DocumentConverter()
    logger.info("Converting PDF with Docling: %s", path.name)
    result = converter.convert(str(path))
    dl_doc = result.document
    page_count = _page_count(dl_doc)
    max_pages = get_settings().pdf_max_pages
    if page_count is not None and page_count > max_pages:
        raise ValueError(f"PDF exceeds maximum page count of {max_pages}")

    markdown = dl_doc.export_to_markdown()
    blocks: list[DocumentBlock] = []
    current_section: str | None = None

    for item, _level in dl_doc.iterate_items():
        text = _item_text(item)
        if not text:
            continue

        block_type = _block_type_for_item(item)
        page_number = _page_number(item)

        if block_type == BlockType.HEADING:
            current_section = text

        blocks.append(
            DocumentBlock(
                block_id=str(uuid.uuid4()),
                block_type=block_type,
                text=text,
                page_number=page_number,
                section=current_section if block_type != BlockType.HEADING else text,
                heading_level=_heading_level(item) if block_type == BlockType.HEADING else None,
            )
        )

    ts = ingestion_timestamp or datetime.now(UTC)
    filename = source_filename or path.name
    title = _infer_title(blocks, filename, markdown)

    return Document(
        document_id=str(uuid.uuid4()),
        title=title,
        filename=filename,
        source_type=SourceType.PDF,
        markdown=markdown,
        blocks=blocks,
        ingestion_timestamp=ts,
        page_count=_page_count(dl_doc),
    )
