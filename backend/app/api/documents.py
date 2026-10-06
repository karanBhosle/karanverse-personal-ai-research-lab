import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.deps import get_ingestion_service
from app.config import get_settings
from app.models.document import Document
from app.models.knowledge_source import DocumentKind, KnowledgeScope
from app.security.uploads import sanitize_filename, validate_pdf_upload
from app.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


class IngestResponse(Document):
    """API response for a successfully ingested document."""

    processed_path: str
    chunk_count: int


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    knowledge_scope: KnowledgeScope = Form(default=KnowledgeScope.PUBLIC_RESEARCH),
    document_kind: DocumentKind = Form(default=DocumentKind.PDF),
    project_id: str | None = Form(default=None),
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    settings = get_settings()
    safe_name = sanitize_filename(file.filename or "upload.pdf", require_extension=".pdf")
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    suffix = Path(safe_name).suffix
    tmp_path: Path | None = None
    try:
        content = await file.read()
        try:
            validate_pdf_upload(content, max_bytes=settings.max_upload_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        document = service.ingest_document(
            tmp_path,
            source_filename=safe_name,
            knowledge_scope=knowledge_scope,
            document_kind=document_kind,
            project_id=project_id,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        logger.exception("Ingestion validation failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail="Document ingestion failed") from exc
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()

    out_dir = service.processed_dir / document.document_id
    return IngestResponse(
        **document.model_dump(),
        processed_path=str(out_dir),
        chunk_count=len(document.chunks),
    )
