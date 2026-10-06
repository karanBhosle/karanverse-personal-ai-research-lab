import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.models.chunking import ChunkingConfig
from app.services.ingestion_service import IngestionService


@pytest.fixture
def ingestion_service(tmp_path: Path) -> IngestionService:
    processed = tmp_path / "processed"
    raw = tmp_path / "raw"
    processed.mkdir()
    raw.mkdir()
    return IngestionService(
        processed_dir=processed,
        raw_dir=raw,
        chunking=ChunkingConfig(chunk_size=400, chunk_overlap=50, min_chunk_size=20),
    )


@pytest.mark.slow
def test_ingestion_pipeline_end_to_end(sample_pdf_path: Path, ingestion_service: IngestionService):
    document = ingestion_service.ingest_document(sample_pdf_path)

    assert document.document_id
    assert document.filename == sample_pdf_path.name
    assert document.title
    assert document.source_type.value == "pdf"
    assert document.markdown.strip()
    assert document.blocks
    assert document.chunks

    out_dir = ingestion_service.processed_dir / document.document_id
    assert (out_dir / "document.json").is_file()
    assert (out_dir / "chunks.json").is_file()
    assert (out_dir / "content.md").is_file()

    saved_chunks = json.loads((out_dir / "chunks.json").read_text(encoding="utf-8"))
    assert len(saved_chunks) == len(document.chunks)
    first = saved_chunks[0]
    for key in (
        "chunk_id",
        "document_id",
        "filename",
        "page_number",
        "section",
        "source_type",
        "ingestion_timestamp",
    ):
        assert key in first
    assert first["document_id"] == document.document_id
    assert first["filename"] == document.filename
    assert first["source_type"] == "pdf"


@pytest.mark.slow
def test_ingest_api_uploads_pdf(sample_pdf_path: Path, tmp_path: Path, monkeypatch):
    processed = tmp_path / "processed"
    raw = tmp_path / "raw"
    processed.mkdir()
    raw.mkdir()

    from app.config.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("DATA_PROCESSED_DIR", str(processed))
    monkeypatch.setenv("DATA_RAW_DIR", str(raw))

    client = TestClient(create_app())
    with sample_pdf_path.open("rb") as pdf_file:
        response = client.post(
            "/documents/ingest",
            files={"file": ("sample_research.pdf", pdf_file, "application/pdf")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"]
    assert body["filename"] == "sample_research.pdf"
    assert body["chunk_count"] >= 1
    assert Path(body["processed_path"]).exists()
