import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.models.document import Chunk, SourceType
from app.models.evidence import RetrievalMethod
from app.models.search import SearchResult
from app.services.citation_service import CitationService


def _chunk(document_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=str(uuid.uuid4()),
        document_id=document_id,
        filename="paper.pdf",
        page_number=4,
        section="Results",
        text=text,
        source_type=SourceType.PDF,
        ingestion_timestamp=datetime(2026, 6, 1, tzinfo=UTC),
    )


def _write_processed(processed_dir: Path, document_id: str, title: str, chunk: Chunk) -> None:
    out = processed_dir / document_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "document.json").write_text(
        json.dumps({"document_id": document_id, "title": title}),
        encoding="utf-8",
    )
    (out / "chunks.json").write_text(
        json.dumps([chunk.model_dump(mode="json")]),
        encoding="utf-8",
    )


def test_citation_labels_and_resolution(tmp_path: Path):
    processed = tmp_path / "processed"
    document_id = "doc-123"
    chunk = _chunk(document_id, "Hybrid retrieval combines bm25 and dense vectors for research.")
    _write_processed(processed, document_id, "Hybrid Retrieval Notes", chunk)

    service = CitationService(processed_dir=processed)
    hit = SearchResult(
        chunk_id=chunk.chunk_id,
        document_id=document_id,
        text=chunk.text,
        score=1.23,
        metadata={
            "filename": "paper.pdf",
            "page_number": 4,
            "section": "Results",
            "source_type": "pdf",
            "ingestion_timestamp": chunk.ingestion_timestamp.isoformat(),
        },
    )
    citations = service.register_search_results([hit], method=RetrievalMethod.BM25)

    assert len(citations) == 1
    assert citations[0].label == "[1]"
    assert CitationService.format_citation_label(2) == "[2]"

    evidence = service.resolve_citation_label("[1]")
    assert evidence.chunk_id == chunk.chunk_id
    assert evidence.document_id == document_id
    assert evidence.title == "Hybrid Retrieval Notes"
    assert evidence.page == 4
    assert evidence.section == "Results"
    assert evidence.retrieval_method == RetrievalMethod.BM25
    assert evidence.retrieval_score == 1.23

    resolved_chunk = service.resolve_to_source_chunk(evidence)
    assert resolved_chunk.chunk_id == chunk.chunk_id
    assert resolved_chunk.text == chunk.text


def test_multiple_citations_number_sequentially(tmp_path: Path):
    service = CitationService(processed_dir=tmp_path)
    results = [
        SearchResult(
            chunk_id=str(uuid.uuid4()),
            document_id="d1",
            text="first",
            score=1.0,
            metadata={"filename": "a.pdf", "source_type": "pdf"},
        ),
        SearchResult(
            chunk_id=str(uuid.uuid4()),
            document_id="d2",
            text="second",
            score=0.8,
            metadata={"filename": "b.pdf", "source_type": "pdf"},
        ),
    ]
    labels = [c.label for c in service.register_search_results(results, method=RetrievalMethod.VECTOR)]
    assert labels == ["[1]", "[2]"]
    assert service.resolve_citation_number(2).text == "second"


def test_map_claim_to_citations_without_llm(tmp_path: Path):
    service = CitationService(processed_dir=tmp_path)
    service.register_search_results(
        [
            SearchResult(
                chunk_id="c1",
                document_id="d1",
                text="Reciprocal rank fusion merges sparse and dense retrieval rankings.",
                score=0.5,
                metadata={"filename": "a.pdf", "source_type": "pdf"},
            ),
            SearchResult(
                chunk_id="c2",
                document_id="d2",
                text="Unrelated graph database indexing notes.",
                score=0.4,
                metadata={"filename": "b.pdf", "source_type": "pdf"},
            ),
        ],
        method=RetrievalMethod.HYBRID,
    )

    claim = "Reciprocal rank fusion merges sparse and dense rankings"
    mapped = service.map_claim_to_citations(claim, max_citations=1, min_token_overlap=2)
    assert len(mapped) == 1
    assert mapped[0].label == "[1]"
    assert mapped[0].claim_text == claim

    sentence = service.attach_citations_to_text(claim, mapped)
    assert sentence.endswith("[1]")


def test_duplicate_chunk_registration_reuses_citation_number(tmp_path: Path):
    service = CitationService(processed_dir=tmp_path)
    hit = SearchResult(
        chunk_id="same-chunk",
        document_id="d1",
        text="shared evidence",
        score=1.0,
        metadata={"filename": "a.pdf", "source_type": "pdf"},
    )
    first = service.register_search_results([hit], method=RetrievalMethod.BM25)
    second = service.register_search_results([hit], method=RetrievalMethod.BM25)
    assert first[0].citation_number == second[0].citation_number == 1
    assert len(service.citations) == 1
