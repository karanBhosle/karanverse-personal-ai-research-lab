import json
import statistics
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.models.document import Chunk, SourceType
from app.retrieval.bm25_retriever import BM25Retriever, get_bm25_retriever


def _chunk(
    text: str,
    *,
    document_id: str = "doc-a",
    filename: str = "paper.pdf",
    page_number: int | None = 1,
    section: str | None = "Introduction",
) -> Chunk:
    ts = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    return Chunk(
        chunk_id=str(uuid.uuid4()),
        document_id=document_id,
        filename=filename,
        page_number=page_number,
        section=section,
        text=text,
        source_type=SourceType.PDF,
        ingestion_timestamp=ts,
    )


@pytest.fixture
def retriever_dirs(tmp_path: Path) -> tuple[Path, Path]:
    processed = tmp_path / "processed"
    index_dir = tmp_path / "index" / "bm25"
    processed.mkdir(parents=True)
    index_dir.mkdir(parents=True)
    return processed, index_dir


def _write_processed_chunks(processed_dir: Path, document_id: str, chunks: list[Chunk]) -> None:
    out = processed_dir / document_id
    out.mkdir(parents=True, exist_ok=True)
    payload = [chunk.model_dump(mode="json") for chunk in chunks]
    (out / "chunks.json").write_text(json.dumps(payload), encoding="utf-8")


def test_bm25_index_persists_and_reloads(retriever_dirs: tuple[Path, Path]):
    processed_dir, index_dir = retriever_dirs
    chunks = [
        _chunk("Hybrid retrieval combines bm25 and dense vectors."),
        _chunk("Reciprocal rank fusion merges ranked lists.", document_id="doc-b", section="Methods"),
    ]
    _write_processed_chunks(processed_dir, "doc-a", [chunks[0]])
    _write_processed_chunks(processed_dir, "doc-b", [chunks[1]])

    builder = BM25Retriever(index_dir=index_dir, processed_dir=processed_dir)
    assert builder.build_from_processed() == 2
    assert (index_dir / "bm25_index.pkl").is_file()
    assert (index_dir / "meta.json").is_file()

    reloaded = BM25Retriever(index_dir=index_dir, processed_dir=processed_dir)
    reloaded.load()
    results = reloaded.retrieve("bm25 fusion", top_k=5)
    assert results
    assert all(result.score > 0 for result in results)


def test_bm25_exact_keyword_retrieval(retriever_dirs: tuple[Path, Path]):
    processed_dir, index_dir = retriever_dirs
    target = _chunk("This chunk contains zzuniquekeyword99 for exact match testing.")
    other = _chunk("Unrelated content about graph databases and citation parsing.")
    _write_processed_chunks(processed_dir, "doc-a", [target, other])

    retriever = BM25Retriever(index_dir=index_dir, processed_dir=processed_dir)
    retriever.build_from_processed()
    results = retriever.retrieve("zzuniquekeyword99", top_k=3)

    assert results
    assert results[0].chunk_id == target.chunk_id
    assert "zzuniquekeyword99" in results[0].text


def test_bm25_ranking_orders_by_relevance(retriever_dirs: tuple[Path, Path]):
    processed_dir, index_dir = retriever_dirs
    weak = _chunk("A short note about retrieval systems and ranking in general.")
    strong = _chunk(
        "bm25 bm25 bm25 ranking benchmark for sparse retrieval ranking quality evaluation."
    )
    _write_processed_chunks(processed_dir, "doc-a", [weak, strong])

    retriever = BM25Retriever(index_dir=index_dir, processed_dir=processed_dir)
    retriever.build_from_processed()
    results = retriever.retrieve("bm25 ranking", top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == strong.chunk_id
    assert results[0].score >= results[1].score


def test_bm25_metadata_preservation(retriever_dirs: tuple[Path, Path]):
    processed_dir, index_dir = retriever_dirs
    chunk = _chunk(
        "Provenance metadata should flow through search results unchanged.",
        page_number=7,
        section="Results",
        filename="experiment.pdf",
    )
    _write_processed_chunks(processed_dir, chunk.document_id, [chunk])

    retriever = BM25Retriever(index_dir=index_dir, processed_dir=processed_dir)
    retriever.build_from_processed()
    result = retriever.retrieve("provenance metadata", top_k=1)[0]

    assert result.document_id == chunk.document_id
    assert result.metadata["filename"] == "experiment.pdf"
    assert result.metadata["page_number"] == 7
    assert result.metadata["section"] == "Results"
    assert result.metadata["source_type"] == "pdf"
    assert result.metadata["ingestion_timestamp"] == chunk.ingestion_timestamp.isoformat()


def test_bm25_ensure_index_rebuilds_when_corpus_changes(retriever_dirs: tuple[Path, Path]):
    processed_dir, index_dir = retriever_dirs
    _write_processed_chunks(processed_dir, "doc-a", [_chunk("initial corpus token")])

    retriever = BM25Retriever(index_dir=index_dir, processed_dir=processed_dir)
    retriever.ensure_index()
    assert retriever.chunk_count == 1

    _write_processed_chunks(
        processed_dir,
        "doc-b",
        [_chunk("second document adds reciprocal rank fusion terms")],
    )
    retriever.ensure_index()
    assert retriever.chunk_count == 2


def test_bm25_search_api(retriever_dirs: tuple[Path, Path], monkeypatch):
    processed_dir, index_dir = retriever_dirs
    _write_processed_chunks(
        processed_dir,
        "doc-a",
        [_chunk("OpenAlex arXiv literature search with bm25 endpoint validation.")],
    )

    from app.config.settings import get_settings

    get_settings.cache_clear()
    get_bm25_retriever.cache_clear()
    monkeypatch.setenv("DATA_PROCESSED_DIR", str(processed_dir))
    monkeypatch.setenv("BM25_INDEX_DIR", str(index_dir))

    client = TestClient(create_app())
    response = client.post(
        "/search/bm25",
        json={"query": "bm25 OpenAlex", "top_k": 5},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["index_chunk_count"] >= 1
    assert body["results"]
    assert body["results"][0]["metadata"]["source_type"] == "pdf"


def test_bm25_retrieval_latency_benchmark(retriever_dirs: tuple[Path, Path]):
    """Benchmark query latency on a small local corpus (synthetic chunks)."""
    processed_dir, index_dir = retriever_dirs
    corpus_chunks = [
        _chunk(
            f"Document {i}: hybrid retrieval uses bm25 sparse scoring and vector search "
            f"with reciprocal rank fusion for research intelligence platform chunk {i}."
        )
        for i in range(120)
    ]
    _write_processed_chunks(processed_dir, "bench-doc", corpus_chunks)

    retriever = BM25Retriever(index_dir=index_dir, processed_dir=processed_dir)
    retriever.build_from_processed()

    queries = [
        "bm25 hybrid retrieval",
        "reciprocal rank fusion",
        "research intelligence platform",
        "vector search sparse scoring",
        "document chunk benchmark",
    ]
    latencies_ms: list[float] = []
    for query in queries:
        start = time.perf_counter()
        hits = retriever.retrieve(query, top_k=10)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)
        assert hits

    mean_ms = statistics.mean(latencies_ms)
    p95_ms = sorted(latencies_ms)[int(0.95 * (len(latencies_ms) - 1))]
    print(
        f"\nBM25 latency benchmark (n={len(queries)} queries, "
        f"corpus={len(corpus_chunks)} chunks): "
        f"mean={mean_ms:.3f}ms p95={p95_ms:.3f}ms"
    )
    assert mean_ms < 50.0
