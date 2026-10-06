import uuid
from datetime import UTC, datetime

import pytest

from app.models.document import Chunk, SourceType
from app.models.knowledge_source import DocumentKind, KnowledgeScope
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.knowledge_scope import infer_knowledge_scopes


def test_infer_personal_query_scopes():
    scopes = infer_knowledge_scopes("What have I previously learned about Graph RAG?")
    assert scopes == {KnowledgeScope.PERSONAL_KNOWLEDGE, KnowledgeScope.PROJECT}


def test_infer_what_have_i_learned():
    scopes = infer_knowledge_scopes("What have I learned about Graph RAG?")
    assert scopes == {KnowledgeScope.PERSONAL_KNOWLEDGE, KnowledgeScope.PROJECT}


def test_infer_project_query_scopes():
    scopes = infer_knowledge_scopes("Which of my projects use hybrid retrieval?")
    assert KnowledgeScope.PROJECT in scopes
    assert KnowledgeScope.PUBLIC_RESEARCH not in scopes


def test_infer_connect_knowledge_query():
    scopes = infer_knowledge_scopes("Connect my knowledge about RAG, agents and MCP.")
    assert KnowledgeScope.PERSONAL_KNOWLEDGE in scopes


def test_default_public_only_for_generic_query():
    scopes = infer_knowledge_scopes("reciprocal rank fusion in literature")
    assert scopes == {KnowledgeScope.PUBLIC_RESEARCH}


def test_allow_mixed_enables_all_scopes():
    scopes = infer_knowledge_scopes("hybrid retrieval", allow_mixed=True)
    assert len(scopes) == len(KnowledgeScope)


def test_bm25_filters_by_knowledge_scope(tmp_path):
    processed = tmp_path / "processed"
    index_dir = tmp_path / "index"
    processed.mkdir()
    index_dir.mkdir()
    doc_id = "doc-1"
    (processed / doc_id).mkdir()
    now = datetime.now(UTC)

    def _chunk(text: str, scope: KnowledgeScope) -> Chunk:
        return Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=doc_id,
            filename="note.md",
            page_number=None,
            section=None,
            text=text,
            source_type=SourceType.MARKDOWN,
            knowledge_scope=scope,
            document_kind=DocumentKind.TECHNICAL_NOTE,
            ingestion_timestamp=now,
        )

    chunks = [
        _chunk("personal graph rag notes", KnowledgeScope.PERSONAL_KNOWLEDGE),
        _chunk("public research paper excerpt", KnowledgeScope.PUBLIC_RESEARCH),
    ]
    (processed / doc_id / "chunks.json").write_text(
        "[" + ",".join(c.model_dump_json() for c in chunks) + "]",
        encoding="utf-8",
    )
    retriever = BM25Retriever(index_dir=index_dir, processed_dir=processed)
    retriever.build_from_processed()

    personal = retriever.retrieve(
        "graph rag",
        top_k=5,
        knowledge_scopes={KnowledgeScope.PERSONAL_KNOWLEDGE},
    )
    assert len(personal) == 1
    assert personal[0].metadata["knowledge_scope"] == KnowledgeScope.PERSONAL_KNOWLEDGE.value
