import uuid
from typing import TypeVar

import pytest
from fastapi.testclient import TestClient

from app.agents.research_update.claims import extract_previous_conclusions
from app.agents.research_update.evidence_overlap import mean_max_jaccard, novel_evidence_ids
from app.config.settings import Settings
from app.llm.provider import LLMProvider
from app.llm.types import LLMRequest, LLMResponse
from app.models.evidence import Evidence, RetrievalMethod, SourceMetadata
from app.models.research_evidence import ResearchEvidenceBundle
from app.models.research_paper import PublicationInfo, ResearchPaper
from app.models.research_plan import ResearchPlan
from app.models.research_report import ReportFinding, ResearchReport
from app.models.research_update import (
    ChangedClaim,
    ResearchUpdateRequest,
    UpdateComparisonLLMOutput,
    UpdateContradiction,
    UpdateQueriesLLMOutput,
)
from app.models.search import HybridSearchResult, RankContribution, RerankedSearchResult
from app.services.research_history_service import create_research_history_service
from app.services.research_update_service import create_research_update_service
from app.sources.registry import create_source_registry

T = TypeVar("T")


def _evidence(text: str, evidence_id: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        document_id="doc-1",
        chunk_id=str(uuid.uuid4()),
        source="notes.pdf",
        title="Agentic RAG Notes",
        text=text,
        retrieval_score=0.8,
        retrieval_method=RetrievalMethod.HYBRID_RERANK,
        source_metadata=SourceMetadata(filename="notes.pdf", source_type="pdf"),
    )


def _plan() -> ResearchPlan:
    return ResearchPlan(
        original_question="What are the best approaches for Agentic RAG in 2026?",
        research_objective="Survey agentic RAG approaches.",
        sub_questions=["What architectures exist?"],
        search_queries=["agentic RAG 2026"],
        required_sources=["openalex"],
        expected_evidence=["Survey papers"],
        research_strategy="Collect and compare.",
    )


def _report() -> ResearchReport:
    return ResearchReport(
        question="What are the best approaches for Agentic RAG in 2026?",
        executive_summary="ReAct-style agents with hybrid retrieval were the default pattern.",
        key_findings=[
            ReportFinding(
                statement="Hybrid retrieval plus tool-using agents is the dominant agentic RAG pattern.",
                basis="evidence",
                evidence_ids=["old-ev-1"],
            )
        ],
        open_questions=["Which 2026 benchmarks matter?"],
    )


def _seed_history(tmp_path):
    history = create_research_history_service(
        Settings(data_processed_dir=tmp_path, research_history_dir=tmp_path / "history")
    )
    plan = _plan()
    old_evidence = _evidence(
        "Hybrid retrieval with BM25 and dense vectors is commonly paired with ReAct agents.",
        "old-ev-1",
    )
    bundle = ResearchEvidenceBundle(
        question=plan.original_question,
        plan=plan,
        papers=[],
        evidence=[old_evidence],
        citations=[],
    )
    record = history.save_run(
        question=plan.original_question,
        plan=plan,
        bundle=bundle,
        critique=None,
        report=_report(),
    )
    return history, record, old_evidence


def _hybrid_hit(text: str, chunk_id: str) -> HybridSearchResult:
    return HybridSearchResult(
        chunk_id=chunk_id,
        document_id="doc-2",
        text=text,
        final_score=0.9,
        bm25_score=1.0,
        vector_score=0.85,
        rank_contribution=RankContribution(bm25_rank=1, vector_rank=1, bm25_rrf=0.02, vector_rrf=0.02),
        metadata={"filename": "update_notes.pdf", "page_number": 3, "section": "Updates", "source_type": "pdf"},
    )


class MockUpdateLLM(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def default_model(self) -> str:
        return "mock-model"

    def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content="{}", model="mock-model", provider="mock")

    def generate_structured(self, request: LLMRequest, response_model: type[T]) -> T:
        self.calls += 1
        if response_model is UpdateQueriesLLMOutput:
            return UpdateQueriesLLMOutput(
                literature_search_queries=["agentic RAG 2026 benchmark"],
                local_retrieval_queries=["graph RAG agentic retrieval update"],
                focus_areas=["benchmarks", "graph RAG"],
                rationale="Target deltas since prior run.",
            )
        if response_model is UpdateComparisonLLMOutput:
            return UpdateComparisonLLMOutput(
                executive_summary="Graph-native agentic RAG stacks gained traction; pure ReAct+hybrid is no longer the only default.",
                changed_claims=[
                    ChangedClaim(
                        previous_conclusion="Hybrid retrieval plus tool-using agents is the dominant agentic RAG pattern.",
                        update_summary="Graph-augmented agent loops are now a peer architecture to hybrid ReAct stacks.",
                        rationale="New local evidence highlights graph traversal tools alongside hybrid retrieval.",
                        previous_evidence_ids=["old-ev-1"],
                        new_evidence_ids=["new-ev-1"],
                        confidence="medium",
                    )
                ],
                unchanged_claims=["Tool-using agents remain central to agentic RAG designs."],
                contradictions=[],
                newly_discovered_information=["MCP tool routers appear in newer agentic RAG prototypes."],
                outdated_information=["Calling hybrid ReAct the sole dominant pattern is too narrow."],
                confidence="medium",
                recommended_action="Run a focused benchmark comparing graph-augmented vs hybrid ReAct agentic RAG on your workload.",
            )
        raise AssertionError(f"Unexpected structured model: {response_model}")


class _StubHybridRetriever:
    def retrieve(self, query, top_k=10, candidate_count=50, knowledge_scopes=None):
        return [
            _hybrid_hit(
                "Graph-augmented agentic RAG systems combine knowledge-graph traversal with MCP tool routers.",
                "chunk-new-1",
            )
        ]


class _StubReranker:
    def rerank(self, query, candidates, top_k=5):
        return [
            RerankedSearchResult(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                text=c.text,
                hybrid_score=c.final_score,
                bm25_score=c.bm25_score,
                vector_score=c.vector_score,
                rank_contribution=c.rank_contribution,
                metadata=c.metadata,
                rerank_score=0.97,
                final_score=0.97,
            )
            for c in candidates[:top_k]
        ]


def test_extract_previous_conclusions(tmp_path):
    history, record, _old = _seed_history(tmp_path)
    conclusions = extract_previous_conclusions(record)
    assert any("Hybrid retrieval" in item for item in conclusions)
    assert any("dominant agentic RAG" in item for item in conclusions)


def test_evidence_overlap_detects_novel_excerpts():
    old = [_evidence("Hybrid retrieval with BM25 and dense vectors.", "old-ev-1")]
    new = [
        _evidence("Graph-augmented agentic RAG with MCP routers.", "new-ev-1"),
        _evidence("Hybrid retrieval with BM25 and dense vectors for agents.", "new-ev-2"),
    ]
    overlap = mean_max_jaccard(old, new)
    assert overlap > 0.2
    novel = novel_evidence_ids(old, new, novelty_threshold=0.5)
    assert "new-ev-1" in novel


def test_research_update_service_with_mocked_evidence(tmp_path):
    history, record, _old = _seed_history(tmp_path)
    settings = Settings(
        data_processed_dir=tmp_path,
        research_history_dir=tmp_path / "history",
        openalex_cache_enabled=False,
        arxiv_cache_enabled=False,
    )
    registry = create_source_registry(settings)
    new_paper = ResearchPaper(
        external_id="W99",
        openalex_id="W99",
        title="Graph Agentic RAG 2026",
        abstract="Surveys graph-native agentic retrieval stacks.",
        publication=PublicationInfo(venue="arXiv", publication_year=2026),
        source="openalex",
    )
    registry.search_papers = lambda source, query, limit=10: [new_paper]

    llm = MockUpdateLLM()
    service = create_research_update_service(
        settings,
        llm=llm,
        history_service=history,
        source_registry=registry,
        hybrid_retriever=_StubHybridRetriever(),
        reranker=_StubReranker(),
    )

    update = service.run(
        ResearchUpdateRequest(
            research_id=record.research_id,
            question="Research this topic again and tell me what changed.",
        )
    )

    assert update.research_id == record.research_id
    assert update.previous_conclusion
    assert update.new_evidence
    assert update.changed_claims
    assert update.newly_discovered_information
    assert update.recommended_action
    assert update.new_literature
    assert llm.calls == 2
    assert update.evidence_overlap_score is not None


def test_research_update_api(tmp_path, monkeypatch: pytest.MonkeyPatch):
    history, record, _old = _seed_history(tmp_path)
    settings = Settings(
        data_processed_dir=tmp_path,
        research_history_dir=tmp_path / "history",
        openalex_cache_enabled=False,
        arxiv_cache_enabled=False,
    )
    registry = create_source_registry(settings)
    registry.search_papers = lambda source, query, limit=10: []

    service = create_research_update_service(
        settings,
        llm=MockUpdateLLM(),
        history_service=history,
        source_registry=registry,
        hybrid_retriever=_StubHybridRetriever(),
        reranker=_StubReranker(),
    )

    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    from app.config.settings import get_settings
    from app.main import create_app
    from app.services.research_update_service import get_research_update_service

    get_settings.cache_clear()
    get_research_update_service.cache_clear()
    app = create_app()
    app.dependency_overrides[get_research_update_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/research/update",
        json={
            "research_id": record.research_id,
            "question": "Research this topic again and tell me what changed.",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["changed_claims"]
    assert body["executive_summary"]
    assert body["new_evidence"]
