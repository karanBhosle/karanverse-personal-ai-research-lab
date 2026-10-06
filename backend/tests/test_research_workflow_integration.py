import uuid
from typing import TypeVar

import pytest
from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.llm.provider import LLMProvider
from app.llm.types import LLMRequest, LLMResponse
from app.models.critique import CritiqueLLMOutput
from app.models.research_plan import ResearchPlanLLMOutput
from app.models.research_report import ReportFinding, SynthesizerLLMOutput
from app.models.search import HybridSearchResult, RankContribution, RerankedSearchResult
from app.services.research_workflow_service import create_research_workflow_service
from app.sources.registry import create_source_registry

T = TypeVar("T")


def _hybrid_hit(text: str) -> HybridSearchResult:
    return HybridSearchResult(
        chunk_id=str(uuid.uuid4()),
        document_id="doc-1",
        text=text,
        final_score=0.7,
        bm25_score=1.0,
        vector_score=0.6,
        rank_contribution=RankContribution(bm25_rank=1, vector_rank=1, bm25_rrf=0.02, vector_rrf=0.02),
        metadata={"filename": "notes.pdf", "page_number": 1, "section": "Intro", "source_type": "pdf"},
    )


class RoutingWorkflowLLM(LLMProvider):
    def __init__(self) -> None:
        self.critic_calls = 0

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def default_model(self) -> str:
        return "mock-model"

    def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content="{}", model="mock-model", provider="mock")

    def generate_structured(self, request: LLMRequest, response_model: type[T]) -> T:
        if response_model is ResearchPlanLLMOutput:
            payload = ResearchPlanLLMOutput(
                research_objective="Survey agentic RAG patterns.",
                sub_questions=["What architectures exist?", "How is retrieval evaluated?"],
                search_queries=["agentic RAG 2026"],
                required_sources=["openalex"],
                expected_evidence=["Survey papers"],
                research_strategy="Search then retrieve local evidence.",
            )
            return response_model.model_validate(payload.model_dump())

        if response_model is CritiqueLLMOutput:
            self.critic_calls += 1
            sufficient = self.critic_calls >= 2
            payload = CritiqueLLMOutput(
                evidence_sufficient=sufficient,
                sufficiency_rationale="ok" if sufficient else "Need more evaluation evidence.",
                additional_research_queries=["agentic RAG evaluation benchmark"],
                research_gaps=["Evaluation benchmarks missing"],
            )
            return response_model.model_validate(payload.model_dump())

        if response_model is SynthesizerLLMOutput:
            payload = SynthesizerLLMOutput(
                executive_summary="Agentic RAG systems use hybrid retrieval [1].",
                key_findings=[
                    ReportFinding(
                        statement="Hybrid retrieval appears in agentic RAG designs.",
                        basis="inference",
                        uncertainty="Based on limited excerpts; verify with more sources.",
                    )
                ],
            )
            return response_model.model_validate(payload.model_dump())

        raise AssertionError(f"Unexpected response model: {response_model}")


class _StubHybridRetriever:
    def retrieve(self, query: str, top_k: int = 10, *, candidate_count: int | None = None, knowledge_scopes=None):
        return [_hybrid_hit(f"Hybrid retrieval evidence for: {query}")]


class _StubReranker:
    @property
    def model_name(self) -> str:
        return "stub"

    def rerank(self, query: str, candidates: list[HybridSearchResult], *, top_k: int):
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
                rerank_score=0.95,
                final_score=0.95,
            )
            for c in candidates[:top_k]
        ]


@pytest.fixture
def workflow_env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    from app.models.research_paper import PublicationInfo, ResearchPaper

    settings = Settings(
        data_processed_dir=tmp_path,
        research_history_dir=tmp_path / "research_history",
        openalex_cache_enabled=False,
        arxiv_cache_enabled=False,
        research_max_iterations=2,
        critic_min_evidence_count=1,
        critic_min_external_papers=1,
    )
    registry = create_source_registry(settings)
    paper = ResearchPaper(
        external_id="W9",
        openalex_id="W9",
        title="Agentic RAG Overview",
        publication=PublicationInfo(venue="arXiv", publication_year=2025),
        source="openalex",
    )
    registry.search_papers = lambda source, query, limit=10: [paper]

    llm = RoutingWorkflowLLM()
    service = create_research_workflow_service(
        settings,
        source_registry=registry,
        hybrid_retriever=_StubHybridRetriever(),
        reranker=_StubReranker(),
        llm=llm,
    )
    return service, llm


def test_workflow_iterates_then_synthesizes(workflow_env):
    service, llm = workflow_env
    state = service.run_with_state("What are the best approaches for Agentic RAG in 2026?", evidence_top_k=2)
    assert state.report is not None
    assert state.plan is not None
    assert state.bundle is not None
    assert state.critique is not None
    assert llm.critic_calls >= 2
    assert state.iteration >= 2
    node_names = [trace.node for trace in state.traces]
    assert "load_history_context" in node_names
    assert "planner" in node_names
    assert "source_search" in node_names
    assert "critic" in node_names
    assert "additional_research" in node_names
    assert "synthesizer" in node_names
    assert state.report.executive_summary


def test_post_research_endpoint(workflow_env, monkeypatch: pytest.MonkeyPatch):
    service, _llm = workflow_env
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    from app.config.settings import get_settings
    from app.main import create_app
    from app.services.research_workflow_service import get_research_workflow_service

    get_settings.cache_clear()
    get_research_workflow_service.cache_clear()
    app = create_app()
    app.dependency_overrides[get_research_workflow_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/research",
        json={"question": "What are the best approaches for Agentic RAG in 2026?", "max_iterations": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["research_id"]
    assert body["report"]["executive_summary"]
    assert body["report"]["references"]
