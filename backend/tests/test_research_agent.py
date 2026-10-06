import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.agents.research_agent.deps import ResearchAgentDeps
from app.agents.research_agent.graph import compile_research_agent
from app.config.settings import Settings
from app.models.research_paper import PublicationInfo, ResearchPaper
from app.models.research_plan import ResearchPlan
from app.models.search import HybridSearchResult, RankContribution, RerankedSearchResult
from app.services.research_agent_service import create_research_agent_service
from app.services.research_planner_service import ResearchPlannerService
from app.sources.registry import create_source_registry


def _hybrid_hit(text: str, score: float = 0.5) -> HybridSearchResult:
    return HybridSearchResult(
        chunk_id=str(uuid.uuid4()),
        document_id="doc-1",
        text=text,
        final_score=score,
        bm25_score=1.0,
        vector_score=0.8,
        rank_contribution=RankContribution(bm25_rank=1, vector_rank=1, bm25_rrf=0.02, vector_rrf=0.02),
        metadata={"filename": "paper.pdf", "page_number": 1, "section": "Intro", "source_type": "pdf"},
    )


def _reranked_from_hybrid(hit: HybridSearchResult, rerank: float = 0.95) -> RerankedSearchResult:
    return RerankedSearchResult(
        chunk_id=hit.chunk_id,
        document_id=hit.document_id,
        text=hit.text,
        hybrid_score=hit.final_score,
        bm25_score=hit.bm25_score,
        vector_score=hit.vector_score,
        rank_contribution=hit.rank_contribution,
        metadata=hit.metadata,
        rerank_score=rerank,
        final_score=rerank,
    )


@pytest.fixture
def sample_plan() -> ResearchPlan:
    return ResearchPlan(
        original_question="What are the best approaches for Agentic RAG in 2026?",
        research_objective="Compare agentic RAG patterns without concluding a winner.",
        sub_questions=[
            "What agentic RAG architectures exist?",
            "How is hybrid retrieval used in agentic RAG?",
        ],
        search_queries=["agentic RAG 2026", "hybrid retrieval agent"],
        required_sources=["openalex", "arxiv"],
        expected_evidence=["System papers", "Benchmarks"],
        research_strategy="Search literature then collect local evidence.",
    )


@pytest.fixture
def sample_paper() -> ResearchPaper:
    return ResearchPaper(
        external_id="W999",
        openalex_id="W999",
        title="Agentic RAG Survey",
        publication=PublicationInfo(venue="arXiv", publication_year=2025),
        source="openalex",
    )


class _StubHybridRetriever:
    def retrieve(self, query: str, top_k: int = 10, *, candidate_count: int | None = None, knowledge_scopes=None):
        return [_hybrid_hit(f"Evidence for: {query}", score=0.4 + 0.1 * len(query))]


class _StubReranker:
    @property
    def model_name(self) -> str:
        return "stub-reranker"

    def rerank(self, query: str, candidates: list[HybridSearchResult], *, top_k: int):
        ranked = sorted(candidates, key=lambda item: item.final_score, reverse=True)
        return [_reranked_from_hybrid(hit) for hit in ranked[:top_k]]


def test_research_agent_graph_collects_evidence_bundle(
    sample_plan: ResearchPlan,
    sample_paper: ResearchPaper,
    tmp_path,
):
    settings = Settings(
        data_processed_dir=tmp_path,
        openalex_cache_enabled=False,
        arxiv_cache_enabled=False,
        research_agent_min_evidence_hits=1,
    )
    registry = create_source_registry(settings)
    registry.search_papers = MagicMock(return_value=[sample_paper])

    deps = ResearchAgentDeps(
        settings=settings,
        plan_question=lambda _q: sample_plan,
        source_registry=registry,
        hybrid_retriever=_StubHybridRetriever(),
        reranker=_StubReranker(),
    )
    graph = compile_research_agent(deps)
    result = graph.invoke({"question": sample_plan.original_question, "evidence_top_k": 2})
    bundle = result["bundle"]

    assert bundle.question == sample_plan.original_question
    assert bundle.plan.research_objective == sample_plan.research_objective
    assert len(bundle.papers) == 1
    assert bundle.papers[0].title == "Agentic RAG Survey"
    assert bundle.evidence
    assert bundle.citations
    assert len(bundle.citations) == len(bundle.evidence)
    assert bundle.source_metadata
    assert bundle.retrieval_scores
    assert not bundle.unresolved_questions
    assert "answer" not in bundle.model_dump()


def test_research_agent_marks_unresolved_sub_questions(sample_plan: ResearchPlan, tmp_path):
    settings = Settings(
        data_processed_dir=tmp_path,
        research_agent_min_evidence_hits=1,
    )

    class _EmptyReranker(_StubReranker):
        def rerank(self, query: str, candidates: list[HybridSearchResult], *, top_k: int):
            return []

    deps = ResearchAgentDeps(
        settings=settings,
        plan_question=lambda _q: sample_plan,
        source_registry=create_source_registry(settings),
        hybrid_retriever=_StubHybridRetriever(),
        reranker=_EmptyReranker(),
    )
    bundle = compile_research_agent(deps).invoke(
        {"question": sample_plan.original_question, "evidence_top_k": 2}
    )["bundle"]

    assert set(bundle.unresolved_questions) == set(sample_plan.sub_questions)
    assert bundle.evidence == []


def test_research_collect_api_with_dependency_override(
    sample_plan: ResearchPlan,
    sample_paper: ResearchPaper,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    settings = Settings(
        data_processed_dir=tmp_path,
        openrouter_api_key="test-key",
        openalex_cache_enabled=False,
        arxiv_cache_enabled=False,
    )
    registry = create_source_registry(settings)
    registry.search_papers = MagicMock(return_value=[sample_paper])

    planner = MagicMock(spec=ResearchPlannerService)
    planner.create_plan.return_value = sample_plan

    agent = create_research_agent_service(
        settings,
        planner=planner,
        source_registry=registry,
        hybrid_retriever=_StubHybridRetriever(),
        reranker=_StubReranker(),
    )

    from app.config.settings import get_settings
    from app.main import create_app
    from app.services.research_agent_service import get_research_agent_service

    get_settings.cache_clear()
    get_research_agent_service.cache_clear()
    app = create_app()
    app.dependency_overrides[get_research_agent_service] = lambda: agent
    client = TestClient(app)

    response = client.post(
        "/research/collect",
        json={"question": sample_plan.original_question, "evidence_top_k": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["question"] == sample_plan.original_question
    assert body["plan"]["sub_questions"]
    assert body["papers"]
    assert body["evidence"]
    assert "unresolved_questions" in body
    planner.create_plan.assert_called_once()
