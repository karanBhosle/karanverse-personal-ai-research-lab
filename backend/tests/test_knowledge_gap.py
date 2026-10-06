import uuid

import pytest
from fastapi.testclient import TestClient

from app.agents.knowledge_gap.signals import detect_gap_signals
from app.agents.knowledge_gap.topic import extract_focus_topic
from app.config.settings import Settings
from app.llm.provider import LLMProvider
from app.llm.types import LLMRequest, LLMResponse
from app.models.critique import ConflictingEvidenceItem, CritiqueResult
from app.models.evidence import Evidence, RetrievalMethod, SourceMetadata
from app.models.knowledge_gap import KnowledgeGapLLMItem, KnowledgeGapLLMOutput, KnowledgeGapRequest
from app.models.research_evidence import ResearchEvidenceBundle
from app.models.research_plan import ResearchPlan
from app.models.research_report import ResearchReport
from app.models.search import HybridSearchResult, RankContribution
from app.services.knowledge_gap_service import create_knowledge_gap_service
from app.services.research_history_service import create_research_history_service


def test_extract_focus_topic():
    assert extract_focus_topic("What don't I know about Agentic RAG?") == "Agentic RAG"
    assert "hybrid" in extract_focus_topic("knowledge gaps in hybrid retrieval").lower()


def _hybrid_hit(text: str, score: float) -> HybridSearchResult:
    return HybridSearchResult(
        chunk_id=str(uuid.uuid4()),
        document_id="doc-1",
        text=text,
        final_score=score,
        bm25_score=score,
        vector_score=score,
        rank_contribution=RankContribution(bm25_rank=1, vector_rank=1),
        metadata={"filename": "notes.md", "knowledge_scope": "PERSONAL_KNOWLEDGE"},
    )


def test_detect_gap_signals_low_evidence_and_conflict():
    plan = ResearchPlan(
        original_question="What are agentic RAG patterns?",
        research_objective="Survey agentic RAG.",
        sub_questions=["How are agents evaluated?"],
        search_queries=["agentic RAG"],
        required_sources=["openalex"],
        expected_evidence=["papers"],
        research_strategy="collect",
    )
    from app.models.research_history import ResearchHistoryRecord
    from datetime import UTC, datetime

    critique = CritiqueResult(
        question=plan.original_question,
        evidence_sufficient=False,
        conflicting_evidence=[ConflictingEvidenceItem(description="Benchmark results disagree on agent planners.")],
        research_gaps=["Production deployment patterns for agentic RAG"],
    )
    record = ResearchHistoryRecord(
        research_id="r1",
        question=plan.original_question,
        created_at=datetime.now(UTC),
        plan=plan,
        evidence=[
            Evidence(
                evidence_id="e1",
                document_id="d1",
                chunk_id="c1",
                source="local",
                title="Notes",
                text="agentic RAG hybrid retrieval",
                retrieval_score=0.8,
                retrieval_method=RetrievalMethod.HYBRID,
                source_metadata=SourceMetadata(filename="n.pdf", source_type="pdf"),
            )
        ],
        report=ResearchReport(
            question=plan.original_question,
            executive_summary="Agentic RAG uses retrieval loops.",
            open_questions=["Which benchmarks matter for agentic RAG?"],
        ),
        conclusions=["Hybrid retrieval is common in agentic RAG."],
        unresolved_questions=["How are agents evaluated?"],
        critique=critique,
    )

    signals = detect_gap_signals(
        [record],
        focus_topic="Agentic RAG",
        personal_hits=[_hybrid_hit("weak personal note", 0.1)],
        project_hits=[],
        public_hits=[_hybrid_hit("public agentic RAG survey", 0.7)],
    )
    types = {signal.gap_type for signal in signals}
    assert "low_evidence" in types
    assert "conflict" in types
    assert "prerequisite" in types


class MockGapLLM(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def default_model(self) -> str:
        return "mock"

    def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content="{}", model="mock", provider="mock")

    def generate_structured(self, request, response_model):
        return KnowledgeGapLLMOutput(
            focus_topic="Agentic RAG",
            coverage_summary="Your notes and one research run touch agentic RAG but evidence is thin.",
            gaps=[
                KnowledgeGapLLMItem(
                    topic="Agentic RAG evaluation benchmarks",
                    reason="Prior research left evaluation benchmarks unresolved with limited evidence.",
                    related_concepts=["benchmarks", "tool-using agents"],
                    supporting_research_ids=["r1"],
                    recommended_learning="Run a focused study on agentic RAG evaluation suites.",
                    priority="high",
                    gap_type="prerequisite",
                )
            ],
        )


def test_knowledge_gap_service_uses_history_and_retrieval(tmp_path):
    history = create_research_history_service(
        Settings(data_processed_dir=tmp_path, research_history_dir=tmp_path / "history")
    )
    plan = ResearchPlan(
        original_question="Agentic RAG architectures",
        research_objective="Map agentic RAG stacks.",
        sub_questions=["What is graph RAG?"],
        search_queries=["agentic RAG"],
        required_sources=["openalex"],
        expected_evidence=["surveys"],
        research_strategy="search",
    )
    history.save_run(
        question=plan.original_question,
        plan=plan,
        bundle=ResearchEvidenceBundle(
            question=plan.original_question,
            plan=plan,
            papers=[],
            evidence=[],
            citations=[],
        ),
        critique=None,
        report=ResearchReport(
            question=plan.original_question,
            executive_summary="Early notes on agentic RAG.",
            open_questions=["Graph-augmented agentic RAG?"],
        ),
        research_id="r1",
    )

    class StubHybrid:
        bm25_retriever = type("B", (), {"ensure_index": lambda self: None})()

        def retrieve(self, query, top_k=10, knowledge_scopes=None):
            scope = next(iter(knowledge_scopes)) if knowledge_scopes else "PUBLIC_RESEARCH"
            text = f"{scope} content about agentic RAG and hybrid retrieval"
            return [_hybrid_hit(text, 0.6)]

    service = create_knowledge_gap_service(
        Settings(data_processed_dir=tmp_path, research_history_dir=tmp_path / "history"),
        llm=MockGapLLM(),
        history_service=history,
        hybrid_retriever=StubHybrid(),
    )
    report = service.analyze(KnowledgeGapRequest(query="What don't I know about Agentic RAG?"))
    assert report.focus_topic == "Agentic RAG"
    assert report.gaps
    assert report.research_runs_analyzed >= 1
    assert report.knowledge_chunks_retrieved >= 3
    assert report.gaps[0].supporting_research


def test_knowledge_gap_api(tmp_path, monkeypatch: pytest.MonkeyPatch):
    history = create_research_history_service(
        Settings(data_processed_dir=tmp_path, research_history_dir=tmp_path / "history")
    )

    class StubHybrid:
        bm25_retriever = type("B", (), {"ensure_index": lambda self: None})()

        def retrieve(self, query, top_k=10, knowledge_scopes=None):
            return [_hybrid_hit("agentic RAG note", 0.5)]

    service = create_knowledge_gap_service(
        Settings(data_processed_dir=tmp_path, research_history_dir=tmp_path / "history"),
        llm=MockGapLLM(),
        history_service=history,
        hybrid_retriever=StubHybrid(),
    )

    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    from app.config.settings import get_settings
    from app.main import create_app
    from app.services.knowledge_gap_service import get_knowledge_gap_service

    get_settings.cache_clear()
    get_knowledge_gap_service.cache_clear()
    app = create_app()
    app.dependency_overrides[get_knowledge_gap_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/knowledge/gaps",
        json={"query": "What don't I know about Agentic RAG?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["focus_topic"] == "Agentic RAG"
    assert body["gaps"]
