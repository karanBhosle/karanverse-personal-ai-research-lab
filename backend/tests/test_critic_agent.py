import uuid

import pytest
from fastapi.testclient import TestClient

from app.agents.critic.analysis import analyze_bundle, find_duplicate_evidence_groups
from app.agents.critic.validation import merge_critique
from app.config.settings import Settings
from app.llm.provider import LLMProvider
from app.llm.types import LLMRequest, LLMResponse
from app.models.critique import ConflictingEvidenceItem, CritiqueLLMOutput
from app.models.evidence import Evidence, RetrievalMethod, SourceMetadata
from app.models.research_evidence import ResearchEvidenceBundle
from app.models.research_paper import PublicationInfo, ResearchPaper
from app.models.research_plan import ResearchPlan
from app.services.critic_service import CriticAgentService


def _evidence(
    text: str,
    *,
    evidence_id: str | None = None,
    score: float = 0.8,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id or str(uuid.uuid4()),
        document_id="doc-1",
        chunk_id=str(uuid.uuid4()),
        source="local",
        title="Agentic RAG Notes",
        page=1,
        section="Intro",
        text=text,
        retrieval_score=score,
        retrieval_method=RetrievalMethod.HYBRID_RERANK,
        source_metadata=SourceMetadata(filename="notes.pdf", source_type="pdf"),
    )


@pytest.fixture
def sample_bundle() -> ResearchEvidenceBundle:
    plan = ResearchPlan(
        original_question="What are the best approaches for Agentic RAG in 2026?",
        research_objective="Survey agentic RAG approaches.",
        sub_questions=["What architectures exist?", "How is retrieval evaluated?"],
        search_queries=["agentic RAG 2026"],
        required_sources=["openalex", "arxiv"],
        expected_evidence=["Survey papers", "Benchmark results"],
        research_strategy="Collect literature and local notes.",
    )
    e1 = _evidence("Agentic RAG combines planners with retrieval loops for tool use.")
    e2 = _evidence("Agentic RAG combines planners with retrieval loops for tool use.")  # near-duplicate
    paper = ResearchPaper(
        external_id="W1",
        openalex_id="W1",
        title="Agentic RAG Overview",
        publication=PublicationInfo(venue="arXiv", publication_year=2025),
        source="openalex",
    )
    return ResearchEvidenceBundle(
        question=plan.original_question,
        plan=plan,
        papers=[paper],
        evidence=[e1, e2],
        citations=[],
        unresolved_questions=["How is retrieval evaluated?"],
    )


class MockCriticLLM(LLMProvider):
    def __init__(self, output: CritiqueLLMOutput) -> None:
        self._output = output
        self.last_request: LLMRequest | None = None

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def default_model(self) -> str:
        return "mock-model"

    def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content=self._output.model_dump_json(), model="mock-model", provider="mock")

    def generate_structured(self, request: LLMRequest, response_model: type):
        self.last_request = request
        return response_model.model_validate(self._output.model_dump())


def test_find_duplicate_evidence_groups(sample_bundle: ResearchEvidenceBundle):
    groups = find_duplicate_evidence_groups(sample_bundle.evidence, min_jaccard=0.8)
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_analyze_bundle_flags_insufficiency(sample_bundle: ResearchEvidenceBundle, tmp_path):
    settings = Settings(data_processed_dir=tmp_path, critic_min_evidence_count=3)
    analysis = analyze_bundle(sample_bundle, settings)
    assert not analysis.deterministic_sufficient
    assert "unresolved_sub_questions" in analysis.flags


def test_merge_critique_strips_invented_evidence_ids(sample_bundle: ResearchEvidenceBundle, tmp_path):
    settings = Settings(data_processed_dir=tmp_path)
    analysis = analyze_bundle(sample_bundle, settings)
    llm_output = CritiqueLLMOutput(
        evidence_sufficient=False,
        sufficiency_rationale="Coverage is thin for evaluation sub-question.",
        source_quality_notes=["Only one external paper in the bundle."],
        conflicting_evidence=[
            ConflictingEvidenceItem(
                description="Apparent tension between excerpts.",
                evidence_ids=[sample_bundle.evidence[0].evidence_id, "invented-id"],
            )
        ],
        unsupported_claims=["No benchmark evidence for evaluation sub-question."],
        missing_perspectives=["Production deployment constraints"],
        research_gaps=["Need evaluation benchmarks"],
        additional_research_required=["Run targeted literature search on agentic RAG evaluation."],
        additional_research_queries=["agentic RAG evaluation benchmark 2025"],
    )
    result = merge_critique(sample_bundle, analysis, llm_output)
    assert not result.evidence_sufficient
    assert len(result.conflicting_evidence) == 1
    assert result.conflicting_evidence[0].evidence_ids == [sample_bundle.evidence[0].evidence_id]
    assert result.duplicate_evidence
    assert result.additional_research_queries
    assert "invented-id" not in str(result.model_dump())


def test_critic_agent_service(sample_bundle: ResearchEvidenceBundle, tmp_path):
    llm_output = CritiqueLLMOutput(
        evidence_sufficient=False,
        sufficiency_rationale="Insufficient local and external coverage.",
        source_quality_notes=["Limited diversity of sources."],
        missing_perspectives=["Graph RAG angle missing"],
        research_gaps=["Evaluation evidence missing"],
        additional_research_required=["Expand literature search"],
        additional_research_queries=["graph RAG agents"],
    )
    settings = Settings(data_processed_dir=tmp_path, critic_min_evidence_count=3)
    service = CriticAgentService(MockCriticLLM(llm_output), settings)
    result = service.critique(sample_bundle)

    assert result.question == sample_bundle.question
    assert not result.evidence_sufficient
    assert result.missing_perspectives
    assert result.additional_research_queries


def test_critique_api_with_dependency_override(sample_bundle: ResearchEvidenceBundle, tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    llm_output = CritiqueLLMOutput(
        evidence_sufficient=False,
        sufficiency_rationale="Need more evidence.",
        additional_research_queries=["hybrid retrieval agentic RAG"],
    )
    settings = Settings(data_processed_dir=tmp_path, openrouter_api_key="test-key")
    service = CriticAgentService(MockCriticLLM(llm_output), settings)

    from app.config.settings import get_settings
    from app.main import create_app
    from app.services.critic_service import get_critic_agent_service

    get_settings.cache_clear()
    get_critic_agent_service.cache_clear()
    app = create_app()
    app.dependency_overrides[get_critic_agent_service] = lambda: service
    client = TestClient(app)

    response = client.post("/research/critique", json={"bundle": sample_bundle.model_dump(mode="json")})
    assert response.status_code == 200
    body = response.json()
    assert body["evidence_sufficient"] is False
    assert body["additional_research_queries"]
