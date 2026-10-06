import uuid

import pytest
from fastapi.testclient import TestClient

from app.agents.synthesizer.citations import build_citation_index
from app.agents.synthesizer.validation import assemble_report
from app.config.settings import Settings
from app.llm.provider import LLMProvider
from app.llm.types import LLMRequest, LLMResponse
from app.models.critique import CritiqueResult
from app.models.evidence import Evidence, RetrievalMethod, SourceMetadata
from app.models.research_evidence import ResearchEvidenceBundle
from app.models.research_paper import PublicationInfo, ResearchPaper
from app.models.research_plan import ResearchPlan
from app.models.research_report import ReportContradiction, ReportFinding, ReportRecommendation, SynthesizerLLMOutput
from app.services.research_synthesizer_service import ResearchSynthesizerService


def _evidence(text: str, evidence_id: str | None = None) -> Evidence:
    return Evidence(
        evidence_id=evidence_id or "ev-1",
        document_id="doc-1",
        chunk_id=str(uuid.uuid4()),
        source="local",
        title="Agentic RAG Notes",
        page=2,
        section="Methods",
        text=text,
        retrieval_score=0.91,
        retrieval_method=RetrievalMethod.HYBRID_RERANK,
        source_metadata=SourceMetadata(filename="notes.pdf", source_type="pdf"),
    )


@pytest.fixture
def plan() -> ResearchPlan:
    return ResearchPlan(
        original_question="What are the best approaches for Agentic RAG in 2026?",
        research_objective="Survey agentic RAG approaches.",
        sub_questions=["What architectures exist?"],
        search_queries=["agentic RAG 2026"],
        required_sources=["openalex"],
        expected_evidence=["Survey papers"],
        research_strategy="Collect evidence then synthesize.",
    )


@pytest.fixture
def bundle(plan: ResearchPlan) -> ResearchEvidenceBundle:
    evidence = _evidence("Hybrid retrieval is commonly combined with agent planners in agentic RAG systems.")
    paper = ResearchPaper(
        external_id="W1",
        openalex_id="W1",
        title="Agentic RAG Overview",
        publication=PublicationInfo(venue="arXiv", publication_year=2025),
        source="openalex",
        landing_page_url="https://example.org/abs/1",
    )
    return ResearchEvidenceBundle(
        question=plan.original_question,
        plan=plan,
        papers=[paper],
        evidence=[evidence],
        citations=[],
    )


@pytest.fixture
def critique(plan: ResearchPlan) -> CritiqueResult:
    return CritiqueResult(
        question=plan.original_question,
        evidence_sufficient=False,
        sufficiency_rationale="Limited evaluation coverage.",
        research_gaps=["Need benchmarks"],
        missing_perspectives=["Production deployment"],
        additional_research_queries=["agentic RAG benchmark"],
    )


class MockSynthesizerLLM(LLMProvider):
    def __init__(self, output: SynthesizerLLMOutput) -> None:
        self._output = output

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def default_model(self) -> str:
        return "mock-model"

    def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content=self._output.model_dump_json(), model="mock-model", provider="mock")

    def generate_structured(self, request: LLMRequest, response_model: type):
        return response_model.model_validate(self._output.model_dump())


def test_assemble_report_drops_unsupported_evidence_claims(bundle: ResearchEvidenceBundle, plan: ResearchPlan, critique: CritiqueResult):
    index = build_citation_index(bundle)
    llm_output = SynthesizerLLMOutput(
        executive_summary="Agentic RAG often uses hybrid retrieval [1].",
        key_findings=[
            ReportFinding(
                statement="Hybrid retrieval appears in agentic RAG designs.",
                basis="evidence",
                evidence_ids=["ev-1"],
                confidence="medium",
            ),
            ReportFinding(
                statement="This should be dropped.",
                basis="evidence",
                evidence_ids=["fabricated"],
                confidence="high",
            ),
            ReportFinding(
                statement="Deployment may require latency tradeoffs.",
                basis="inference",
                confidence="low",
                uncertainty="Not directly supported by excerpts.",
            ),
        ],
        contradictions=[
            ReportContradiction(description="Conflict", evidence_ids=["fabricated"]),
        ],
        limitations=["Small corpus"],
        practical_recommendations=[
            ReportRecommendation(
                recommendation="Collect more benchmark papers.",
                basis="inference",
                confidence="low",
            )
        ],
        open_questions=["How to evaluate agents?"],
        uncertainty_notes=["Coverage is limited"],
    )
    report = assemble_report(
        question=plan.original_question,
        plan=plan,
        bundle=bundle,
        critique=critique,
        index=index,
        llm_output=llm_output,
    )

    assert report.executive_summary.startswith("Agentic RAG")
    assert len(report.key_findings) == 2
    assert report.key_findings[0].citation_labels == ["[1]"]
    assert report.key_findings[0].evidence_ids == ["ev-1"]
    assert report.key_findings[1].basis == "inference"
    assert report.contradictions == []
    assert len(report.evidence) == 1
    assert report.evidence[0].citation_label == "[1]"
    assert any(ref.reference_type == "external_literature" for ref in report.references)
    assert "benchmark" in " ".join(report.open_questions).lower()
    assert report.limitations


def test_synthesizer_service(bundle: ResearchEvidenceBundle, plan: ResearchPlan, critique: CritiqueResult, tmp_path):
    llm_output = SynthesizerLLMOutput(
        executive_summary="Hybrid retrieval is used in agentic RAG [1].",
        key_findings=[
            ReportFinding(
                statement="Hybrid retrieval is used in agentic RAG systems.",
                basis="evidence",
                evidence_ids=["ev-1"],
            )
        ],
    )
    service = ResearchSynthesizerService(MockSynthesizerLLM(llm_output), Settings(data_processed_dir=tmp_path))
    report = service.synthesize(question=plan.original_question, plan=plan, bundle=bundle, critique=critique)
    assert report.question == plan.original_question
    assert report.key_findings[0].citation_labels == ["[1]"]
    assert report.references


def test_synthesize_api_with_dependency_override(
    bundle: ResearchEvidenceBundle,
    plan: ResearchPlan,
    critique: CritiqueResult,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    llm_output = SynthesizerLLMOutput(
        executive_summary="Summary with citation [1].",
        key_findings=[
            ReportFinding(
                statement="Finding",
                basis="evidence",
                evidence_ids=["ev-1"],
            )
        ],
    )
    settings = Settings(data_processed_dir=tmp_path, openrouter_api_key="test-key")
    service = ResearchSynthesizerService(MockSynthesizerLLM(llm_output), settings)

    from app.config.settings import get_settings
    from app.main import create_app
    from app.services.research_synthesizer_service import get_research_synthesizer_service

    get_settings.cache_clear()
    get_research_synthesizer_service.cache_clear()
    app = create_app()
    app.dependency_overrides[get_research_synthesizer_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/research/synthesize",
        json={
            "question": plan.original_question,
            "plan": plan.model_dump(mode="json"),
            "bundle": bundle.model_dump(mode="json"),
            "critique": critique.model_dump(mode="json"),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["executive_summary"]
    assert body["evidence"]
    assert body["references"]
    assert body["open_questions"]
