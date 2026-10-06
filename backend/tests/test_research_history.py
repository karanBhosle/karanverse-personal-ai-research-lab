import uuid

import pytest
from fastapi.testclient import TestClient

from app.agents.research_planner.prompts import build_planner_user_prompt
from app.config.settings import Settings
from app.models.evidence import Evidence, RetrievalMethod, SourceMetadata
from app.models.research_evidence import ResearchEvidenceBundle
from app.models.research_paper import PublicationInfo, ResearchPaper
from app.models.research_plan import ResearchPlan
from app.models.research_report import ReportFinding, ResearchReport
from app.services.research_history_service import ResearchHistoryService, create_research_history_service


def _plan(question: str, objective: str, *, search_queries: list[str] | None = None) -> ResearchPlan:
    return ResearchPlan(
        original_question=question,
        research_objective=objective,
        sub_questions=["What methods apply?"],
        search_queries=search_queries or [question],
        required_sources=["openalex"],
        expected_evidence=["Surveys"],
        research_strategy="Search then synthesize.",
    )


def _report(question: str) -> ResearchReport:
    if "sourdough" in question.lower():
        summary = "Sourdough baking relies on long fermentation and starter maintenance."
    else:
        summary = "Hybrid retrieval and agent planners are common in agentic RAG stacks."
    return ResearchReport(
        question=question,
        executive_summary=summary,
        key_findings=[
            ReportFinding(
                statement="Agentic RAG often combines hybrid retrieval with tool-using agents.",
                basis="evidence",
                evidence_ids=["ev-1"],
            )
        ],
        open_questions=(
            ["How long should sourdough bulk ferment?"]
            if "sourdough" in question.lower()
            else ["Which agentic RAG benchmarks matter in 2026?"]
        ),
        references=[],
    )


def _bundle(plan: ResearchPlan) -> ResearchEvidenceBundle:
    if "sourdough" in plan.original_question.lower():
        text = "Maintaining a sourdough starter requires regular feeding and temperature control."
    else:
        text = "Hybrid retrieval supports agentic RAG."
    evidence = Evidence(
        evidence_id="ev-1",
        document_id="doc-1",
        chunk_id=str(uuid.uuid4()),
        source="local",
        title="Notes",
        text=text,
        retrieval_score=0.9,
        retrieval_method=RetrievalMethod.HYBRID_RERANK,
        source_metadata=SourceMetadata(filename="notes.pdf", source_type="pdf"),
    )
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
        evidence=[evidence],
        citations=[],
        unresolved_questions=["How is retrieval evaluated?"],
    )


@pytest.fixture
def history_service(tmp_path) -> ResearchHistoryService:
    return create_research_history_service(
        Settings(data_processed_dir=tmp_path, research_history_dir=tmp_path / "research_history")
    )


def test_save_and_get_record(history_service: ResearchHistoryService):
    plan = _plan("What is agentic RAG?", "Understand agentic RAG.")
    bundle = _bundle(plan)
    report = _report(plan.original_question)
    saved = history_service.save_run(
        question=plan.original_question,
        plan=plan,
        bundle=bundle,
        critique=None,
        report=report,
    )
    loaded = history_service.get_record(saved.research_id)
    assert loaded is not None
    assert loaded.research_id == saved.research_id
    assert loaded.plan.research_objective == plan.research_objective
    assert loaded.sources
    assert loaded.evidence
    assert loaded.conclusions
    assert loaded.unresolved_questions


def test_retrieve_relevant_history_uses_token_overlap(history_service: ResearchHistoryService):
    q1 = "What are hybrid retrieval patterns for agentic RAG?"
    q2 = "How do I bake sourdough bread at home?"
    for question, objective in (
        (q1, "Survey hybrid retrieval for agentic RAG systems."),
        (q2, "Learn sourdough fermentation techniques."),
    ):
        plan = _plan(
            question,
            objective,
            search_queries=["sourdough starter fermentation"] if "sourdough" in question else ["hybrid retrieval agentic RAG"],
        )
        history_service.save_run(
            question=question,
            plan=plan,
            bundle=_bundle(plan),
            critique=None,
            report=_report(question),
        )

    snippets = history_service.retrieve_relevant(
        "agentic RAG hybrid retrieval benchmarks",
        top_k=2,
    )
    assert snippets
    assert snippets[0].research_id
    assert "agentic" in snippets[0].question.lower() or "hybrid" in snippets[0].excerpt.lower()
    assert snippets[0].score >= (snippets[1].score if len(snippets) > 1 else 0)


def test_retrieve_does_not_return_full_report_body(history_service: ResearchHistoryService):
    plan = _plan("Agentic RAG overview", "Map the landscape.")
    long_summary = "X" * 5000
    report = _report(plan.original_question)
    report = report.model_copy(update={"executive_summary": long_summary})
    history_service.save_run(
        question=plan.original_question,
        plan=plan,
        bundle=_bundle(plan),
        critique=None,
        report=report,
    )
    snippet = history_service.retrieve_relevant("agentic RAG overview", top_k=1)[0]
    assert len(snippet.excerpt) <= 400


def test_planner_prompt_includes_retrieved_snippets_not_full_report():
    history_block = "[Prior research abc]\nQuestion: Agentic RAG\nBrief excerpt: short"
    prompt = build_planner_user_prompt("New question", history_context=history_block)
    assert "Prior research abc" in prompt
    assert "retrieved for overlap" in prompt


def test_list_and_get_api(history_service: ResearchHistoryService):
    from app.main import create_app
    from app.services.research_history_service import get_research_history_service

    plan = _plan("List endpoint test", "Test persistence.")
    record = history_service.save_run(
        question=plan.original_question,
        plan=plan,
        bundle=_bundle(plan),
        critique=None,
        report=_report(plan.original_question),
    )

    app = create_app()
    app.dependency_overrides[get_research_history_service] = lambda: history_service
    client = TestClient(app)

    list_response = client.get("/research/history")
    assert list_response.status_code == 200
    payload = list_response.json()
    assert payload["count"] >= 1
    assert any(item["research_id"] == record.research_id for item in payload["items"])

    get_response = client.get(f"/research/{record.research_id}")
    assert get_response.status_code == 200
    body = get_response.json()
    assert body["research_id"] == record.research_id
    assert body["report"]["executive_summary"]

    missing = client.get("/research/does-not-exist")
    assert missing.status_code == 404
