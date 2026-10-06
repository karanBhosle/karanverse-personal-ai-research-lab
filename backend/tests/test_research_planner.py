from typing import TypeVar

import pytest
from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.llm.provider import LLMProvider
from app.llm.types import LLMMessage, LLMRequest, LLMResponse
from app.models.research_plan import ResearchPlanLLMOutput
from app.services.research_planner_service import ResearchPlannerService
from app.sources.registry import SourceRegistry, create_source_registry

T = TypeVar("T")


class MockPlannerLLM(LLMProvider):
    def __init__(self, output: ResearchPlanLLMOutput, *, capture_request: bool = False) -> None:
        self._output = output
        self.capture_request = capture_request
        self.last_request: LLMRequest | None = None

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def default_model(self) -> str:
        return "mock-model"

    def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content=self._output.model_dump_json(), model="mock-model", provider="mock")

    def generate_structured(self, request: LLMRequest, response_model: type[T]) -> T:
        if self.capture_request:
            self.last_request = request
        return response_model.model_validate(self._output.model_dump())


@pytest.fixture
def agentic_rag_plan_output() -> ResearchPlanLLMOutput:
    return ResearchPlanLLMOutput(
        research_objective=(
            "Survey and compare leading Agentic RAG approaches as of 2026, including architecture patterns, "
            "retrieval strategies, and evaluation practices—without concluding which is universally best."
        ),
        sub_questions=[
            "What agentic RAG architectures combine tool use, planning, and retrieval loops?",
            "How are hybrid sparse-dense retrieval and reranking used in agentic RAG systems?",
            "What graph-based or knowledge-graph RAG patterns appear in recent agent stacks?",
            "How are agentic RAG systems evaluated (benchmarks, human eval, task success)?",
            "What production challenges (latency, cost, safety, observability) are reported?",
        ],
        search_queries=[
            "agentic RAG architecture 2025 2026",
            "hybrid retrieval reciprocal rank fusion agent",
            "graph RAG knowledge graph retrieval agent",
            "agentic RAG evaluation benchmark",
            "production agentic RAG latency cost",
        ],
        required_sources=["openalex", "arxiv"],
        expected_evidence=[
            "Survey or position papers on agentic RAG",
            "System papers describing retrieval + agent loops",
            "Benchmarks and ablation studies on retrieval components",
            "Engineering write-ups on deployment constraints",
        ],
        research_strategy=(
            "1) Run literature searches per query on openalex and arxiv. "
            "2) Cluster papers by theme (architecture, retrieval, graph RAG, eval, production). "
            "3) Extract methods and reported tradeoffs into an evidence table. "
            "4) Synthesize only after evidence collection—not during planning."
        ),
    )


@pytest.fixture
def planner_registry(tmp_path) -> SourceRegistry:
    settings = Settings(
        data_processed_dir=tmp_path,
        openalex_cache_enabled=False,
        arxiv_cache_enabled=False,
    )
    return create_source_registry(settings)


def test_research_planner_creates_structured_plan(
    agentic_rag_plan_output: ResearchPlanLLMOutput,
    planner_registry: SourceRegistry,
    tmp_path,
):
    llm = MockPlannerLLM(agentic_rag_plan_output, capture_request=True)
    settings = Settings(data_processed_dir=tmp_path)
    service = ResearchPlannerService(llm, planner_registry, settings)

    question = "What are the best approaches for Agentic RAG in 2026?"
    plan = service.create_plan(question)

    assert plan.original_question == question
    assert plan.research_objective
    assert len(plan.sub_questions) == 5
    assert "agentic rag" in plan.sub_questions[0].lower()
    assert plan.search_queries
    assert plan.required_sources == ["openalex", "arxiv"]
    assert plan.expected_evidence
    assert "evidence" in plan.research_strategy.lower() or "literature" in plan.research_strategy.lower()

    assert llm.last_request is not None
    system_prompt = llm.last_request.messages[0].content
    user_prompt = llm.last_request.messages[1].content
    assert "must NOT answer" in system_prompt or "Do not" in system_prompt
    assert question in user_prompt
    assert "Do not answer" in user_prompt


def test_research_planner_normalizes_unknown_sources(
    agentic_rag_plan_output: ResearchPlanLLMOutput,
    planner_registry: SourceRegistry,
    tmp_path,
):
    output = agentic_rag_plan_output.model_copy(update={"required_sources": ["openalex", "semantic_scholar"]})
    llm = MockPlannerLLM(output)
    settings = Settings(data_processed_dir=tmp_path)
    service = ResearchPlannerService(llm, planner_registry, settings)

    plan = service.create_plan("Test question?")
    assert plan.required_sources == ["openalex"]


def test_research_plan_api_with_dependency_override(
    agentic_rag_plan_output: ResearchPlanLLMOutput,
    planner_registry: SourceRegistry,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    llm = MockPlannerLLM(agentic_rag_plan_output)
    settings = Settings(
        data_processed_dir=tmp_path,
        openrouter_api_key="test-key",
    )
    service = ResearchPlannerService(llm, planner_registry, settings)

    from app.config.settings import get_settings
    from app.main import create_app
    from app.services.research_planner_service import get_research_planner_service

    get_settings.cache_clear()
    get_research_planner_service.cache_clear()
    app = create_app()
    app.dependency_overrides[get_research_planner_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/research/plan",
        json={"question": "What are the best approaches for Agentic RAG in 2026?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["plan"]["original_question"].startswith("What are the best approaches")
    assert len(body["plan"]["sub_questions"]) == 5
    assert "arxiv" in body["plan"]["required_sources"]
