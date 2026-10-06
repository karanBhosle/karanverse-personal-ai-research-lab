import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.llm.provider import LLMProvider
from app.llm.types import LLMMessage, LLMRequest, LLMResponse
from app.main import create_app
from app.models.evidence import RetrievalMethod
from app.models.rag import GroundedAnswerLLMOutput
from app.models.search import RankContribution, RerankedSearchResult
from app.services.rag_service import GroundedRAGService


class _MockLLM(LLMProvider):
    def __init__(self, output: GroundedAnswerLLMOutput) -> None:
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


def _reranked_hit(text: str, score: float = 0.9) -> RerankedSearchResult:
    return RerankedSearchResult(
        chunk_id=str(uuid.uuid4()),
        document_id="doc-1",
        text=text,
        hybrid_score=0.04,
        bm25_score=1.2,
        vector_score=0.7,
        rank_contribution=RankContribution(bm25_rank=1, vector_rank=1, bm25_rrf=0.01, vector_rrf=0.01),
        metadata={
            "filename": "paper.pdf",
            "page_number": 2,
            "section": "Methods",
            "source_type": "pdf",
            "ingestion_timestamp": datetime(2026, 6, 1, tzinfo=UTC).isoformat(),
        },
        rerank_score=score,
        final_score=score,
    )


def test_grounded_rag_pipeline_returns_answer_and_citations(tmp_path):
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        _reranked_hit("Hybrid retrieval combines bm25 and dense vectors with reciprocal rank fusion.")
    ]
    llm = _MockLLM(
        GroundedAnswerLLMOutput(
            answer="Hybrid retrieval combines bm25 and dense vectors [1].",
            citation_numbers=[1],
            evidence_sufficient=True,
        )
    )
    settings = Settings(
        data_processed_dir=tmp_path,
        rerank_candidate_count=10,
        hybrid_candidate_pool=20,
        reranker_model="test-reranker",
        rrf_k=60,
    )
    service = GroundedRAGService(retriever=retriever, llm=llm, settings=settings)
    response = service.answer("What is hybrid retrieval?", top_k=3)

    assert "bm25" in response.answer.lower()
    assert response.evidence_sufficient is True
    assert len(response.evidence) == 1
    assert response.citations[0].label == "[1]"
    assert response.citations[0].evidence.retrieval_method == RetrievalMethod.HYBRID_RERANK
    assert response.retrieval_metadata.selected_evidence_count == 1
    retriever.retrieve.assert_called_once()


def test_grounded_rag_insufficient_when_no_retrieval_results(tmp_path):
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    llm = _MockLLM(
        GroundedAnswerLLMOutput(answer="should not be used", citation_numbers=[], evidence_sufficient=True)
    )
    settings = Settings(data_processed_dir=tmp_path)
    service = GroundedRAGService(retriever=retriever, llm=llm, settings=settings)
    response = service.answer("unknown topic", top_k=3)

    assert response.evidence_sufficient is False
    assert response.evidence == []
    assert "insufficient" in response.answer.lower()


def test_research_answer_api_with_dependency_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://example.test/v1")

    retriever = MagicMock()
    retriever.retrieve.return_value = [_reranked_hit("Evidence about bm25 sparse retrieval.")]
    llm = _MockLLM(
        GroundedAnswerLLMOutput(
            answer="bm25 is a sparse retrieval method [1].",
            citation_numbers=[1],
            evidence_sufficient=True,
        )
    )
    settings = Settings(
        data_processed_dir=tmp_path,
        openrouter_api_key="test-key",
        openrouter_base_url="https://example.test/v1",
    )
    rag_service = GroundedRAGService(retriever=retriever, llm=llm, settings=settings)

    from app.config.settings import get_settings
    from app.services.rag_service import get_grounded_rag_service

    get_settings.cache_clear()
    get_grounded_rag_service.cache_clear()

    app = create_app()
    app.dependency_overrides[get_grounded_rag_service] = lambda: rag_service

    client = TestClient(app)
    response = client.post(
        "/research/answer",
        json={"question": "What is bm25?", "top_k": 3},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["citations"]
    assert body["evidence"]
    assert body["retrieval_metadata"]["top_k"] == 3
