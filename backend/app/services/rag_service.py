import logging
import re
from functools import lru_cache

from app.config.settings import Settings, get_settings
from app.llm.factory import create_llm_provider
from app.llm.provider import LLMProvider
from app.models.evidence import Evidence
from app.models.rag import (
    CitationView,
    GroundedAnswerLLMOutput,
    ResearchAnswerResponse,
    RetrievalMetadata,
)
from app.models.search import RerankedSearchResult
from app.security.knowledge_guard import filter_evidence_for_llm, include_private_in_llm_context, research_retrieval_scopes
from app.retrieval.reranking_hybrid_retriever import RerankingHybridRetriever, get_reranking_hybrid_retriever
from app.services.citation_service import CitationService
from app.services.rag_context import (
    build_evidence_context,
    build_grounded_system_prompt,
    build_grounded_user_prompt,
)
from app.llm.types import LLMMessage, LLMRequest

logger = logging.getLogger(__name__)

_CITATION_IN_TEXT_RE = re.compile(r"\[(\d+)\]")


class GroundedRAGService:
    """Baseline grounded RAG: hybrid retrieval → rerank → evidence → LLM → citations."""

    def __init__(
        self,
        retriever: RerankingHybridRetriever,
        llm: LLMProvider,
        settings: Settings,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.settings = settings

    def answer(
        self,
        question: str,
        top_k: int = 5,
        *,
        allow_private_knowledge: bool = False,
    ) -> ResearchAnswerResponse:
        candidate_count = max(top_k, self.settings.rerank_candidate_count)
        scopes = research_retrieval_scopes(
            question,
            self.settings,
            user_allow_private=allow_private_knowledge,
            allow_mixed=False,
        )
        reranked = self.retriever.retrieve(
            question,
            final_top_k=top_k,
            candidate_count=candidate_count,
            knowledge_scopes=scopes,
        )
        selected = self._select_evidence(reranked, top_k=top_k)

        citation_service = CitationService(processed_dir=self.settings.data_processed_dir)
        citations = citation_service.register_reranked_results(selected)
        evidence_items = [citation_service.get_evidence(c.evidence_id) for c in citations]
        allow_private_llm = include_private_in_llm_context(
            self.settings,
            user_opt_in=allow_private_knowledge,
        )
        evidence_items = filter_evidence_for_llm(evidence_items, allow_private=allow_private_llm)
        citations = [c for c in citations if c.evidence_id in {e.evidence_id for e in evidence_items}]
        evidence_by_id = {item.evidence_id: item for item in evidence_items}

        metadata = RetrievalMetadata(
            question=question,
            top_k=top_k,
            candidate_count=candidate_count,
            hybrid_candidate_pool=self.settings.hybrid_candidate_pool,
            reranker_model=self.settings.reranker_model,
            rrf_k=self.settings.rrf_k,
            retrieved_after_rerank=len(reranked),
            selected_evidence_count=len(selected),
            scores=[
                {
                    "chunk_id": hit.chunk_id,
                    "rerank_score": hit.rerank_score,
                    "hybrid_score": hit.hybrid_score,
                    "bm25_score": hit.bm25_score,
                    "vector_score": hit.vector_score,
                }
                for hit in selected
            ],
        )

        if not selected:
            return ResearchAnswerResponse(
                answer=(
                    "The available evidence is insufficient to answer this question. "
                    "No relevant chunks were retrieved from the corpus."
                ),
                citations=[],
                evidence=[],
                retrieval_metadata=metadata,
                evidence_sufficient=False,
            )

        context = build_evidence_context(citations, evidence_by_id)
        llm_output = self._generate_grounded_answer(question, context)
        citation_views = self._build_citation_views(
            citation_service,
            llm_output,
            citations,
            evidence_by_id,
        )

        return ResearchAnswerResponse(
            answer=llm_output.answer,
            citations=citation_views,
            evidence=evidence_items,
            retrieval_metadata=metadata,
            evidence_sufficient=llm_output.evidence_sufficient,
        )

    @staticmethod
    def _select_evidence(
        reranked: list[RerankedSearchResult],
        *,
        top_k: int,
    ) -> list[RerankedSearchResult]:
        return reranked[:top_k]

    def _generate_grounded_answer(self, question: str, evidence_context: str) -> GroundedAnswerLLMOutput:
        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=build_grounded_system_prompt()),
                LLMMessage(
                    role="user",
                    content=build_grounded_user_prompt(question, evidence_context),
                ),
            ],
            temperature=0.1,
        )
        return self.llm.generate_structured(request, GroundedAnswerLLMOutput)

    def _build_citation_views(
        self,
        citation_service: CitationService,
        llm_output: GroundedAnswerLLMOutput,
        citations: list,
        evidence_by_id: dict[str, Evidence],
    ) -> list[CitationView]:
        numbers = set(llm_output.citation_numbers)
        numbers.update(int(match) for match in _CITATION_IN_TEXT_RE.findall(llm_output.answer))

        views: list[CitationView] = []
        for number in sorted(numbers):
            try:
                evidence = citation_service.resolve_citation_number(number)
            except KeyError:
                continue
            citation = next(
                (c for c in citations if c.evidence_id == evidence.evidence_id),
                None,
            )
            if citation is None:
                continue
            views.append(
                CitationView(
                    citation_number=citation.citation_number,
                    label=citation.label,
                    evidence=evidence_by_id[evidence.evidence_id],
                )
            )
        if not views:
            views = [
                CitationView(
                    citation_number=citation.citation_number,
                    label=citation.label,
                    evidence=evidence_by_id[citation.evidence_id],
                )
                for citation in citations
            ]
        return views


@lru_cache
def get_grounded_rag_service() -> GroundedRAGService:
    settings = get_settings()
    return GroundedRAGService(
        retriever=get_reranking_hybrid_retriever(),
        llm=create_llm_provider(settings),
        settings=settings,
    )
