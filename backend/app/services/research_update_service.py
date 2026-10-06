import logging
from datetime import UTC, datetime
from functools import lru_cache

from app.agents.research_agent.nodes import _dedupe_hybrid, _dedupe_reranked
from app.agents.research_update.claims import extract_previous_conclusions
from app.agents.research_update.context import evidence_to_context_item, paper_to_context_item
from app.agents.research_update.evidence_overlap import mean_max_jaccard, novel_evidence_ids
from app.agents.research_update.prompts import (
    build_update_comparison_system_prompt,
    build_update_comparison_user_prompt,
    build_update_queries_system_prompt,
    build_update_queries_user_prompt,
)
from app.agents.research_update.validation import merge_update
from app.config.settings import Settings, get_settings
from app.llm.factory import create_llm_provider
from app.llm.provider import LLMProvider
from app.llm.types import LLMMessage, LLMRequest
from app.models.research_history import ResearchHistoryRecord
from app.models.research_paper import ResearchPaper
from app.models.research_update import ResearchUpdate, ResearchUpdateRequest, UpdateComparisonLLMOutput, UpdateQueriesLLMOutput
from app.models.search import HybridSearchResult
from app.retrieval.hybrid_retriever import HybridRetriever, get_hybrid_retriever
from app.retrieval.knowledge_scope import infer_knowledge_scopes
from app.retrieval.reranker import Reranker, create_cross_encoder_reranker
from app.services.citation_service import CitationService
from app.services.research_history_service import ResearchHistoryService, create_research_history_service
from app.sources.exceptions import ResearchSourceError
from app.sources.registry import SourceRegistry, get_source_registry

logger = logging.getLogger(__name__)


class ResearchUpdateService:
    def __init__(
        self,
        settings: Settings,
        llm: LLMProvider,
        history_service: ResearchHistoryService,
        source_registry: SourceRegistry,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._history = history_service
        self._sources = source_registry
        self._hybrid = hybrid_retriever
        self._reranker = reranker

    def _resolve_record(self, request: ResearchUpdateRequest) -> ResearchHistoryRecord:
        if request.research_id:
            record = self._history.get_record(request.research_id)
            if record is None:
                raise LookupError(f"Research run not found: {request.research_id}")
            return record
        if not request.question:
            raise ValueError("Either research_id or question is required to locate prior research.")
        matches = self._history.retrieve_relevant(request.question, top_k=1)
        if not matches:
            raise LookupError("No prior research history matches this question.")
        record = self._history.get_record(matches[0].research_id)
        if record is None:
            raise LookupError("Matched research history record is missing on disk.")
        return record

    def _generate_update_queries(
        self,
        record: ResearchHistoryRecord,
        conclusions: list[str],
        steer: str | None,
    ) -> UpdateQueriesLLMOutput:
        question = steer.strip() if steer else record.question
        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=build_update_queries_system_prompt()),
                LLMMessage(
                    role="user",
                    content=build_update_queries_user_prompt(
                        question=question,
                        previous_research_at=record.created_at.isoformat(),
                        conclusions=conclusions,
                        open_questions=list(
                            dict.fromkeys(record.unresolved_questions + record.report.open_questions)
                        ),
                        research_objective=record.plan.research_objective,
                    ),
                ),
            ],
            temperature=self._settings.research_update_temperature,
        )
        return self._llm.generate_structured(request, UpdateQueriesLLMOutput)

    def _search_literature(
        self,
        record: ResearchHistoryRecord,
        queries: list[str],
    ) -> list[ResearchPaper]:
        per_query = self._settings.research_agent_papers_per_query
        sources = record.plan.required_sources or self._sources.list_sources()
        papers_by_key: dict[tuple[str, str], ResearchPaper] = {}
        for query in queries:
            for source in sources:
                try:
                    batch = self._sources.search_papers(source, query, limit=per_query)
                except ResearchSourceError as exc:
                    logger.warning("Update literature search failed source=%s query=%r: %s", source, query, exc)
                    continue
                for paper in batch:
                    papers_by_key[(paper.source, paper.external_id)] = paper
        return list(papers_by_key.values())

    def _retrieve_new_evidence(
        self,
        record: ResearchHistoryRecord,
        queries: list[str],
        papers: list[ResearchPaper],
        *,
        evidence_top_k: int,
        steer: str | None,
    ) -> list:
        from app.models.evidence import Evidence

        question = steer.strip() if steer else record.question
        scopes = infer_knowledge_scopes(question, allow_mixed=False)
        pool = self._settings.research_agent_document_pool
        search_queries = list(dict.fromkeys([question] + queries))
        for paper in papers[: self._settings.research_agent_max_paper_title_queries]:
            if paper.title:
                search_queries.append(paper.title)

        hits: list[HybridSearchResult] = []
        for query in search_queries:
            hits.extend(
                self._hybrid.retrieve(
                    query,
                    top_k=self._settings.research_agent_hybrid_top_k,
                    candidate_count=max(pool, self._settings.hybrid_candidate_pool),
                    knowledge_scopes=scopes,
                )
            )
        candidates = _dedupe_hybrid(hits)
        if not candidates:
            return []
        reranked = self._reranker.rerank(question, candidates, top_k=evidence_top_k)
        unique = _dedupe_reranked(reranked)
        citation_service = CitationService(processed_dir=self._settings.data_processed_dir)
        citations = citation_service.register_reranked_results(unique)
        return [citation_service.get_evidence(citation.evidence_id) for citation in citations]

    def _compare(
        self,
        record: ResearchHistoryRecord,
        conclusions: list[str],
        new_evidence: list,
        papers: list[ResearchPaper],
        overlap_score: float,
        novel_ids: list[str],
        steer: str | None,
    ) -> UpdateComparisonLLMOutput:
        max_chars = self._settings.research_update_max_excerpt_chars
        prior_context = [
            evidence_to_context_item(item, max_chars=max_chars) for item in record.evidence
        ]
        new_context = [evidence_to_context_item(item, max_chars=max_chars) for item in new_evidence]
        literature_context = [paper_to_context_item(paper) for paper in papers[:10]]
        question = steer.strip() if steer else record.question
        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=build_update_comparison_system_prompt()),
                LLMMessage(
                    role="user",
                    content=build_update_comparison_user_prompt(
                        question=question,
                        previous_conclusions=conclusions,
                        prior_evidence=prior_context,
                        new_evidence=new_context,
                        new_literature=literature_context,
                        overlap_score=overlap_score,
                        novel_evidence_ids=novel_ids,
                    ),
                ),
            ],
            temperature=self._settings.research_update_temperature,
        )
        return self._llm.generate_structured(request, UpdateComparisonLLMOutput)

    def run(self, request: ResearchUpdateRequest) -> ResearchUpdate:
        record = self._resolve_record(request)
        conclusions = extract_previous_conclusions(record)
        top_k = request.evidence_top_k or self._settings.research_agent_evidence_top_k

        query_plan = self._generate_update_queries(record, conclusions, request.question)
        literature_queries = [q.strip() for q in query_plan.literature_search_queries if q.strip()]
        local_queries = [q.strip() for q in query_plan.local_retrieval_queries if q.strip()]
        all_queries = list(dict.fromkeys(literature_queries + local_queries))

        papers: list[ResearchPaper] = []
        if request.include_literature_search and literature_queries:
            papers = self._search_literature(record, literature_queries)

        new_evidence = self._retrieve_new_evidence(
            record,
            local_queries or all_queries,
            papers,
            evidence_top_k=top_k,
            steer=request.question,
        )

        overlap = mean_max_jaccard(record.evidence, new_evidence)
        novel_ids = novel_evidence_ids(record.evidence, new_evidence)

        allowed_ids = {item.evidence_id for item in record.evidence} | {
            item.evidence_id for item in new_evidence
        }
        comparison = self._compare(
            record,
            conclusions,
            new_evidence,
            papers,
            overlap,
            novel_ids,
            request.question,
        )

        base = ResearchUpdate(
            research_id=record.research_id,
            question=request.question.strip() if request.question else record.question,
            previous_research_at=record.created_at,
            update_generated_at=datetime.now(UTC),
            previous_conclusion=conclusions,
            new_evidence=new_evidence,
            new_literature=papers,
            update_queries_used=all_queries,
            evidence_overlap_score=round(overlap, 4),
        )
        return merge_update(base=base, comparison=comparison, allowed_evidence_ids=allowed_ids)


def create_research_update_service(
    settings: Settings,
    *,
    llm: LLMProvider | None = None,
    history_service: ResearchHistoryService | None = None,
    source_registry: SourceRegistry | None = None,
    hybrid_retriever: HybridRetriever | None = None,
    reranker: Reranker | None = None,
) -> ResearchUpdateService:
    return ResearchUpdateService(
        settings=settings,
        llm=llm or create_llm_provider(settings),
        history_service=history_service or create_research_history_service(settings),
        source_registry=source_registry or get_source_registry(),
        hybrid_retriever=hybrid_retriever or get_hybrid_retriever(),
        reranker=reranker or create_cross_encoder_reranker(settings.reranker_model),
    )


@lru_cache
def get_research_update_service() -> ResearchUpdateService:
    settings = get_settings()
    return create_research_update_service(settings)
