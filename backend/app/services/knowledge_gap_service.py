import logging
from functools import lru_cache

from app.agents.knowledge_gap.context import build_gap_context_json
from app.security.knowledge_guard import include_private_in_llm_context
from app.agents.knowledge_gap.prompts import build_knowledge_gap_system_prompt, build_knowledge_gap_user_prompt
from app.agents.knowledge_gap.signals import detect_gap_signals
from app.agents.knowledge_gap.topic import extract_focus_topic
from app.agents.knowledge_gap.validation import merge_llm_gaps, signals_to_fallback_gaps
from app.config.settings import Settings, get_settings
from app.llm.factory import create_llm_provider
from app.llm.provider import LLMProvider
from app.llm.types import LLMMessage, LLMRequest
from app.models.knowledge_gap import KnowledgeGapReport, KnowledgeGapRequest, KnowledgeGapLLMOutput
from app.models.knowledge_source import KnowledgeScope
from app.models.research_history import ResearchHistoryRecord
from app.retrieval.hybrid_retriever import HybridRetriever, get_hybrid_retriever
from app.services.research_history_service import ResearchHistoryService, create_research_history_service

logger = logging.getLogger(__name__)


class KnowledgeGapService:
    def __init__(
        self,
        settings: Settings,
        llm: LLMProvider,
        history_service: ResearchHistoryService,
        hybrid_retriever: HybridRetriever,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._history = history_service
        self._hybrid = hybrid_retriever

    def _retrieve_scoped(self, query: str, scope: KnowledgeScope, top_k: int):
        self._hybrid.bm25_retriever.ensure_index()
        return self._hybrid.retrieve(query, top_k=top_k, knowledge_scopes={scope})

    def _relevant_history(self, query: str, focus_topic: str, history_top_k: int) -> list[ResearchHistoryRecord]:
        snippets = self._history.retrieve_relevant(query, top_k=history_top_k)
        if not snippets:
            snippets = self._history.retrieve_relevant(focus_topic, top_k=history_top_k)
        records: list[ResearchHistoryRecord] = []
        for snippet in snippets:
            record = self._history.get_record(snippet.research_id)
            if record is not None:
                records.append(record)
        return records

    def analyze(self, request: KnowledgeGapRequest) -> KnowledgeGapReport:
        focus_topic = extract_focus_topic(request.query)
        search_query = f"{focus_topic} {request.query}".strip()

        history_records = self._relevant_history(request.query, focus_topic, request.history_top_k)
        all_history = self._history.list_records()
        if not history_records and all_history:
            history_records = all_history[: request.history_top_k]

        personal_hits = self._retrieve_scoped(search_query, KnowledgeScope.PERSONAL_KNOWLEDGE, request.knowledge_top_k)
        project_hits = self._retrieve_scoped(search_query, KnowledgeScope.PROJECT, request.knowledge_top_k)
        public_hits = self._retrieve_scoped(search_query, KnowledgeScope.PUBLIC_RESEARCH, request.knowledge_top_k)

        signals = detect_gap_signals(
            all_history,
            focus_topic=focus_topic,
            personal_hits=personal_hits,
            project_hits=project_hits,
            public_hits=public_hits,
            low_evidence_threshold=self._settings.knowledge_gap_low_evidence_threshold,
            weak_score_threshold=self._settings.knowledge_gap_weak_score_threshold,
        )

        allow_private_llm = include_private_in_llm_context(
            self._settings,
            user_opt_in=request.allow_private_knowledge,
        )
        context_json = build_gap_context_json(
            query=request.query,
            focus_topic=focus_topic,
            history_records=history_records,
            personal_hits=personal_hits,
            project_hits=project_hits,
            public_hits=public_hits,
            signals=signals,
            max_excerpt_chars=self._settings.knowledge_gap_max_excerpt_chars,
            allow_private_knowledge=allow_private_llm,
        )

        records_by_id = {record.research_id: record for record in all_history}
        allowed_ids = set(records_by_id)

        gaps: list = []
        coverage_summary = ""
        try:
            llm_request = LLMRequest(
                messages=[
                    LLMMessage(role="system", content=build_knowledge_gap_system_prompt()),
                    LLMMessage(role="user", content=build_knowledge_gap_user_prompt(context_json)),
                ],
                temperature=self._settings.knowledge_gap_temperature,
            )
            llm_output = self._llm.generate_structured(llm_request, KnowledgeGapLLMOutput)
            coverage_summary = llm_output.coverage_summary
            gaps = merge_llm_gaps(llm_output, records_by_id, allowed_research_ids=allowed_ids)
            if not gaps and signals:
                gaps = signals_to_fallback_gaps(signals, {r.research_id: r for r in all_history})
        except Exception as exc:
            logger.warning("Knowledge gap LLM synthesis failed, using deterministic gaps: %s", exc)
            gaps = signals_to_fallback_gaps(signals, {r.research_id: r for r in all_history})
            coverage_summary = (
                f"Deterministic analysis for '{focus_topic}' based on {len(history_records)} research run(s) "
                f"and {len(personal_hits) + len(project_hits) + len(public_hits)} knowledge chunk(s)."
            )

        return KnowledgeGapReport(
            query=request.query,
            focus_topic=focus_topic,
            gaps=gaps,
            coverage_summary=coverage_summary,
            research_runs_analyzed=len(history_records),
            knowledge_chunks_retrieved=len(personal_hits) + len(project_hits) + len(public_hits),
            detected_signals=[signal.message for signal in signals],
        )


def create_knowledge_gap_service(
    settings: Settings,
    *,
    llm: LLMProvider | None = None,
    history_service: ResearchHistoryService | None = None,
    hybrid_retriever: HybridRetriever | None = None,
) -> KnowledgeGapService:
    return KnowledgeGapService(
        settings=settings,
        llm=llm or create_llm_provider(settings),
        history_service=history_service or create_research_history_service(settings),
        hybrid_retriever=hybrid_retriever or get_hybrid_retriever(),
    )


@lru_cache
def get_knowledge_gap_service() -> KnowledgeGapService:
    return create_knowledge_gap_service(get_settings())
