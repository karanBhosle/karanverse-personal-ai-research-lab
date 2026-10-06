import logging

from app.agents.research_agent.deps import ResearchAgentDeps
from app.agents.research_agent.state import ResearchAgentState
from app.models.evidence import SourceMetadata
from app.models.research_evidence import ResearchEvidenceBundle, RetrievalScoreRecord
from app.models.research_paper import ResearchPaper
from app.models.search import HybridSearchResult, RerankedSearchResult
from app.services.citation_service import CitationService
from app.security.knowledge_guard import research_retrieval_scopes, sanitize_literature_search_query
from app.sources.exceptions import ResearchSourceError

logger = logging.getLogger(__name__)


def _dedupe_hybrid(hits: list[HybridSearchResult]) -> list[HybridSearchResult]:
    best: dict[str, HybridSearchResult] = {}
    for hit in hits:
        existing = best.get(hit.chunk_id)
        if existing is None or hit.final_score > existing.final_score:
            best[hit.chunk_id] = hit
    return sorted(best.values(), key=lambda item: item.final_score, reverse=True)


def _dedupe_reranked(hits: list[RerankedSearchResult]) -> list[RerankedSearchResult]:
    best: dict[str, RerankedSearchResult] = {}
    for hit in hits:
        existing = best.get(hit.chunk_id)
        if existing is None or hit.final_score > existing.final_score:
            best[hit.chunk_id] = hit
    return sorted(best.values(), key=lambda item: item.final_score, reverse=True)


def run_research_planner(state: ResearchAgentState, deps: ResearchAgentDeps) -> dict:
    question = state["question"]
    plan = deps.plan_question(question)
    return {"plan": plan}


def run_sub_question_generation(state: ResearchAgentState, deps: ResearchAgentDeps) -> dict:
    plan = state["plan"]
    sub_questions = [q.strip() for q in plan.sub_questions if q.strip()]
    if not sub_questions:
        sub_questions = [state["question"].strip()]
    logger.info("Research agent sub-questions=%d", len(sub_questions))
    return {"sub_questions": sub_questions}


def run_literature_search(state: ResearchAgentState, deps: ResearchAgentDeps) -> dict:
    plan = state["plan"]
    per_query = deps.settings.research_agent_papers_per_query
    papers_by_key: dict[tuple[str, str], ResearchPaper] = {}
    for existing in state.get("papers") or []:
        papers_by_key[(existing.source, existing.external_id)] = existing

    queries = state.get("active_search_queries") or plan.search_queries
    for query in queries:
        public_query = sanitize_literature_search_query(query)
        for source in plan.required_sources:
            try:
                batch = deps.source_registry.search_papers(source, public_query, limit=per_query)
            except ResearchSourceError as exc:
                logger.warning(
                    "Literature search failed source=%s query=%r: %s",
                    source,
                    query,
                    exc,
                )
                continue
            for paper in batch:
                papers_by_key[(paper.source, paper.external_id)] = paper

    papers = list(papers_by_key.values())
    logger.info("Research agent literature papers=%d", len(papers))
    return {"papers": papers}


def _knowledge_scopes_for_state(state: ResearchAgentState, deps: ResearchAgentDeps):
    allow_mixed = bool(state.get("allow_mixed_sources"))
    user_allow_private = bool(state.get("allow_private_knowledge"))
    return research_retrieval_scopes(
        state["question"],
        deps.settings,
        user_allow_private=user_allow_private,
        allow_mixed=allow_mixed,
    )


def run_document_retrieval(state: ResearchAgentState, deps: ResearchAgentDeps) -> dict:
    pool = deps.settings.research_agent_document_pool
    scopes = _knowledge_scopes_for_state(state, deps)
    queries = [state["question"]]
    for paper in (state.get("papers") or [])[: deps.settings.research_agent_max_paper_title_queries]:
        if paper.title:
            queries.append(paper.title)

    hits: list[HybridSearchResult] = []
    for query in queries:
        hits.extend(
            deps.hybrid_retriever.retrieve(
                query,
                top_k=pool,
                candidate_count=max(pool, deps.settings.hybrid_candidate_pool),
                knowledge_scopes=scopes,
            )
        )

    document_candidates = _dedupe_hybrid(hits)
    logger.info("Research agent document candidates=%d", len(document_candidates))
    return {"document_candidates": document_candidates}


def run_hybrid_retrieval(state: ResearchAgentState, deps: ResearchAgentDeps) -> dict:
    sub_questions = state["sub_questions"]
    pool = deps.settings.hybrid_candidate_pool
    top_k = deps.settings.research_agent_hybrid_top_k
    scopes = _knowledge_scopes_for_state(state, deps)
    hybrid_candidates_by_sub_question: dict[str, list[HybridSearchResult]] = {}

    for sub_question in sub_questions:
        hits = deps.hybrid_retriever.retrieve(
            sub_question,
            top_k=top_k,
            candidate_count=pool,
            knowledge_scopes=scopes,
        )
        hybrid_candidates_by_sub_question[sub_question] = _dedupe_hybrid(hits)

    logger.info("Research agent hybrid retrieval sub_questions=%d", len(hybrid_candidates_by_sub_question))
    return {"hybrid_candidates_by_sub_question": hybrid_candidates_by_sub_question}


def run_reranking(state: ResearchAgentState, deps: ResearchAgentDeps) -> dict:
    sub_questions = state["sub_questions"]
    document_candidates = state.get("document_candidates") or []
    hybrid_map = state.get("hybrid_candidates_by_sub_question") or {}
    top_k = state.get("evidence_top_k") or deps.settings.research_agent_evidence_top_k

    reranked_by_sub_question: dict[str, list[RerankedSearchResult]] = {}
    for sub_question in sub_questions:
        sub_hits = hybrid_map.get(sub_question, [])
        pool = _dedupe_hybrid(document_candidates + sub_hits)
        if not pool:
            reranked_by_sub_question[sub_question] = []
            continue
        reranked_by_sub_question[sub_question] = deps.reranker.rerank(
            sub_question,
            pool,
            top_k=top_k,
        )

    logger.info("Research agent reranked sub_questions=%d", len(reranked_by_sub_question))
    return {"reranked_by_sub_question": reranked_by_sub_question}


def run_evidence_collection(state: ResearchAgentState, deps: ResearchAgentDeps) -> dict:
    reranked_map = state.get("reranked_by_sub_question") or {}
    min_hits = deps.settings.research_agent_min_evidence_hits
    unresolved: list[str] = []

    citation_service = CitationService(processed_dir=deps.settings.data_processed_dir)
    retrieval_scores: list[RetrievalScoreRecord] = []
    all_reranked: list[RerankedSearchResult] = []

    for sub_question, hits in reranked_map.items():
        if len(hits) < min_hits:
            unresolved.append(sub_question)
        for hit in hits:
            all_reranked.append(hit)
            retrieval_scores.append(
                RetrievalScoreRecord(
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    sub_question=sub_question,
                    hybrid_score=hit.hybrid_score,
                    bm25_score=hit.bm25_score,
                    vector_score=hit.vector_score,
                    rerank_score=hit.rerank_score,
                    final_score=hit.final_score,
                )
            )

    unique_reranked = _dedupe_reranked(all_reranked)
    citations = citation_service.register_reranked_results(unique_reranked)
    evidence = [citation_service.get_evidence(c.evidence_id) for c in citations]

    source_metadata: list[SourceMetadata] = []
    seen_sources: set[tuple[str, str, str]] = set()
    for item in evidence:
        key = (item.document_id, item.chunk_id, item.source_metadata.filename)
        if key in seen_sources:
            continue
        seen_sources.add(key)
        source_metadata.append(item.source_metadata)

    bundle = ResearchEvidenceBundle(
        question=state["question"],
        plan=state["plan"],
        papers=state.get("papers") or [],
        evidence=evidence,
        citations=citations,
        source_metadata=source_metadata,
        retrieval_scores=retrieval_scores,
        unresolved_questions=unresolved,
    )
    logger.info(
        "Research evidence bundle evidence=%d papers=%d unresolved=%d",
        len(evidence),
        len(bundle.papers),
        len(unresolved),
    )
    return {"bundle": bundle}
