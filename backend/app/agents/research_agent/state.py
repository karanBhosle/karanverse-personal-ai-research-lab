from typing import TypedDict

from app.models.research_evidence import ResearchEvidenceBundle
from app.models.research_paper import ResearchPaper
from app.models.research_plan import ResearchPlan
from app.models.search import HybridSearchResult, RerankedSearchResult


class ResearchAgentState(TypedDict, total=False):
    question: str
    evidence_top_k: int
    allow_mixed_sources: bool
    allow_private_knowledge: bool
    plan: ResearchPlan
    sub_questions: list[str]
    papers: list[ResearchPaper]
    document_candidates: list[HybridSearchResult]
    hybrid_candidates_by_sub_question: dict[str, list[HybridSearchResult]]
    reranked_by_sub_question: dict[str, list[RerankedSearchResult]]
    bundle: ResearchEvidenceBundle
