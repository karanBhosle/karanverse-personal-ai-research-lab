import operator
from typing import Annotated, TypedDict

from app.models.critique import CritiqueResult
from app.models.research_evidence import ResearchEvidenceBundle
from app.models.research_plan import ResearchPlan
from app.models.research_report import ResearchReport
from app.models.research_state import NodeTraceEntry
from app.models.research_paper import ResearchPaper
from app.models.search import HybridSearchResult, RerankedSearchResult


class ResearchWorkflowState(TypedDict, total=False):
    question: str
    evidence_top_k: int
    iteration: int
    max_iterations: int
    use_research_history: bool
    allow_private_knowledge: bool
    history_top_k: int
    history_context_text: str
    history_context_snippets: list[dict]
    plan: ResearchPlan
    sub_questions: list[str]
    active_search_queries: list[str]
    papers: list[ResearchPaper]
    document_candidates: list[HybridSearchResult]
    hybrid_candidates_by_sub_question: dict[str, list[HybridSearchResult]]
    reranked_by_sub_question: dict[str, list[RerankedSearchResult]]
    bundle: ResearchEvidenceBundle
    critique: CritiqueResult
    report: ResearchReport
    node_traces: Annotated[list[NodeTraceEntry], operator.add]
