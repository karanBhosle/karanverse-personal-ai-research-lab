from collections.abc import Callable
from dataclasses import dataclass

from app.config.settings import Settings
from app.models.research_plan import ResearchPlan
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.sources.registry import SourceRegistry


@dataclass(frozen=True, slots=True)
class ResearchAgentDeps:
    settings: Settings
    plan_question: Callable[[str], ResearchPlan]
    source_registry: SourceRegistry
    hybrid_retriever: HybridRetriever
    reranker: Reranker
