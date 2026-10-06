import logging
from functools import lru_cache

from app.agents.research_agent.deps import ResearchAgentDeps
from app.agents.research_agent.graph import compile_research_agent
from app.config.settings import Settings, get_settings
from app.models.research_evidence import ResearchEvidenceBundle
from app.retrieval.hybrid_retriever import HybridRetriever, get_hybrid_retriever
from app.retrieval.reranker import Reranker, create_cross_encoder_reranker
from app.services.research_planner_service import ResearchPlannerService, get_research_planner_service
from app.sources.registry import SourceRegistry, get_source_registry

logger = logging.getLogger(__name__)


class ResearchAgentService:
    """LangGraph research agent: plan → search → retrieve → rerank → evidence (no synthesis)."""

    def __init__(self, deps: ResearchAgentDeps) -> None:
        self._deps = deps
        self._graph = compile_research_agent(deps)

    def collect_evidence(
        self,
        question: str,
        *,
        evidence_top_k: int | None = None,
    ) -> ResearchEvidenceBundle:
        top_k = evidence_top_k or self._deps.settings.research_agent_evidence_top_k
        result = self._graph.invoke(
            {
                "question": question.strip(),
                "evidence_top_k": top_k,
            }
        )
        bundle = result.get("bundle")
        if bundle is None:
            raise RuntimeError("Research agent graph did not produce an evidence bundle")
        return bundle


def create_research_agent_service(
    settings: Settings,
    *,
    planner: ResearchPlannerService,
    source_registry: SourceRegistry | None = None,
    hybrid_retriever: HybridRetriever | None = None,
    reranker: Reranker | None = None,
) -> ResearchAgentService:
    deps = ResearchAgentDeps(
        settings=settings,
        plan_question=planner.create_plan,
        source_registry=source_registry or get_source_registry(),
        hybrid_retriever=hybrid_retriever or get_hybrid_retriever(),
        reranker=reranker or create_cross_encoder_reranker(settings.reranker_model),
    )
    return ResearchAgentService(deps)


@lru_cache
def get_research_agent_service() -> ResearchAgentService:
    settings = get_settings()
    return create_research_agent_service(settings, planner=get_research_planner_service())
