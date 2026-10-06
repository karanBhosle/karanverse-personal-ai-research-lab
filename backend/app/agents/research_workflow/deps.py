from dataclasses import dataclass

from app.config.settings import Settings
from app.llm.provider import LLMProvider
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.services.research_history_service import ResearchHistoryService
from app.sources.registry import SourceRegistry


@dataclass(frozen=True, slots=True)
class ResearchWorkflowDeps:
    settings: Settings
    llm: LLMProvider
    source_registry: SourceRegistry
    hybrid_retriever: HybridRetriever
    reranker: Reranker
    history_service: ResearchHistoryService

    def agent_deps(self, plan_question):
        from app.agents.research_agent.deps import ResearchAgentDeps

        return ResearchAgentDeps(
            settings=self.settings,
            plan_question=plan_question,
            source_registry=self.source_registry,
            hybrid_retriever=self.hybrid_retriever,
            reranker=self.reranker,
        )
