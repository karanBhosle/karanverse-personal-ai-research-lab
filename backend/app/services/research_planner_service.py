import logging
from functools import lru_cache

from app.agents.research_planner.graph import compile_research_planner
from app.config.settings import Settings, get_settings
from app.llm.factory import create_llm_provider
from app.llm.provider import LLMProvider
from app.models.research_plan import ResearchPlan
from app.sources.registry import SourceRegistry, get_source_registry

logger = logging.getLogger(__name__)


class ResearchPlannerService:
    """Runs the Research Planner LangGraph agent."""

    def __init__(
        self,
        llm: LLMProvider,
        source_registry: SourceRegistry,
        settings: Settings,
    ) -> None:
        sources = source_registry.list_sources()
        self._graph = compile_research_planner(
            llm,
            available_sources=sources,
            temperature=settings.research_planner_temperature,
        )

    def create_plan(self, question: str) -> ResearchPlan:
        result = self._graph.invoke({"question": question, "plan": None})
        plan = result.get("plan")
        if plan is None:
            raise RuntimeError("Research planner graph did not produce a plan")
        return plan


@lru_cache
def get_research_planner_service() -> ResearchPlannerService:
    settings = get_settings()
    return ResearchPlannerService(
        llm=create_llm_provider(settings),
        source_registry=get_source_registry(),
        settings=settings,
    )
