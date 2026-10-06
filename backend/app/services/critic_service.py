import logging
from functools import lru_cache

from app.agents.critic.graph import compile_critic_agent
from app.config.settings import Settings, get_settings
from app.llm.factory import create_llm_provider
from app.llm.provider import LLMProvider
from app.models.critique import CritiqueResult
from app.models.research_evidence import ResearchEvidenceBundle

logger = logging.getLogger(__name__)


class CriticAgentService:
    """LangGraph critic: deterministic bundle analysis + structured LLM critique."""

    def __init__(self, llm: LLMProvider, settings: Settings) -> None:
        self._graph = compile_critic_agent(llm, settings)

    def critique(self, bundle: ResearchEvidenceBundle) -> CritiqueResult:
        result = self._graph.invoke({"bundle": bundle})
        critique = result.get("critique")
        if critique is None:
            raise RuntimeError("Critic agent graph did not produce a critique")
        return critique


@lru_cache
def get_critic_agent_service() -> CriticAgentService:
    settings = get_settings()
    return CriticAgentService(llm=create_llm_provider(settings), settings=settings)
