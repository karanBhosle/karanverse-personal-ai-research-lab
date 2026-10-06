import logging
from functools import lru_cache

from app.agents.synthesizer.graph import compile_synthesizer_agent
from app.config.settings import Settings, get_settings
from app.llm.factory import create_llm_provider
from app.llm.provider import LLMProvider
from app.models.critique import CritiqueResult
from app.models.research_evidence import ResearchEvidenceBundle
from app.models.research_plan import ResearchPlan
from app.models.research_report import ResearchReport

logger = logging.getLogger(__name__)


class ResearchSynthesizerService:
    """LangGraph synthesizer: grounded report from plan, evidence, and critic."""

    def __init__(self, llm: LLMProvider, settings: Settings) -> None:
        self._graph = compile_synthesizer_agent(llm, settings)

    def synthesize(
        self,
        *,
        question: str,
        plan: ResearchPlan,
        bundle: ResearchEvidenceBundle,
        critique: CritiqueResult,
    ) -> ResearchReport:
        result = self._graph.invoke(
            {
                "question": question.strip(),
                "plan": plan,
                "bundle": bundle,
                "critique": critique,
            }
        )
        report = result.get("report")
        if report is None:
            raise RuntimeError("Synthesizer agent graph did not produce a report")
        return report


@lru_cache
def get_research_synthesizer_service() -> ResearchSynthesizerService:
    settings = get_settings()
    return ResearchSynthesizerService(llm=create_llm_provider(settings), settings=settings)
