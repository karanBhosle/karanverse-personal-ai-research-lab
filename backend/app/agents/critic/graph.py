import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.critic.analysis import DeterministicBundleAnalysis, analyze_bundle
from app.agents.critic.context import build_critic_context
from app.agents.critic.prompts import build_critic_system_prompt, build_critic_user_prompt
from app.agents.critic.validation import merge_critique
from app.config.settings import Settings
from app.llm.provider import LLMProvider
from app.llm.types import LLMMessage, LLMRequest
from app.models.critique import CritiqueLLMOutput, CritiqueResult
from app.models.research_evidence import ResearchEvidenceBundle

logger = logging.getLogger(__name__)


class CriticState(TypedDict, total=False):
    bundle: ResearchEvidenceBundle
    analysis: DeterministicBundleAnalysis
    critique: CritiqueResult


def compile_critic_agent(
    llm: LLMProvider,
    settings: Settings,
    *,
    temperature: float | None = None,
):
    temp = temperature if temperature is not None else settings.critic_temperature

    def analyze_node(state: CriticState) -> dict:
        bundle = state["bundle"]
        analysis = analyze_bundle(bundle, settings)
        return {"analysis": analysis}

    def critique_node(state: CriticState) -> dict:
        bundle = state["bundle"]
        analysis = state["analysis"]
        context = build_critic_context(bundle, max_excerpt_chars=settings.critic_max_excerpt_chars)
        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=build_critic_system_prompt()),
                LLMMessage(role="user", content=build_critic_user_prompt(context)),
            ],
            temperature=temp,
        )
        llm_output = llm.generate_structured(request, CritiqueLLMOutput)
        critique = merge_critique(bundle, analysis, llm_output)
        logger.info(
            "Critic completed sufficient=%s duplicates=%d queries=%d",
            critique.evidence_sufficient,
            len(critique.duplicate_evidence),
            len(critique.additional_research_queries),
        )
        return {"critique": critique}

    graph = StateGraph(CriticState)
    graph.add_node("analyze_bundle", analyze_node)
    graph.add_node("critique", critique_node)
    graph.add_edge(START, "analyze_bundle")
    graph.add_edge("analyze_bundle", "critique")
    graph.add_edge("critique", END)
    return graph.compile()
