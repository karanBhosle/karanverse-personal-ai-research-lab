import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.synthesizer.citations import CitationIndex, build_citation_index
from app.agents.synthesizer.context import build_synthesizer_context
from app.agents.synthesizer.prompts import build_synthesizer_system_prompt, build_synthesizer_user_prompt
from app.agents.synthesizer.validation import assemble_report
from app.config.settings import Settings
from app.llm.provider import LLMProvider
from app.llm.types import LLMMessage, LLMRequest
from app.models.critique import CritiqueResult
from app.models.research_evidence import ResearchEvidenceBundle
from app.models.research_plan import ResearchPlan
from app.models.research_report import ResearchReport, SynthesizerLLMOutput

logger = logging.getLogger(__name__)


class SynthesizerState(TypedDict, total=False):
    question: str
    plan: ResearchPlan
    bundle: ResearchEvidenceBundle
    critique: CritiqueResult
    citation_index: CitationIndex
    report: ResearchReport


def compile_synthesizer_agent(
    llm: LLMProvider,
    settings: Settings,
    *,
    temperature: float | None = None,
):
    temp = temperature if temperature is not None else settings.synthesizer_temperature

    def index_node(state: SynthesizerState) -> dict:
        bundle = state["bundle"]
        return {"citation_index": build_citation_index(bundle)}

    def synthesize_node(state: SynthesizerState) -> dict:
        question = state["question"]
        plan = state["plan"]
        bundle = state["bundle"]
        critique = state["critique"]
        index = state["citation_index"]
        context = build_synthesizer_context(
            question=question,
            plan=plan,
            bundle=bundle,
            critique=critique,
            citation_index=index,
            max_excerpt_chars=settings.synthesizer_max_excerpt_chars,
        )
        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=build_synthesizer_system_prompt()),
                LLMMessage(role="user", content=build_synthesizer_user_prompt(context)),
            ],
            temperature=temp,
        )
        llm_output = llm.generate_structured(request, SynthesizerLLMOutput)
        report = assemble_report(
            question=question,
            plan=plan,
            bundle=bundle,
            critique=critique,
            index=index,
            llm_output=llm_output,
        )
        logger.info(
            "Research report synthesized findings=%d references=%d",
            len(report.key_findings),
            len(report.references),
        )
        return {"report": report}

    graph = StateGraph(SynthesizerState)
    graph.add_node("index_citations", index_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_edge(START, "index_citations")
    graph.add_edge("index_citations", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()
