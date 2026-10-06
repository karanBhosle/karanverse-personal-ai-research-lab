import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.research_planner.prompts import build_planner_system_prompt, build_planner_user_prompt
from app.llm.provider import LLMProvider
from app.llm.types import LLMMessage, LLMRequest, LLMRequestObservability
from app.models.research_plan import ResearchPlan, ResearchPlanLLMOutput

logger = logging.getLogger(__name__)


class PlannerState(TypedDict, total=False):
    question: str
    plan: ResearchPlan | None
    history_context: str | None


def _normalize_required_sources(sources: list[str], available: set[str]) -> list[str]:
    normalized: list[str] = []
    for raw in sources:
        key = raw.strip().lower()
        if key and key in available and key not in normalized:
            normalized.append(key)
    if normalized:
        return normalized
    return sorted(available)


def _llm_output_to_plan(question: str, output: ResearchPlanLLMOutput, available_sources: list[str]) -> ResearchPlan:
    available = {s.lower() for s in available_sources}
    return ResearchPlan(
        original_question=question.strip(),
        research_objective=output.research_objective.strip(),
        sub_questions=[q.strip() for q in output.sub_questions if q.strip()],
        search_queries=[q.strip() for q in output.search_queries if q.strip()],
        required_sources=_normalize_required_sources(output.required_sources, available),
        expected_evidence=[e.strip() for e in output.expected_evidence if e.strip()],
        research_strategy=output.research_strategy.strip(),
    )


def create_research_planner_graph(
    llm: LLMProvider,
    *,
    available_sources: list[str],
    temperature: float = 0.2,
) -> StateGraph:
    """LangGraph workflow with a single planning node (extensible for future steps)."""

    def plan_node(state: PlannerState) -> dict[str, ResearchPlan]:
        question = state["question"]
        history_context = state.get("history_context")
        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=build_planner_system_prompt(available_sources)),
                LLMMessage(
                    role="user",
                    content=build_planner_user_prompt(question, history_context=history_context),
                ),
            ],
            temperature=temperature,
            observability=LLMRequestObservability(agent_name="planner"),
        )
        llm_output = llm.generate_structured(request, ResearchPlanLLMOutput)
        plan = _llm_output_to_plan(question, llm_output, available_sources)
        logger.info(
            "Research plan created sub_questions=%d search_queries=%d sources=%s",
            len(plan.sub_questions),
            len(plan.search_queries),
            plan.required_sources,
        )
        return {"plan": plan}

    graph = StateGraph(PlannerState)
    graph.add_node("plan", plan_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", END)
    return graph


def compile_research_planner(
    llm: LLMProvider,
    *,
    available_sources: list[str],
    temperature: float = 0.2,
):
    return create_research_planner_graph(
        llm,
        available_sources=available_sources,
        temperature=temperature,
    ).compile()
