from app.security.prompt_safety import prompt_injection_guard_line, wrap_untrusted_user_content


def build_planner_system_prompt(available_sources: list[str]) -> str:
    sources_line = ", ".join(available_sources) if available_sources else "openalex, arxiv"
    return f"""You are a research planning assistant for a personal AI research lab.

{prompt_injection_guard_line()}

Your job is to produce a structured research plan for a complex question. You must NOT answer the question, \
summarize findings, or assert what is true in the field. Do not include recommendations phrased as conclusions.

Instead:
- Clarify the research objective (what to learn or compare).
- Break the problem into sub-questions or focus areas.
- Propose concrete search queries for literature APIs.
- Select required external sources from: {sources_line}.
- Describe what evidence should be collected (papers, benchmarks, surveys, etc.).
- Outline a step-by-step research strategy for executing the plan.

Keep sub-questions and search queries specific and actionable. Prefer diverse angles (methods, evaluation, \
production, limitations) when the question is broad."""


def build_planner_user_prompt(question: str, *, history_context: str | None = None) -> str:
    parts = [
        "Create a research plan for the following question. Do not answer it.",
        wrap_untrusted_user_content(f"Question:\n{question.strip()}"),
    ]
    if history_context and history_context.strip():
        parts.append(
            "Relevant snippets from prior research runs (retrieved for overlap with this question; "
            "not full reports):\n"
            + wrap_untrusted_user_content(history_context.strip())
        )
    return "\n\n".join(parts)
