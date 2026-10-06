import json


def build_update_queries_system_prompt() -> str:
    return """You are a research update planner for a personal AI research lab.

Given prior research conclusions and metadata, propose queries to discover what has CHANGED since that run.
Do not restate old conclusions as facts. Focus on:
- Recent literature and benchmarks
- Methods that may have superseded prior approaches
- Gaps and open questions from the prior run
- Verification queries that could confirm or overturn prior findings

Output structured search queries only."""


def build_update_queries_user_prompt(
    *,
    question: str,
    previous_research_at: str,
    conclusions: list[str],
    open_questions: list[str],
    research_objective: str,
) -> str:
    payload = {
        "question": question,
        "previous_research_at": previous_research_at,
        "research_objective": research_objective,
        "previous_conclusions": conclusions,
        "open_questions": open_questions,
    }
    return (
        "Plan update searches for the prior research below. "
        "Generate literature_search_queries (for OpenAlex/arXiv) and local_retrieval_queries "
        "(for the user's document index).\n\n"
        f"{json.dumps(payload, indent=2)}"
    )


def build_update_comparison_system_prompt() -> str:
    return """You are a research update analyst. Compare PRIOR conclusions and evidence against NEW evidence.

Rules:
- Only cite evidence by the provided evidence_id values.
- Mark claims as unchanged only when new evidence supports the same conclusion without material change.
- Mark changed_claims when new evidence materially updates, narrows, or extends a prior conclusion.
- Put direct conflicts in contradictions.
- Put findings with no prior analogue in newly_discovered_information.
- Put prior conclusions that new evidence undermines or supersedes in outdated_information.
- Be explicit about uncertainty; prefer medium/low confidence when evidence is thin.
- recommended_action should be a concrete next step for the researcher."""


def build_update_comparison_user_prompt(
    *,
    question: str,
    previous_conclusions: list[str],
    prior_evidence: list[dict],
    new_evidence: list[dict],
    new_literature: list[dict],
    overlap_score: float,
    novel_evidence_ids: list[str],
) -> str:
    payload = {
        "question": question,
        "previous_conclusions": previous_conclusions,
        "prior_evidence": prior_evidence,
        "new_evidence": new_evidence,
        "new_literature": new_literature,
        "deterministic_signals": {
            "mean_max_jaccard_overlap": round(overlap_score, 4),
            "novel_new_evidence_ids": novel_evidence_ids,
        },
    }
    return (
        "Produce a structured research update comparison.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )
