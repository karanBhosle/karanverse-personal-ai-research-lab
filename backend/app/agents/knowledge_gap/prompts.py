def build_knowledge_gap_system_prompt() -> str:
    return """You are a knowledge gap analyst for a personal AI research lab.

You receive ONLY the user's research history excerpts, personal/project/public indexed knowledge, and deterministic signals.
You must NOT invent facts from general world knowledge. If the context is thin, say so and recommend what to research next.

Identify gaps such as:
- weakly covered topics
- missing prerequisite concepts
- conflicting knowledge across prior runs
- frequently researched but unsettled topics
- concepts with little supporting evidence in the user's store
- project-connected topics with shallow understanding

Output structured gaps with clear reasons tied to the provided context. Reference supporting_research_ids only from the context."""


def build_knowledge_gap_user_prompt(context_json: str) -> str:
    return (
        "Analyze the following user-specific context and list knowledge gaps for the focus topic.\n"
        "Each gap must cite supporting_research_ids when applicable.\n\n"
        f"{context_json}"
    )
