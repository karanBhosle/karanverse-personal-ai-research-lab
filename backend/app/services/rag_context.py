from app.models.evidence import Citation, Evidence
from app.security.prompt_safety import prompt_injection_guard_line, wrap_untrusted_user_content


def build_evidence_context(citations: list[Citation], evidence_by_id: dict[str, Evidence]) -> str:
    blocks: list[str] = []
    for citation in citations:
        evidence = evidence_by_id[citation.evidence_id]
        page = evidence.page if evidence.page is not None else "n/a"
        section = evidence.section or "n/a"
        blocks.append(
            f"{citation.label} "
            f"(source={evidence.source}, title={evidence.title}, page={page}, section={section})\n"
            f"{evidence.text}"
        )
    return "\n\n".join(blocks)


def build_grounded_system_prompt() -> str:
    return (
        "You are a research assistant for a personal research lab. "
        "Answer ONLY using the provided evidence blocks. "
        "Do not use outside knowledge. "
        "When making factual claims, include the matching citation labels like [1] or [2]. "
        "If the evidence is insufficient to answer the question, set evidence_sufficient to false "
        "and clearly state what is missing in the answer. "
        f"{prompt_injection_guard_line()}"
    )


def build_grounded_user_prompt(question: str, evidence_context: str) -> str:
    return (
        f"{wrap_untrusted_user_content(f'Question:\\n{question}')}\n\n"
        f"{wrap_untrusted_user_content(f'Evidence:\\n{evidence_context}')}\n\n"
        "Respond in JSON with fields: answer, citation_numbers, evidence_sufficient."
    )
