from app.models.research_history import ResearchHistoryContextSnippet, ResearchHistoryRecord
from app.retrieval.tokenizer import tokenize


def _searchable_text(record: ResearchHistoryRecord) -> str:
    parts = [
        record.question,
        record.plan.research_objective,
        " ".join(record.plan.sub_questions),
        " ".join(record.plan.search_queries),
        record.report.executive_summary,
        " ".join(record.conclusions),
        " ".join(record.unresolved_questions),
        " ".join(record.report.open_questions),
    ]
    return " ".join(p for p in parts if p).lower()


def _snippet_excerpt(record: ResearchHistoryRecord, *, max_chars: int = 400) -> str:
    text = record.report.executive_summary.strip()
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def retrieve_relevant_history(
    records: list[ResearchHistoryRecord],
    query: str,
    *,
    top_k: int = 3,
    exclude_research_id: str | None = None,
) -> list[ResearchHistoryContextSnippet]:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return []

    scored: list[tuple[float, ResearchHistoryRecord]] = []
    for record in records:
        if exclude_research_id and record.research_id == exclude_research_id:
            continue
        doc_tokens = set(tokenize(_searchable_text(record)))
        overlap = len(query_tokens & doc_tokens)
        if overlap == 0:
            continue
        score = overlap / max(len(query_tokens), 1)
        scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    snippets: list[ResearchHistoryContextSnippet] = []
    for score, record in scored[:top_k]:
        snippets.append(
            ResearchHistoryContextSnippet(
                research_id=record.research_id,
                question=record.question,
                created_at=record.created_at,
                score=round(score, 4),
                research_objective=record.plan.research_objective,
                conclusions=record.conclusions[:5],
                open_questions=list(
                    dict.fromkeys(record.unresolved_questions + record.report.open_questions)
                )[:5],
                excerpt=_snippet_excerpt(record),
            )
        )
    return snippets


def format_snippets_for_planner(snippets: list[ResearchHistoryContextSnippet]) -> str:
    if not snippets:
        return ""
    blocks: list[str] = []
    for snippet in snippets:
        conclusions = "; ".join(snippet.conclusions) if snippet.conclusions else "n/a"
        open_q = "; ".join(snippet.open_questions) if snippet.open_questions else "n/a"
        objective = snippet.research_objective or "n/a"
        blocks.append(
            f"[Prior research {snippet.research_id}]\n"
            f"Question: {snippet.question}\n"
            f"Objective: {objective}\n"
            f"Prior conclusions: {conclusions}\n"
            f"Open questions: {open_q}\n"
            f"Brief excerpt: {snippet.excerpt}"
        )
    footer = (
        "Use prior research only for continuity and gap-filling. "
        "Do not treat prior conclusions as ground truth; plan fresh evidence collection."
    )
    return "\n\n".join(blocks) + f"\n\n{footer}"
