from app.models.research_history import ResearchHistoryRecord


def extract_previous_conclusions(record: ResearchHistoryRecord) -> list[str]:
    """Derive atomic prior conclusions from stored history (no LLM)."""
    seen: set[str] = set()
    ordered: list[str] = []

    def add(statement: str) -> None:
        text = statement.strip()
        if not text:
            return
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        ordered.append(text)

    for conclusion in record.conclusions:
        add(conclusion)
    for finding in record.report.key_findings:
        add(finding.statement)
    for recommendation in record.report.practical_recommendations:
        add(recommendation.recommendation)
    if not ordered and record.report.executive_summary.strip():
        add(record.report.executive_summary.strip())

    return ordered
