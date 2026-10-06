import json
from typing import Any

from app.agents.knowledge_gap.signals import GapSignal
from app.models.research_history import ResearchHistoryRecord
from app.models.search import HybridSearchResult


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _hits_payload(hits: list[HybridSearchResult], *, max_chars: int, label: str) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for hit in hits:
        metadata = hit.metadata or {}
        payload.append(
            {
                "scope": label,
                "chunk_id": hit.chunk_id,
                "document_id": hit.document_id,
                "score": round(hit.final_score, 4),
                "filename": metadata.get("filename"),
                "project_id": metadata.get("project_id"),
                "excerpt": _truncate(hit.text, max_chars),
            }
        )
    return payload


def build_history_payload(
    records: list[ResearchHistoryRecord],
    *,
    max_excerpt_chars: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for record in records:
        items.append(
            {
                "research_id": record.research_id,
                "question": record.question,
                "created_at": record.created_at.isoformat(),
                "research_objective": record.plan.research_objective,
                "conclusions": record.conclusions[:8],
                "open_questions": list(
                    dict.fromkeys(record.unresolved_questions + record.report.open_questions)
                )[:8],
                "evidence_count": len(record.evidence),
                "sub_questions": record.plan.sub_questions[:6],
                "executive_summary_excerpt": _truncate(record.report.executive_summary, max_excerpt_chars),
                "critique_gaps": (record.critique.research_gaps[:5] if record.critique else []),
                "critique_conflicts": (
                    [c.description for c in record.critique.conflicting_evidence[:5]]
                    if record.critique
                    else []
                ),
            }
        )
    return items


def build_gap_context_json(
    *,
    query: str,
    focus_topic: str,
    history_records: list[ResearchHistoryRecord],
    personal_hits: list[HybridSearchResult],
    project_hits: list[HybridSearchResult],
    public_hits: list[HybridSearchResult],
    signals: list[GapSignal],
    max_excerpt_chars: int,
    allow_private_knowledge: bool = False,
) -> str:
    if not allow_private_knowledge:
        personal_hits = []
        project_hits = []
        history_records = []
    payload = {
        "user_query": query,
        "focus_topic": focus_topic,
        "deterministic_signals": [
            {
                "gap_type": signal.gap_type,
                "message": signal.message,
                "topic_hint": signal.topic_hint,
                "research_ids": list(signal.research_ids),
                "priority": signal.priority,
            }
            for signal in signals
        ],
        "research_history": build_history_payload(history_records, max_excerpt_chars=max_excerpt_chars),
        "personal_knowledge": _hits_payload(personal_hits, max_chars=max_excerpt_chars, label="PERSONAL_KNOWLEDGE"),
        "project_knowledge": _hits_payload(project_hits, max_chars=max_excerpt_chars, label="PROJECT"),
        "public_research_corpus": _hits_payload(public_hits, max_chars=max_excerpt_chars, label="PUBLIC_RESEARCH"),
    }
    return json.dumps(payload, indent=2)
