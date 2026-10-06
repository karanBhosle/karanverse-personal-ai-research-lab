from dataclasses import dataclass

from app.models.research_history import ResearchHistoryRecord
from app.models.search import HybridSearchResult
from app.retrieval.tokenizer import tokenize


@dataclass(frozen=True, slots=True)
class GapSignal:
    gap_type: str
    message: str
    topic_hint: str
    research_ids: tuple[str, ...] = ()
    priority: str = "medium"


def _topic_tokens(topic: str) -> set[str]:
    return set(tokenize(topic))


def _record_matches_topic(record: ResearchHistoryRecord, topic_tokens: set[str]) -> bool:
    if not topic_tokens:
        return True
    haystack = " ".join(
        [
            record.question,
            record.plan.research_objective,
            " ".join(record.plan.sub_questions),
            " ".join(record.conclusions),
            record.report.executive_summary,
        ]
    ).lower()
    doc_tokens = set(tokenize(haystack))
    overlap = len(topic_tokens & doc_tokens)
    return overlap >= max(1, min(2, len(topic_tokens) // 3))


def detect_gap_signals(
    records: list[ResearchHistoryRecord],
    *,
    focus_topic: str,
    personal_hits: list[HybridSearchResult],
    project_hits: list[HybridSearchResult],
    public_hits: list[HybridSearchResult],
    low_evidence_threshold: int = 3,
    weak_score_threshold: float = 0.25,
) -> list[GapSignal]:
    topic_tokens = _topic_tokens(focus_topic)
    relevant = [record for record in records if _record_matches_topic(record, topic_tokens)]
    signals: list[GapSignal] = []

    if len(relevant) >= 2:
        signals.append(
            GapSignal(
                gap_type="frequent_topic",
                message=(
                    f"You have {len(relevant)} prior research run(s) touching '{focus_topic}' — "
                    "review whether conclusions stabilized or still conflict."
                ),
                topic_hint=focus_topic,
                research_ids=tuple(record.research_id for record in relevant[:5]),
                priority="medium",
            )
        )

    for record in relevant:
        if len(record.evidence) < low_evidence_threshold:
            signals.append(
                GapSignal(
                    gap_type="low_evidence",
                    message=(
                        f"Research '{record.question[:80]}' has only {len(record.evidence)} evidence item(s) "
                        f"on related topics."
                    ),
                    topic_hint=focus_topic,
                    research_ids=(record.research_id,),
                    priority="high" if len(record.evidence) == 0 else "medium",
                )
            )

        for open_q in record.unresolved_questions + record.report.open_questions:
            if topic_tokens and not (set(tokenize(open_q)) & topic_tokens):
                continue
            signals.append(
                GapSignal(
                    gap_type="prerequisite",
                    message=f"Unresolved from prior research: {open_q}",
                    topic_hint=open_q,
                    research_ids=(record.research_id,),
                    priority="high",
                )
            )

        critique = record.critique
        if critique:
            for conflict in critique.conflicting_evidence:
                signals.append(
                    GapSignal(
                        gap_type="conflict",
                        message=f"Conflicting evidence noted: {conflict.description}",
                        topic_hint=focus_topic,
                        research_ids=(record.research_id,),
                        priority="high",
                    )
                )
            for gap in critique.research_gaps:
                if topic_tokens and not (set(tokenize(gap)) & topic_tokens):
                    continue
                signals.append(
                    GapSignal(
                        gap_type="weak_coverage",
                        message=f"Prior critique flagged gap: {gap}",
                        topic_hint=gap,
                        research_ids=(record.research_id,),
                        priority="medium",
                    )
                )

    personal_weak = [hit for hit in personal_hits if hit.final_score < weak_score_threshold]
    if personal_hits and len(personal_weak) == len(personal_hits):
        signals.append(
            GapSignal(
                gap_type="weak_coverage",
                message=(
                    f"Personal knowledge retrieval for '{focus_topic}' returned only weak matches — "
                    "topic may be thinly documented in your notes."
                ),
                topic_hint=focus_topic,
                priority="high",
            )
        )

    if project_hits and not public_hits and not relevant:
        signals.append(
            GapSignal(
                gap_type="project_gap",
                message=(
                    "Project-related material mentions this topic, but you have little indexed research history "
                    "or public corpus coverage — understanding may be project-local only."
                ),
                topic_hint=focus_topic,
                priority="medium",
            )
        )

    if project_hits and relevant:
        avg_evidence = sum(len(record.evidence) for record in relevant) / max(len(relevant), 1)
        if avg_evidence < low_evidence_threshold:
            signals.append(
                GapSignal(
                    gap_type="project_gap",
                    message=(
                        "This topic appears in your projects and research questions, but supporting evidence "
                        "in past runs is limited."
                    ),
                    topic_hint=focus_topic,
                    research_ids=tuple(record.research_id for record in relevant[:3]),
                    priority="high",
                )
            )

    if not relevant and not personal_hits and not public_hits:
        signals.append(
            GapSignal(
                gap_type="weak_coverage",
                message=f"No strong matches in your research history or knowledge index for '{focus_topic}'.",
                topic_hint=focus_topic,
                priority="high",
            )
        )

    return _dedupe_signals(signals)


def _dedupe_signals(signals: list[GapSignal]) -> list[GapSignal]:
    seen: set[tuple[str, str]] = set()
    unique: list[GapSignal] = []
    for signal in signals:
        key = (signal.gap_type, signal.message)
        if key in seen:
            continue
        seen.add(key)
        unique.append(signal)
    return unique
