from app.models.knowledge_gap import KnowledgeGap, KnowledgeGapLLMItem, KnowledgeGapLLMOutput, SupportingResearchRef
from app.models.research_history import ResearchHistoryRecord


def merge_llm_gaps(
    llm_output: KnowledgeGapLLMOutput,
    records_by_id: dict[str, ResearchHistoryRecord],
    *,
    allowed_research_ids: set[str],
) -> list[KnowledgeGap]:
    gaps: list[KnowledgeGap] = []
    for item in llm_output.gaps:
        refs: list[SupportingResearchRef] = []
        for research_id in item.supporting_research_ids:
            if research_id not in allowed_research_ids:
                continue
            record = records_by_id.get(research_id)
            if record is None:
                continue
            refs.append(
                SupportingResearchRef(
                    research_id=research_id,
                    question=record.question,
                    excerpt=record.report.executive_summary[:280] if record.report.executive_summary else None,
                )
            )
        gaps.append(
            KnowledgeGap(
                topic=item.topic,
                reason=item.reason,
                related_concepts=item.related_concepts,
                supporting_research=refs,
                recommended_learning=item.recommended_learning,
                priority=item.priority,
                gap_type=item.gap_type,
            )
        )
    return gaps


def signals_to_fallback_gaps(
    signals: list,
    records_by_id: dict[str, ResearchHistoryRecord],
) -> list[KnowledgeGap]:
    gaps: list[KnowledgeGap] = []
    for signal in signals:
        refs = []
        for research_id in signal.research_ids:
            record = records_by_id.get(research_id)
            if record is None:
                continue
            refs.append(
                SupportingResearchRef(
                    research_id=research_id,
                    question=record.question,
                    excerpt=record.report.executive_summary[:200] if record.report.executive_summary else None,
                )
            )
        gaps.append(
            KnowledgeGap(
                topic=signal.topic_hint,
                reason=signal.message,
                related_concepts=[],
                supporting_research=refs,
                recommended_learning=f"Study or run targeted research on: {signal.topic_hint}",
                priority=signal.priority if signal.priority in {"high", "medium", "low"} else "medium",
                gap_type=signal.gap_type,
            )
        )
    return gaps
