from app.models.research_update import ChangedClaim, ResearchUpdate, UpdateComparisonLLMOutput, UpdateContradiction


def _filter_evidence_ids(ids: list[str], allowed: set[str]) -> list[str]:
    return [item for item in ids if item in allowed]


def sanitize_comparison_output(
    output: UpdateComparisonLLMOutput,
    *,
    allowed_evidence_ids: set[str],
) -> UpdateComparisonLLMOutput:
    changed: list[ChangedClaim] = []
    for claim in output.changed_claims:
        changed.append(
            claim.model_copy(
                update={
                    "previous_evidence_ids": _filter_evidence_ids(
                        claim.previous_evidence_ids, allowed_evidence_ids
                    ),
                    "new_evidence_ids": _filter_evidence_ids(claim.new_evidence_ids, allowed_evidence_ids),
                }
            )
        )
    contradictions: list[UpdateContradiction] = []
    for item in output.contradictions:
        contradictions.append(
            item.model_copy(
                update={
                    "previous_evidence_ids": _filter_evidence_ids(item.previous_evidence_ids, allowed_evidence_ids),
                    "new_evidence_ids": _filter_evidence_ids(item.new_evidence_ids, allowed_evidence_ids),
                }
            )
        )
    return output.model_copy(update={"changed_claims": changed, "contradictions": contradictions})


def merge_update(
    *,
    base: ResearchUpdate,
    comparison: UpdateComparisonLLMOutput,
    allowed_evidence_ids: set[str],
) -> ResearchUpdate:
    clean = sanitize_comparison_output(comparison, allowed_evidence_ids=allowed_evidence_ids)
    return base.model_copy(
        update={
            "executive_summary": clean.executive_summary,
            "changed_claims": clean.changed_claims,
            "unchanged_claims": clean.unchanged_claims,
            "contradictions": clean.contradictions,
            "newly_discovered_information": clean.newly_discovered_information,
            "outdated_information": clean.outdated_information,
            "confidence": clean.confidence,
            "recommended_action": clean.recommended_action,
        }
    )
