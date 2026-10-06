from app.agents.critic.analysis import DeterministicBundleAnalysis
from app.models.critique import (
    ConflictingEvidenceItem,
    CritiqueLLMOutput,
    CritiqueResult,
    DuplicateEvidenceGroup,
)
from app.models.research_evidence import ResearchEvidenceBundle


def _filter_evidence_ids(ids: list[str], allowed: set[str]) -> list[str]:
    seen: list[str] = []
    for eid in ids:
        if eid in allowed and eid not in seen:
            seen.append(eid)
    return seen


def merge_critique(
    bundle: ResearchEvidenceBundle,
    analysis: DeterministicBundleAnalysis,
    llm_output: CritiqueLLMOutput,
) -> CritiqueResult:
    allowed = {item.evidence_id for item in bundle.evidence}

    conflicting: list[ConflictingEvidenceItem] = []
    for item in llm_output.conflicting_evidence:
        refs = _filter_evidence_ids(item.evidence_ids, allowed)
        if not refs:
            continue
        conflicting.append(item.model_copy(update={"evidence_ids": refs}))

    duplicate_evidence = [
        DuplicateEvidenceGroup(evidence_ids=list(group), reason="High textual overlap between excerpts")
        for group in analysis.duplicate_groups
    ]

    evidence_sufficient = llm_output.evidence_sufficient and analysis.deterministic_sufficient
    additional_queries = list(dict.fromkeys(llm_output.additional_research_queries))
    if not evidence_sufficient and not additional_queries:
        for sq in analysis.uncovered_sub_questions:
            additional_queries.append(sq)
        for query in bundle.plan.search_queries:
            additional_queries.append(query)

    additional_required = list(dict.fromkeys(llm_output.additional_research_required))
    if analysis.uncovered_sub_questions:
        additional_required.append(
            "Retrieve and rerank local evidence for unresolved sub-questions: "
            + "; ".join(analysis.uncovered_sub_questions[:5])
        )

    research_gaps = list(dict.fromkeys(llm_output.research_gaps + analysis.uncovered_sub_questions))

    return CritiqueResult(
        question=bundle.question,
        evidence_sufficient=evidence_sufficient,
        source_quality=list(llm_output.source_quality_notes),
        conflicting_evidence=conflicting,
        unsupported_claims=list(llm_output.unsupported_claims),
        missing_perspectives=list(llm_output.missing_perspectives),
        duplicate_evidence=duplicate_evidence,
        research_gaps=research_gaps,
        additional_research_required=additional_required,
        additional_research_queries=additional_queries,
        sufficiency_rationale=llm_output.sufficiency_rationale.strip(),
        deterministic_flags=analysis.flags,
    )
