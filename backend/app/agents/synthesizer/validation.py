from app.agents.synthesizer.citations import CitationIndex
from app.models.critique import CritiqueResult
from app.models.research_evidence import ResearchEvidenceBundle
from app.models.research_plan import ResearchPlan
from app.models.research_report import (
    ReportContradiction,
    ReportEvidenceItem,
    ReportFinding,
    ReportRecommendation,
    ReportReference,
    ResearchReport,
    SynthesizerLLMOutput,
)


def _filter_ids(ids: list[str], allowed: set[str]) -> list[str]:
    out: list[str] = []
    for eid in ids:
        if eid in allowed and eid not in out:
            out.append(eid)
    return out


def _sanitize_finding(finding: ReportFinding, index: CitationIndex, allowed: set[str]) -> ReportFinding | None:
    evidence_ids = _filter_ids(finding.evidence_ids, allowed)
    if finding.basis == "evidence" and not evidence_ids:
        return None
    labels = index.labels_for(evidence_ids)
    return finding.model_copy(update={"evidence_ids": evidence_ids, "citation_labels": labels})


def _sanitize_contradiction(item: ReportContradiction, index: CitationIndex, allowed: set[str]) -> ReportContradiction | None:
    evidence_ids = _filter_ids(item.evidence_ids, allowed)
    if not evidence_ids:
        return None
    return item.model_copy(
        update={"evidence_ids": evidence_ids, "citation_labels": index.labels_for(evidence_ids)}
    )


def _sanitize_recommendation(
    item: ReportRecommendation,
    index: CitationIndex,
    allowed: set[str],
) -> ReportRecommendation:
    evidence_ids = _filter_ids(item.evidence_ids, allowed)
    if item.basis == "evidence" and not evidence_ids:
        return item.model_copy(update={"basis": "inference", "evidence_ids": [], "citation_labels": []})
    return item.model_copy(
        update={"evidence_ids": evidence_ids, "citation_labels": index.labels_for(evidence_ids)}
    )


def _build_evidence_section(bundle: ResearchEvidenceBundle, index: CitationIndex) -> list[ReportEvidenceItem]:
    items: list[ReportEvidenceItem] = []
    for evidence in bundle.evidence:
        label = index.label_by_evidence_id.get(evidence.evidence_id, "")
        excerpt = evidence.text.strip()
        if len(excerpt) > 800:
            excerpt = excerpt[:797] + "..."
        items.append(
            ReportEvidenceItem(
                evidence_id=evidence.evidence_id,
                citation_label=label,
                document_id=evidence.document_id,
                chunk_id=evidence.chunk_id,
                title=evidence.title,
                excerpt=excerpt,
                page=evidence.page,
                section=evidence.section,
                retrieval_score=evidence.retrieval_score,
                source_filename=evidence.source_metadata.filename or None,
            )
        )
    return items


def _build_references(bundle: ResearchEvidenceBundle, index: CitationIndex) -> list[ReportReference]:
    refs: list[ReportReference] = []
    for evidence in bundle.evidence:
        label = index.label_by_evidence_id.get(evidence.evidence_id)
        refs.append(
            ReportReference(
                reference_id=f"evidence:{evidence.evidence_id}",
                reference_type="local_evidence",
                title=evidence.title,
                citation_label=label,
                evidence_id=evidence.evidence_id,
            )
        )
    for paper in bundle.papers:
        refs.append(
            ReportReference(
                reference_id=f"{paper.source}:{paper.external_id}",
                reference_type="external_literature",
                title=paper.title,
                external_source=paper.source,
                external_id=paper.external_id,
                url=str(paper.landing_page_url) if paper.landing_page_url else None,
            )
        )
    return refs


def assemble_report(
    *,
    question: str,
    plan: ResearchPlan,
    bundle: ResearchEvidenceBundle,
    critique: CritiqueResult,
    index: CitationIndex,
    llm_output: SynthesizerLLMOutput,
) -> ResearchReport:
    allowed = set(index.evidence_by_id.keys())

    key_findings: list[ReportFinding] = []
    for finding in llm_output.key_findings:
        sanitized = _sanitize_finding(finding, index, allowed)
        if sanitized is not None:
            key_findings.append(sanitized)

    contradictions: list[ReportContradiction] = []
    for item in llm_output.contradictions:
        sanitized = _sanitize_contradiction(item, index, allowed)
        if sanitized is not None:
            contradictions.append(sanitized)

    for conflict in critique.conflicting_evidence:
        sanitized = _sanitize_contradiction(
            ReportContradiction(
                description=conflict.description,
                evidence_ids=conflict.evidence_ids,
            ),
            index,
            allowed,
        )
        if sanitized is not None:
            contradictions.append(sanitized)

    limitations = list(
        dict.fromkeys(
            llm_output.limitations
            + critique.source_quality
            + critique.unsupported_claims
            + [f"Research gap: {gap}" for gap in critique.research_gaps]
        )
    )
    if not critique.evidence_sufficient:
        limitations.append(f"Evidence sufficiency: {critique.sufficiency_rationale}")

    recommendations = [_sanitize_recommendation(item, index, allowed) for item in llm_output.practical_recommendations]

    open_questions = list(
        dict.fromkeys(
            llm_output.open_questions
            + critique.missing_perspectives
            + bundle.unresolved_questions
            + critique.additional_research_queries
        )
    )

    uncertainty_notes = list(dict.fromkeys(llm_output.uncertainty_notes))
    if not critique.evidence_sufficient:
        uncertainty_notes.append("Overall evidence coverage is limited; conclusions should be treated cautiously.")

    return ResearchReport(
        question=question,
        executive_summary=llm_output.executive_summary.strip(),
        key_findings=key_findings,
        evidence=_build_evidence_section(bundle, index),
        contradictions=contradictions,
        limitations=limitations,
        practical_recommendations=recommendations,
        open_questions=open_questions,
        references=_build_references(bundle, index),
        uncertainty_notes=uncertainty_notes,
    )
