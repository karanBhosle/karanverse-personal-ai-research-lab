from dataclasses import dataclass

from app.models.evidence import Evidence
from app.models.research_evidence import ResearchEvidenceBundle
from app.services.citation_service import CitationService


@dataclass(frozen=True, slots=True)
class CitationIndex:
    label_by_evidence_id: dict[str, str]
    number_by_evidence_id: dict[str, int]
    evidence_by_id: dict[str, Evidence]

    def labels_for(self, evidence_ids: list[str]) -> list[str]:
        labels: list[str] = []
        for eid in evidence_ids:
            label = self.label_by_evidence_id.get(eid)
            if label and label not in labels:
                labels.append(label)
        return labels


def build_citation_index(bundle: ResearchEvidenceBundle) -> CitationIndex:
    evidence_by_id = {item.evidence_id: item for item in bundle.evidence}
    label_by_id: dict[str, str] = {}
    number_by_id: dict[str, int] = {}

    if bundle.citations:
        for citation in bundle.citations:
            if citation.evidence_id in evidence_by_id:
                label_by_id[citation.evidence_id] = citation.label
                number_by_id[citation.evidence_id] = citation.citation_number

    next_number = max(number_by_id.values(), default=0) + 1
    for evidence in bundle.evidence:
        if evidence.evidence_id in label_by_id:
            continue
        label = CitationService.format_citation_label(next_number)
        label_by_id[evidence.evidence_id] = label
        number_by_id[evidence.evidence_id] = next_number
        next_number += 1

    return CitationIndex(
        label_by_evidence_id=label_by_id,
        number_by_evidence_id=number_by_id,
        evidence_by_id=evidence_by_id,
    )
