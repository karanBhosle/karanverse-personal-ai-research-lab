import json

from app.agents.synthesizer.citations import CitationIndex
from app.models.critique import CritiqueResult
from app.models.research_evidence import ResearchEvidenceBundle
from app.models.research_plan import ResearchPlan
from app.security.knowledge_guard import filter_evidence_for_llm


def build_synthesizer_context(
    *,
    question: str,
    plan: ResearchPlan,
    bundle: ResearchEvidenceBundle,
    critique: CritiqueResult,
    citation_index: CitationIndex,
    max_excerpt_chars: int,
    allow_private_knowledge: bool = False,
) -> str:
    visible_evidence = filter_evidence_for_llm(bundle.evidence, allow_private=allow_private_knowledge)
    evidence_blocks = []
    for evidence in visible_evidence:
        label = citation_index.label_by_evidence_id.get(evidence.evidence_id, "")
        text = evidence.text.strip()
        if len(text) > max_excerpt_chars:
            text = text[: max_excerpt_chars - 3] + "..."
        evidence_blocks.append(
            {
                "citation_label": label,
                "evidence_id": evidence.evidence_id,
                "document_id": evidence.document_id,
                "chunk_id": evidence.chunk_id,
                "title": evidence.title,
                "page": evidence.page,
                "section": evidence.section,
                "retrieval_score": evidence.retrieval_score,
                "source_metadata": evidence.source_metadata.model_dump(),
                "excerpt": text,
            }
        )

    papers = [
        {
            "source": paper.source,
            "external_id": paper.external_id,
            "title": paper.title,
            "publication_year": paper.publication.publication_year,
            "landing_page_url": str(paper.landing_page_url) if paper.landing_page_url else None,
            "note": "Bibliographic metadata only—not primary evidence unless linked local excerpt exists.",
        }
        for paper in bundle.papers
    ]

    payload = {
        "question": question,
        "research_plan": plan.model_dump(),
        "critic": critique.model_dump(),
        "unresolved_sub_questions": bundle.unresolved_questions,
        "local_evidence": evidence_blocks,
        "external_literature_metadata": papers,
        "allowed_evidence_ids": list(citation_index.evidence_by_id.keys()),
        "allowed_citation_labels": sorted(set(citation_index.label_by_evidence_id.values())),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
