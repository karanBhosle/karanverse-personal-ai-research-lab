import json

from app.models.research_evidence import ResearchEvidenceBundle
from app.security.knowledge_guard import filter_evidence_for_llm


def build_critic_context(
    bundle: ResearchEvidenceBundle,
    *,
    max_excerpt_chars: int = 500,
    allow_private_knowledge: bool = False,
) -> str:
    """Serialize bundle facts for the critic LLM (no hidden corpus)."""
    visible_evidence = filter_evidence_for_llm(bundle.evidence, allow_private=allow_private_knowledge)
    evidence_entries = []
    for item in visible_evidence:
        text = item.text.strip()
        if len(text) > max_excerpt_chars:
            text = text[: max_excerpt_chars - 3] + "..."
        evidence_entries.append(
            {
                "evidence_id": item.evidence_id,
                "document_id": item.document_id,
                "chunk_id": item.chunk_id,
                "source": item.source,
                "title": item.title,
                "page": item.page,
                "section": item.section,
                "retrieval_score": item.retrieval_score,
                "retrieval_method": item.retrieval_method.value,
                "source_metadata": item.source_metadata.model_dump(),
                "excerpt": text,
            }
        )

    papers = [
        {
            "source": paper.source,
            "external_id": paper.external_id,
            "title": paper.title,
            "publication_year": paper.publication.publication_year,
            "categories": paper.categories,
            "cited_by_count": paper.cited_by_count,
            "landing_page_url": str(paper.landing_page_url) if paper.landing_page_url else None,
        }
        for paper in bundle.papers
    ]

    payload = {
        "question": bundle.question,
        "research_objective": bundle.plan.research_objective,
        "sub_questions": bundle.plan.sub_questions,
        "expected_evidence": bundle.plan.expected_evidence,
        "unresolved_sub_questions": bundle.unresolved_questions,
        "external_papers": papers,
        "local_evidence": evidence_entries,
        "retrieval_scores": [score.model_dump() for score in bundle.retrieval_scores],
        "allowed_evidence_ids": [item["evidence_id"] for item in evidence_entries],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
