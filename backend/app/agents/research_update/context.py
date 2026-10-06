from app.models.evidence import Evidence
from app.models.research_paper import ResearchPaper


def evidence_to_context_item(evidence: Evidence, *, max_chars: int) -> dict:
    text = evidence.text.strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return {
        "evidence_id": evidence.evidence_id,
        "title": evidence.title,
        "source": evidence.source,
        "page": evidence.page,
        "section": evidence.section,
        "retrieval_score": evidence.retrieval_score,
        "excerpt": text,
    }


def paper_to_context_item(paper: ResearchPaper, *, max_abstract_chars: int = 400) -> dict:
    abstract = (paper.abstract or "").strip()
    if len(abstract) > max_abstract_chars:
        abstract = abstract[: max_abstract_chars - 3] + "..."
    return {
        "source": paper.source,
        "external_id": paper.external_id,
        "title": paper.title,
        "publication_year": paper.publication.publication_year,
        "abstract_excerpt": abstract,
    }
