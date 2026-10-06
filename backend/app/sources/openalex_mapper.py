from app.models.research_paper import (
    PublicationInfo,
    ResearchAuthor,
    ResearchConcept,
    ResearchPaper,
)


def _abstract_from_inverted_index(index: dict | None) -> str | None:
    if not index:
        return None
    max_pos = 0
    for positions in index.values():
        if positions:
            max_pos = max(max_pos, max(positions))
    words = [""] * (max_pos + 1)
    for word, positions in index.items():
        for pos in positions:
            if 0 <= pos < len(words):
                words[pos] = word
    text = " ".join(word for word in words if word).strip()
    return text or None


def _openalex_short_id(openalex_id: str | None) -> str | None:
    if not openalex_id:
        return None
    return openalex_id.rsplit("/", 1)[-1]


def map_openalex_work(work: dict) -> ResearchPaper:
    openalex_id = _openalex_short_id(work.get("id")) or "unknown"
    doi = work.get("doi")
    if isinstance(doi, str) and doi.startswith("https://doi.org/"):
        doi = doi.removeprefix("https://doi.org/")

    authors: list[ResearchAuthor] = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        institutions = [
            inst.get("display_name", "")
            for inst in (authorship.get("institutions") or [])
            if inst.get("display_name")
        ]
        authors.append(
            ResearchAuthor(
                openalex_id=_openalex_short_id(author.get("id")),
                name=author.get("display_name") or "Unknown author",
                institutions=institutions,
                orcid=author.get("orcid"),
            )
        )

    concepts: list[ResearchConcept] = []
    for concept in work.get("concepts") or []:
        if not concept.get("display_name"):
            continue
        concepts.append(
            ResearchConcept(
                openalex_id=_openalex_short_id(concept.get("id")),
                name=concept["display_name"],
                score=concept.get("score"),
            )
        )

    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    biblio = work.get("biblio") or {}
    pages = None
    if biblio.get("first_page") or biblio.get("last_page"):
        pages = f"{biblio.get('first_page', '')}-{biblio.get('last_page', '')}".strip("-")

    publication = PublicationInfo(
        venue=source.get("display_name"),
        publisher=source.get("host_organization_name"),
        publication_year=work.get("publication_year"),
        publication_date=work.get("publication_date"),
        volume=biblio.get("volume"),
        issue=biblio.get("issue"),
        pages=pages,
        type=work.get("type"),
    )

    open_access = work.get("open_access") or {}
    oa_url = open_access.get("oa_url")
    landing_page = primary_location.get("landing_page_url") or work.get("id")

    return ResearchPaper(
        external_id=openalex_id,
        openalex_id=openalex_id,
        doi=doi,
        title=work.get("display_name") or work.get("title") or "Untitled",
        abstract=_abstract_from_inverted_index(work.get("abstract_inverted_index")),
        authors=authors,
        concepts=concepts,
        publication=publication,
        cited_by_count=work.get("cited_by_count"),
        open_access_url=oa_url,
        landing_page_url=landing_page,
        source="openalex",
    )
