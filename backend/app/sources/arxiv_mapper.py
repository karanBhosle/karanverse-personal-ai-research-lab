import re
from xml.etree import ElementTree as ET

from app.models.research_paper import PublicationInfo, ResearchAuthor, ResearchConcept, ResearchPaper

ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"
NS = {"atom": ATOM_NS, "arxiv": ARXIV_NS}

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?$")


def _local_tag(element: ET.Element) -> str:
    tag = element.tag
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _text(element: ET.Element | None) -> str | None:
    if element is None or element.text is None:
        return None
    return " ".join(element.text.split())


def normalize_arxiv_id(paper_id: str) -> str:
    paper_id = paper_id.strip()
    if paper_id.lower().startswith("arxiv:"):
        paper_id = paper_id.split(":", 1)[1].strip()
    if "arxiv.org" in paper_id:
        paper_id = paper_id.rstrip("/").rsplit("/", 1)[-1]
    if paper_id.upper().startswith("ARXIV:"):
        paper_id = paper_id[6:].strip()
    match = _ARXIV_ID_RE.search(paper_id)
    if match:
        return match.group(1)
    return paper_id


def _arxiv_id_from_entry_id(entry_id: str | None) -> str:
    if not entry_id:
        return "unknown"
    return normalize_arxiv_id(entry_id)


def _links(entry: ET.Element) -> dict[str, str]:
    links: dict[str, str] = {}
    for link in entry.findall("atom:link", NS):
        rel = link.get("rel") or "alternate"
        href = link.get("href")
        link_type = link.get("type") or ""
        if not href:
            continue
        if rel == "related" and link_type == "application/pdf":
            links["pdf"] = href
        elif rel == "alternate":
            links["abs"] = href
    return links


def map_arxiv_entry(entry: ET.Element) -> ResearchPaper:
    entry_id = _text(entry.find("atom:id", NS))
    arxiv_id = _arxiv_id_from_entry_id(entry_id)
    title = _text(entry.find("atom:title", NS)) or "Untitled"
    abstract = _text(entry.find("atom:summary", NS))

    authors: list[ResearchAuthor] = []
    for author_el in entry.findall("atom:author", NS):
        name = _text(author_el.find("atom:name", NS)) or "Unknown author"
        authors.append(ResearchAuthor(name=name))

    categories: list[str] = []
    concepts: list[ResearchConcept] = []
    for category in entry.findall("atom:category", NS):
        term = category.get("term")
        if not term:
            continue
        categories.append(term)
        concepts.append(ResearchConcept(name=term))

    primary_category = entry.find("arxiv:primary_category", NS)
    primary_term = primary_category.get("term") if primary_category is not None else None

    published = _text(entry.find("atom:published", NS))
    publication_year = None
    publication_date = None
    if published:
        publication_date = published[:10] if len(published) >= 10 else published
        try:
            publication_year = int(publication_date[:4])
        except ValueError:
            publication_year = None

    links = _links(entry)
    abs_url = links.get("abs") or f"https://arxiv.org/abs/{arxiv_id}"
    pdf_url = links.get("pdf") or f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    doi_el = entry.find("arxiv:doi", NS)
    doi = _text(doi_el)

    publication = PublicationInfo(
        venue="arXiv",
        publication_year=publication_year,
        publication_date=publication_date,
        type=primary_term or (categories[0] if categories else "preprint"),
    )

    return ResearchPaper(
        external_id=arxiv_id,
        arxiv_id=arxiv_id,
        doi=doi,
        title=title,
        abstract=abstract,
        authors=authors,
        concepts=concepts,
        categories=categories,
        publication=publication,
        pdf_url=pdf_url,
        open_access_url=pdf_url,
        landing_page_url=abs_url,
        source="arxiv",
    )


def parse_arxiv_atom_feed(xml_text: str) -> list[ResearchPaper]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid arXiv Atom feed: {exc}") from exc

    papers: list[ResearchPaper] = []
    for entry in root.findall("atom:entry", NS):
        if _local_tag(entry) != "entry":
            continue
        papers.append(map_arxiv_entry(entry))
    return papers
