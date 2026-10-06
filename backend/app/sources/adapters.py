from dataclasses import dataclass

from app.models.research_paper import ResearchPaper
from app.sources.arxiv_client import ArxivClient
from app.sources.openalex_client import OpenAlexClient


@dataclass(frozen=True, slots=True)
class OpenAlexResearchSource:
    client: OpenAlexClient

    @property
    def name(self) -> str:
        return "openalex"

    def search_papers(self, query: str, limit: int = 10) -> list[ResearchPaper]:
        return self.client.search_papers(query, limit=limit)

    def get_paper(self, paper_id: str) -> ResearchPaper:
        return self.client.get_paper(paper_id)


@dataclass(frozen=True, slots=True)
class ArxivResearchSource:
    client: ArxivClient

    @property
    def name(self) -> str:
        return "arxiv"

    def search_papers(self, query: str, limit: int = 10) -> list[ResearchPaper]:
        return self.client.search_papers(query, limit=limit)

    def get_paper(self, paper_id: str) -> ResearchPaper:
        return self.client.get_paper(paper_id)
