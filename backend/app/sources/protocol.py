from typing import Protocol

from app.models.research_paper import ResearchPaper


class ResearchLiteratureSource(Protocol):
    """Contract for external literature backends (no LLM coupling)."""

    @property
    def name(self) -> str: ...

    def search_papers(self, query: str, limit: int = 10) -> list[ResearchPaper]: ...

    def get_paper(self, paper_id: str) -> ResearchPaper: ...
