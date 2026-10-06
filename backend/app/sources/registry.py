from functools import lru_cache

from app.config.settings import Settings, get_settings
from app.models.research_paper import ResearchPaper
from app.sources.adapters import ArxivResearchSource, OpenAlexResearchSource
from app.sources.arxiv_client import create_arxiv_client
from app.sources.exceptions import UnknownResearchSourceError
from app.sources.openalex_client import create_openalex_client
from app.sources.protocol import ResearchLiteratureSource


class SourceRegistry:
    """Lookup for external literature sources; extend by registering new backends."""

    def __init__(self, sources: dict[str, ResearchLiteratureSource] | None = None) -> None:
        self._sources: dict[str, ResearchLiteratureSource] = {}
        if sources:
            for name, source in sources.items():
                self.register(name, source)

    def register(self, name: str, source: ResearchLiteratureSource) -> None:
        key = name.strip().lower()
        if not key:
            raise ValueError("Source name must be non-empty")
        self._sources[key] = source

    def list_sources(self) -> list[str]:
        return sorted(self._sources.keys())

    def _get(self, source: str) -> ResearchLiteratureSource:
        key = source.strip().lower()
        impl = self._sources.get(key)
        if impl is None:
            raise UnknownResearchSourceError(source)
        return impl

    def search_papers(self, source: str, query: str, limit: int = 10) -> list[ResearchPaper]:
        return self._get(source).search_papers(query, limit=limit)

    def get_paper(self, source: str, paper_id: str) -> ResearchPaper:
        return self._get(source).get_paper(paper_id)


def create_source_registry(settings: Settings) -> SourceRegistry:
    registry = SourceRegistry()
    registry.register("openalex", OpenAlexResearchSource(create_openalex_client(settings)))
    registry.register("arxiv", ArxivResearchSource(create_arxiv_client(settings)))
    return registry


@lru_cache
def get_source_registry() -> SourceRegistry:
    return create_source_registry(get_settings())
