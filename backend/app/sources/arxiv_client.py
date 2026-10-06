import logging
import re
from functools import lru_cache
from urllib.parse import urlencode

import httpx

from app.config.settings import Settings, get_settings
from app.security.http_urls import validate_http_service_base_url

_ARXIV_ALLOWED_HOSTS = frozenset({"export.arxiv.org", "arxiv.org"})
from app.models.research_paper import ResearchPaper
from app.sources.arxiv_mapper import normalize_arxiv_id, parse_arxiv_atom_feed
from app.sources.cache import JSONResponseCache
from app.sources.exceptions import ResearchSourceError

logger = logging.getLogger(__name__)


class ArxivError(ResearchSourceError):
    """arXiv API error."""


def _escape_search_query(query: str) -> str:
    cleaned = query.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.replace('"', "")


class ArxivClient:
    """External literature client for arXiv Atom API (no LLM coupling)."""

    def __init__(
        self,
        *,
        base_url: str = "https://export.arxiv.org/api",
        timeout_seconds: float = 30.0,
        cache: JSONResponseCache | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.cache = cache

    def _cache_key(self, params: dict[str, str | int]) -> str:
        query = urlencode(sorted((str(k), str(v)) for k, v in params.items()))
        return f"{self.base_url}/query?{query}"

    def _get_atom(self, params: dict[str, str | int]) -> str:
        cache_key = self._cache_key(params)
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None and isinstance(cached.get("xml"), str):
                return cached["xml"]

        url = f"{self.base_url}/query"
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.get(url, params=params)
        except httpx.TimeoutException as exc:
            raise ArxivError("arXiv request timed out") from exc
        except httpx.HTTPError as exc:
            raise ArxivError(f"arXiv HTTP error: {exc}") from exc

        if response.status_code == 404:
            raise ArxivError("arXiv resource not found")
        if response.status_code >= 400:
            raise ArxivError(f"arXiv request failed ({response.status_code}): {response.text}")

        xml_text = response.text
        if self.cache is not None:
            self.cache.set(cache_key, {"xml": xml_text})
        return xml_text

    def _parse_feed(self, xml_text: str) -> list[ResearchPaper]:
        try:
            return parse_arxiv_atom_feed(xml_text)
        except ValueError as exc:
            raise ArxivError(str(exc)) from exc

    def search_papers(self, query: str, limit: int = 10) -> list[ResearchPaper]:
        if limit < 1:
            return []
        max_results = min(limit, 200)
        search_query = f"all:{_escape_search_query(query)}"
        xml_text = self._get_atom(
            {
                "search_query": search_query,
                "start": 0,
                "max_results": max_results,
            }
        )
        papers = self._parse_feed(xml_text)[:limit]
        logger.info("arXiv search query=%r returned %d papers", query, len(papers))
        return papers

    def get_paper(self, paper_id: str) -> ResearchPaper:
        arxiv_id = normalize_arxiv_id(paper_id)
        xml_text = self._get_atom({"id_list": arxiv_id, "max_results": 1})
        papers = self._parse_feed(xml_text)
        if not papers:
            raise ArxivError("arXiv resource not found")
        paper = papers[0]
        if paper.arxiv_id and normalize_arxiv_id(paper.arxiv_id) != arxiv_id:
            raise ArxivError("arXiv resource not found")
        return paper


def _arxiv_base_url(settings: Settings) -> str:
    return validate_http_service_base_url(settings.arxiv_base_url, allowed_hosts=_ARXIV_ALLOWED_HOSTS)


def get_arxiv_client() -> ArxivClient:
    settings = get_settings()
    cache = None
    if settings.arxiv_cache_enabled:
        cache = JSONResponseCache(
            cache_dir=settings.arxiv_cache_dir,
            ttl_seconds=settings.arxiv_cache_ttl_seconds,
        )
    return ArxivClient(
        base_url=_arxiv_base_url(settings),
        timeout_seconds=settings.arxiv_timeout_seconds,
        cache=cache,
    )


def create_arxiv_client(settings: Settings) -> ArxivClient:
    cache = None
    if settings.arxiv_cache_enabled:
        cache = JSONResponseCache(
            cache_dir=settings.arxiv_cache_dir,
            ttl_seconds=settings.arxiv_cache_ttl_seconds,
        )
    return ArxivClient(
        base_url=_arxiv_base_url(settings),
        timeout_seconds=settings.arxiv_timeout_seconds,
        cache=cache,
    )
