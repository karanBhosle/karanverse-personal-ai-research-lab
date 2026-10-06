import logging
from functools import lru_cache
from urllib.parse import quote, urlencode

import httpx

from app.config.settings import Settings, get_settings
from app.security.http_urls import validate_http_service_base_url

_OPENALEX_ALLOWED_HOSTS = frozenset({"api.openalex.org", "openalex.org"})
from app.models.research_paper import ResearchPaper
from app.sources.cache import JSONResponseCache
from app.sources.exceptions import ResearchSourceError
from app.sources.openalex_mapper import map_openalex_work

logger = logging.getLogger(__name__)


class OpenAlexError(ResearchSourceError):
    """OpenAlex API error."""


class OpenAlexClient:
    """External literature client for OpenAlex (no LLM coupling)."""

    def __init__(
        self,
        *,
        base_url: str = "https://api.openalex.org",
        mailto: str | None = None,
        timeout_seconds: float = 30.0,
        cache: JSONResponseCache | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.mailto = mailto
        self.timeout_seconds = timeout_seconds
        self.cache = cache

    def _params(self, extra: dict[str, str | int]) -> dict[str, str | int]:
        params = dict(extra)
        if self.mailto:
            params["mailto"] = self.mailto
        return params

    def _cache_key(self, path: str, params: dict[str, str | int]) -> str:
        query = urlencode(sorted((str(k), str(v)) for k, v in params.items()))
        return f"{self.base_url}{path}?{query}"

    def _get_json(self, path: str, params: dict[str, str | int]) -> dict:
        cache_key = self._cache_key(path, params)
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.get(url, params=self._params(params))
        except httpx.TimeoutException as exc:
            raise OpenAlexError("OpenAlex request timed out") from exc
        except httpx.HTTPError as exc:
            raise OpenAlexError(f"OpenAlex HTTP error: {exc}") from exc

        if response.status_code == 404:
            raise OpenAlexError("OpenAlex resource not found")
        if response.status_code >= 400:
            raise OpenAlexError(f"OpenAlex request failed ({response.status_code}): {response.text}")

        data = response.json()
        if self.cache is not None:
            self.cache.set(cache_key, data)
        return data

    @staticmethod
    def _normalize_paper_id(paper_id: str) -> str:
        paper_id = paper_id.strip()
        if paper_id.startswith("https://openalex.org/"):
            return paper_id
        if paper_id.upper().startswith("W"):
            return f"https://openalex.org/{paper_id.upper()}"
        if paper_id.startswith("10."):
            return f"https://doi.org/{paper_id}"
        return paper_id

    def search_papers(self, query: str, limit: int = 10) -> list[ResearchPaper]:
        if limit < 1:
            return []
        per_page = min(limit, 200)
        payload = self._get_json(
            "/works",
            {"search": query, "per_page": per_page},
        )
        results = payload.get("results") or []
        papers = [map_openalex_work(item) for item in results[:limit]]
        logger.info("OpenAlex search query=%r returned %d papers", query, len(papers))
        return papers

    def get_paper(self, paper_id: str) -> ResearchPaper:
        normalized_id = self._normalize_paper_id(paper_id)
        encoded = quote(normalized_id, safe="")
        payload = self._get_json(f"/works/{encoded}", {})
        return map_openalex_work(payload)


@lru_cache
def get_openalex_client() -> OpenAlexClient:
    settings = get_settings()
    cache = None
    if settings.openalex_cache_enabled:
        cache = JSONResponseCache(
            cache_dir=settings.openalex_cache_dir,
            ttl_seconds=settings.openalex_cache_ttl_seconds,
        )
    return OpenAlexClient(
        base_url=settings.openalex_base_url,
        mailto=settings.openalex_mailto,
        timeout_seconds=settings.openalex_timeout_seconds,
        cache=cache,
    )


def create_openalex_client(settings: Settings) -> OpenAlexClient:
    cache = None
    if settings.openalex_cache_enabled:
        cache = JSONResponseCache(
            cache_dir=settings.openalex_cache_dir,
            ttl_seconds=settings.openalex_cache_ttl_seconds,
        )
    base_url = validate_http_service_base_url(settings.openalex_base_url, allowed_hosts=_OPENALEX_ALLOWED_HOSTS)
    return OpenAlexClient(
        base_url=base_url,
        mailto=settings.openalex_mailto,
        timeout_seconds=settings.openalex_timeout_seconds,
        cache=cache,
    )
