import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

_HTTPX_CLIENT = httpx.Client

from app.config.settings import Settings
from app.sources.exceptions import UnknownResearchSourceError
from app.sources.registry import SourceRegistry, create_source_registry

FIXTURES = Path(__file__).parent / "fixtures"


def test_registry_lists_openalex_and_arxiv(tmp_path: Path):
    settings = Settings(
        data_processed_dir=tmp_path,
        openalex_cache_enabled=False,
        arxiv_cache_enabled=False,
    )
    registry = create_source_registry(settings)
    assert registry.list_sources() == ["arxiv", "openalex"]


def test_registry_unknown_source_raises():
    registry = SourceRegistry()
    with pytest.raises(UnknownResearchSourceError):
        registry.search_papers("semantic_scholar", "query", limit=1)


def test_registry_routes_to_arxiv(monkeypatch, tmp_path: Path):
    arxiv_xml = (FIXTURES / "arxiv_atom_search_response.xml").read_text(encoding="utf-8")
    openalex_payload = json.loads((FIXTURES / "openalex_search_response.json").read_text(encoding="utf-8"))

    def handler(request: httpx.Request) -> httpx.Response:
        if "export.arxiv.org" in str(request.url):
            return httpx.Response(200, text=arxiv_xml)
        return httpx.Response(200, json=openalex_payload)

    transport = httpx.MockTransport(handler)

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return _HTTPX_CLIENT(*args, **kwargs)

    monkeypatch.setattr("app.sources.arxiv_client.httpx.Client", _client)
    monkeypatch.setattr("app.sources.openalex_client.httpx.Client", _client)

    settings = Settings(
        data_processed_dir=tmp_path,
        openalex_cache_enabled=False,
        arxiv_cache_enabled=False,
    )
    registry = create_source_registry(settings)

    arxiv_papers = registry.search_papers("arxiv", "hybrid", limit=5)
    openalex_papers = registry.search_papers("openalex", "hybrid", limit=5)

    assert arxiv_papers[0].source == "arxiv"
    assert openalex_papers[0].source == "openalex"


def test_research_search_api_arxiv_source(monkeypatch):
    arxiv_xml = (FIXTURES / "arxiv_atom_search_response.xml").read_text(encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=arxiv_xml)

    transport = httpx.MockTransport(handler)

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return _HTTPX_CLIENT(*args, **kwargs)

    monkeypatch.setattr("app.sources.arxiv_client.httpx.Client", _client)

    from app.config.settings import get_settings
    from app.main import create_app
    from app.sources.registry import get_source_registry

    get_settings.cache_clear()
    get_source_registry.cache_clear()
    app = create_app()
    client = TestClient(app)
    response = client.post(
        "/research/search",
        json={"query": "hybrid retrieval", "limit": 5, "source": "arxiv"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "arxiv"
    assert body["papers"][0]["arxiv_id"] == "2401.12345"


def test_research_search_unknown_source_returns_404():
    from app.config.settings import get_settings
    from app.main import create_app
    from app.sources.registry import get_source_registry

    get_settings.cache_clear()
    get_source_registry.cache_clear()
    app = create_app()
    client = TestClient(app)
    response = client.post(
        "/research/search",
        json={"query": "test", "source": "unknown"},
    )
    assert response.status_code == 404
