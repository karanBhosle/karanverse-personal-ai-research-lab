import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

_HTTPX_CLIENT = httpx.Client

from app.sources.cache import JSONResponseCache
from app.sources.openalex_client import OpenAlexClient, OpenAlexError

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def search_payload() -> dict:
    return json.loads((FIXTURES / "openalex_search_response.json").read_text(encoding="utf-8"))


def test_search_papers_maps_normalized_metadata(search_payload: dict, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/works"
        assert "hybrid" in str(request.url)
        return httpx.Response(200, json=search_payload)

    transport = httpx.MockTransport(handler)

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return _HTTPX_CLIENT(*args, **kwargs)

    monkeypatch.setattr("app.sources.openalex_client.httpx.Client", _client)
    client = OpenAlexClient(cache=None)
    papers = client.search_papers("hybrid retrieval", limit=5)

    assert len(papers) == 1
    paper = papers[0]
    assert paper.external_id == "W123456789"
    assert paper.openalex_id == "W123456789"
    assert paper.doi == "10.1234/example"
    assert paper.title.startswith("Hybrid Retrieval")
    assert paper.abstract and "bm25" in paper.abstract
    assert paper.authors[0].name == "Ada Researcher"
    assert paper.authors[0].institutions == ["Example University"]
    assert paper.concepts[0].name == "Information retrieval"
    assert paper.publication.venue == "Journal of Retrieval"
    assert paper.publication.publication_year == 2024
    assert paper.cited_by_count == 42


def test_get_paper_by_openalex_id(search_payload: dict, monkeypatch):
    work = search_payload["results"][0]

    def handler(request: httpx.Request) -> httpx.Response:
        assert "/works/" in request.url.path
        return httpx.Response(200, json=work)

    transport = httpx.MockTransport(handler)

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return _HTTPX_CLIENT(*args, **kwargs)

    monkeypatch.setattr("app.sources.openalex_client.httpx.Client", _client)
    client = OpenAlexClient(cache=None)
    paper = client.get_paper("W123456789")
    assert paper.external_id == "W123456789"
    assert paper.openalex_id == "W123456789"


def test_cache_avoids_second_http_call(search_payload: dict, monkeypatch, tmp_path: Path):
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json=search_payload)

    transport = httpx.MockTransport(handler)

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return _HTTPX_CLIENT(*args, **kwargs)

    monkeypatch.setattr("app.sources.openalex_client.httpx.Client", _client)
    cache = JSONResponseCache(tmp_path / "openalex", ttl_seconds=3600)
    client = OpenAlexClient(cache=cache)

    first = client.search_papers("hybrid", limit=1)
    second = client.search_papers("hybrid", limit=1)
    assert len(first) == 1
    assert len(second) == 1
    assert calls["count"] == 1


def test_get_paper_not_found(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    transport = httpx.MockTransport(handler)

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return _HTTPX_CLIENT(*args, **kwargs)

    monkeypatch.setattr("app.sources.openalex_client.httpx.Client", _client)
    client = OpenAlexClient(cache=None)
    with pytest.raises(OpenAlexError):
        client.get_paper("W404")


def test_research_search_api(monkeypatch, search_payload: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=search_payload)

    transport = httpx.MockTransport(handler)

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return _HTTPX_CLIENT(*args, **kwargs)

    monkeypatch.setattr("app.sources.openalex_client.httpx.Client", _client)

    from app.config.settings import get_settings
    from app.main import create_app
    from app.sources.registry import get_source_registry

    get_settings.cache_clear()
    get_source_registry.cache_clear()
    app = create_app()
    client = TestClient(app)
    response = client.post("/research/search", json={"query": "hybrid retrieval", "limit": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "openalex"
    assert body["count"] == 1
    assert body["papers"][0]["openalex_id"] == "W123456789"
