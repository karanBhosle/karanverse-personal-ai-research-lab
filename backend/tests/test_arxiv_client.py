from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

_HTTPX_CLIENT = httpx.Client

from app.sources.arxiv_client import ArxivClient, ArxivError
from app.sources.arxiv_mapper import normalize_arxiv_id, parse_arxiv_atom_feed
from app.sources.cache import JSONResponseCache

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def arxiv_atom_xml() -> str:
    return (FIXTURES / "arxiv_atom_search_response.xml").read_text(encoding="utf-8")


def test_normalize_arxiv_id():
    assert normalize_arxiv_id("2401.12345v1") == "2401.12345"
    assert normalize_arxiv_id("arxiv:2401.12345") == "2401.12345"
    assert normalize_arxiv_id("https://arxiv.org/abs/2401.12345") == "2401.12345"


def test_parse_arxiv_atom_maps_metadata(arxiv_atom_xml: str):
    papers = parse_arxiv_atom_feed(arxiv_atom_xml)
    assert len(papers) == 1
    paper = papers[0]
    assert paper.external_id == "2401.12345"
    assert paper.arxiv_id == "2401.12345"
    assert paper.source == "arxiv"
    assert "Hybrid Retrieval" in paper.title
    assert paper.abstract and "BM25" in paper.abstract
    assert [a.name for a in paper.authors] == ["Grace Hopper", "Alan Turing"]
    assert paper.categories == ["cs.IR", "cs.LG"]
    assert paper.publication.publication_date == "2024-01-10"
    assert paper.publication.publication_year == 2024
    assert str(paper.pdf_url).endswith("2401.12345v1.pdf")
    assert paper.doi == "10.48550/arXiv.2401.12345"


def test_search_papers_keyword_search(arxiv_atom_xml: str, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/query")
        assert "search_query" in str(request.url)
        return httpx.Response(200, text=arxiv_atom_xml)

    transport = httpx.MockTransport(handler)

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return _HTTPX_CLIENT(*args, **kwargs)

    monkeypatch.setattr("app.sources.arxiv_client.httpx.Client", _client)
    client = ArxivClient(cache=None)
    papers = client.search_papers("hybrid retrieval", limit=5)
    assert len(papers) == 1
    assert papers[0].arxiv_id == "2401.12345"


def test_get_paper_by_arxiv_id(arxiv_atom_xml: str, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert "id_list=2401.12345" in str(request.url)
        return httpx.Response(200, text=arxiv_atom_xml)

    transport = httpx.MockTransport(handler)

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return _HTTPX_CLIENT(*args, **kwargs)

    monkeypatch.setattr("app.sources.arxiv_client.httpx.Client", _client)
    client = ArxivClient(cache=None)
    paper = client.get_paper("2401.12345v1")
    assert paper.arxiv_id == "2401.12345"


def test_arxiv_cache_avoids_second_http_call(arxiv_atom_xml: str, monkeypatch, tmp_path: Path):
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, text=arxiv_atom_xml)

    transport = httpx.MockTransport(handler)

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return _HTTPX_CLIENT(*args, **kwargs)

    monkeypatch.setattr("app.sources.arxiv_client.httpx.Client", _client)
    cache = JSONResponseCache(tmp_path / "arxiv", ttl_seconds=3600)
    client = ArxivClient(cache=cache)

    client.search_papers("hybrid", limit=1)
    client.search_papers("hybrid", limit=1)
    assert calls["count"] == 1


def test_get_paper_not_found(monkeypatch):
    empty_feed = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=empty_feed)

    transport = httpx.MockTransport(handler)

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return _HTTPX_CLIENT(*args, **kwargs)

    monkeypatch.setattr("app.sources.arxiv_client.httpx.Client", _client)
    client = ArxivClient(cache=None)
    with pytest.raises(ArxivError):
        client.get_paper("9999.99999")
