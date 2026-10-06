from pathlib import Path

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _clear_cached_singletons() -> None:
    from app.config.settings import get_settings
    from app.knowledge_graph.service import get_knowledge_graph_service
    from app.llm.factory import get_llm_provider
    from app.observability.service import get_observability_service
    from app.retrieval.bm25_retriever import get_bm25_retriever
    from app.retrieval.hybrid_retriever import get_hybrid_retriever
    from app.retrieval.qdrant_retriever import get_qdrant_retriever
    from app.retrieval.reranking_hybrid_retriever import get_reranking_hybrid_retriever
    from app.services.critic_service import get_critic_agent_service
    from app.services.knowledge_gap_service import get_knowledge_gap_service
    from app.services.rag_service import get_grounded_rag_service
    from app.services.research_agent_service import get_research_agent_service
    from app.services.research_history_service import get_research_history_service
    from app.services.research_planner_service import get_research_planner_service
    from app.services.research_synthesizer_service import get_research_synthesizer_service
    from app.services.research_update_service import get_research_update_service
    from app.services.research_workflow_service import get_research_workflow_service
    from app.sources.registry import get_source_registry

    for cached in (
        get_settings,
        get_bm25_retriever,
        get_qdrant_retriever,
        get_hybrid_retriever,
        get_reranking_hybrid_retriever,
        get_llm_provider,
        get_grounded_rag_service,
        get_research_planner_service,
        get_research_agent_service,
        get_critic_agent_service,
        get_research_synthesizer_service,
        get_research_workflow_service,
        get_research_history_service,
        get_research_update_service,
        get_knowledge_gap_service,
        get_knowledge_graph_service,
        get_source_registry,
        get_observability_service,
    ):
        cached.cache_clear()


@pytest.fixture(scope="session")
def sample_pdf_path() -> Path:
    """Minimal two-page PDF with headings and body text for Docling ingestion tests."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIXTURES_DIR / "sample_research.pdf"
    if pdf_path.exists():
        return pdf_path

    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 750, "Hybrid Retrieval Notes")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, 720, "Introduction")
    c.setFont("Helvetica", 11)
    c.drawString(
        72,
        700,
        "This document describes BM25, dense vectors, and reciprocal rank fusion.",
    )
    c.showPage()
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, 750, "Methods")
    c.setFont("Helvetica", 11)
    c.drawString(72, 730, "Chunks retain page numbers and section metadata for provenance.")
    c.save()
    return pdf_path


@pytest.fixture(autouse=True)
def disable_qdrant_indexing_by_default(monkeypatch: pytest.MonkeyPatch):
    """Avoid requiring Qdrant for unrelated API/ingestion tests."""
    monkeypatch.setenv("QDRANT_INDEXING_ENABLED", "false")
    monkeypatch.setenv("NEO4J_ENABLED", "false")
    _clear_cached_singletons()
    yield
    _clear_cached_singletons()
