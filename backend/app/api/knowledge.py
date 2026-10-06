import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_ingestion_service
from app.config import get_settings
from app.config.paths import get_project_root
from app.security.uploads import sanitize_filename, validate_text_ingest
from app.llm.exceptions import LLMConfigurationError, LLMError
from app.llm.factory import llm_configuration_status
from app.models.knowledge_gap import KnowledgeGapReport, KnowledgeGapRequest
from app.models.knowledge_source import DocumentKind, KnowledgeScope
from app.services.knowledge_gap_service import KnowledgeGapService, get_knowledge_gap_service
from app.models.graph_viz import GraphNodeDetailResponse, GraphQueryRequest, GraphVizResponse
from app.services.graph_visualization_service import GraphVisualizationService, get_graph_visualization_service
from app.models.scoped_search import ScopedSearchOptions
from app.models.search import HybridSearchResult
from app.retrieval.hybrid_retriever import HybridRetriever, get_hybrid_retriever
from app.services.ingestion_service import IngestionService
from app.services.portfolio_import_service import import_portfolio_seed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


class KnowledgeScopesResponse(BaseModel):
    scopes: list[str]


@router.get("/scopes", response_model=KnowledgeScopesResponse)
def list_knowledge_scopes() -> KnowledgeScopesResponse:
    return KnowledgeScopesResponse(scopes=[scope.value for scope in KnowledgeScope])


class PersonalSearchRequest(ScopedSearchOptions):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)


class PersonalSearchResponse(BaseModel):
    query: str
    knowledge_scopes: list[str]
    results: list[HybridSearchResult]


@router.post("/graph/query", response_model=GraphVizResponse)
def query_knowledge_graph(
    body: GraphQueryRequest,
    graph_service: GraphVisualizationService = Depends(get_graph_visualization_service),
) -> GraphVizResponse:
    return graph_service.query(body)


@router.get("/graph/node/{node_id:path}", response_model=GraphNodeDetailResponse)
def get_graph_node_detail(
    node_id: str,
    graph_service: GraphVisualizationService = Depends(get_graph_visualization_service),
) -> GraphNodeDetailResponse:
    detail = graph_service.get_node_detail(node_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Graph node not found: {node_id}")
    return detail


@router.post("/gaps", response_model=KnowledgeGapReport)
def detect_knowledge_gaps(
    body: KnowledgeGapRequest,
    gap_service: KnowledgeGapService = Depends(get_knowledge_gap_service),
) -> KnowledgeGapReport:
    llm_status = llm_configuration_status()
    if not llm_status["ready"]:
        raise HTTPException(
            status_code=503,
            detail=f"LLM provider is not configured: {llm_status.get('detail')}",
        )
    try:
        return gap_service.analyze(body)
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMError as exc:
        logger.exception("Knowledge gap detection failed during LLM call")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/search", response_model=PersonalSearchResponse)
def search_personal_knowledge(
    body: PersonalSearchRequest,
    hybrid: HybridRetriever = Depends(get_hybrid_retriever),
) -> PersonalSearchResponse:
    scopes = body.resolved_scopes(body.query)
    hybrid.bm25_retriever.ensure_index()
    results = hybrid.retrieve(body.query, top_k=body.top_k, knowledge_scopes=scopes)
    return PersonalSearchResponse(
        query=body.query,
        knowledge_scopes=sorted(scope.value for scope in scopes),
        results=results,
    )


class IngestTextRequest(BaseModel):
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    filename: str = Field(default="note.md", min_length=1)
    knowledge_scope: KnowledgeScope = KnowledgeScope.PERSONAL_KNOWLEDGE
    document_kind: DocumentKind = DocumentKind.TECHNICAL_NOTE
    project_id: str | None = None
    tags: list[str] = Field(default_factory=list)


class IngestTextResponse(BaseModel):
    document_id: str
    chunk_count: int
    knowledge_scope: str


@router.post("/ingest/text", response_model=IngestTextResponse)
def ingest_text_knowledge(
    body: IngestTextRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestTextResponse:
    settings = get_settings()
    try:
        validate_text_ingest(body.text, max_chars=settings.max_ingest_text_chars)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    safe_filename = sanitize_filename(body.filename, fallback="note.md")
    document = service.ingest_text_document(
        title=body.title,
        text=body.text,
        filename=safe_filename,
        knowledge_scope=body.knowledge_scope,
        document_kind=body.document_kind,
        project_id=body.project_id,
        tags=body.tags,
    )
    return IngestTextResponse(
        document_id=document.document_id,
        chunk_count=len(document.chunks),
        knowledge_scope=document.knowledge_scope.value,
    )


class ImportPortfolioResponse(BaseModel):
    imported_document_ids: list[str]
    seed_path: str


@router.post("/import/portfolio", response_model=ImportPortfolioResponse)
def import_portfolio(
    service: IngestionService = Depends(get_ingestion_service),
) -> ImportPortfolioResponse:
    seed_path = get_project_root() / "data" / "personal" / "portfolio_seed.json"
    try:
        ids = import_portfolio_seed(service, seed_path=seed_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ImportPortfolioResponse(imported_document_ids=ids, seed_path=str(seed_path))
