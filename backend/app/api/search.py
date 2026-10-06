import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.models.scoped_search import ScopedSearchOptions
from app.models.search import HybridSearchResult, RerankedSearchResult, SearchResult
from app.config import get_settings
from app.config.settings import Settings
from app.retrieval.bm25_retriever import BM25Retriever, get_bm25_retriever
from app.retrieval.hybrid_retriever import HybridRetriever, get_hybrid_retriever
from app.retrieval.reranking_hybrid_retriever import (
    RerankingHybridRetriever,
    get_reranking_hybrid_retriever,
)
from app.retrieval.qdrant_retriever import QdrantRetriever, get_qdrant_retriever

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


class BM25SearchRequest(ScopedSearchOptions):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    rebuild_index: bool = Field(
        default=False,
        description="Force rebuild of the BM25 index from processed chunks before search",
    )


class BM25SearchResponse(BaseModel):
    query: str
    top_k: int
    knowledge_scopes: list[str]
    results: list[SearchResult]
    index_chunk_count: int


@router.post("/bm25", response_model=BM25SearchResponse)
def search_bm25(
    body: BM25SearchRequest,
    retriever: BM25Retriever = Depends(get_bm25_retriever),
) -> BM25SearchResponse:
    if body.rebuild_index:
        retriever.build_from_processed()
    else:
        retriever.ensure_index()

    scopes = body.resolved_scopes(body.query)
    results = retriever.retrieve(body.query, top_k=body.top_k, knowledge_scopes=scopes)
    logger.info("BM25 search query=%r hits=%d", body.query, len(results))
    return BM25SearchResponse(
        query=body.query,
        top_k=body.top_k,
        knowledge_scopes=sorted(s.value for s in scopes),
        results=results,
        index_chunk_count=retriever.chunk_count,
    )


class VectorSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=100)
    collection_name: str | None = Field(
        default=None,
        description="Optional override of configured Qdrant collection",
    )


class VectorSearchResponse(BaseModel):
    query: str
    top_k: int
    collection_name: str
    embedding_model: str
    vector_dimension: int
    results: list[SearchResult]


@router.post("/vector", response_model=VectorSearchResponse)
def search_vector(
    body: VectorSearchRequest,
    settings: Settings = Depends(get_settings),
    retriever: QdrantRetriever = Depends(get_qdrant_retriever),
) -> VectorSearchResponse:
    if body.collection_name and body.collection_name != settings.qdrant_collection:
        from app.llm.embedding import create_embedding_service
        from app.retrieval.qdrant_retriever import create_qdrant_client
        from app.retrieval.qdrant_store import QdrantVectorStore

        embedding = create_embedding_service(
            model_name=settings.embedding_model,
            vector_dimension=settings.vector_dimension,
        )
        store = QdrantVectorStore(
            client=create_qdrant_client(settings.qdrant_url, settings.qdrant_api_key),
            collection_name=body.collection_name,
            embedding_service=embedding,
            vector_dimension=settings.vector_dimension,
        )
        retriever = QdrantRetriever(
            store=store,
            embedding_service=embedding,
            default_top_k=settings.vector_search_top_k,
        )

    top_k = body.top_k or settings.vector_search_top_k
    results = retriever.retrieve(body.query, top_k=top_k)
    return VectorSearchResponse(
        query=body.query,
        top_k=top_k,
        collection_name=body.collection_name or settings.qdrant_collection,
        embedding_model=settings.embedding_model,
        vector_dimension=settings.vector_dimension,
        results=results,
    )


class HybridSearchRequest(ScopedSearchOptions):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    rebuild_bm25_index: bool = Field(default=False)
    bm25_weight: float | None = Field(default=None, gt=0)
    vector_weight: float | None = Field(default=None, gt=0)


class HybridSearchResponse(BaseModel):
    query: str
    top_k: int
    knowledge_scopes: list[str]
    rrf_k: int
    bm25_weight: float
    vector_weight: float
    candidate_pool: int
    results: list[HybridSearchResult]


@router.post("/hybrid", response_model=HybridSearchResponse)
def search_hybrid(
    body: HybridSearchRequest,
    settings: Settings = Depends(get_settings),
    hybrid: HybridRetriever = Depends(get_hybrid_retriever),
) -> HybridSearchResponse:
    if body.rebuild_bm25_index:
        hybrid.bm25_retriever.build_from_processed()
    else:
        hybrid.bm25_retriever.ensure_index()

    bm25_weight = body.bm25_weight or settings.hybrid_bm25_weight
    vector_weight = body.vector_weight or settings.hybrid_vector_weight

    retriever = HybridRetriever(
        bm25_retriever=hybrid.bm25_retriever,
        vector_retriever=hybrid.vector_retriever,
        rrf_k=settings.rrf_k,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        candidate_pool=settings.hybrid_candidate_pool,
    )
    scopes = body.resolved_scopes(body.query)
    results = retriever.retrieve(body.query, top_k=body.top_k, knowledge_scopes=scopes)
    return HybridSearchResponse(
        query=body.query,
        top_k=body.top_k,
        knowledge_scopes=sorted(s.value for s in scopes),
        rrf_k=settings.rrf_k,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        candidate_pool=settings.hybrid_candidate_pool,
        results=results,
    )


class HybridRerankSearchRequest(ScopedSearchOptions):
    query: str = Field(min_length=1)
    final_top_k: int | None = Field(default=None, ge=1, le=100)
    candidate_count: int | None = Field(default=None, ge=1, le=200)
    rebuild_bm25_index: bool = Field(default=False)
    bm25_weight: float | None = Field(default=None, gt=0)
    vector_weight: float | None = Field(default=None, gt=0)


class HybridRerankSearchResponse(BaseModel):
    query: str
    knowledge_scopes: list[str]
    final_top_k: int
    candidate_count: int
    reranker_model: str
    rrf_k: int
    bm25_weight: float
    vector_weight: float
    results: list[RerankedSearchResult]


@router.post("/hybrid/rerank", response_model=HybridRerankSearchResponse)
def search_hybrid_rerank(
    body: HybridRerankSearchRequest,
    settings: Settings = Depends(get_settings),
    pipeline: RerankingHybridRetriever = Depends(get_reranking_hybrid_retriever),
) -> HybridRerankSearchResponse:
    if body.rebuild_bm25_index:
        pipeline.hybrid_retriever.bm25_retriever.build_from_processed()
    else:
        pipeline.hybrid_retriever.bm25_retriever.ensure_index()

    bm25_weight = body.bm25_weight or settings.hybrid_bm25_weight
    vector_weight = body.vector_weight or settings.hybrid_vector_weight
    candidate_count = body.candidate_count or settings.rerank_candidate_count
    final_top_k = body.final_top_k or settings.rerank_final_top_k

    hybrid = HybridRetriever(
        bm25_retriever=pipeline.hybrid_retriever.bm25_retriever,
        vector_retriever=pipeline.hybrid_retriever.vector_retriever,
        rrf_k=settings.rrf_k,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        candidate_pool=max(settings.hybrid_candidate_pool, candidate_count),
    )
    reranking = RerankingHybridRetriever(
        hybrid_retriever=hybrid,
        reranker=pipeline.reranker,
        candidate_count=candidate_count,
        final_top_k=final_top_k,
    )
    scopes = body.resolved_scopes(body.query)
    results = reranking.retrieve(
        body.query,
        final_top_k=final_top_k,
        candidate_count=candidate_count,
        knowledge_scopes=scopes,
    )
    return HybridRerankSearchResponse(
        query=body.query,
        knowledge_scopes=sorted(s.value for s in scopes),
        final_top_k=final_top_k,
        candidate_count=candidate_count,
        reranker_model=settings.reranker_model,
        rrf_k=settings.rrf_k,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        results=results,
    )
