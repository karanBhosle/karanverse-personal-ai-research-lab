from dataclasses import dataclass

from app.config.settings import Settings
from app.evaluation.strategies import RetrievalBackend
from app.llm.embedding import create_embedding_service
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.chunk_store import load_chunks_from_processed
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.qdrant_retriever import QdrantRetriever, create_qdrant_client, get_qdrant_retriever
from app.retrieval.qdrant_store import QdrantVectorStore
from app.retrieval.reranker import Reranker, create_cross_encoder_reranker
from app.retrieval.reranking_hybrid_retriever import RerankingHybridRetriever


@dataclass(slots=True)
class BuiltRetrievalBackend(RetrievalBackend):
    bm25: BM25Retriever
    dense: QdrantRetriever
    hybrid: HybridRetriever
    hybrid_rerank: RerankingHybridRetriever | None
    reranker: Reranker | None
    candidate_count: int
    hybrid_top_k: int
    vector_backend: str


def build_vector_retriever(settings: Settings, *, use_remote_qdrant: bool) -> tuple[QdrantRetriever, str]:
    if use_remote_qdrant:
        return get_qdrant_retriever(), "remote_qdrant"

    embedding = create_embedding_service(
        model_name=settings.embedding_model,
        vector_dimension=settings.vector_dimension,
    )
    client = create_qdrant_client(":memory:")
    store = QdrantVectorStore(
        client=client,
        collection_name=f"{settings.qdrant_collection}_eval",
        embedding_service=embedding,
        vector_dimension=settings.vector_dimension,
    )
    chunks = load_chunks_from_processed(settings.data_processed_dir)
    if chunks:
        store.upsert_chunks(chunks)
    retriever = QdrantRetriever(
        store=store,
        embedding_service=embedding,
        default_top_k=settings.vector_search_top_k,
    )
    return retriever, "in_memory_from_processed"


def build_retrieval_backend(
    settings: Settings,
    *,
    use_remote_qdrant: bool,
    include_reranker: bool,
    candidate_count: int | None = None,
    hybrid_top_k: int | None = None,
) -> BuiltRetrievalBackend:
    bm25 = BM25Retriever(settings.bm25_index_dir, settings.data_processed_dir)
    bm25.ensure_index()
    dense, vector_backend = build_vector_retriever(settings, use_remote_qdrant=use_remote_qdrant)
    pool = candidate_count or settings.rerank_candidate_count
    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        vector_retriever=dense,
        rrf_k=settings.rrf_k,
        bm25_weight=settings.hybrid_bm25_weight,
        vector_weight=settings.hybrid_vector_weight,
        candidate_pool=max(settings.hybrid_candidate_pool, pool),
    )
    reranker: Reranker | None = None
    hybrid_rerank: RerankingHybridRetriever | None = None
    if include_reranker:
        reranker = create_cross_encoder_reranker(settings.reranker_model)
        hybrid_rerank = RerankingHybridRetriever(
            hybrid_retriever=hybrid,
            reranker=reranker,
            candidate_count=pool,
            final_top_k=settings.rerank_final_top_k,
        )
    return BuiltRetrievalBackend(
        bm25=bm25,
        dense=dense,
        hybrid=hybrid,
        hybrid_rerank=hybrid_rerank,
        reranker=reranker,
        candidate_count=pool,
        hybrid_top_k=hybrid_top_k or settings.research_agent_hybrid_top_k,
        vector_backend=vector_backend,
    )
