from fastapi import Depends

from app.config import get_settings
from app.config.settings import Settings
from app.llm.embedding import create_embedding_service
from app.models.chunking import ChunkingConfig
from app.retrieval.qdrant_retriever import create_qdrant_client
from app.retrieval.qdrant_store import QdrantVectorStore
from app.services.ingestion_service import IngestionService
from app.services.vector_index_service import VectorIndexService


def get_vector_index_service(settings: Settings = Depends(get_settings)) -> VectorIndexService | None:
    if not settings.qdrant_indexing_enabled:
        return None
    embedding = create_embedding_service(
        model_name=settings.embedding_model,
        vector_dimension=settings.vector_dimension,
    )
    client = create_qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
    store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding_service=embedding,
        vector_dimension=settings.vector_dimension,
    )
    return VectorIndexService(store=store, enabled=True)


def get_ingestion_service(
    settings: Settings = Depends(get_settings),
    vector_indexer: VectorIndexService | None = Depends(get_vector_index_service),
) -> IngestionService:
    chunking = ChunkingConfig(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_chunk_size=settings.chunk_min_size,
    )
    settings.data_processed_dir.mkdir(parents=True, exist_ok=True)
    settings.data_raw_dir.mkdir(parents=True, exist_ok=True)
    return IngestionService(
        processed_dir=settings.data_processed_dir,
        raw_dir=settings.data_raw_dir,
        chunking=chunking,
        vector_indexer=vector_indexer,
    )
