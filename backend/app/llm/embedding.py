from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class EmbeddingService(ABC):
    """Provider-agnostic text embedding interface."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        raise NotImplementedError


class SentenceTransformerEmbeddingService(EmbeddingService):
    """sentence-transformers backed embedding service."""

    def __init__(self, model_name: str, *, expected_dimension: int | None = None) -> None:
        self._model_name = model_name
        self._expected_dimension = expected_dimension
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            if self._expected_dimension is not None:
                dim_fn = getattr(self._model, "get_embedding_dimension", None) or getattr(
                    self._model, "get_sentence_embedding_dimension"
                )
                actual = int(dim_fn())
                if actual != self._expected_dimension:
                    raise ValueError(
                        f"VECTOR_DIMENSION={self._expected_dimension} does not match "
                        f"model dimension {actual} for {self._model_name}"
                    )
        return self._model

    @property
    def dimension(self) -> int:
        if self._expected_dimension is not None:
            return self._expected_dimension
        model = self._get_model()
        dim_fn = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension"
        )
        return int(dim_fn())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._get_model().encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, query: str) -> list[float]:
        vector = self._get_model().encode(query, convert_to_numpy=True, show_progress_bar=False)
        return vector.tolist()


def create_embedding_service(model_name: str, vector_dimension: int | None = None) -> EmbeddingService:
    return SentenceTransformerEmbeddingService(
        model_name=model_name,
        expected_dimension=vector_dimension,
    )
