import logging
from functools import lru_cache
from typing import Any

from app.config.settings import Settings, get_settings
from app.knowledge_graph.backends import (
    DisabledKnowledgeGraphBackend,
    InMemoryKnowledgeGraphBackend,
    KnowledgeGraphBackend,
    Neo4jKnowledgeGraphBackend,
)
from app.knowledge_graph.types import (
    GraphEntity,
    GraphEntityLabel,
    GraphEntityView,
    GraphRelationship,
    GraphRelationshipType,
)

logger = logging.getLogger(__name__)


class KnowledgeGraphService:
    """Optional knowledge graph operations (safe no-op when disabled/unavailable)."""

    def __init__(self, backend: KnowledgeGraphBackend) -> None:
        self._backend = backend
        self._schema_ready = False

    @property
    def enabled(self) -> bool:
        return self._backend.is_enabled

    def _ensure_ready(self) -> bool:
        if not self._backend.is_enabled:
            return False
        if self._schema_ready:
            return True
        try:
            self._backend.ensure_schema()
            self._schema_ready = True
            return True
        except Exception as exc:
            logger.warning("Knowledge graph schema initialization failed: %s", exc)
            return False

    def add_entity(
        self,
        label: GraphEntityLabel,
        entity_id: str,
        properties: dict[str, Any] | None = None,
    ) -> GraphEntityView | None:
        entity = GraphEntity(label=label, entity_id=entity_id, properties=properties or {})
        if not self._backend.is_enabled:
            return self._backend.add_entity(entity)
        if not self._ensure_ready():
            return None
        try:
            return self._backend.add_entity(entity)
        except Exception as exc:
            logger.warning("Knowledge graph add_entity failed label=%s id=%s: %s", label, entity_id, exc)
            return None

    def add_relationship(
        self,
        relationship_type: GraphRelationshipType,
        from_label: GraphEntityLabel,
        from_id: str,
        to_label: GraphEntityLabel,
        to_id: str,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        relationship = GraphRelationship(
            relationship_type=relationship_type,
            from_label=from_label,
            from_id=from_id,
            to_label=to_label,
            to_id=to_id,
            properties=properties or {},
        )
        if not self._backend.is_enabled:
            self._backend.add_relationship(relationship)
            return False
        if not self._ensure_ready():
            return False
        try:
            self._backend.add_relationship(relationship)
            return True
        except Exception as exc:
            logger.warning(
                "Knowledge graph add_relationship failed %s %s->%s: %s",
                relationship_type,
                from_id,
                to_id,
                exc,
            )
            return False

    def find_related_concepts(self, concept_id: str, *, limit: int = 10) -> list[GraphEntityView]:
        if not self._backend.is_enabled or not self._ensure_ready():
            return []
        try:
            return self._backend.find_related_concepts(concept_id, limit=limit)
        except Exception as exc:
            logger.warning("Knowledge graph find_related_concepts failed: %s", exc)
            return []

    def find_supporting_evidence(self, question_id: str, *, limit: int = 10) -> list[GraphEntityView]:
        if not self._backend.is_enabled or not self._ensure_ready():
            return []
        try:
            return self._backend.find_supporting_evidence(question_id, limit=limit)
        except Exception as exc:
            logger.warning("Knowledge graph find_supporting_evidence failed: %s", exc)
            return []

    def find_related_projects(
        self,
        entity_id: str,
        entity_label: GraphEntityLabel,
        *,
        limit: int = 10,
    ) -> list[GraphEntityView]:
        if not self._backend.is_enabled or not self._ensure_ready():
            return []
        try:
            return self._backend.find_related_projects(entity_id, entity_label, limit=limit)
        except Exception as exc:
            logger.warning("Knowledge graph find_related_projects failed: %s", exc)
            return []


def create_knowledge_graph_backend(settings: Settings) -> KnowledgeGraphBackend:
    if not settings.neo4j_enabled:
        return DisabledKnowledgeGraphBackend()
    if not settings.neo4j_password:
        logger.warning("NEO4J_ENABLED=true but NEO4J_PASSWORD is missing; knowledge graph disabled")
        return DisabledKnowledgeGraphBackend()
    try:
        backend = Neo4jKnowledgeGraphBackend(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
        backend.ensure_schema()
        return backend
    except Exception as exc:
        logger.warning("Neo4j unavailable at startup; knowledge graph disabled: %s", exc)
        return DisabledKnowledgeGraphBackend()


def create_knowledge_graph_service(settings: Settings | None = None) -> KnowledgeGraphService:
    settings = settings or get_settings()
    return KnowledgeGraphService(create_knowledge_graph_backend(settings))


@lru_cache
def get_knowledge_graph_service() -> KnowledgeGraphService:
    return create_knowledge_graph_service(get_settings())
