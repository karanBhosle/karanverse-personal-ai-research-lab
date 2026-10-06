import logging
from abc import ABC, abstractmethod
from typing import Any

from app.knowledge_graph.types import (
    GraphEntity,
    GraphEntityLabel,
    GraphEntityView,
    GraphRelationship,
    GraphRelationshipType,
)

logger = logging.getLogger(__name__)


class KnowledgeGraphBackend(ABC):
    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def ensure_schema(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_entity(self, entity: GraphEntity) -> GraphEntityView:
        raise NotImplementedError

    @abstractmethod
    def add_relationship(self, relationship: GraphRelationship) -> None:
        raise NotImplementedError

    @abstractmethod
    def find_related_concepts(self, concept_id: str, *, limit: int = 10) -> list[GraphEntityView]:
        raise NotImplementedError

    @abstractmethod
    def find_supporting_evidence(self, question_id: str, *, limit: int = 10) -> list[GraphEntityView]:
        raise NotImplementedError

    @abstractmethod
    def find_related_projects(self, entity_id: str, entity_label: GraphEntityLabel, *, limit: int = 10) -> list[GraphEntityView]:
        raise NotImplementedError


class DisabledKnowledgeGraphBackend(KnowledgeGraphBackend):
    @property
    def is_enabled(self) -> bool:
        return False

    def ensure_schema(self) -> None:
        return None

    def add_entity(self, entity: GraphEntity) -> GraphEntityView:
        return GraphEntityView(label=entity.label, entity_id=entity.entity_id, properties=entity.properties)

    def add_relationship(self, relationship: GraphRelationship) -> None:
        return None

    def find_related_concepts(self, concept_id: str, *, limit: int = 10) -> list[GraphEntityView]:
        return []

    def find_supporting_evidence(self, question_id: str, *, limit: int = 10) -> list[GraphEntityView]:
        return []

    def find_related_projects(
        self,
        entity_id: str,
        entity_label: GraphEntityLabel,
        *,
        limit: int = 10,
    ) -> list[GraphEntityView]:
        return []


class InMemoryKnowledgeGraphBackend(KnowledgeGraphBackend):
    """Test backend mirroring graph query behavior."""

    def __init__(self) -> None:
        self.entities: dict[tuple[GraphEntityLabel, str], GraphEntityView] = {}
        self.relationships: list[GraphRelationship] = []

    @property
    def is_enabled(self) -> bool:
        return True

    def ensure_schema(self) -> None:
        return None

    def add_entity(self, entity: GraphEntity) -> GraphEntityView:
        key = (entity.label, entity.entity_id)
        view = GraphEntityView(label=entity.label, entity_id=entity.entity_id, properties=dict(entity.properties))
        self.entities[key] = view
        return view

    def add_relationship(self, relationship: GraphRelationship) -> None:
        self.relationships.append(relationship)

    def _neighbors(self, label: GraphEntityLabel, entity_id: str, rel_type: GraphRelationshipType | None = None):
        for rel in self.relationships:
            if rel_type is not None and rel.relationship_type != rel_type:
                continue
            if rel.from_label == label and rel.from_id == entity_id:
                yield rel.to_label, rel.to_id
            if rel.to_label == label and rel.to_id == entity_id:
                yield rel.from_label, rel.from_id

    def find_related_concepts(self, concept_id: str, *, limit: int = 10) -> list[GraphEntityView]:
        seen: set[tuple[GraphEntityLabel, str]] = set()
        results: list[GraphEntityView] = []
        frontier = [(GraphEntityLabel.CONCEPT, concept_id)]
        depth = 0
        while frontier and depth < 2 and len(results) < limit:
            next_frontier: list[tuple[GraphEntityLabel, str]] = []
            for label, eid in frontier:
                for n_label, n_id in self._neighbors(label, eid, GraphRelationshipType.RELATED_TO):
                    if n_label != GraphEntityLabel.CONCEPT:
                        continue
                    key = (n_label, n_id)
                    if key in seen or n_id == concept_id:
                        continue
                    seen.add(key)
                    entity = self.entities.get(key)
                    if entity is not None:
                        results.append(entity)
                        next_frontier.append(key)
            frontier = next_frontier
            depth += 1
        return results[:limit]

    def find_supporting_evidence(self, question_id: str, *, limit: int = 10) -> list[GraphEntityView]:
        results: list[GraphEntityView] = []
        for rel in self.relationships:
            if rel.relationship_type != GraphRelationshipType.SUPPORTS:
                continue
            if rel.to_label == GraphEntityLabel.RESEARCH_QUESTION and rel.to_id == question_id:
                key = (rel.from_label, rel.from_id)
                if rel.from_label == GraphEntityLabel.EVIDENCE and key in self.entities:
                    results.append(self.entities[key])
        return results[:limit]

    def find_related_projects(
        self,
        entity_id: str,
        entity_label: GraphEntityLabel,
        *,
        limit: int = 10,
    ) -> list[GraphEntityView]:
        seen: set[tuple[GraphEntityLabel, str]] = set()
        results: list[GraphEntityView] = []
        for rel in self.relationships:
            if rel.relationship_type == GraphRelationshipType.USED_IN_PROJECT:
                if rel.from_label == entity_label and rel.from_id == entity_id:
                    key = (GraphEntityLabel.PROJECT, rel.to_id)
                    if key in self.entities and key not in seen:
                        seen.add(key)
                        results.append(self.entities[key])
            if rel.relationship_type in {GraphRelationshipType.MENTIONS, GraphRelationshipType.RELATED_TO}:
                candidates: list[tuple[GraphEntityLabel, str]] = []
                if rel.from_label == entity_label and rel.from_id == entity_id:
                    candidates.append((rel.to_label, rel.to_id))
                if rel.to_label == entity_label and rel.to_id == entity_id:
                    candidates.append((rel.from_label, rel.from_id))
                for n_label, n_id in candidates:
                    for rel2 in self.relationships:
                        if rel2.relationship_type != GraphRelationshipType.USED_IN_PROJECT:
                            continue
                        if rel2.from_label == n_label and rel2.from_id == n_id:
                            key = (GraphEntityLabel.PROJECT, rel2.to_id)
                            if key in self.entities and key not in seen:
                                seen.add(key)
                                results.append(self.entities[key])
        return results[:limit]


class Neo4jKnowledgeGraphBackend(KnowledgeGraphBackend):
    def __init__(self, uri: str, user: str, password: str, database: str | None = None) -> None:
        from neo4j import GraphDatabase

        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    @property
    def is_enabled(self) -> bool:
        return True

    def close(self) -> None:
        self._driver.close()

    def _session(self):
        if self._database:
            return self._driver.session(database=self._database)
        return self._driver.session()

    def ensure_schema(self) -> None:
        labels = [label.value for label in GraphEntityLabel]
        statements = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (n:GraphEntity) REQUIRE n.entity_id IS UNIQUE",
        ]
        for label in labels:
            statements.append(
                f"CREATE CONSTRAINT {label.lower()}_entity_id IF NOT EXISTS "
                f"FOR (n:`{label}`) REQUIRE n.entity_id IS UNIQUE"
            )
        with self._session() as session:
            for stmt in statements:
                session.run(stmt)

    def add_entity(self, entity: GraphEntity) -> GraphEntityView:
        label = entity.label.value
        props = {"entity_id": entity.entity_id, **entity.properties}
        query = (
            f"MERGE (n:`{label}` {{entity_id: $entity_id}}) "
            "SET n += $props, n:GraphEntity "
            "RETURN n"
        )
        with self._session() as session:
            record = session.run(query, entity_id=entity.entity_id, props=props).single()
        node = dict(record["n"]) if record else props
        clean_props = {k: v for k, v in node.items() if k != "entity_id"}
        return GraphEntityView(label=entity.label, entity_id=entity.entity_id, properties=clean_props)

    def add_relationship(self, relationship: GraphRelationship) -> None:
        rel = relationship.relationship_type.value
        from_label = relationship.from_label.value
        to_label = relationship.to_label.value
        query = (
            f"MATCH (a:`{from_label}` {{entity_id: $from_id}}) "
            f"MATCH (b:`{to_label}` {{entity_id: $to_id}}) "
            f"MERGE (a)-[r:`{rel}`]->(b) "
            "SET r += $props"
        )
        with self._session() as session:
            session.run(
                query,
                from_id=relationship.from_id,
                to_id=relationship.to_id,
                props=relationship.properties,
            )

    @staticmethod
    def _node_to_view(node: dict[str, Any], label: GraphEntityLabel) -> GraphEntityView:
        entity_id = node.get("entity_id", "")
        props = {k: v for k, v in node.items() if k != "entity_id"}
        return GraphEntityView(label=label, entity_id=entity_id, properties=props)

    def find_related_concepts(self, concept_id: str, *, limit: int = 10) -> list[GraphEntityView]:
        query = (
            "MATCH (c:Concept {entity_id: $concept_id})-[:RELATED_TO*1..2]-(related:Concept) "
            "WHERE related.entity_id <> $concept_id "
            "RETURN DISTINCT related LIMIT $limit"
        )
        with self._session() as session:
            records = session.run(query, concept_id=concept_id, limit=limit)
            nodes = [dict(record["related"]) for record in records]
        return [self._node_to_view(node, GraphEntityLabel.CONCEPT) for node in nodes]

    def find_supporting_evidence(self, question_id: str, *, limit: int = 10) -> list[GraphEntityView]:
        query = (
            "MATCH (e:Evidence)-[:SUPPORTS]->(q:ResearchQuestion {entity_id: $question_id}) "
            "RETURN e LIMIT $limit"
        )
        with self._session() as session:
            records = session.run(query, question_id=question_id, limit=limit)
            nodes = [dict(record["e"]) for record in records]
        return [self._node_to_view(node, GraphEntityLabel.EVIDENCE) for node in nodes]

    def find_related_projects(
        self,
        entity_id: str,
        entity_label: GraphEntityLabel,
        *,
        limit: int = 10,
    ) -> list[GraphEntityView]:
        label = entity_label.value
        query = (
            f"MATCH (n:`{label}` {{entity_id: $entity_id}}) "
            "OPTIONAL MATCH (n)-[:USED_IN_PROJECT]->(p:Project) "
            "OPTIONAL MATCH (n)-[:MENTIONS|RELATED_TO*1..2]-(x)-[:USED_IN_PROJECT]->(p2:Project) "
            "WITH collect(DISTINCT p) + collect(DISTINCT p2) AS projects "
            "UNWIND projects AS project "
            "WITH DISTINCT project WHERE project IS NOT NULL "
            "RETURN project LIMIT $limit"
        )
        with self._session() as session:
            records = session.run(query, entity_id=entity_id, limit=limit)
            nodes = [dict(record["project"]) for record in records]
        return [self._node_to_view(node, GraphEntityLabel.PROJECT) for node in nodes]
