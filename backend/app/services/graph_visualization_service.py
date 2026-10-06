import logging
from datetime import datetime
from functools import lru_cache

from app.config.settings import Settings, get_settings
from app.knowledge_graph.service import KnowledgeGraphService, get_knowledge_graph_service
from app.knowledge_graph.types import GraphEntityLabel
from app.models.graph_viz import (
    GraphNodeDetailResponse,
    GraphNodeEvidenceItem,
    GraphNodeResearchItem,
    GraphQueryRequest,
    GraphVizEdge,
    GraphVizNode,
    GraphVizResponse,
)
from app.retrieval.tokenizer import tokenize
from app.services.graph_materializer import MaterializedGraph, materialize_graph
from app.services.research_history_service import ResearchHistoryService, create_research_history_service

logger = logging.getLogger(__name__)


class GraphVisualizationService:
    def __init__(
        self,
        settings: Settings,
        history_service: ResearchHistoryService,
        kg_service: KnowledgeGraphService,
    ) -> None:
        self._settings = settings
        self._history = history_service
        self._kg = kg_service
        self._graph_cache: MaterializedGraph | None = None
        self._cache_built_at: datetime | None = None

    def _materialized(self) -> MaterializedGraph:
        # Rebuild when history may have changed (simple: always rebuild for correctness)
        records = self._history.list_records()
        return materialize_graph(records, processed_dir=self._settings.data_processed_dir)

    def _matches_query(self, node: GraphVizNode, query_tokens: set[str]) -> bool:
        if not query_tokens:
            return True
        haystack = " ".join(
            [
                node.title,
                node.subtitle or "",
                str(node.properties.get("excerpt", "")),
                node.label.value,
            ]
        ).lower()
        doc_tokens = set(tokenize(haystack))
        return bool(query_tokens & doc_tokens)

    def _passes_filters(
        self,
        node: GraphVizNode,
        *,
        labels: set[GraphEntityLabel] | None,
        source_types: set[str] | None,
        date_from: datetime | None,
        date_to: datetime | None,
    ) -> bool:
        if labels and node.label not in labels:
            return False
        if source_types and node.source_type and node.source_type.lower() not in source_types:
            return False
        if date_from and node.created_at and node.created_at < date_from:
            return False
        if date_to and node.created_at and node.created_at > date_to:
            return False
        return True

    def _subgraph(
        self,
        graph: MaterializedGraph,
        seed_ids: set[str],
        *,
        depth: int,
        limit: int,
    ) -> tuple[list[GraphVizNode], list[GraphVizEdge]]:
        visited: set[str] = set()
        frontier = set(seed_ids)
        for _ in range(depth + 1):
            next_frontier: set[str] = set()
            for node_id in frontier:
                if node_id in visited or node_id not in graph.nodes:
                    continue
                visited.add(node_id)
                if len(visited) >= limit:
                    break
                for neighbor in graph.adjacency.get(node_id, set()):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
            frontier = next_frontier
            if len(visited) >= limit:
                break

        node_list = [graph.nodes[nid] for nid in visited if nid in graph.nodes]
        node_ids = {node.id for node in node_list}
        edges = [edge for edge in graph.edges if edge.source in node_ids and edge.target in node_ids]
        return node_list, edges

    def query(self, request: GraphQueryRequest) -> GraphVizResponse:
        graph = self._materialized()
        query_tokens = set(tokenize(request.query or ""))
        labels = set(request.node_labels) if request.node_labels else None
        source_types = {s.lower() for s in request.source_types} if request.source_types else None

        if request.center_node_id and request.center_node_id in graph.nodes:
            seeds = {request.center_node_id}
        else:
            seeds = {
                node.id
                for node in graph.nodes.values()
                if self._matches_query(node, query_tokens)
                and self._passes_filters(
                    node,
                    labels=labels,
                    source_types=source_types,
                    date_from=request.date_from,
                    date_to=request.date_to,
                )
            }
            if not seeds and query_tokens:
                seeds = {
                    node.id
                    for node in graph.nodes.values()
                    if self._passes_filters(
                        node,
                        labels=labels,
                        source_types=source_types,
                        date_from=request.date_from,
                        date_to=request.date_to,
                    )
                }

        nodes, edges = self._subgraph(graph, seeds, depth=request.expand_depth, limit=request.limit)
        backend = "hybrid" if self._kg.enabled else "materialized"
        return GraphVizResponse(
            query=request.query,
            nodes=nodes,
            edges=edges,
            backend=backend,
        )

    def get_node_detail(self, node_id: str) -> GraphNodeDetailResponse | None:
        graph = self._materialized()
        node = graph.nodes.get(node_id)
        if node is None:
            return None

        neighbors, edges = self._subgraph(graph, {node_id}, depth=1, limit=80)
        neighbor_nodes = [n for n in neighbors if n.id != node_id]

        evidence: list[GraphNodeEvidenceItem] = []
        research_items: list[GraphNodeResearchItem] = []
        papers: list[GraphVizNode] = []

        records_by_id = {record.research_id: record for record in self._history.list_records()}

        if node.label == GraphEntityLabel.EVIDENCE:
            rid = node.properties.get("research_id")
            evidence.append(
                GraphNodeEvidenceItem(
                    evidence_id=str(node.properties.get("evidence_id", "")),
                    title=node.title,
                    excerpt=str(node.properties.get("excerpt", "")),
                    research_id=str(rid) if rid else None,
                    source_type=node.source_type,
                )
            )
        elif node.label == GraphEntityLabel.RESEARCH_QUESTION:
            rid = node.properties.get("research_id") or node_id.split(":", 1)[-1]
            record = records_by_id.get(str(rid))
            if record:
                research_items.append(
                    GraphNodeResearchItem(
                        research_id=record.research_id,
                        question=record.question,
                        created_at=record.created_at,
                        excerpt=record.report.executive_summary[:400],
                    )
                )
                for ev in record.evidence:
                    evidence.append(
                        GraphNodeEvidenceItem(
                            evidence_id=ev.evidence_id,
                            title=ev.title,
                            excerpt=ev.text[:400],
                            research_id=record.research_id,
                            source_type=ev.source_metadata.source_type if ev.source_metadata else None,
                        )
                    )

        for edge in edges:
            other_id = edge.target if edge.source == node_id else edge.source
            other = graph.nodes.get(other_id)
            if other is None:
                continue
            if other.label == GraphEntityLabel.EVIDENCE and edge.relationship_type.value == "SUPPORTS":
                evidence.append(
                    GraphNodeEvidenceItem(
                        evidence_id=str(other.properties.get("evidence_id", other.id)),
                        title=other.title,
                        excerpt=str(other.properties.get("excerpt", "")),
                        research_id=str(other.properties.get("research_id") or ""),
                        source_type=other.source_type,
                    )
                )
            if other.label == GraphEntityLabel.RESEARCH_PAPER:
                papers.append(other)
            if other.label == GraphEntityLabel.RESEARCH_QUESTION:
                rid = other.properties.get("research_id")
                record = records_by_id.get(str(rid)) if rid else None
                if record:
                    research_items.append(
                        GraphNodeResearchItem(
                            research_id=record.research_id,
                            question=record.question,
                            created_at=record.created_at,
                            excerpt=record.report.executive_summary[:300],
                        )
                    )

        if node.label == GraphEntityLabel.CONCEPT and self._kg.enabled:
            concept_slug = node.properties.get("concept_slug") or node.id.split(":", 1)[-1]
            for project in self._kg.find_related_projects(concept_slug, GraphEntityLabel.CONCEPT, limit=10):
                pid = f"{GraphEntityLabel.PROJECT.value}:{project.entity_id}"
                if pid in graph.nodes:
                    neighbor_nodes.append(graph.nodes[pid])

        # dedupe
        seen_e = set()
        deduped_evidence = []
        for item in evidence:
            if item.evidence_id in seen_e:
                continue
            seen_e.add(item.evidence_id)
            deduped_evidence.append(item)

        return GraphNodeDetailResponse(
            node=node,
            neighbors=neighbor_nodes,
            edges=edges,
            evidence=deduped_evidence[:20],
            research_history=research_items[:10],
            supporting_papers=papers[:15],
        )


def create_graph_visualization_service(
    settings: Settings,
    *,
    history_service: ResearchHistoryService | None = None,
    kg_service: KnowledgeGraphService | None = None,
) -> GraphVisualizationService:
    return GraphVisualizationService(
        settings=settings,
        history_service=history_service or create_research_history_service(settings),
        kg_service=kg_service or get_knowledge_graph_service(),
    )


@lru_cache
def get_graph_visualization_service() -> GraphVisualizationService:
    return create_graph_visualization_service(get_settings())
