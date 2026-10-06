import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.knowledge_graph.types import GraphEntityLabel, GraphRelationshipType
from app.models.graph_viz import GraphVizEdge, GraphVizNode
from app.models.research_history import ResearchHistoryRecord
from app.retrieval.chunk_store import load_chunks_from_processed
from app.retrieval.tokenizer import tokenize


def _slug(text: str) -> str:
    tokens = tokenize(text)
    if not tokens:
        return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "item"
    return "-".join(tokens[:8])


def _node_key(label: GraphEntityLabel, entity_id: str) -> str:
    return f"{label.value}:{entity_id}"


@dataclass
class MaterializedGraph:
    nodes: dict[str, GraphVizNode] = field(default_factory=dict)
    edges: list[GraphVizEdge] = field(default_factory=list)
    adjacency: dict[str, set[str]] = field(default_factory=dict)

    def add_node(self, node: GraphVizNode) -> None:
        self.nodes[node.id] = node

    def add_edge(
        self,
        *,
        source: str,
        target: str,
        relationship_type: GraphRelationshipType,
        properties: dict | None = None,
    ) -> None:
        edge_id = f"{source}|{relationship_type.value}|{target}"
        if any(edge.id == edge_id for edge in self.edges):
            return
        self.edges.append(
            GraphVizEdge(
                id=edge_id,
                source=source,
                target=target,
                relationship_type=relationship_type,
                properties=properties or {},
            )
        )
        self.adjacency.setdefault(source, set()).add(target)
        self.adjacency.setdefault(target, set()).add(source)


def materialize_graph(
    records: list[ResearchHistoryRecord],
    *,
    processed_dir: Path,
) -> MaterializedGraph:
    graph = MaterializedGraph()

    for record in records:
        rq_id = _node_key(GraphEntityLabel.RESEARCH_QUESTION, record.research_id)
        graph.add_node(
            GraphVizNode(
                id=rq_id,
                label=GraphEntityLabel.RESEARCH_QUESTION,
                title=record.question,
                subtitle=record.plan.research_objective,
                source_type="research_history",
                created_at=record.created_at,
                properties={
                    "research_id": record.research_id,
                    "unresolved_count": len(record.unresolved_questions),
                },
            )
        )

        for text in record.plan.sub_questions + record.plan.search_queries + record.conclusions:
            concept_id = _slug(text)
            cid = _node_key(GraphEntityLabel.CONCEPT, concept_id)
            if cid not in graph.nodes:
                graph.add_node(
                    GraphVizNode(
                        id=cid,
                        label=GraphEntityLabel.CONCEPT,
                        title=text,
                        source_type="derived",
                        created_at=record.created_at,
                        properties={"concept_slug": concept_id},
                    )
                )
            graph.add_edge(
                source=rq_id,
                target=cid,
                relationship_type=GraphRelationshipType.MENTIONS,
            )

        for source in record.sources:
            if not source.external_id:
                continue
            paper_id = _node_key(GraphEntityLabel.RESEARCH_PAPER, f"{source.source}:{source.external_id}")
            if paper_id not in graph.nodes:
                graph.add_node(
                    GraphVizNode(
                        id=paper_id,
                        label=GraphEntityLabel.RESEARCH_PAPER,
                        title=source.title,
                        source_type=source.source,
                        created_at=record.created_at,
                        properties={
                            "external_id": source.external_id,
                            "url": source.url,
                        },
                    )
                )
            graph.add_edge(
                source=paper_id,
                target=rq_id,
                relationship_type=GraphRelationshipType.SUPPORTS,
            )
            for text in record.plan.search_queries[:3]:
                cid = _node_key(GraphEntityLabel.CONCEPT, _slug(text))
                if cid in graph.nodes:
                    graph.add_edge(
                        source=paper_id,
                        target=cid,
                        relationship_type=GraphRelationshipType.MENTIONS,
                    )

        for evidence in record.evidence:
            eid = _node_key(GraphEntityLabel.EVIDENCE, evidence.evidence_id)
            source_type = evidence.source_metadata.source_type if evidence.source_metadata else "local"
            graph.add_node(
                GraphVizNode(
                    id=eid,
                    label=GraphEntityLabel.EVIDENCE,
                    title=evidence.title,
                    subtitle=evidence.section,
                    source_type=source_type,
                    created_at=record.created_at,
                    properties={
                        "evidence_id": evidence.evidence_id,
                        "chunk_id": evidence.chunk_id,
                        "document_id": evidence.document_id,
                        "excerpt": evidence.text[:500],
                        "research_id": record.research_id,
                    },
                )
            )
            graph.add_edge(
                source=eid,
                target=rq_id,
                relationship_type=GraphRelationshipType.SUPPORTS,
            )
            doc_id = _node_key(GraphEntityLabel.DOCUMENT, evidence.document_id)
            if doc_id not in graph.nodes:
                graph.add_node(
                    GraphVizNode(
                        id=doc_id,
                        label=GraphEntityLabel.DOCUMENT,
                        title=evidence.title,
                        source_type=source_type,
                        created_at=record.created_at,
                        properties={"document_id": evidence.document_id},
                    )
                )
            graph.add_edge(
                source=eid,
                target=doc_id,
                relationship_type=GraphRelationshipType.DERIVED_FROM,
            )
            project_id = evidence.source_metadata.project_id if evidence.source_metadata else None
            if project_id:
                pid = _node_key(GraphEntityLabel.PROJECT, project_id)
                if pid not in graph.nodes:
                    graph.add_node(
                        GraphVizNode(
                            id=pid,
                            label=GraphEntityLabel.PROJECT,
                            title=project_id,
                            source_type="project",
                            created_at=record.created_at,
                            properties={"project_id": project_id},
                        )
                    )
                graph.add_edge(
                    source=doc_id,
                    target=pid,
                    relationship_type=GraphRelationshipType.USED_IN_PROJECT,
                )
                graph.add_edge(
                    source=rq_id,
                    target=pid,
                    relationship_type=GraphRelationshipType.USED_IN_PROJECT,
                )

    for chunk in load_chunks_from_processed(processed_dir):
        project_id = chunk.project_id
        if not project_id:
            continue
        pid = _node_key(GraphEntityLabel.PROJECT, str(project_id))
        if pid not in graph.nodes:
            graph.add_node(
                GraphVizNode(
                    id=pid,
                    label=GraphEntityLabel.PROJECT,
                    title=str(project_id),
                    source_type="project",
                    properties={"project_id": project_id},
                )
            )
        doc_id = _node_key(GraphEntityLabel.DOCUMENT, chunk.document_id)
        if doc_id not in graph.nodes:
            graph.add_node(
                GraphVizNode(
                    id=doc_id,
                    label=GraphEntityLabel.DOCUMENT,
                    title=chunk.filename or chunk.document_id,
                    source_type=chunk.source_type.value,
                    properties={"document_id": chunk.document_id},
                )
            )
        graph.add_edge(
            source=doc_id,
            target=pid,
            relationship_type=GraphRelationshipType.USED_IN_PROJECT,
        )

    return graph
