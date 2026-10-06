"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchGraphNodeDetail, fetchHealth, queryKnowledgeGraph } from "@/lib/api";
import type { GraphNodeDetailResponse, GraphVizNode } from "@/lib/types";
import { Badge } from "@/components/ui/Badge";
import { cn } from "@/lib/cn";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

const LABEL_OPTIONS = [
  "Concept",
  "ResearchPaper",
  "Project",
  "ResearchQuestion",
  "Evidence",
  "Document",
  "Author",
  "Topic",
];

const SOURCE_OPTIONS = ["openalex", "arxiv", "local", "pdf", "research_history", "project", "derived"];

const NODE_COLORS: Record<string, string> = {
  Concept: "#22d3ee",
  ResearchPaper: "#a78bfa",
  Project: "#fbbf24",
  ResearchQuestion: "#34d399",
  Evidence: "#fb7185",
  Document: "#94a3b8",
  Author: "#c084fc",
  Topic: "#60a5fa",
};

const EXAMPLE_QUERIES = [
  "How is Graph RAG connected to my projects?",
  "Graph RAG hybrid retrieval",
  "Agentic RAG papers",
];

type GraphData = {
  nodes: Array<GraphVizNode & { name: string; val: number }>;
  links: Array<{ source: string; target: string; type: string }>;
};

export function GraphExplorer() {
  const [query, setQuery] = useState("How is Graph RAG connected to my projects?");
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
  const [selectedLabels, setSelectedLabels] = useState<string[]>(LABEL_OPTIONS);
  const [selectedSources, setSelectedSources] = useState<string[]>([]);
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();
  const [selectedNode, setSelectedNode] = useState<GraphVizNode | null>(null);
  const [detail, setDetail] = useState<GraphNodeDetailResponse | null>(null);
  const [neo4jConnected, setNeo4jConnected] = useState(false);

  useEffect(() => {
    fetchHealth()
      .then((h) => setNeo4jConnected(Boolean(h.neo4j_connected)))
      .catch(() => setNeo4jConnected(false));
  }, []);

  const loadGraph = useCallback(async () => {
    setLoading(true);
    setError(undefined);
    try {
      const response = await queryKnowledgeGraph({
        query,
        expand_depth: 2,
        limit: 150,
        node_labels: selectedLabels.length ? selectedLabels : undefined,
        source_types: selectedSources.length ? selectedSources : undefined,
        date_from: dateFrom ? new Date(dateFrom).toISOString() : undefined,
        date_to: dateTo ? new Date(dateTo).toISOString() : undefined,
      });
      setGraphData({
        nodes: response.nodes.map((node) => ({
          ...node,
          name: node.title,
          val: node.label === "ResearchQuestion" ? 8 : node.label === "Project" ? 7 : 5,
        })),
        links: response.edges.map((edge) => ({
          source: edge.source,
          target: edge.target,
          type: edge.relationship_type,
        })),
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load graph");
      setGraphData({ nodes: [], links: [] });
    } finally {
      setLoading(false);
    }
  }, [query, selectedLabels, selectedSources, dateFrom, dateTo]);

  useEffect(() => {
    loadGraph();
  }, [loadGraph]);

  const onNodeClick = useCallback(async (node: GraphVizNode & { id?: string }) => {
    const graphNode = graphData.nodes.find((n) => n.id === node.id) ?? (node as GraphVizNode);
    setSelectedNode(graphNode);
    setDetail(null);
    try {
      const response = await fetchGraphNodeDetail(graphNode.id);
      setDetail(response);
    } catch {
      setDetail(null);
    }
  }, [graphData.nodes]);

  const legend = useMemo(
    () =>
      LABEL_OPTIONS.map((label) => ({
        label,
        color: NODE_COLORS[label] ?? "#94a3b8",
        count: graphData.nodes.filter((n) => n.label === label).length,
      })),
    [graphData.nodes],
  );

  return (
    <div className="flex h-full flex-col">
      <header className="border-b border-zinc-800 px-6 py-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-2xl font-semibold">Knowledge graph</h1>
            <p className="text-sm text-zinc-500">
              Explore how concepts, papers, projects, and research questions connect across your lab.
            </p>
          </div>
          <Badge tone={neo4jConnected ? "success" : "neutral"}>
            {neo4jConnected ? "Neo4j + materialized view" : "Materialized from history"}
          </Badge>
        </div>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 p-4 xl:grid-cols-[280px_1fr_320px]">
        <aside className="space-y-4 overflow-y-auto rounded-xl border border-zinc-800 bg-zinc-900/30 p-4">
          <div>
            <label className="text-xs uppercase tracking-wide text-zinc-500">Explore</label>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="mt-2 w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm"
              placeholder="Graph RAG projects"
            />
            <div className="mt-2 flex flex-wrap gap-1">
              {EXAMPLE_QUERIES.map((example) => (
                <button
                  key={example}
                  type="button"
                  onClick={() => setQuery(example)}
                  className="rounded-md border border-zinc-700 px-2 py-1 text-[10px] text-zinc-400 hover:border-zinc-600"
                >
                  {example.slice(0, 28)}…
                </button>
              ))}
            </div>
            <button
              type="button"
              onClick={loadGraph}
              disabled={loading}
              className="mt-3 w-full rounded-lg bg-cyan-600 py-2 text-sm font-medium text-zinc-950 disabled:opacity-50"
            >
              {loading ? "Loading…" : "Refresh graph"}
            </button>
          </div>

          <div>
            <p className="text-xs uppercase tracking-wide text-zinc-500">Node types</p>
            <div className="mt-2 flex flex-wrap gap-1">
              {LABEL_OPTIONS.map((label) => {
                const on = selectedLabels.includes(label);
                return (
                  <button
                    key={label}
                    type="button"
                    onClick={() =>
                      setSelectedLabels((prev) =>
                        on ? prev.filter((item) => item !== label) : [...prev, label],
                      )
                    }
                    className={cn(
                      "rounded-md border px-2 py-1 text-[10px]",
                      on ? "border-cyan-700 text-cyan-200" : "border-zinc-700 text-zinc-500",
                    )}
                  >
                    {label}
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <p className="text-xs uppercase tracking-wide text-zinc-500">Source type</p>
            <div className="mt-2 flex flex-wrap gap-1">
              {SOURCE_OPTIONS.map((source) => {
                const on = selectedSources.includes(source);
                return (
                  <button
                    key={source}
                    type="button"
                    onClick={() =>
                      setSelectedSources((prev) =>
                        on ? prev.filter((item) => item !== source) : [...prev, source],
                      )
                    }
                    className={cn(
                      "rounded-md border px-2 py-1 text-[10px] capitalize",
                      on ? "border-amber-700 text-amber-200" : "border-zinc-700 text-zinc-500",
                    )}
                  >
                    {source}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <label className="text-[10px] text-zinc-500">
              From
              <input
                type="date"
                value={dateFrom}
                onChange={(e) => setDateFrom(e.target.value)}
                className="mt-1 w-full rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs"
              />
            </label>
            <label className="text-[10px] text-zinc-500">
              To
              <input
                type="date"
                value={dateTo}
                onChange={(e) => setDateTo(e.target.value)}
                className="mt-1 w-full rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs"
              />
            </label>
          </div>

          <div className="space-y-1 text-xs text-zinc-500">
            {legend.map((item) => (
              <div key={item.label} className="flex items-center justify-between">
                <span className="flex items-center gap-2">
                  <span className="inline-block h-2 w-2 rounded-full" style={{ background: item.color }} />
                  {item.label}
                </span>
                <span>{item.count}</span>
              </div>
            ))}
          </div>
          {error ? <p className="text-xs text-rose-400">{error}</p> : null}
        </aside>

        <section className="relative min-h-[420px] overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950">
          {graphData.nodes.length ? (
            <ForceGraph2D
              graphData={graphData}
              nodeLabel={(node) => `${(node as GraphVizNode).label}: ${(node as GraphVizNode).title}`}
              linkDirectionalArrowLength={3.5}
              linkDirectionalArrowRelPos={1}
              linkColor={() => "rgba(148, 163, 184, 0.35)"}
              nodeCanvasObject={(node, ctx, globalScale) => {
                const n = node as GraphVizNode & { x?: number; y?: number };
                const label = n.label ?? "Node";
                const color = NODE_COLORS[label] ?? "#94a3b8";
                const size = Math.sqrt(Math.max(4, (node as { val?: number }).val ?? 4)) * 3;
                if (n.x == null || n.y == null) return;
                ctx.beginPath();
                ctx.arc(n.x, n.y, size, 0, 2 * Math.PI, false);
                ctx.fillStyle = color;
                ctx.fill();
                if (globalScale > 1.2) {
                  ctx.font = `${10 / globalScale}px Sans-Serif`;
                  ctx.fillStyle = "#e4e4e7";
                  ctx.fillText(n.title.slice(0, 24), n.x + size + 2, n.y + 3);
                }
              }}
              onNodeClick={(node) => onNodeClick(node as GraphVizNode)}
            />
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-zinc-500">
              {loading ? "Building graph…" : "No nodes match your filters. Run research or ingest knowledge first."}
            </div>
          )}
          <p className="pointer-events-none absolute bottom-2 left-3 text-[10px] text-zinc-600">
            Scroll to zoom · drag background to pan · click a node to inspect evidence
          </p>
        </section>

        <aside className="overflow-y-auto rounded-xl border border-zinc-800 bg-zinc-900/30 p-4">
          <h2 className="text-sm font-semibold text-zinc-200">Selection</h2>
          {!selectedNode ? (
            <p className="mt-3 text-xs text-zinc-500">Select a node to explore relationships, evidence, and research history.</p>
          ) : (
            <div className="mt-3 space-y-4 text-sm">
              <div>
                <Badge tone="accent">{selectedNode.label}</Badge>
                <p className="mt-2 font-medium text-zinc-100">{selectedNode.title}</p>
                {selectedNode.subtitle ? <p className="text-xs text-zinc-500">{selectedNode.subtitle}</p> : null}
                {selectedNode.source_type ? (
                  <p className="mt-1 text-xs text-zinc-600">Source: {selectedNode.source_type}</p>
                ) : null}
              </div>

              {detail?.supporting_papers.length ? (
                <div>
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Supporting papers</h3>
                  <ul className="mt-2 space-y-2 text-xs text-zinc-400">
                    {detail.supporting_papers.map((paper) => (
                      <li key={paper.id} className="rounded border border-zinc-800 p-2">{paper.title}</li>
                    ))}
                  </ul>
                </div>
              ) : null}

              {detail?.evidence.length ? (
                <div>
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Evidence</h3>
                  <ul className="mt-2 space-y-2 text-xs text-zinc-400">
                    {detail.evidence.map((item) => (
                      <li key={item.evidence_id} className="rounded border border-zinc-800 p-2">
                        <p className="font-medium text-zinc-300">{item.title}</p>
                        <p className="mt-1 line-clamp-4">{item.excerpt}</p>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}

              {detail?.research_history.length ? (
                <div>
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Research history</h3>
                  <ul className="mt-2 space-y-2">
                    {detail.research_history.map((item) => (
                      <li key={item.research_id} className="rounded border border-zinc-800 p-2 text-xs">
                        <Link href={`/research/${item.research_id}`} className="text-cyan-400 hover:text-cyan-300">
                          {item.question}
                        </Link>
                        {item.excerpt ? <p className="mt-1 line-clamp-3 text-zinc-500">{item.excerpt}</p> : null}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}

              {detail?.neighbors.length ? (
                <div>
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Neighbors</h3>
                  <ul className="mt-2 space-y-1 text-xs text-zinc-500">
                    {detail.neighbors.slice(0, 8).map((neighbor) => (
                      <li key={neighbor.id}>
                        <button
                          type="button"
                          className="text-left hover:text-cyan-300"
                          onClick={() => onNodeClick(neighbor)}
                        >
                          {neighbor.label}: {neighbor.title}
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
