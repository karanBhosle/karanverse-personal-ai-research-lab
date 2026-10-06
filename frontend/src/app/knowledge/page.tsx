"use client";

import { useState } from "react";
import { detectKnowledgeGaps, searchKnowledge } from "@/lib/api";
import type { KnowledgeGap } from "@/lib/types";
import { Badge } from "@/components/ui/Badge";

export default function KnowledgePage() {
  const [query, setQuery] = useState("What have I learned about Graph RAG?");
  const [searchResults, setSearchResults] = useState<Array<{ chunk_id: string; text: string; final_score: number }>>([]);
  const [gaps, setGaps] = useState<KnowledgeGap[]>([]);
  const [summary, setSummary] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();

  const runSearch = async () => {
    setLoading(true);
    setError(undefined);
    try {
      const data = await searchKnowledge(query, 8);
      setSearchResults(data.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setLoading(false);
    }
  };

  const runGaps = async () => {
    setLoading(true);
    setError(undefined);
    try {
      const data = await detectKnowledgeGaps(query);
      setGaps(data.gaps);
      setSummary(data.coverage_summary);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Gap detection failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <header className="border-b border-zinc-800 px-6 py-5">
        <h1 className="text-2xl font-semibold">Knowledge</h1>
        <p className="text-sm text-zinc-500">Search personal/project/public corpora and detect knowledge gaps.</p>
      </header>

      <div className="grid gap-4 p-6 lg:grid-cols-2">
        <section className="rounded-xl border border-zinc-800 bg-zinc-900/30 p-4">
          <label className="text-xs uppercase tracking-wide text-zinc-500">Query</label>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            rows={4}
            className="mt-2 w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm"
          />
          <div className="mt-3 flex gap-2">
            <button
              type="button"
              onClick={runSearch}
              disabled={loading}
              className="rounded-lg border border-zinc-700 px-3 py-2 text-sm hover:bg-zinc-800"
            >
              Search knowledge
            </button>
            <button
              type="button"
              onClick={runGaps}
              disabled={loading}
              className="rounded-lg bg-cyan-600 px-3 py-2 text-sm font-medium text-zinc-950"
            >
              Detect gaps
            </button>
          </div>
          {error ? <p className="mt-2 text-xs text-rose-400">{error}</p> : null}
        </section>

        <section className="rounded-xl border border-zinc-800 bg-zinc-900/30 p-4">
          <h2 className="text-sm font-semibold">Knowledge gaps</h2>
          {summary ? <p className="mt-2 text-sm text-zinc-400">{summary}</p> : null}
          <ul className="mt-3 space-y-3">
            {gaps.map((gap) => (
              <li key={`${gap.topic}-${gap.reason}`} className="rounded-lg border border-zinc-800 bg-zinc-950/50 p-3 text-sm">
                <div className="flex items-center gap-2">
                  <p className="font-medium text-zinc-200">{gap.topic}</p>
                  <Badge tone={gap.priority === "high" ? "danger" : gap.priority === "medium" ? "warn" : "neutral"}>
                    {gap.priority}
                  </Badge>
                </div>
                <p className="mt-2 text-zinc-500">{gap.reason}</p>
                <p className="mt-2 text-xs text-cyan-400/90">{gap.recommended_learning}</p>
              </li>
            ))}
          </ul>
        </section>

        <section className="rounded-xl border border-zinc-800 bg-zinc-900/30 p-4 lg:col-span-2">
          <h2 className="text-sm font-semibold">Retrieved chunks</h2>
          <ul className="mt-3 grid gap-2 md:grid-cols-2">
            {searchResults.map((hit) => (
              <li key={hit.chunk_id} className="rounded-lg border border-zinc-800 p-3 text-xs text-zinc-500">
                <p className="text-zinc-400">score {hit.final_score.toFixed(3)}</p>
                <p className="mt-1 line-clamp-4">{hit.text}</p>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}
