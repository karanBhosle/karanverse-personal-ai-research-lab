"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { fetchResearchHistory } from "@/lib/api";
import type { ResearchHistorySummary } from "@/lib/types";

export default function HistoryPage() {
  const [items, setItems] = useState<ResearchHistorySummary[]>([]);
  const [error, setError] = useState<string>();

  useEffect(() => {
    fetchResearchHistory()
      .then((data) => setItems(data.items))
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load history"));
  }, []);

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <header className="border-b border-zinc-800 px-6 py-5">
        <h1 className="text-2xl font-semibold">Research history</h1>
        <p className="text-sm text-zinc-500">Persisted runs with plans, evidence, and reports.</p>
      </header>

      <div className="p-6">
        {error ? <p className="text-sm text-rose-400">{error}</p> : null}
        <div className="space-y-2">
          {items.map((item) => (
            <Link
              key={item.research_id}
              href={`/research/${item.research_id}`}
              className="block rounded-xl border border-zinc-800 bg-zinc-900/30 px-4 py-4 hover:border-zinc-700"
            >
              <p className="text-sm font-medium text-zinc-200">{item.question}</p>
              <p className="mt-1 text-xs text-zinc-500">{item.research_objective}</p>
              <div className="mt-2 flex gap-4 text-xs text-zinc-600">
                <span>{new Date(item.created_at).toLocaleString()}</span>
                <span>{item.conclusion_count} conclusions</span>
                <span>{item.unresolved_question_count} open</span>
              </div>
            </Link>
          ))}
          {!items.length && !error ? <p className="text-sm text-zinc-500">No saved research yet.</p> : null}
        </div>
      </div>
    </div>
  );
}
