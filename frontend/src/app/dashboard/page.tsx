"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { fetchHealth, fetchResearchHistory } from "@/lib/api";
import { Badge } from "@/components/ui/Badge";

export default function DashboardPage() {
  const [health, setHealth] = useState<{ neo4j_connected: boolean; neo4j_enabled: boolean } | null>(null);
  const [historyCount, setHistoryCount] = useState(0);
  const [recent, setRecent] = useState<Array<{ research_id: string; question: string; created_at: string }>>([]);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
    fetchResearchHistory()
      .then((data) => {
        setHistoryCount(data.count);
        setRecent(data.items.slice(0, 5));
      })
      .catch(() => undefined);
  }, []);

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <header className="border-b border-zinc-800 px-6 py-5">
        <h1 className="text-2xl font-semibold">Dashboard</h1>
        <p className="text-sm text-zinc-500">Overview of your research lab activity and system status.</p>
      </header>

      <div className="grid gap-4 p-6 md:grid-cols-3">
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
          <p className="text-xs uppercase tracking-wide text-zinc-500">Research runs</p>
          <p className="mt-2 text-3xl font-semibold">{historyCount}</p>
        </div>
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
          <p className="text-xs uppercase tracking-wide text-zinc-500">Knowledge graph</p>
          <p className="mt-2 flex items-center gap-2">
            {health?.neo4j_connected ? <Badge tone="success">Connected</Badge> : <Badge tone="warn">Offline</Badge>}
          </p>
        </div>
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
          <p className="text-xs uppercase tracking-wide text-zinc-500">Quick action</p>
          <Link href="/research" className="mt-2 inline-block text-sm text-cyan-400 hover:text-cyan-300">
            Start new research →
          </Link>
        </div>
      </div>

      <section className="px-6 pb-8">
        <h2 className="mb-3 text-sm font-semibold text-zinc-300">Recent research</h2>
        <div className="space-y-2">
          {recent.length ? (
            recent.map((item) => (
              <Link
                key={item.research_id}
                href={`/research/${item.research_id}`}
                className="block rounded-lg border border-zinc-800 bg-zinc-900/30 px-4 py-3 hover:border-zinc-700"
              >
                <p className="text-sm text-zinc-200">{item.question}</p>
                <p className="mt-1 text-xs text-zinc-500">{new Date(item.created_at).toLocaleString()}</p>
              </Link>
            ))
          ) : (
            <p className="text-sm text-zinc-500">No research history yet. Run your first study from the Research page.</p>
          )}
        </div>
      </section>
    </div>
  );
}
