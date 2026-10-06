"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { fetchResearchRecord } from "@/lib/api";
import type { ResearchHistoryRecord } from "@/lib/types";
import { Badge } from "@/components/ui/Badge";

export default function ResearchDetailPage() {
  const params = useParams<{ id: string }>();
  const [record, setRecord] = useState<ResearchHistoryRecord | null>(null);
  const [error, setError] = useState<string>();

  useEffect(() => {
    if (!params.id) return;
    fetchResearchRecord(params.id)
      .then(setRecord)
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load record"));
  }, [params.id]);

  if (error) {
    return <div className="p-6 text-sm text-rose-400">{error}</div>;
  }

  if (!record) {
    return <div className="p-6 text-sm text-zinc-500">Loading research record…</div>;
  }

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <header className="border-b border-zinc-800 px-6 py-5">
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="text-xl font-semibold">Research record</h1>
          <Badge tone="accent">{record.research_id.slice(0, 8)}</Badge>
        </div>
        <p className="mt-2 text-sm text-zinc-400">{record.question}</p>
        <p className="text-xs text-zinc-600">{new Date(record.created_at).toLocaleString()}</p>
      </header>

      <div className="grid gap-4 p-6 lg:grid-cols-2">
        <article className="rounded-xl border border-zinc-800 bg-zinc-900/30 p-4 text-sm">
          <h2 className="font-semibold text-zinc-200">Report</h2>
          <p className="mt-3 text-zinc-300">{record.report.executive_summary}</p>
          <ul className="mt-4 space-y-2">
            {record.report.key_findings.map((f) => (
              <li key={f.statement} className="rounded-lg border border-zinc-800 bg-zinc-950/50 p-3">
                {f.statement}
                <div className="mt-2">
                  <Badge tone="neutral">{f.confidence}</Badge>
                </div>
              </li>
            ))}
          </ul>
        </article>

        <aside className="space-y-4">
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/30 p-4 text-sm">
            <h2 className="font-semibold text-zinc-200">Evidence ({record.evidence.length})</h2>
            <ul className="mt-3 max-h-[420px] space-y-2 overflow-y-auto">
              {record.evidence.map((ev) => (
                <li key={ev.evidence_id} className="rounded-lg border border-zinc-800 p-2 text-xs text-zinc-500">
                  <p className="font-medium text-zinc-300">{ev.title}</p>
                  <p className="mt-1 line-clamp-3">{ev.text}</p>
                </li>
              ))}
            </ul>
          </div>
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/30 p-4 text-sm">
            <h2 className="font-semibold text-zinc-200">Open questions</h2>
            <ul className="mt-2 list-disc pl-5 text-zinc-500">
              {record.unresolved_questions.map((q) => (
                <li key={q}>{q}</li>
              ))}
            </ul>
          </div>
        </aside>
      </div>
    </div>
  );
}
