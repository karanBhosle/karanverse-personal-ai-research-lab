"use client";

import { STAGE_ORDER, stageIndex, stageLabel } from "@/lib/research-progress";
import type { ProgressStage } from "@/lib/types";
import { cn } from "@/lib/cn";

export function ProgressPipeline({
  activeStage,
  statusMessage,
  running,
}: {
  activeStage: ProgressStage;
  statusMessage?: string;
  running: boolean;
}) {
  const activeIndex = stageIndex(activeStage);

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold text-zinc-200">Research progress</h2>
        {running ? (
          <span className="text-xs text-cyan-400 animate-pulse">Running workflow…</span>
        ) : activeStage === "complete" ? (
          <span className="text-xs text-emerald-400">Complete</span>
        ) : null}
      </div>
      <ol className="flex flex-wrap items-center gap-2 text-xs">
        {STAGE_ORDER.map((stage, index) => {
          const done = index < activeIndex || activeStage === "complete";
          const current = index === activeIndex && running;
          return (
            <li key={stage} className="flex items-center gap-2">
              <span
                className={cn(
                  "rounded-full border px-2.5 py-1 font-medium",
                  done && "border-emerald-800 bg-emerald-950/50 text-emerald-200",
                  current && "border-cyan-600 bg-cyan-950/60 text-cyan-100 ring-1 ring-cyan-500/40",
                  !done && !current && "border-zinc-700 bg-zinc-900 text-zinc-500",
                )}
              >
                {stageLabel(stage)}
              </span>
              {index < STAGE_ORDER.length - 1 ? <span className="text-zinc-600">→</span> : null}
            </li>
          );
        })}
      </ol>
      {statusMessage ? <p className="mt-3 text-xs text-zinc-400">{statusMessage}</p> : null}
    </div>
  );
}
