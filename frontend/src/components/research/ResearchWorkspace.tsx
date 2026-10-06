"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  collectEvidence,
  createResearchPlan,
  critiqueBundle,
  fetchResearchSources,
  synthesizeReport,
} from "@/lib/api";
import { stageFromNode, stageLabel } from "@/lib/research-progress";
import { streamResearchWorkflow } from "@/lib/research-stream";
import type {
  Evidence,
  ProgressStage,
  ResearchMode,
  ResearchPlan,
  ResearchReport,
  ResearchWorkflowResponse,
} from "@/lib/types";
import { ProgressPipeline } from "./ProgressPipeline";
import { Badge } from "@/components/ui/Badge";
import { cn } from "@/lib/cn";

function confidenceTone(level: string) {
  if (level === "high") return "success" as const;
  if (level === "low") return "warn" as const;
  return "neutral" as const;
}

export function ResearchWorkspace() {
  const router = useRouter();
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<ResearchMode>("stream");
  const [sources, setSources] = useState<string[]>([]);
  const [selectedSources, setSelectedSources] = useState<string[]>([]);
  const [maxIterations, setMaxIterations] = useState(2);
  const [evidenceTopK, setEvidenceTopK] = useState(5);
  const [useHistory, setUseHistory] = useState(true);

  const [running, setRunning] = useState(false);
  const [activeStage, setActiveStage] = useState<ProgressStage>("planner");
  const [statusMessage, setStatusMessage] = useState<string>();
  const [error, setError] = useState<string>();

  const [plan, setPlan] = useState<ResearchPlan | null>(null);
  const [evidence, setEvidence] = useState<Evidence[]>([]);
  const [report, setReport] = useState<ResearchReport | null>(null);
  const [researchId, setResearchId] = useState<string | null>(null);
  const [logLines, setLogLines] = useState<string[]>([]);

  useEffect(() => {
    fetchResearchSources()
      .then((data) => {
        setSources(data.sources);
        setSelectedSources(data.sources);
      })
      .catch(() => setSources(["openalex", "arxiv"]));
  }, []);

  const resetRun = () => {
    setPlan(null);
    setEvidence([]);
    setReport(null);
    setResearchId(null);
    setLogLines([]);
    setError(undefined);
    setActiveStage("planner");
    setStatusMessage(undefined);
  };

  const appendLog = (line: string) => setLogLines((prev) => [...prev.slice(-12), line]);

  const runStream = useCallback(async () => {
    resetRun();
    setRunning(true);
    setActiveStage("planner");
    await streamResearchWorkflow(
      {
        question,
        evidence_top_k: evidenceTopK,
        max_iterations: maxIterations,
        use_research_history: useHistory,
      },
      (event) => {
        if (event.type === "progress") {
          const stage = stageFromNode(event.node);
          setActiveStage(stage);
          setStatusMessage(event.message);
          appendLog(`${stageLabel(stage)}: ${event.message}`);
        } else if (event.type === "complete") {
          const result = event.result as ResearchWorkflowResponse;
          setResearchId(event.research_id);
          setReport(result.report);
          if (result.report.evidence?.length) {
            setEvidence(
              result.report.evidence.map((item) => ({
                evidence_id: item.evidence_id,
                document_id: "",
                chunk_id: item.evidence_id,
                source: item.title,
                title: item.title,
                text: item.excerpt,
                retrieval_score: 0,
              })),
            );
          }
          setActiveStage("complete");
          setStatusMessage("Research report ready.");
          setRunning(false);
        } else if (event.type === "error") {
          setError(event.message);
          setActiveStage("error");
          setRunning(false);
        }
      },
    );
    setRunning(false);
  }, [question, evidenceTopK, maxIterations, useHistory]);

  const runGuided = useCallback(async () => {
    resetRun();
    setRunning(true);
    try {
      setActiveStage("planner");
      setStatusMessage("Creating structured research plan…");
      const planRes = await createResearchPlan(question);
      const nextPlan = { ...planRes.plan, required_sources: selectedSources.length ? selectedSources : planRes.plan.required_sources };
      setPlan(nextPlan);
      appendLog("Planner: plan created");

      setActiveStage("searching");
      setStatusMessage("Searching literature and retrieving evidence…");
      const bundle = await collectEvidence(question, evidenceTopK);
      setEvidence(bundle.evidence ?? []);
      appendLog(`Retrieving: ${bundle.evidence?.length ?? 0} evidence items`);

      setActiveStage("reranking");
      setStatusMessage("Reranking and packaging evidence bundle…");
      setActiveStage("critiquing");
      setStatusMessage("Running critic on evidence…");
      const critique = await critiqueBundle(bundle);

      setActiveStage("synthesizing");
      setStatusMessage("Synthesizing grounded report…");
      const synthesized = await synthesizeReport({
        question,
        plan: nextPlan,
        bundle,
        critique,
      });
      setReport(synthesized);
      setActiveStage("complete");
      setStatusMessage("Guided research complete.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Guided research failed");
      setActiveStage("error");
    } finally {
      setRunning(false);
    }
  }, [question, evidenceTopK, selectedSources]);

  const onSubmit = () => {
    if (!question.trim() || running) return;
    if (mode === "stream") runStream();
    else runGuided();
  };

  const citations = useMemo(() => report?.references ?? [], [report]);

  return (
    <div className="flex h-full flex-col">
      <header className="border-b border-zinc-800 px-6 py-4">
        <h1 className="text-xl font-semibold text-zinc-50">Research workspace</h1>
        <p className="text-sm text-zinc-500">Plan, collect evidence, critique, and synthesize a cited report.</p>
      </header>

      <div className="grid flex-1 grid-cols-1 gap-4 overflow-hidden p-4 xl:grid-cols-[320px_1fr_340px]">
        <section className="flex flex-col gap-4 overflow-y-auto rounded-xl border border-zinc-800 bg-zinc-900/30 p-4">
          <div>
            <label className="text-xs font-medium uppercase tracking-wide text-zinc-500">Research question</label>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              rows={5}
              placeholder="How should my personal research lab combine hybrid retrieval with agentic workflows?"
              className="mt-2 w-full resize-none rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 outline-none ring-cyan-500/0 focus:ring-2 focus:ring-cyan-500/40"
            />
          </div>

          <div>
            <label className="text-xs font-medium uppercase tracking-wide text-zinc-500">Research mode</label>
            <div className="mt-2 grid grid-cols-2 gap-2">
              {(["stream", "guided"] as ResearchMode[]).map((value) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setMode(value)}
                  className={cn(
                    "rounded-lg border px-3 py-2 text-left text-sm",
                    mode === value
                      ? "border-cyan-700 bg-cyan-950/40 text-cyan-100"
                      : "border-zinc-700 bg-zinc-950 text-zinc-400 hover:border-zinc-600",
                  )}
                >
                  <p className="font-medium capitalize">{value}</p>
                  <p className="mt-0.5 text-xs opacity-80">
                    {value === "stream" ? "Live workflow stream" : "Stepwise API pipeline"}
                  </p>
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="text-xs font-medium uppercase tracking-wide text-zinc-500">Literature sources</label>
            <div className="mt-2 flex flex-wrap gap-2">
              {sources.map((source) => {
                const on = selectedSources.includes(source);
                return (
                  <button
                    key={source}
                    type="button"
                    onClick={() =>
                      setSelectedSources((prev) =>
                        on ? prev.filter((s) => s !== source) : [...prev, source],
                      )
                    }
                    className={cn(
                      "rounded-md border px-2 py-1 text-xs capitalize",
                      on ? "border-cyan-700 bg-cyan-950/50 text-cyan-100" : "border-zinc-700 text-zinc-500",
                    )}
                  >
                    {source}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3 text-sm">
            <label className="flex flex-col gap-1 text-xs text-zinc-500">
              Evidence top-K
              <input
                type="number"
                min={1}
                max={20}
                value={evidenceTopK}
                onChange={(e) => setEvidenceTopK(Number(e.target.value))}
                className="rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-zinc-100"
              />
            </label>
            <label className="flex flex-col gap-1 text-xs text-zinc-500">
              Max iterations
              <input
                type="number"
                min={1}
                max={5}
                value={maxIterations}
                onChange={(e) => setMaxIterations(Number(e.target.value))}
                className="rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-zinc-100"
              />
            </label>
          </div>

          <label className="flex items-center gap-2 text-xs text-zinc-400">
            <input type="checkbox" checked={useHistory} onChange={(e) => setUseHistory(e.target.checked)} />
            Use prior research history as planner context
          </label>

          <button
            type="button"
            disabled={running || !question.trim()}
            onClick={onSubmit}
            className="rounded-lg bg-cyan-600 px-4 py-2.5 text-sm font-semibold text-zinc-950 enabled:hover:bg-cyan-500 disabled:opacity-40"
          >
            {running ? "Running research…" : "Run research"}
          </button>

          {error ? <p className="text-xs text-rose-400">{error}</p> : null}
        </section>

        <section className="flex min-h-0 flex-col gap-4 overflow-y-auto">
          <ProgressPipeline activeStage={activeStage} statusMessage={statusMessage} running={running} />

          {plan ? (
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/30 p-4 text-sm">
              <h3 className="font-semibold text-zinc-200">Research plan</h3>
              <p className="mt-2 text-zinc-400">{plan.research_objective}</p>
              <ul className="mt-2 list-disc space-y-1 pl-5 text-zinc-500">
                {plan.sub_questions.slice(0, 5).map((q) => (
                  <li key={q}>{q}</li>
                ))}
              </ul>
            </div>
          ) : null}

          <article className="rounded-xl border border-zinc-800 bg-zinc-900/30 p-5">
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <h2 className="text-lg font-semibold text-zinc-100">Final report</h2>
              {researchId ? <Badge tone="accent">ID {researchId.slice(0, 8)}</Badge> : null}
            </div>
            {report ? (
              <div className="space-y-4 text-sm leading-relaxed text-zinc-300">
                <p className="text-base text-zinc-100">{report.executive_summary}</p>
                <div>
                  <h3 className="mb-2 font-medium text-zinc-200">Key findings</h3>
                  <ul className="space-y-3">
                    {report.key_findings.map((finding) => (
                      <li key={finding.statement} className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3">
                        <p>{finding.statement}</p>
                        <div className="mt-2 flex flex-wrap gap-2">
                          <Badge tone={confidenceTone(finding.confidence)}>{finding.confidence} confidence</Badge>
                          <Badge tone="neutral">{finding.basis}</Badge>
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
                {report.open_questions.length ? (
                  <div>
                    <h3 className="mb-2 font-medium text-zinc-200">Open questions</h3>
                    <ul className="list-disc pl-5 text-zinc-500">
                      {report.open_questions.map((q) => (
                        <li key={q}>{q}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
                {researchId ? (
                  <button
                    type="button"
                    onClick={() => router.push(`/research/${researchId}`)}
                    className="text-sm text-cyan-400 hover:text-cyan-300"
                  >
                    Open saved research record →
                  </button>
                ) : null}
              </div>
            ) : (
              <p className="text-sm text-zinc-500">Run research to generate an evidence-backed report.</p>
            )}
          </article>

          {logLines.length ? (
            <div className="rounded-xl border border-zinc-800 bg-zinc-950/50 p-3 font-mono text-xs text-zinc-500">
              {logLines.map((line, i) => (
                <div key={`${line}-${i}`}>{line}</div>
              ))}
            </div>
          ) : null}
        </section>

        <section className="flex min-h-0 flex-col gap-4 overflow-y-auto">
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/30 p-4">
            <h3 className="text-sm font-semibold text-zinc-200">Evidence panel</h3>
            <div className="mt-3 space-y-3">
              {evidence.length ? (
                evidence.map((item) => (
                  <div key={item.evidence_id} className="rounded-lg border border-zinc-800 bg-zinc-950/50 p-3 text-xs">
                    <div className="flex items-center justify-between gap-2">
                      <p className="font-medium text-zinc-200">{item.title}</p>
                      <Badge tone="neutral">{item.retrieval_score.toFixed(2)}</Badge>
                    </div>
                    <p className="mt-2 line-clamp-4 text-zinc-500">{item.text}</p>
                  </div>
                ))
              ) : report?.evidence?.length ? (
                report.evidence.map((item) => (
                  <div key={item.evidence_id} className="rounded-lg border border-zinc-800 bg-zinc-950/50 p-3 text-xs">
                    <p className="font-medium text-zinc-200">
                      {item.citation_label} {item.title}
                    </p>
                    <p className="mt-2 line-clamp-4 text-zinc-500">{item.excerpt}</p>
                  </div>
                ))
              ) : (
                <p className="text-xs text-zinc-500">Evidence will appear after retrieval or from the final report.</p>
              )}
            </div>
          </div>

          <div className="rounded-xl border border-zinc-800 bg-zinc-900/30 p-4">
            <h3 className="text-sm font-semibold text-zinc-200">Citations</h3>
            <ul className="mt-3 space-y-2 text-xs text-zinc-400">
              {citations.length ? (
                citations.map((ref) => (
                  <li key={ref.reference_id} className="rounded-md border border-zinc-800 px-2 py-2">
                    <span className="text-cyan-400">{ref.citation_label ?? ref.reference_id}</span> {ref.title}
                  </li>
                ))
              ) : (
                <li className="text-zinc-500">No references yet.</li>
              )}
            </ul>
          </div>
        </section>
      </div>
    </div>
  );
}
