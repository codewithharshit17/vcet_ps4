import { Bar } from "react-chartjs-2";
import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  Tooltip,
} from "chart.js";
import type { Compression, PathState, Sentence } from "./types";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

/** The single rule of this UI: never render a number we did not measure. */
export const DASH = "—";

export const fmt = (v: number | null | undefined, digits = 0, suffix = ""): string =>
  v === null || v === undefined || Number.isNaN(v)
    ? DASH
    : `${v.toLocaleString(undefined, {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      })}${suffix}`;

export function Panel({
  title,
  right,
  children,
  className = "",
}: {
  title: string;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`border border-[#1b212c] bg-[#0d1017] rounded ${className}`}
    >
      <header className="flex items-center justify-between px-3 py-1.5 border-b border-[#1b212c]">
        <h2 className="text-[11px] uppercase tracking-wider text-[#7d8899]">
          {title}
        </h2>
        {right}
      </header>
      <div className="p-3">{children}</div>
    </section>
  );
}

export function StatCard({
  label,
  value,
  sub,
  tone = "neutral",
}: {
  label: string;
  value: string;
  sub: React.ReactNode;
  tone?: "good" | "warn" | "bad" | "neutral";
}) {
  const colour = {
    good: "text-emerald-400",
    warn: "text-amber-400",
    bad: "text-red-400",
    neutral: "text-[#d7dce3]",
  }[tone];
  return (
    <div className="border border-[#1b212c] bg-[#0d1017] rounded px-3 py-2.5">
      <div className="text-[10px] uppercase tracking-wider text-[#7d8899]">
        {label}
      </div>
      <div className={`num text-2xl font-semibold mt-1 ${colour}`}>{value}</div>
      <div className="text-[11px] text-[#6b7686] mt-0.5 num">{sub}</div>
    </div>
  );
}

const STAGE_COLOURS: Record<string, string> = {
  retrieval_ms: "#3b82f6",
  sentence_split_ms: "#8b5cf6",
  rerank_ms: "#ec4899",
  mmr_ms: "#f59e0b",
  select_ms: "#10b981",
  strip_ms: "#14b8a6",
  assemble_ms: "#64748b",
  llm_wait_ms: "#475569",
};

/**
 * Stacked horizontal bars, baseline vs compressed, segmented by stage.
 * This is the chart that makes the overhead trade-off visible instead of
 * hidden -- if compression loses, you see exactly which stage spent the time.
 */
export function Waterfall({
  comp,
  baseline,
  compressed,
}: {
  comp: Compression | null;
  baseline: PathState;
  compressed: PathState;
}) {
  if (!comp) return <div className="text-[#6b7686] text-xs">No run yet.</div>;

  const stageKeys = [
    "retrieval_ms",
    "sentence_split_ms",
    "rerank_ms",
    "mmr_ms",
    "select_ms",
    "strip_ms",
    "assemble_ms",
  ];

  // Compressed pays every pipeline stage; baseline pays retrieval only.
  const compStages = stageKeys.map((k) => comp.stages[k] ?? 0);
  const baseStages = stageKeys.map((k) =>
    k === "retrieval_ms" ? comp.stages[k] ?? 0 : 0
  );

  // Remaining time to first token is the LLM's own latency.
  const compOverhead = compStages.reduce((a, b) => a + b, 0);
  const baseOverhead = baseStages.reduce((a, b) => a + b, 0);
  const compLlm =
    compressed.ttft_ms === null ? null : Math.max(0, compressed.ttft_ms - compOverhead);
  const baseLlm =
    baseline.ttft_ms === null ? null : Math.max(0, baseline.ttft_ms - baseOverhead);

  const datasets = [
    ...stageKeys.map((k, i) => ({
      label: k.replace(/_ms$/, ""),
      data: [baseStages[i], compStages[i]],
      backgroundColor: STAGE_COLOURS[k],
      borderWidth: 0,
    })),
    {
      label: "llm wait",
      data: [baseLlm ?? 0, compLlm ?? 0],
      backgroundColor: STAGE_COLOURS.llm_wait_ms,
      borderWidth: 0,
    },
  ];

  return (
    <div>
      <div style={{ height: 130 }}>
        <Bar
          data={{ labels: ["baseline", "compressed"], datasets }}
          options={{
            indexAxis: "y" as const,
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: {
                stacked: true,
                ticks: { color: "#6b7686", font: { size: 10 } },
                grid: { color: "#161b24" },
                title: {
                  display: true,
                  text: "ms to first token",
                  color: "#6b7686",
                  font: { size: 10 },
                },
              },
              y: {
                stacked: true,
                ticks: { color: "#9aa4b2", font: { size: 11 } },
                grid: { display: false },
              },
            },
            plugins: {
              legend: {
                labels: { color: "#7d8899", boxWidth: 9, font: { size: 10 } },
                position: "bottom" as const,
              },
              tooltip: {
                callbacks: {
                  label: (c) => `${c.dataset.label}: ${(c.raw as number).toFixed(1)} ms`,
                },
              },
            },
          }}
        />
      </div>
      {baseline.ttft_ms === null && (
        <p className="text-[11px] text-[#6b7686] mt-2">
          LLM segment unmeasured — showing pipeline stages only.
        </p>
      )}
    </div>
  );
}

function tint(rel: number): string {
  // Opacity encodes relevance so the eye can rank sentences at a glance.
  const a = 0.10 + Math.min(1, Math.max(0, rel)) * 0.42;
  return `rgba(16,185,129,${a.toFixed(3)})`;
}

export function ContextDiff({ comp }: { comp: Compression | null }) {
  if (!comp) return <div className="text-[#6b7686] text-xs">No run yet.</div>;

  const kept = comp.sentences.filter((s) => s.selected);
  const byDoc = new Map<string, Sentence[]>();
  for (const s of comp.sentences) {
    if (!byDoc.has(s.doc_id)) byDoc.set(s.doc_id, []);
    byDoc.get(s.doc_id)!.push(s);
  }

  return (
    <div className="grid grid-cols-2 gap-3">
      <div className="flex items-center justify-between col-span-2 text-[11px]">
        <span className="text-[#7d8899]">
          All {comp.sentences.length} candidate sentences · kept{" "}
          <span className="text-emerald-400 num">{kept.length}</span>
        </span>
        <span className="num text-[#7d8899]">
          baseline{" "}
          <span className="text-[#d7dce3]">
            {comp.baseline_context_tokens.toLocaleString()}
          </span>{" "}
          tok → compressed{" "}
          <span className="text-emerald-400">
            {comp.compressed_context_tokens.toLocaleString()}
          </span>{" "}
          tok
        </span>
      </div>

      <div className="col-span-2 max-h-[460px] overflow-y-auto pr-1 space-y-3">
        {[...byDoc.entries()].map(([doc, sents]) => (
          <div key={doc}>
            <div className="text-[10px] uppercase tracking-wider text-[#5c6675] sticky top-0 bg-[#0d1017] py-1">
              {doc}
            </div>
            <div className="space-y-0.5">
              {sents.map((s) => (
                <div
                  key={s.sid}
                  title={
                    s.selected
                      ? `KEPT · relevance ${s.relevance.toFixed(3)} · ce ${s.ce_norm.toFixed(
                          2
                        )} · bm25 ${s.bm25_norm.toFixed(2)} · ${s.tokens} tok`
                      : `DROPPED · ${s.drop_reason ?? "unknown"} · relevance ${s.relevance.toFixed(
                          3
                        )} · ${s.tokens} tok`
                  }
                  className={
                    "px-2 py-1 rounded text-[12px] leading-snug cursor-help " +
                    (s.selected
                      ? "text-[#e6ebf2]"
                      : "text-[#4b5563] line-through decoration-[#374151]")
                  }
                  style={
                    s.selected ? { background: tint(s.relevance) } : undefined
                  }
                >
                  {s.kind !== "prose" && (
                    <span className="text-[9px] uppercase mr-1.5 px-1 rounded bg-[#1f2937] text-[#93a3b8] no-underline">
                      {s.kind}
                    </span>
                  )}
                  {s.text.length > 260 ? s.text.slice(0, 260) + "…" : s.text}
                  <span className="num text-[10px] text-[#5c6675] ml-2">
                    {s.tokens}t
                  </span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function AnswerPane({
  label,
  path,
  tone,
}: {
  label: string;
  path: PathState;
  tone: string;
}) {
  return (
    <div className="border border-[#1b212c] bg-[#0d1017] rounded flex flex-col min-h-[220px]">
      <header className="flex items-center justify-between px-3 py-1.5 border-b border-[#1b212c]">
        <span className={`text-[11px] uppercase tracking-wider ${tone}`}>
          {label}
        </span>
        <span className="num text-[10px] text-[#6b7686]">
          ttft {fmt(path.ttft_ms, 0, "ms")} · in {fmt(path.prompt_tokens)} · out{" "}
          {fmt(path.output_tokens)}
          {path.contaminated && (
            <span className="text-amber-400 ml-2">cache-contaminated</span>
          )}
        </span>
      </header>
      <div className="p-3 text-[12.5px] leading-relaxed whitespace-pre-wrap overflow-y-auto max-h-[300px]">
        {path.error ? (
          <span className="text-red-400">{path.error}</span>
        ) : path.answer ? (
          path.answer
        ) : (
          <span className="text-[#4b5563]">—</span>
        )}
      </div>
    </div>
  );
}
