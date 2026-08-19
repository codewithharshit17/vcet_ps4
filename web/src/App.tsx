import { useCallback, useEffect, useRef, useState } from "react";
import {
  AnswerPane,
  ContextDiff,
  DASH,
  Panel,
  StatCard,
  Waterfall,
  fmt,
} from "./components";
import { EvalTab } from "./EvalTab";
import type { Compression, Config, PathState } from "./types";
import { emptyPath } from "./types";

const DEFAULT_QUERY =
  "What are the principal risk factors related to supply chain and manufacturing?";

/**
 * Fractional range inputs snap unpredictably: with step=0.05 the browser
 * validates against min + n*step in floating point, and a nominal 0.7 was
 * silently landing on 0.8 -- so the dashboard ran a different lambda than it
 * displayed, and than config.py declared. The slider therefore works in
 * integer units internally and only converts at the edges.
 */
function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  scale = 1,
  fmtVal,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  scale?: number;
  fmtVal?: (v: number) => string;
}) {
  return (
    <label className="flex flex-col gap-1 min-w-[132px]">
      <span className="text-[10px] uppercase tracking-wider text-[#7d8899] flex justify-between">
        {label}
        <span className="num text-[#d7dce3]">
          {fmtVal ? fmtVal(value) : value}
        </span>
      </span>
      <input
        type="range"
        min={Math.round(min * scale)}
        max={Math.round(max * scale)}
        step={Math.round(step * scale)}
        value={Math.round(value * scale)}
        onChange={(e) => onChange(parseInt(e.target.value, 10) / scale)}
        className="w-full"
      />
    </label>
  );
}

export default function App() {
  const [cfg, setCfg] = useState<Config | null>(null);
  const [tab, setTab] = useState<"live" | "eval">("live");

  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [budget, setBudget] = useState(800);
  const [lambda, setLambda] = useState(0.7);
  const [threshold, setThreshold] = useState(0);
  const [topK, setTopK] = useState(20);
  const [strip, setStrip] = useState(false);

  const [comp, setComp] = useState<Compression | null>(null);
  const [baseline, setBaseline] = useState<PathState>(emptyPath());
  const [compressed, setCompressed] = useState<PathState>(emptyPath());
  const [running, setRunning] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    fetch("/api/config")
      .then((r) => r.json())
      .then((c: Config) => {
        setCfg(c);
        setBudget(c.token_budget);
        setLambda(c.mmr_lambda);
        setThreshold(c.relevance_threshold);
        setTopK(c.top_k);
        setStrip(c.aggressive_strip);
      })
      .catch(() => setNotice("Backend unreachable. Start it with `make api`."));
  }, []);

  const run = useCallback(() => {
    esRef.current?.close();
    setComp(null);
    setBaseline(emptyPath());
    setCompressed(emptyPath());
    setNotice(null);
    setRunning(true);

    const qs = new URLSearchParams({
      query,
      budget: String(budget),
      mmr_lambda: String(lambda),
      threshold: String(threshold),
      top_k: String(topK),
      strip: String(strip),
    });
    const es = new EventSource(`/api/run?${qs}`);
    esRef.current = es;

    const patch = (path: string, p: Partial<PathState>) =>
      (path === "baseline" ? setBaseline : setCompressed)((s) => ({ ...s, ...p }));

    es.addEventListener("compression", (e) =>
      setComp(JSON.parse((e as MessageEvent).data))
    );
    es.addEventListener("ttft", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      patch(d.path, { ttft_ms: d.ttft_ms });
    });
    es.addEventListener("delta", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      (d.path === "baseline" ? setBaseline : setCompressed)((s) => ({
        ...s,
        answer: s.answer + d.text,
      }));
    });
    es.addEventListener("path_done", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      patch(d.path, { ...d, done: true });
    });
    es.addEventListener("path_error", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      patch(d.path, { error: d.error, done: true });
    });
    es.addEventListener("llm_unavailable", (e) => {
      setNotice(JSON.parse((e as MessageEvent).data).reason);
    });
    es.addEventListener("done", () => {
      setRunning(false);
      es.close();
    });
    es.onerror = () => {
      setRunning(false);
      es.close();
    };
  }, [query, budget, lambda, threshold, topK, strip]);

  // Hero metrics. Everything derives from measured values or stays an em dash.
  const ratio = comp?.compression_ratio ?? null;
  const ttftDelta =
    baseline.ttft_ms !== null && compressed.ttft_ms !== null
      ? baseline.ttft_ms - compressed.ttft_ms
      : null;
  const costOf = (p: PathState) =>
    cfg && p.prompt_tokens !== null && p.output_tokens !== null
      ? (p.prompt_tokens * cfg.price_in_per_mtok +
          p.output_tokens * cfg.price_out_per_mtok) /
        1e6
      : null;
  const bCost = costOf(baseline);
  const cCost = costOf(compressed);

  return (
    <div className="min-h-full p-4 max-w-[1600px] mx-auto">
      <header className="flex items-baseline gap-3 mb-3">
        <h1 className="text-[15px] font-semibold tracking-tight">
          Token-Diet
          <span className="text-[#5c6675] font-normal ml-2">
            post-retrieval context compressor
          </span>
        </h1>
        <nav className="ml-auto flex gap-1">
          {(["live", "eval"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={
                "px-3 py-1 rounded text-[11px] uppercase tracking-wider " +
                (tab === t
                  ? "bg-[#1b212c] text-[#d7dce3]"
                  : "text-[#6b7686] hover:text-[#9aa4b2]")
              }
            >
              {t}
            </button>
          ))}
        </nav>
      </header>

      {cfg && (
        <div className="num text-[10px] text-[#5c6675] mb-3">
          {cfg.answer_model} · ${cfg.price_in_per_mtok}/${cfg.price_out_per_mtok}
          per Mtok · warm-up{" "}
          {Object.entries(cfg.warmup_ms)
            .map(([k, v]) => `${k.replace("_ms", "")} ${v.toFixed(0)}ms`)
            .join(" · ") || "pending"}{" "}
          (discarded)
          {!cfg.llm_configured && (
            <span className="text-amber-400 ml-2">
              · GROQ_API_KEY not set — LLM metrics unmeasured
            </span>
          )}
        </div>
      )}

      {tab === "eval" ? (
        <EvalTab />
      ) : (
        <>
          <div className="flex gap-2 mb-3">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !running && run()}
              placeholder="Ask something about the indexed filings…"
              className="flex-1 bg-[#0d1017] border border-[#1b212c] rounded px-3 py-2 text-[13px] outline-none focus:border-[#2f3a4c]"
            />
            <button
              onClick={run}
              disabled={running}
              className="px-5 rounded bg-[#1d4ed8] hover:bg-[#1e40af] disabled:opacity-40 text-white text-[12px] font-medium"
            >
              {running ? "running…" : "Run A/B"}
            </button>
          </div>

          <div className="flex flex-wrap gap-5 items-end mb-3 px-3 py-2.5 border border-[#1b212c] bg-[#0d1017] rounded">
            <Slider label="token budget" value={budget} min={100} max={4000} step={50} onChange={setBudget} />
            <Slider label="MMR λ" value={lambda} min={0} max={1} step={0.05} scale={100} onChange={setLambda} fmtVal={(v) => v.toFixed(2)} />
            <Slider label="relevance min" value={threshold} min={0} max={1} step={0.01} scale={100} onChange={setThreshold} fmtVal={(v) => v.toFixed(2)} />
            <Slider label="top-K" value={topK} min={5} max={50} step={1} onChange={setTopK} />
            <label className="flex items-center gap-2 text-[11px] text-[#9aa4b2]">
              <input type="checkbox" checked={strip} onChange={(e) => setStrip(e.target.checked)} />
              aggressive strip
              <span className="text-[#5c6675]">(off by default)</span>
            </label>
            <span className="text-[10px] text-[#5c6675] ml-auto">
              re-runs on demand; scores are cached so sliders are cheap
            </span>
          </div>

          {notice && (
            <div className="mb-3 px-3 py-2 rounded border border-amber-900/60 bg-amber-950/30 text-amber-300 text-[11px]">
              {notice}
            </div>
          )}

          <div className="grid grid-cols-4 gap-3 mb-3">
            <StatCard
              label="compression ratio"
              value={ratio === null ? DASH : `${(ratio * 100).toFixed(1)}%`}
              sub={
                comp
                  ? `${comp.tokens_saved.toLocaleString()} tokens saved`
                  : DASH
              }
              tone={ratio === null ? "neutral" : "good"}
            />
            <StatCard
              label="TTFT drop"
              value={ttftDelta === null ? DASH : `${ttftDelta > 0 ? "" : "+"}${(-ttftDelta).toFixed(0)}ms`}
              sub={`${fmt(baseline.ttft_ms, 0, "ms")} → ${fmt(compressed.ttft_ms, 0, "ms")}`}
              tone={ttftDelta === null ? "neutral" : ttftDelta > 0 ? "good" : "bad"}
            />
            <StatCard
              label="cost / query"
              value={cCost === null ? DASH : `$${cCost.toFixed(6)}`}
              sub={bCost === null ? DASH : `baseline $${bCost.toFixed(6)}`}
              tone={cCost === null ? "neutral" : "good"}
            />
            <StatCard
              label="answer quality"
              value={DASH}
              sub="retention — run eval harness"
              tone="neutral"
            />
          </div>

          <div className="grid grid-cols-2 gap-3 mb-3">
            <AnswerPane label="baseline · uncompressed" path={baseline} tone="text-[#94a3b8]" />
            <AnswerPane label="compressed" path={compressed} tone="text-emerald-400" />
          </div>

          <div className="grid grid-cols-[1fr_420px] gap-3">
            <Panel title="context diff — kept vs dropped (hover for reason)">
              <ContextDiff comp={comp} />
            </Panel>
            <div className="space-y-3">
              <Panel title="latency waterfall">
                <Waterfall comp={comp} baseline={baseline} compressed={compressed} />
              </Panel>
              <Panel title="stage timings">
                <table className="w-full num text-[11px]">
                  <tbody>
                    {comp &&
                      Object.entries(comp.stages).map(([k, v]) => (
                        <tr key={k} className="border-b border-[#141922] last:border-0">
                          <td className="py-0.5 text-[#7d8899]">{k}</td>
                          <td className="py-0.5 text-right">
                            {v === null ? (
                              <span className="text-[#4b5563]">{DASH}</span>
                            ) : (
                              `${v.toFixed(1)} ms`
                            )}
                          </td>
                        </tr>
                      ))}
                    {comp && (
                      <tr className="border-t border-[#1b212c]">
                        <td className="py-1 text-[#9aa4b2]">pipeline overhead</td>
                        <td className="py-1 text-right text-[#d7dce3]">
                          {comp.pipeline_overhead_ms.toFixed(1)} ms
                        </td>
                      </tr>
                    )}
                    {!comp && (
                      <tr>
                        <td className="text-[#4b5563]">no run yet</td>
                      </tr>
                    )}
                  </tbody>
                </table>
                {comp && (
                  <div className="text-[10px] text-[#5c6675] mt-2 num">
                    cache — ce {comp.cache_stats.ce_hits ?? 0}h/
                    {comp.cache_stats.ce_misses ?? 0}m · emb{" "}
                    {comp.cache_stats.emb_hits ?? 0}h/
                    {comp.cache_stats.emb_misses ?? 0}m
                  </div>
                )}
              </Panel>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
