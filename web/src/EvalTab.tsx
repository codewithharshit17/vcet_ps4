import { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import {
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LineElement,
  LinearScale,
  PointElement,
  Tooltip,
} from "chart.js";
import { DASH, Panel } from "./components";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend
);

type EvalPayload = {
  available: boolean;
  reason?: string;
  generated_at?: string;
  model?: string;
  sweep?: {
    budget: number | null;
    compression_ratio: number | null;
    fact_retention: number | null;
    semantic_similarity: number | null;
    judge_support: number | null;
    ttft_delta_ms: number | null;
  }[];
  by_query?: {
    query: string;
    difficulty: string;
    budget: number | null;
    fact_retention: number | null;
    missing_facts: string[];
  }[];
};

export function EvalTab() {
  const [data, setData] = useState<EvalPayload | null>(null);

  useEffect(() => {
    fetch("/api/eval")
      .then((r) => r.json())
      .then(setData)
      .catch(() => setData({ available: false, reason: "Backend unreachable." }));
  }, []);

  if (!data) return <div className="text-[#6b7686] text-xs">Loading…</div>;

  if (!data.available) {
    return (
      <Panel title="quality vs compression">
        <p className="text-[#7d8899] text-[12px]">
          No eval results yet. {data.reason}
        </p>
        <p className="text-[#5c6675] text-[11px] mt-2">
          The curve deliberately renders nothing rather than a placeholder — an
          unmeasured quality number is worse than an absent one.
        </p>
      </Panel>
    );
  }

  const sweep = data.sweep ?? [];
  const labels = sweep.map((s) => (s.budget === null ? "∞" : String(s.budget)));

  return (
    <div className="space-y-3">
      <Panel
        title="quality vs compression — where does it fall off a cliff?"
        right={
          <span className="num text-[10px] text-[#5c6675]">
            {data.model} · {data.generated_at}
          </span>
        }
      >
        <div style={{ height: 300 }}>
          <Line
            data={{
              labels,
              datasets: [
                {
                  label: "fact retention",
                  data: sweep.map((s) => s.fact_retention),
                  borderColor: "#10b981",
                  backgroundColor: "#10b981",
                  tension: 0.25,
                },
                {
                  label: "semantic similarity",
                  data: sweep.map((s) => s.semantic_similarity),
                  borderColor: "#3b82f6",
                  backgroundColor: "#3b82f6",
                  tension: 0.25,
                },
                {
                  label: "judge: factual support",
                  data: sweep.map((s) =>
                    s.judge_support === null ? null : s.judge_support / 5
                  ),
                  borderColor: "#f59e0b",
                  backgroundColor: "#f59e0b",
                  tension: 0.25,
                },
                {
                  label: "compression ratio",
                  data: sweep.map((s) => s.compression_ratio),
                  borderColor: "#64748b",
                  borderDash: [4, 3],
                  backgroundColor: "#64748b",
                  tension: 0.25,
                },
              ],
            }}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              scales: {
                y: {
                  min: 0,
                  max: 1,
                  ticks: { color: "#6b7686", font: { size: 10 } },
                  grid: { color: "#161b24" },
                },
                x: {
                  ticks: { color: "#6b7686", font: { size: 10 } },
                  grid: { color: "#161b24" },
                  title: {
                    display: true,
                    text: "token budget",
                    color: "#6b7686",
                    font: { size: 10 },
                  },
                },
              },
              plugins: {
                legend: {
                  labels: { color: "#7d8899", boxWidth: 9, font: { size: 10 } },
                },
              },
            }}
          />
        </div>
      </Panel>

      <Panel title="gold set — per query">
        <div className="max-h-[380px] overflow-y-auto">
          <table className="w-full text-[11px]">
            <thead className="text-[#7d8899] text-left sticky top-0 bg-[#0d1017]">
              <tr>
                <th className="py-1 font-normal">query</th>
                <th className="py-1 font-normal">difficulty</th>
                <th className="py-1 font-normal text-right">budget</th>
                <th className="py-1 font-normal text-right">fact retention</th>
                <th className="py-1 font-normal">missing</th>
              </tr>
            </thead>
            <tbody>
              {(data.by_query ?? []).map((r, i) => (
                <tr key={i} className="border-t border-[#141922]">
                  <td className="py-1 pr-2">{r.query}</td>
                  <td className="py-1 text-[#7d8899]">{r.difficulty}</td>
                  <td className="py-1 text-right num">
                    {r.budget === null ? "∞" : r.budget}
                  </td>
                  <td
                    className={
                      "py-1 text-right num " +
                      (r.fact_retention === null
                        ? "text-[#4b5563]"
                        : r.fact_retention >= 0.9
                        ? "text-emerald-400"
                        : r.fact_retention >= 0.75
                        ? "text-amber-400"
                        : "text-red-400")
                    }
                  >
                    {r.fact_retention === null
                      ? DASH
                      : r.fact_retention.toFixed(2)}
                  </td>
                  <td className="py-1 text-[#6b7686]">
                    {r.missing_facts?.join("; ") || "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
