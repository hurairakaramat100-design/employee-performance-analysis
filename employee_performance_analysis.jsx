import { useState, useEffect, useCallback } from "react";

// ── Palette ──────────────────────────────────────────────────────────────────
// Deep navy canvas  #0D1B2A
// Steel blue mid    #1E3A5F
// Electric teal     #00C2B2  (accent / high scores)
// Amber warning     #F5A623  (mid scores)
// Coral low         #E8523A  (low scores)
// Light slate       #A8B8CC
// Near-white        #EDF2F7

const PALETTE = {
  bg:        "#0D1B2A",
  mid:       "#1E3A5F",
  teal:      "#00C2B2",
  amber:     "#F5A623",
  coral:     "#E8523A",
  slate:     "#A8B8CC",
  light:     "#EDF2F7",
};

// ── ML Model (linear regression simulation) ──────────────────────────────────
function predict({ experience, workingHours, trainingHours, projectsCompleted }) {
  // Weights derived from a simulated trained model
  const w = {
    experience:        3.8,
    workingHours:      2.1,
    trainingHours:     1.9,
    projectsCompleted: 4.2,
    bias:             20.0,
  };
  const raw =
    w.experience        * experience +
    w.workingHours      * workingHours +
    w.trainingHours     * trainingHours +
    w.projectsCompleted * projectsCompleted +
    w.bias;
  return Math.min(100, Math.max(0, Math.round(raw * 10) / 10));
}

// Batch training: generate synthetic employees
function generateDataset(n = 120) {
  const rng = (min, max) => +(Math.random() * (max - min) + min).toFixed(1);
  return Array.from({ length: n }, (_, i) => {
    const exp   = rng(0, 15);
    const wh    = rng(4, 12);
    const th    = rng(0, 10);
    const proj  = rng(0, 10);
    return {
      id: i + 1,
      name: `EMP-${String(i + 1).padStart(3, "0")}`,
      experience: exp,
      workingHours: wh,
      trainingHours: th,
      projectsCompleted: proj,
      score: predict({ experience: exp, workingHours: wh, trainingHours: th, projectsCompleted: proj }),
    };
  });
}

const DATASET = generateDataset();

// ── Feature Importance (SHAP-style normalized values) ────────────────────────
const FEATURE_WEIGHTS = [
  { label: "Projects Completed", value: 0.38, color: PALETTE.teal },
  { label: "Experience (yrs)",   value: 0.28, color: "#6D8DD6" },
  { label: "Working Hours/day",  value: 0.19, color: PALETTE.amber },
  { label: "Training Hours",     value: 0.15, color: "#9B6DD6" },
];

// ── Helpers ───────────────────────────────────────────────────────────────────
function scoreColor(s) {
  if (s >= 75) return PALETTE.teal;
  if (s >= 50) return PALETTE.amber;
  return PALETTE.coral;
}
function scoreLabel(s) {
  if (s >= 75) return "High";
  if (s >= 50) return "Medium";
  return "Low";
}
function avg(arr, key) {
  return +(arr.reduce((a, b) => a + b[key], 0) / arr.length).toFixed(1);
}

// ── Sub-components ─────────────────────────────────────────────────────────────
function Slider({ label, min, max, step = 0.5, value, onChange, unit = "" }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div style={{ marginBottom: 18 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
        <span style={{ color: PALETTE.slate, fontSize: 13, letterSpacing: "0.04em", textTransform: "uppercase" }}>
          {label}
        </span>
        <span style={{ color: PALETTE.teal, fontWeight: 700, fontSize: 15 }}>
          {value}{unit}
        </span>
      </div>
      <div style={{ position: "relative", height: 6, borderRadius: 3, background: "#1a3050" }}>
        <div style={{
          position: "absolute", left: 0, top: 0, height: "100%",
          width: `${pct}%`, borderRadius: 3,
          background: `linear-gradient(90deg, ${PALETTE.teal}, #00e5d3)`,
          transition: "width 0.15s",
        }} />
        <input
          type="range" min={min} max={max} step={step} value={value}
          onChange={e => onChange(+e.target.value)}
          style={{
            position: "absolute", top: -8, left: 0, width: "100%",
            height: 22, opacity: 0, cursor: "pointer", zIndex: 2,
          }}
        />
      </div>
    </div>
  );
}

function GaugeDial({ score }) {
  const color = scoreColor(score);
  const r = 64, cx = 80, cy = 80;
  const circumference = Math.PI * r; // half circle
  const dash = (score / 100) * circumference;

  return (
    <svg viewBox="0 0 160 100" style={{ width: "100%", maxWidth: 200, display: "block", margin: "0 auto" }}>
      {/* Track */}
      <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
        fill="none" stroke="#1a3050" strokeWidth={12} strokeLinecap="round" />
      {/* Value */}
      <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
        fill="none" stroke={color} strokeWidth={12} strokeLinecap="round"
        strokeDasharray={`${dash} ${circumference}`}
        style={{ transition: "stroke-dasharray 0.6s cubic-bezier(.4,0,.2,1), stroke 0.4s" }}
      />
      {/* Score text */}
      <text x={cx} y={cy - 8} textAnchor="middle"
        fontSize={28} fontWeight="800" fill={color}
        style={{ transition: "fill 0.4s" }}>
        {score}
      </text>
      <text x={cx} y={cy + 10} textAnchor="middle"
        fontSize={11} fill={PALETTE.slate} letterSpacing="2">
        SCORE
      </text>
    </svg>
  );
}

function FeatureBar({ label, value, color }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ color: PALETTE.slate, fontSize: 12 }}>{label}</span>
        <span style={{ color, fontWeight: 700, fontSize: 12 }}>{(value * 100).toFixed(0)}%</span>
      </div>
      <div style={{ height: 5, borderRadius: 3, background: "#1a3050", overflow: "hidden" }}>
        <div style={{
          height: "100%", width: `${value * 100}%`, borderRadius: 3, background: color,
          transition: "width 0.8s cubic-bezier(.4,0,.2,1)",
        }} />
      </div>
    </div>
  );
}

function HistogramBar({ label, count, maxCount, color }) {
  const h = Math.max(4, (count / maxCount) * 100);
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", flex: 1 }}>
      <span style={{ color: PALETTE.slate, fontSize: 10, marginBottom: 4 }}>{count}</span>
      <div style={{ width: "70%", height: `${h}px`, background: color, borderRadius: "3px 3px 0 0", transition: "height 0.6s" }} />
      <span style={{ color: PALETTE.slate, fontSize: 9, marginTop: 4 }}>{label}</span>
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState("predict");
  const [exp, setExp]   = useState(5);
  const [wh,  setWh]    = useState(8);
  const [th,  setTh]    = useState(3);
  const [proj, setProj] = useState(4);
  const [searching, setSearching] = useState("");
  const [sort, setSort] = useState({ key: "score", dir: -1 });
  const [animScore, setAnimScore] = useState(0);

  const score = predict({ experience: exp, workingHours: wh, trainingHours: th, projectsCompleted: proj });

  // Animate gauge
  useEffect(() => {
    let frame, current = animScore;
    const step = () => {
      const diff = score - current;
      if (Math.abs(diff) < 0.5) { setAnimScore(score); return; }
      current += diff * 0.12;
      setAnimScore(Math.round(current * 10) / 10);
      frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [score]);

  // Stats
  const high   = DATASET.filter(d => d.score >= 75).length;
  const medium = DATASET.filter(d => d.score >= 50 && d.score < 75).length;
  const low    = DATASET.filter(d => d.score < 50).length;

  // Histogram bins
  const bins = [
    { label: "0-20",  color: PALETTE.coral },
    { label: "21-40", color: "#E87840" },
    { label: "41-60", color: PALETTE.amber },
    { label: "61-80", color: "#6DB8E8" },
    { label: "81-100",color: PALETTE.teal },
  ];
  const binCounts = bins.map((b, i) => {
    const lo = i * 20, hi = lo + 20;
    return { ...b, count: DATASET.filter(d => d.score > lo && d.score <= hi + (i === 4 ? 1 : 0)).length };
  });
  const maxBin = Math.max(...binCounts.map(b => b.count));

  // Table
  const filtered = DATASET.filter(d =>
    d.name.includes(searching.toUpperCase()) || searching === ""
  ).sort((a, b) => sort.dir * (a[sort.key] - b[sort.key]));

  const toggleSort = key => setSort(s => ({ key, dir: s.key === key ? -s.dir : -1 }));

  // Styles
  const card = {
    background: PALETTE.mid,
    borderRadius: 12,
    padding: "20px 22px",
    border: "1px solid #234",
  };

  const tabStyle = active => ({
    padding: "8px 20px",
    borderRadius: 8,
    border: "none",
    cursor: "pointer",
    fontWeight: 700,
    fontSize: 13,
    letterSpacing: "0.05em",
    textTransform: "uppercase",
    background: active ? PALETTE.teal : "transparent",
    color: active ? "#0D1B2A" : PALETTE.slate,
    transition: "all 0.2s",
  });

  return (
    <div style={{
      minHeight: "100vh",
      background: PALETTE.bg,
      fontFamily: "'Inter', 'Segoe UI', sans-serif",
      color: PALETTE.light,
      padding: "28px 20px",
      boxSizing: "border-box",
    }}>
      {/* Header */}
      <div style={{ maxWidth: 900, margin: "0 auto" }}>
        <div style={{ marginBottom: 28 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
            <div style={{
              width: 36, height: 36, borderRadius: 8,
              background: `linear-gradient(135deg, ${PALETTE.teal}, #6D8DD6)`,
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 18,
            }}>📊</div>
            <span style={{ color: PALETTE.slate, fontSize: 12, letterSpacing: "0.15em", textTransform: "uppercase" }}>
              ML Analytics · Employee Intelligence
            </span>
          </div>
          <h1 style={{
            fontSize: "clamp(22px, 4vw, 32px)", fontWeight: 900,
            margin: 0, lineHeight: 1.15,
            background: `linear-gradient(90deg, ${PALETTE.light}, ${PALETTE.teal})`,
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          }}>
            Performance Analysis
          </h1>
          <p style={{ color: PALETTE.slate, fontSize: 14, margin: "6px 0 0", lineHeight: 1.5 }}>
            Predict & understand what drives employee performance scores
          </p>
        </div>

        {/* KPI strip */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginBottom: 24 }}>
          {[
            { label: "High Performers", value: high, color: PALETTE.teal, sub: "≥ 75 score" },
            { label: "Mid Performers",  value: medium, color: PALETTE.amber, sub: "50–74 score" },
            { label: "Low Performers",  value: low, color: PALETTE.coral, sub: "< 50 score" },
          ].map(k => (
            <div key={k.label} style={{ ...card, textAlign: "center" }}>
              <div style={{ fontSize: "clamp(22px,5vw,36px)", fontWeight: 900, color: k.color }}>{k.value}</div>
              <div style={{ fontSize: 11, fontWeight: 700, color: PALETTE.light, marginBottom: 2 }}>{k.label}</div>
              <div style={{ fontSize: 10, color: PALETTE.slate }}>{k.sub}</div>
            </div>
          ))}
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 6, marginBottom: 20, background: "#0a1520", borderRadius: 10, padding: 4, width: "fit-content" }}>
          {[["predict", "🎯 Predict"], ["explore", "📈 Explore"], ["dataset", "🗃 Dataset"]].map(([key, label]) => (
            <button key={key} onClick={() => setTab(key)} style={tabStyle(tab === key)}>{label}</button>
          ))}
        </div>

        {/* ── PREDICT TAB ─────────────────────────────────────────── */}
        {tab === "predict" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            {/* Controls */}
            <div style={card}>
              <div style={{ color: PALETTE.slate, fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 18 }}>
                Input Features
              </div>
              <Slider label="Experience" min={0} max={15} step={0.5} value={exp} onChange={setExp} unit=" yrs" />
              <Slider label="Working Hours / Day" min={4} max={12} step={0.5} value={wh} onChange={setWh} unit=" hrs" />
              <Slider label="Training Hours" min={0} max={10} step={0.5} value={th} onChange={setTh} unit=" hrs" />
              <Slider label="Projects Completed" min={0} max={10} step={0.5} value={proj} onChange={setProj} />
            </div>

            {/* Output */}
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              {/* Gauge */}
              <div style={{ ...card, textAlign: "center" }}>
                <div style={{ color: PALETTE.slate, fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 12 }}>
                  Performance Score
                </div>
                <GaugeDial score={animScore} />
                <div style={{
                  display: "inline-block", marginTop: 10,
                  padding: "4px 16px", borderRadius: 20,
                  background: scoreColor(score) + "22",
                  border: `1px solid ${scoreColor(score)}55`,
                  color: scoreColor(score), fontWeight: 700, fontSize: 13,
                }}>
                  {scoreLabel(score)} Performance
                </div>
              </div>

              {/* Feature importance */}
              <div style={card}>
                <div style={{ color: PALETTE.slate, fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 14 }}>
                  Feature Importance
                </div>
                {FEATURE_WEIGHTS.map(f => (
                  <FeatureBar key={f.label} label={f.label} value={f.value} color={f.color} />
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ── EXPLORE TAB ─────────────────────────────────────────── */}
        {tab === "explore" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            {/* Distribution histogram */}
            <div style={card}>
              <div style={{ color: PALETTE.slate, fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 16 }}>
                Score Distribution (n={DATASET.length})
              </div>
              <div style={{ display: "flex", alignItems: "flex-end", height: 110, gap: 6 }}>
                {binCounts.map(b => (
                  <HistogramBar key={b.label} label={b.label} count={b.count} maxCount={maxBin} color={b.color} />
                ))}
              </div>
              <div style={{ borderTop: "1px solid #234", marginTop: 14, paddingTop: 12 }}>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
                  {[
                    ["Avg Score", `${avg(DATASET, "score")}`],
                    ["Avg Experience", `${avg(DATASET, "experience")} yrs`],
                    ["Avg Work Hrs", `${avg(DATASET, "workingHours")} /day`],
                    ["Avg Projects", avg(DATASET, "projectsCompleted")],
                  ].map(([k, v]) => (
                    <div key={k} style={{ background: "#0a1520", borderRadius: 8, padding: "8px 10px" }}>
                      <div style={{ color: PALETTE.slate, fontSize: 10 }}>{k}</div>
                      <div style={{ color: PALETTE.light, fontWeight: 700, fontSize: 15 }}>{v}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Scatter-like breakdown */}
            <div style={card}>
              <div style={{ color: PALETTE.slate, fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 16 }}>
                Score by Experience Bracket
              </div>
              {[
                { label: "0–5 yrs",  filter: d => d.experience <= 5 },
                { label: "5–10 yrs", filter: d => d.experience > 5 && d.experience <= 10 },
                { label: "10+ yrs",  filter: d => d.experience > 10 },
              ].map(bracket => {
                const group = DATASET.filter(bracket.filter);
                const mean = avg(group, "score");
                const color = scoreColor(mean);
                return (
                  <div key={bracket.label} style={{ marginBottom: 14 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                      <span style={{ color: PALETTE.slate, fontSize: 12 }}>{bracket.label}</span>
                      <span style={{ color, fontWeight: 700, fontSize: 12 }}>{mean} avg · {group.length} employees</span>
                    </div>
                    <div style={{ height: 8, borderRadius: 4, background: "#1a3050", overflow: "hidden" }}>
                      <div style={{ height: "100%", width: `${mean}%`, background: color, borderRadius: 4, transition: "width 0.8s" }} />
                    </div>
                  </div>
                );
              })}

              <div style={{ borderTop: "1px solid #234", marginTop: 16, paddingTop: 14 }}>
                <div style={{ color: PALETTE.slate, fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 12 }}>
                  Performance Tier Breakdown
                </div>
                {[
                  { label: "High (≥75)", count: high, color: PALETTE.teal },
                  { label: "Medium (50–74)", count: medium, color: PALETTE.amber },
                  { label: "Low (<50)", count: low, color: PALETTE.coral },
                ].map(t => (
                  <div key={t.label} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                    <div style={{ width: 10, height: 10, borderRadius: "50%", background: t.color, flexShrink: 0 }} />
                    <div style={{ flex: 1, height: 6, background: "#1a3050", borderRadius: 3, overflow: "hidden" }}>
                      <div style={{ height: "100%", width: `${(t.count / DATASET.length) * 100}%`, background: t.color, transition: "width 0.8s" }} />
                    </div>
                    <span style={{ color: PALETTE.slate, fontSize: 11, minWidth: 90 }}>
                      {t.label} · {t.count}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ── DATASET TAB ─────────────────────────────────────────── */}
        {tab === "dataset" && (
          <div style={card}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16, flexWrap: "wrap", gap: 10 }}>
              <div style={{ color: PALETTE.slate, fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase" }}>
                Employee Records · {DATASET.length} total
              </div>
              <input
                placeholder="Search EMP-..."
                value={searching}
                onChange={e => setSearching(e.target.value)}
                style={{
                  background: "#0a1520", border: "1px solid #234", borderRadius: 8,
                  padding: "6px 12px", color: PALETTE.light, fontSize: 13, outline: "none",
                  width: 160,
                }}
              />
            </div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead>
                  <tr>
                    {[
                      ["name", "Employee"],
                      ["experience", "Exp (yrs)"],
                      ["workingHours", "Work Hrs"],
                      ["trainingHours", "Train Hrs"],
                      ["projectsCompleted", "Projects"],
                      ["score", "Score"],
                    ].map(([key, label]) => (
                      <th key={key}
                        onClick={() => toggleSort(key)}
                        style={{
                          textAlign: "left", padding: "8px 10px",
                          color: sort.key === key ? PALETTE.teal : PALETTE.slate,
                          fontWeight: 700, fontSize: 11, letterSpacing: "0.06em",
                          textTransform: "uppercase", cursor: "pointer",
                          borderBottom: "1px solid #234",
                          whiteSpace: "nowrap",
                        }}>
                        {label} {sort.key === key ? (sort.dir === -1 ? "↓" : "↑") : ""}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filtered.slice(0, 40).map((d, i) => (
                    <tr key={d.id} style={{ background: i % 2 === 0 ? "#0a1520" : "transparent" }}>
                      <td style={{ padding: "7px 10px", color: PALETTE.light, fontWeight: 600 }}>{d.name}</td>
                      <td style={{ padding: "7px 10px", color: PALETTE.slate }}>{d.experience}</td>
                      <td style={{ padding: "7px 10px", color: PALETTE.slate }}>{d.workingHours}</td>
                      <td style={{ padding: "7px 10px", color: PALETTE.slate }}>{d.trainingHours}</td>
                      <td style={{ padding: "7px 10px", color: PALETTE.slate }}>{d.projectsCompleted}</td>
                      <td style={{ padding: "7px 10px" }}>
                        <span style={{
                          display: "inline-block", padding: "2px 10px", borderRadius: 12,
                          background: scoreColor(d.score) + "22",
                          border: `1px solid ${scoreColor(d.score)}44`,
                          color: scoreColor(d.score), fontWeight: 700, fontSize: 11,
                        }}>
                          {d.score}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {filtered.length > 40 && (
                <div style={{ color: PALETTE.slate, fontSize: 11, textAlign: "center", paddingTop: 10 }}>
                  Showing 40 of {filtered.length} records
                </div>
              )}
            </div>
          </div>
        )}

        {/* Footer */}
        <div style={{ marginTop: 24, color: PALETTE.slate, fontSize: 11, textAlign: "center", opacity: 0.6 }}>
          Linear Regression Model · 4 Features · Simulated Dataset of 120 Employees
        </div>
      </div>
    </div>
  );
}
