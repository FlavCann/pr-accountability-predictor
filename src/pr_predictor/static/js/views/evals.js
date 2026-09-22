// The model-evaluation view: the benchmark's scoreboard across runs.

import { api } from "../lib/api.js";
import { $, bindTips } from "../lib/dom.js";
import { esc, f2, fmtDay, fmtWhen, pct } from "../lib/format.js";
import { state } from "../state.js";

const SERIES = [
  ["f1", "F1", "var(--series-1)"],
  ["precision", "Precision", "var(--series-2)"],
  ["recall", "Recall", "var(--series-3)"],
];
const SLICE_LABEL = {
  fiscal_deadline: "Fiscal-year deadlines",
  qa: "Said in Q&A",
  restatement: "Restatements",
  asr_garble: "Garbled words",
  moderator: "Said by a moderator",
  revised_deadline: "Moved targets",
  chunk_boundary: "Across a chunk edge",
  unseen_whole: "Never seen whole",
  ai_drafted: "AI-drafted labels",
  pooled: "Added after blind pass",
};
const promptName = (r) => r.prompt_version + (r.baseline_truncate_words ? ` (first ${r.baseline_truncate_words} words)` : "");
const chunking = (r) => r.harness ? `${r.harness.chunk_words} / ${r.harness.chunk_overlap_words}` : "—";

export async function renderEvals() {
  $("#main").innerHTML = `<p class="empty">Loading evaluation runs…</p>`;
  let list, detail;
  try {
    list = await api("/evals");
  } catch (e) {
    $("#main").innerHTML = `<p class="empty">Could not load evaluation runs: ${esc(e.message)}</p>`;
    return;
  }
  if (state.view !== "evals") return;
  const runs = list.runs;
  if (!runs.length) {
    $("#main").innerHTML = `<p class="empty">No evaluation runs recorded yet. Run <code>python -m pr_predictor eval</code>.</p>`;
    return;
  }
  const current = runs.find((r) => r.is_champion) || runs[runs.length - 1];
  try {
    detail = await api("/evals/" + encodeURIComponent(current.run_id));
  } catch (e) {
    $("#main").innerHTML = `<p class="empty">Could not load run ${esc(current.run_id)}: ${esc(e.message)}</p>`;
    return;
  }
  if (state.view !== "evals") return;

  const i = runs.indexOf(current);
  const prev = i > 0 ? runs[i - 1] : null;
  const o = current.overall;
  const drafted = current.not_human_labelled || [];
  const splits = current.splits || {};
  const splitText = Object.entries(splits).map(([k, v]) => `${v.documents} ${k}`).join(", ");

  $("#main").innerHTML = `
    <h1>Model evaluation</h1>
    <div class="sub">${current.is_champion ? "Current champion" : "Latest run (no champion yet)"}:
      <b>${esc(promptName(current))}</b> on <b>${esc(current.model)}</b> ·
      recorded ${esc(fmtWhen(current.recorded_at))} · ${current.documents} transcripts scored (${esc(splitText)}) ·
      ${current.repeats} ${current.repeats === 1 ? "repeat" : "repeats"}</div>

    ${drafted.length ? `<div class="note warn"><b>Not a validated benchmark.</b> The answer key for ${drafted.length} of the
      ${current.documents} scored transcripts was not made by a person (${drafted.map(esc).join(", ")}). These scores show the
      pipeline runs end to end, not how good it is.</div>` : ""}
    ${current.underpowered ? `<div class="note">Only ${current.documents} transcripts are scored; at least 5 are needed before a
      likely range can be reported. Treat small differences between runs as possibly luck.</div>` : ""}
    ${current.unjudged ? `<div class="note">${current.unjudged} extractions matched no label, so precision is not final.</div>` : ""}

    <section class="eval">
      <div class="card">
        <h2>How the ${current.is_champion ? "champion" : "latest run"} scores</h2>
        <div class="tiles">
          ${evalTile("F1", f2(o.f1), current.f1_ci95 ? `likely ${f2(current.f1_ci95[0])}–${f2(current.f1_ci95[1])}` : "headline score", o.f1, prev && prev.overall.f1, true, f2)}
          ${evalTile("Precision", f2(o.precision), "its promises that were real", o.precision, prev && prev.overall.precision, true, f2)}
          ${evalTile("Recall", f2(o.recall), "real promises it found", o.recall, prev && prev.overall.recall, true, f2)}
          ${evalTile("Span fidelity", pct(o.span_fidelity), "quotes found in the transcript", o.span_fidelity, prev && prev.overall.span_fidelity, true, (x) => pct(x))}
          ${evalTile("Traps picked up", f2(o.forbidden_extracted), "per run; lower is better", o.forbidden_extracted, prev && prev.overall.forbidden_extracted, false, f2)}
          ${evalTile("Type · hedge · deadline", `${f2(o.type_accuracy)} · ${f2(o.hedge_accuracy)} · ${f2(o.deadline_accuracy)}`, "accuracy on promises it found", null, null, true, f2)}
          ${evalTile("Stability", current.repeat_f1 && current.repeat_f1.length > 1
              ? `${f2(Math.min(...current.repeat_f1))}–${f2(Math.max(...current.repeat_f1))}` : "—",
              "F1 range across repeats", null, null, true, f2)}
          ${evalTile("Cost per run", "$" + Number(current.cost_usd || 0).toFixed(2),
              current.latency_s ? Math.round(current.latency_s / 60) + " min" : "", null, null, true, f2)}
        </div>
        ${prev ? `<div class="muted" style="font-size:12px;margin-top:8px">Changes are against the previous run:
          ${esc(promptName(prev))} on ${esc(prev.model)}, ${esc(fmtWhen(prev.recorded_at))}.</div>` : ""}
      </div>

      <div class="card">
        <h2>Progress across versions</h2>
        <div class="legend-row">${SERIES.map(([, l, c]) =>
          `<span class="key" style="display:inline-flex;align-items:center;gap:6px"><span class="sw" style="background:${c}"></span>${l}</span>`).join("")}
          <span class="key">★ champion</span></div>
        <div class="chart scroll-x">${progressChart(runs)}</div>
        <h3>All runs</h3>
        <div class="scroll-x">${runsTable(runs)}</div>
      </div>

      <div class="split">
        <div class="card">
          <h2>By transcript</h2>
          <div class="sub" style="margin-bottom:8px">${Object.entries(splits).map(([k, v]) =>
            `${esc(k[0].toUpperCase() + k.slice(1))} F1 <b class="num">${f2(v.f1)}</b> (${v.documents})`).join(" · ")}</div>
          <div class="scroll-x">${docTable(detail.per_document)}</div>
        </div>
        <div class="card">
          <h2>Hard cases: recall by slice</h2>
          <div class="sub" style="margin-bottom:10px">Share of labelled promises of each kind that were found.
            The dark tick marks overall recall (${f2(o.recall)}); bars short of it are weaker spots.</div>
          ${sliceBars(detail.slices, current.repeats)}
        </div>
      </div>

      <details class="card">
        <summary>What produced this score</summary>
        <div class="manifest">${esc(JSON.stringify(Object.assign({ run_id: current.run_id,
          gold_fingerprint: current.gold_fingerprint }, detail.manifest || {}), null, 2))}</div>
      </details>
    </section>`;

  bindTips($("#main"));
  bindChart();
}

function evalTile(label, value, foot, cur, prev, higherIsBetter, fmt) {
  let delta = "";
  if (cur != null && prev != null) {
    const d = cur - prev;
    const better = higherIsBetter ? d > 0 : d < 0;
    // The arrow shows the direction of change; the colour shows whether that is good.
    const cls = Math.abs(d) < 1e-9 ? "flat" : better ? "up" : "down";
    const arrow = Math.abs(d) < 1e-9 ? "•" : d > 0 ? "▲" : "▼";
    const sign = d > 0 ? "+" : d < 0 ? "−" : "±";
    const mag = fmt === f2 ? Math.abs(d).toFixed(2) : Math.round(Math.abs(d) * 100) + " pts";
    delta = `<div class="delta ${cls}">${arrow} ${sign}${mag} vs previous</div>`;
  }
  const compact = String(value).length > 10 ? " compact" : "";
  return `<div class="tile"><div class="label">${esc(label)}</div>
    <div class="value num${compact}">${value}</div>${delta}
    <div class="foot">${esc(foot)}</div></div>`;
}

function progressChart(runs) {
  const W = 760, H = 290, m = { l: 40, r: 118, t: 14, b: 62 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const x = (i) => runs.length === 1 ? m.l + iw / 2 : m.l + (i * iw) / (runs.length - 1);
  const y = (v) => m.t + ih * (1 - Math.max(0, Math.min(1, v)));
  const every = Math.max(1, Math.ceil(runs.length / 8));

  let g = "";
  [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
    g += `<line class="grid-line" x1="${m.l}" x2="${m.l + iw}" y1="${y(t)}" y2="${y(t)}"/>
          <text class="axis-text" x="${m.l - 8}" y="${y(t) + 4}" text-anchor="end">${t.toFixed(2)}</text>`;
  });
  runs.forEach((r, i) => {
    if (i % every && i !== runs.length - 1 && !r.is_champion) return;
    const cx = x(i);
    g += `<text class="x-text" x="${cx}" y="${m.t + ih + 18}" text-anchor="middle">${esc(fmtDay(r.recorded_at))}</text>
          <text class="x-sub" x="${cx}" y="${m.t + ih + 32}" text-anchor="middle">${esc(r.prompt_version)}</text>
          ${r.is_champion ? `<text class="champ-text" x="${cx}" y="${m.t + ih + 47}" text-anchor="middle">★ champion</text>` : ""}`;
  });

  g += `<line class="crosshair" id="xhair" x1="0" x2="0" y1="${m.t}" y2="${m.t + ih}" visibility="hidden"/>`;

  SERIES.forEach(([k, , c]) => {
    const pts = runs.map((r, i) => [x(i), y(r.overall[k] || 0)]);
    if (pts.length > 1) {
      g += `<path d="${pts.map((p, j) => (j ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ")}"
              fill="none" stroke="${c}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
    }
    pts.forEach((p) => {
      g += `<circle cx="${p[0]}" cy="${p[1]}" r="4.5" fill="${c}" stroke="var(--surface)" stroke-width="2"/>`;
    });
  });

  // Direct labels at the latest run, nudged apart so they never collide.
  const last = runs[runs.length - 1], lx = x(runs.length - 1) + 12;
  const labels = SERIES.map(([k, l]) => ({ y: y(last.overall[k] || 0), text: `${l} ${f2(last.overall[k])}` }))
    .sort((a, b) => a.y - b.y);
  for (let j = 1; j < labels.length; j++) labels[j].y = Math.max(labels[j].y, labels[j - 1].y + 14);
  labels.forEach((l) => { g += `<text class="dlabel" x="${lx}" y="${l.y + 4}">${esc(l.text)}</text>`; });

  // Hit columns: the whole height, halfway to each neighbour.
  runs.forEach((r, i) => {
    const left = i === 0 ? m.l - 12 : (x(i - 1) + x(i)) / 2;
    const right = i === runs.length - 1 ? m.l + iw + 12 : (x(i) + x(i + 1)) / 2;
    const tip = [
      `${promptName(r)} · ${r.model}${r.is_champion ? "  ★ champion" : ""}`,
      fmtWhen(r.recorded_at),
      `F1 ${f2(r.overall.f1)} · precision ${f2(r.overall.precision)} · recall ${f2(r.overall.recall)}`,
      `span fidelity ${pct(r.overall.span_fidelity)} · traps ${f2(r.overall.forbidden_extracted)}`,
      `${r.documents} transcripts · ${r.repeats} repeats · $${Number(r.cost_usd || 0).toFixed(2)}`,
    ].join("\n");
    g += `<rect class="hit" x="${left}" y="${m.t}" width="${Math.max(1, right - left)}" height="${ih}"
            data-cx="${x(i)}" data-chart-tip="${esc(tip)}"/>`;
  });

  const summary = runs.map((r) => `${r.prompt_version} F1 ${f2(r.overall.f1)}`).join(", ");
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="F1, precision and recall per evaluation run: ${esc(summary)}">${g}</svg>`;
}

function bindChart() {
  const tip = $("#tip"), xh = document.getElementById("xhair");
  document.querySelectorAll("[data-chart-tip]").forEach((el) => {
    el.addEventListener("mousemove", (e) => {
      tip.textContent = el.dataset.chartTip;
      tip.style.display = "block";
      tip.style.left = Math.min(e.clientX + 12, window.innerWidth - tip.offsetWidth - 8) + "px";
      tip.style.top = (e.clientY + 14) + "px";
      if (xh) { xh.setAttribute("x1", el.dataset.cx); xh.setAttribute("x2", el.dataset.cx); xh.setAttribute("visibility", "visible"); }
    });
    el.addEventListener("mouseleave", () => {
      tip.style.display = "none";
      if (xh) xh.setAttribute("visibility", "hidden");
    });
  });
}

function runsTable(runs) {
  const rows = runs.slice().reverse();
  return `<table>
    <thead><tr><th>Recorded</th><th>Prompt</th><th>Model</th><th class="n">Chunking</th><th class="n">Transcripts</th>
      <th class="n">F1</th><th class="n">Repeats F1</th><th class="n">Precision</th><th class="n">Recall</th>
      <th class="n">Fidelity</th><th class="n">Traps</th><th class="n">Cost</th><th></th></tr></thead>
    <tbody>${rows.map((r) => `
      <tr class="${r.is_champion ? "champion" : ""}">
        <td>${esc(fmtWhen(r.recorded_at))}</td>
        <td>${esc(promptName(r))}</td>
        <td>${esc(r.model)}</td>
        <td class="n">${esc(chunking(r))}</td>
        <td class="n">${r.documents}</td>
        <td class="n"><b>${f2(r.overall.f1)}</b>${r.f1_ci95 ? `<div class="muted">${f2(r.f1_ci95[0])}–${f2(r.f1_ci95[1])}</div>` : ""}</td>
        <td class="n">${(r.repeat_f1 || []).map(f2).join(" · ")}</td>
        <td class="n">${f2(r.overall.precision)}</td>
        <td class="n">${f2(r.overall.recall)}</td>
        <td class="n">${pct(r.overall.span_fidelity)}</td>
        <td class="n">${f2(r.overall.forbidden_extracted)}</td>
        <td class="n">$${Number(r.cost_usd || 0).toFixed(2)}</td>
        <td>${r.is_champion ? `<span class="pill">★ champion</span>` : ""}</td>
      </tr>`).join("")}</tbody>
  </table>
  <div class="muted" style="font-size:12px;margin-top:6px">Chunking is words per chunk / overlap. Runs scored against different
    answer keys are not directly comparable.</div>`;
}

function docTable(docs) {
  const rows = (docs || []).slice().sort((a, b) => String(a.split).localeCompare(String(b.split)) || a.f1 - b.f1);
  return `<table>
    <thead><tr><th>Transcript</th><th>Split</th><th class="n">F1</th><th class="n">Precision</th>
      <th class="n">Recall</th><th class="n">Traps</th></tr></thead>
    <tbody>${rows.map((d) => `
      <tr><td>${esc(d.title || d.source_url)}</td><td><span class="tag">${esc(d.split)}</span></td>
        <td class="n"><b>${f2(d.f1)}</b></td><td class="n">${f2(d.precision)}</td>
        <td class="n">${f2(d.recall)}</td><td class="n">${f2(d.forbidden_extracted)}</td></tr>`).join("")}</tbody>
  </table>`;
}

function sliceBars(slices, repeats) {
  if (!slices) return `<p class="muted">No slice data for this run.</p>`;
  const overall = slices.all ? slices.all.recall : null;
  const rows = Object.entries(slices)
    .filter(([k, v]) => k !== "all" && !k.startsWith("split:") && v.recall != null)
    .sort((a, b) => a[1].recall - b[1].recall);
  if (!rows.length) return `<p class="muted">No tagged promises in this answer key.</p>`;
  const reps = Math.max(1, repeats || 1);
  return `<div class="hbars">${rows.map(([k, v]) => {
    const n = Math.round(v.gold_observations / reps);
    const label = SLICE_LABEL[k] || k;
    return `<div class="hbar" data-tip="${esc(label)}: recall ${f2(v.recall)} on ${n} labelled ${n === 1 ? "promise" : "promises"}${
        v.deadline_accuracy != null ? ` · deadline accuracy ${f2(v.deadline_accuracy)}` : ""}">
      <span>${esc(label)}</span>
      <span class="track"><span class="fill" style="display:block;width:${(v.recall * 100).toFixed(1)}%"></span>
        ${overall != null ? `<span class="ref" style="left:calc(${(overall * 100).toFixed(1)}% - 1px)"></span>` : ""}</span>
      <span class="val">${f2(v.recall)} · ${n}</span>
    </div>`;
  }).join("")}</div>`;
}
