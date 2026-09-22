// The company view: profile, summary and rating, the commitments, then the breakdowns.

import { badge } from "../components/badge.js";
import { mountRating, ratingCard } from "../components/rating.js";
import { HEDGE_LABEL, STATUSES, TYPE_LABEL } from "../constants.js";
import { api } from "../lib/api.js";
import { $, bindTips } from "../lib/dom.js";
import { esc, fmtDate, pct } from "../lib/format.js";
import { state, writeHash } from "../state.js";
import { selectThread } from "./thread.js";

export async function selectCompany(name, threadId) {
  state.company = name;
  state.thread = null;
  document.querySelectorAll("#companies .chip").forEach((b) =>
    b.setAttribute("aria-pressed", String(b.dataset.company === name)));
  $("#main").innerHTML = `<p class="empty">Loading ${esc(name)}…</p>`;
  const q = encodeURIComponent(name);
  const [profile, threads] = await Promise.all([
    api("/companies/" + q + "/profile"),
    api("/threads?company=" + q),
  ]);
  if (state.view !== "companies") return;  // the user switched tabs meanwhile
  state.profile = profile;
  state.threads = threads;
  renderCompany();
  if (threadId && threads.some((t) => t.id === threadId)) selectThread(threadId);
  else writeHash();
}

function renderCompany() {
  const p = state.profile, c = p.coverage;
  const counts = Object.assign({}, p.verdicts, { unadjudicated: c.unadjudicated_threads });
  const partialCoverage = c.adjudicated_threads < c.threads;
  const plural = (n, one, many) => n + " " + (n === 1 ? one : many);
  const about = p.about;

  $("#main").innerHTML = `
    <section class="card company-hero">
      <div>
        <h1>${esc(p.company)}</h1>
        <div class="sub">${plural(c.sources, "presentation", "presentations")} from ${fmtDate(c.first_source)} to ${fmtDate(c.last_source)} ·
          ${c.promises} promises about ${c.threads} commitments</div>
        ${about && about.description ? `<p class="about">${esc(about.description)}</p>` : ""}
      </div>
      ${about && about.logo ? `<img class="company-logo" src="${esc(about.logo)}" alt="${esc(p.company)} logo">` : ""}
    </section>

    <section class="summary-row">
      <div class="card">
        <h2>What happened to its ${c.threads} commitments</h2>
        ${verdictBar(counts, c.threads)}
        ${partialCoverage ? `<div class="note">So far we've checked <b>${c.adjudicated_threads} of ${c.threads}</b> commitments.
          The numbers below are based on those ${c.adjudicated_threads}.</div>` : ""}
        <div class="tiles">
          ${tile("Checked", c.adjudicated_threads, "of " + c.threads, pct(c.adjudicated_threads / (c.threads || 1)) + " so far")}
          ${tile("Kept", counts.kept, p.record.resolved ? "of " + p.record.resolved : "", p.record.resolved ? "of those with an outcome" : "no outcomes yet")}
          ${tile("Quietly dropped", counts.quietly_dropped, "", "promised, then never followed up")}
          ${tile("Missed", counts.missed, "", counts.partial ? "plus " + counts.partial + " partly kept" : "promised, and not delivered")}
          ${tile("Can't tell yet", counts.too_early + counts.no_evidence, "", "too early, or no evidence found")}
          ${tile("Repeated", c.restated_threads, "", "promised more than once")}
        </div>
      </div>
      ${p.rating ? ratingCard(p.rating) : ""}
    </section>

    <section class="workspace">
      <div class="card">
        <h2>Commitments</h2>
        <div class="filters">
          <input id="f-q" type="search" placeholder="Search promises" value="${esc(state.filter.q)}" aria-label="Search promises">
          <select id="f-status" aria-label="Outcome">
            <option value="">All outcomes</option>
            ${STATUSES.map(([k, l]) => `<option value="${k}">${l} (${counts[k] || 0})</option>`).join("")}
          </select>
          <select id="f-type" aria-label="Kind of promise">
            <option value="">All kinds</option>
            ${Object.entries(TYPE_LABEL).map(([k, l]) => `<option value="${k}">${l}</option>`).join("")}
          </select>
        </div>
        <div class="muted" id="count" style="font-size:13px;margin-bottom:8px"></div>
        <div class="list" id="list" role="listbox" aria-label="Commitments"></div>
      </div>
      <div class="card detail" id="detail"><p class="empty">Pick a commitment to see what was said, where, and what happened.</p></div>
    </section>

    <section class="profile">

      <div class="split">
        <div class="card">
          <h2>What kind of promises</h2>
          <div class="scroll-x">${typeTable(p.by_type)}</div>
        </div>
        <div class="card">
          <h2>How firmly they were made</h2>
          <table>
            <thead><tr><th>Wording</th><th class="n">Promises</th><th class="n">Share</th></tr></thead>
            <tbody>${Object.entries(p.hedge_mix).map(([k, n]) => `
              <tr><td>${esc(HEDGE_LABEL[k] || k)}</td><td class="n">${n}</td>
              <td class="n">${pct(n / (c.promises || 1))}</td></tr>`).join("")}</tbody>
          </table>
          <h3>Gone quiet</h3>
          <div class="sub"><b>${plural(p.silence.silent_threads, "commitment was", "commitments were")}</b> never repeated in the
            presentations that followed. That alone doesn't mean they were dropped (some were simply delivered),
            but it's where we look first.</div>
          <p class="muted" style="font-size:13px;margin:16px 0 0">
            ${plural(p.escalated, "outcome was", "outcomes were")} double-checked by a second model ·
            ${p.reviewed ? plural(p.reviewed, "was", "were") + " reviewed by a person" : "none reviewed by a person yet"}</p>
        </div>
      </div>
    </section>`;

  $("#f-status").value = state.filter.status;
  $("#f-type").value = state.filter.type;
  $("#f-q").addEventListener("input", (e) => { state.filter.q = e.target.value; renderList(); });
  $("#f-status").addEventListener("change", (e) => { state.filter.status = e.target.value; renderList(); });
  $("#f-type").addEventListener("change", (e) => { state.filter.type = e.target.value; renderList(); });
  $("#list").addEventListener("click", (e) => {
    const r = e.target.closest("[data-thread]");
    if (r) selectThread(r.dataset.thread);
  });
  if (p.rating) mountRating($("#rating"), p.rating);
  bindTips($("#main"));
  renderList();
}

function tile(label, value, unit, foot) {
  return `<div class="tile"><div class="label">${esc(label)}</div>
    <div class="value">${value}${unit ? ` <small>${esc(unit)}</small>` : ""}</div>
    <div class="foot">${esc(foot)}</div></div>`;
}

function verdictBar(counts, total) {
  const present = STATUSES.filter(([k]) => counts[k] > 0);
  const segs = present.map(([k, l]) =>
    `<div class="seg" style="flex:${counts[k]};background:var(--v-${k})"
       data-tip="${esc(l)}: ${counts[k]} of ${total} commitments (${pct(counts[k] / total)})"></div>`).join("");
  const legend = STATUSES.map(([k, l]) =>
    `<span class="key"><span class="sw" style="background:var(--v-${k})"></span>${l} <b class="num">${counts[k] || 0}</b></span>`).join("");
  return `<div class="bar" role="img" aria-label="${esc(present.map(([k, l]) => l + " " + counts[k]).join(", "))}">${segs}</div>
          <div class="legend">${legend}</div>`;
}

function typeTable(byType) {
  const cols = STATUSES.filter(([k]) => Object.values(byType).some((r) => r[k]));
  const rows = Object.entries(byType).sort((a, b) => b[1].threads - a[1].threads);
  return `<table>
    <thead><tr><th>Kind</th><th class="n">Commitments</th>${cols.map(([k, l]) =>
      `<th class="n"><span class="sw" style="background:var(--v-${k});margin-right:4px"></span>${l}</th>`).join("")}</tr></thead>
    <tbody>${rows.map(([t, r]) => `<tr><td>${esc(TYPE_LABEL[t] || t)}</td><td class="n">${r.threads}</td>
      ${cols.map(([k]) => `<td class="n">${r[k] || `<span class="muted">0</span>`}</td>`).join("")}</tr>`).join("")}</tbody>
  </table>`;
}

// Thread list

function renderList() {
  const f = state.filter, q = f.q.trim().toLowerCase();
  const rows = state.threads
    .filter((t) => !f.status || (t.verdict || "unadjudicated") === f.status)
    .filter((t) => !f.type || t.promise_type === f.type)
    .filter((t) => !q || t.claim.toLowerCase().includes(q))
    // Checked first, then most recently mentioned.
    .sort((a, b) => (!!b.verdict - !!a.verdict) || String(b.last_seen).localeCompare(String(a.last_seen)));

  $("#count").textContent = rows.length === state.threads.length
    ? rows.length + " commitments" : rows.length + " of " + state.threads.length + " commitments";
  $("#list").innerHTML = rows.map((t) => `
    <button class="row" role="option" data-thread="${t.id}" aria-selected="${t.id === state.thread}">
      <div class="claim">${esc(t.claim)}</div>
      <div class="meta">
        ${badge(t.verdict)}
        <span class="tag">${esc(TYPE_LABEL[t.promise_type] || t.promise_type)}</span>
        <span>${t.observations === 1 ? "said once" : "said " + t.observations + " times"}</span>
        <span>${fmtDate(t.first_seen)}${t.last_seen !== t.first_seen ? " → " + fmtDate(t.last_seen) : ""}</span>
      </div>
    </button>`).join("") || `<p class="empty">Nothing matches.</p>`;
}
