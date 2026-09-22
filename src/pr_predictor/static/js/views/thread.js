// One commitment in full: its outcome and the evidence behind it, then every
// time it was said, with a link to the moment. Pipeline internals sit folded at the end.

import { badge } from "../components/badge.js";
import { HEDGE_LABEL, STATUS_LABEL, TYPE_LABEL } from "../constants.js";
import { api } from "../lib/api.js";
import { $ } from "../lib/dom.js";
import { esc, fmtDate, fmtTime, pct } from "../lib/format.js";
import { state, writeHash } from "../state.js";

export async function selectThread(id) {
  state.thread = id;
  writeHash();
  document.querySelectorAll("#list .row").forEach((r) =>
    r.setAttribute("aria-selected", String(r.dataset.thread === id)));
  const box = $("#detail");
  box.innerHTML = `<p class="empty">Loading…</p>`;
  let a;
  try { a = await api("/threads/" + encodeURIComponent(id) + "/audit"); }
  catch (e) { box.innerHTML = `<p class="empty">${esc(e.message)}</p>`; return; }
  if (state.thread !== id) return;  // a later click won
  box.innerHTML = renderAudit(a);
  box.onclick = (e) => {
    const link = e.target.closest("a.evd");
    if (link) { e.preventDefault(); jump(link); }
  };
  box.scrollTop = 0;
  if (window.innerWidth <= 1000) box.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderAudit(a) {
  const v = a.verdicts[0];
  const s = a.silence;
  const n = a.promises.length;
  const said = n === 1 ? "Once, on " + fmtDate(a.first_seen)
    : n + " times, " + fmtDate(a.first_seen) + " → " + fmtDate(a.last_seen);
  const since = !s.sources_since ? "Nothing newer yet"
    : s.sources_since + " more " + (s.sources_since === 1 ? "presentation" : "presentations")
      + (s.silent ? ", and it wasn't repeated in any" : "");

  return `
    <div class="claim-lg">${esc(a.claim)}</div>
    <dl class="facts">
      <dt>Kind</dt><dd>${esc(TYPE_LABEL[a.promise_type] || a.promise_type)}</dd>
      ${a.metric ? `<dt>Measured by</dt><dd>${esc(a.metric)}</dd>` : ""}
      <dt>Deadline</dt><dd>${a.deadline ? fmtDate(a.deadline) : `<span class="muted">No fixed date</span>`}</dd>
      <dt>Said</dt><dd>${said}</dd>
      <dt>Since then</dt><dd>${since}</dd>
    </dl>

    <h3>Outcome</h3>
    ${v ? renderVerdict(v) : `<p class="muted">We haven't checked this one yet.</p>`}
    ${a.verdicts.length > 1 ? `<details><summary>Earlier outcomes (${a.verdicts.length - 1})</summary>
      ${a.verdicts.slice(1).map((ev) => `<div class="obs">${renderVerdict(ev, true)}</div>`).join("")}</details>` : ""}

    <h3>Where they said it</h3>
    ${a.promises.map((p) => renderPromise(p, a.claim)).join("")}

    ${renderTechnical(a)}`;
}

function renderPromise(p, threadClaim) {
  const pv = p.provenance;
  const ts = fmtTime(pv.t_start);
  return `<div class="obs">
    <div class="head">
      <span><b>${esc(pv.source_title || "Untitled source")}</b> · ${fmtDate(pv.published)}</span>
      ${ts ? `<a class="watch" href="${esc(pv.video_url)}" target="_blank" rel="noopener">▶ Watch at ${ts}</a>`
           : `<a class="watch" href="${esc(pv.source_url)}" target="_blank" rel="noopener">Open source</a>`}
    </div>
    <blockquote>“${esc(pv.verbatim)}”</blockquote>
    ${p.claim !== threadClaim ? `<div class="sub">In short: ${esc(p.claim)}</div>` : ""}
    <div class="meta">
      <span class="tag">${esc(HEDGE_LABEL[p.hedge_level] || p.hedge_level)}</span>
      ${p.deadline_raw ? `<span class="tag">Deadline: “${esc(p.deadline_raw)}”</span>` : ""}
    </div>
  </div>`;
}

// Evidence the rationale cites, numbered in the order it is first cited.
function footnotes(v) {
  const byId = new Map(v.evidence.map((e) => [e.id, e]));
  const order = [];
  (v.rationale.match(/evd_[0-9a-f]{12}/g) || []).forEach((id) => {
    if (byId.has(id) && !order.includes(id)) order.push(id);
  });
  return { byId, order, num: (id) => order.indexOf(id) + 1 };
}

function renderVerdict(v, compact) {
  const fn = footnotes(v);
  const rationale = esc(v.rationale)
    .replace(/\s*\(?(evd_[0-9a-f]{12}(?:,\s*evd_[0-9a-f]{12})*)\)?/g, (m, ids) => {
      const list = ids.split(/,\s*/);
      if (!list.every((id) => fn.byId.has(id))) return m;  // cites evidence we don't have: leave as written
      return list.map((id) =>
        `<a class="evd" href="#ev-${v.id}-${id}" title="Source ${fn.num(id)}">[${fn.num(id)}]</a>`).join("");
    });
  const cited = fn.order.map((id) => fn.byId.get(id));
  const rest = v.evidence.filter((e) => !fn.order.includes(e.id));
  const reviewed = v.reviews.length ? v.reviews[v.reviews.length - 1] : null;

  return `
    <div class="verdict-head">
      ${badge(v.status)}
      <span>${pct(v.confidence)} confident</span>
      <span class="muted">checked ${fmtDate(v.as_of)}</span>
    </div>
    ${reviewed ? `<div class="muted" style="font-size:13px;margin-top:6px">Reviewed by ${esc(reviewed.reviewer)}${
        v.status !== v.model_status ? `, who changed it from ${esc(STATUS_LABEL[v.model_status] || v.model_status)}` : ""}</div>` : ""}
    ${reviewed && reviewed.note ? `<div class="note">Reviewer's note: ${esc(reviewed.note)}</div>` : ""}
    <div class="rationale">${rationale}</div>
    ${compact ? "" : `
      ${cited.length ? `<h3>What we based this on</h3>${cited.map((e) => renderEvidence(v, e, fn.num(e.id))).join("")}` : ""}
      ${rest.length ? `<details><summary>Other passages we looked at (${rest.length})</summary>
        ${rest.map((e) => renderEvidence(v, e)).join("")}</details>` : ""}`}`;
}

function renderEvidence(v, e, n) {
  const src = e.source || {};
  const where = src.title
    ? `${esc(src.title)} · ${fmtDate(src.published)}${src.url ? ` · <a href="${esc(src.url)}" target="_blank" rel="noopener">open</a>` : ""}`
    : e.external_url ? `<a href="${esc(e.external_url)}" target="_blank" rel="noopener">${esc(e.external_url)}</a>` : "Unknown source";
  return `<div class="evidence${n ? " cited" : ""}" id="ev-${v.id}-${e.id}">
    <div class="src">${n ? `<b>[${n}]</b>` : ""}${where}</div>
    <div class="excerpt">“…${esc(e.excerpt.trim())}…”</div>
  </div>`;
}

// The pipeline's own record of how each claim was produced. Folded away, but
// never dropped: it is what makes a claim about a named company checkable.
function renderTechnical(a) {
  const verdictLines = a.verdicts.map((v) => {
    const parts = [v.id, "status " + v.status, "decided by " + (v.adjudicated_by || "—"),
      "confidence " + v.confidence, "as of " + v.as_of];
    if (v.escalated) parts.push("escalated to " + (v.advisor_model || "advisor")
      + (v.initial_status ? " (initially " + v.initial_status + ")" : ""));
    if (v.status !== v.model_status) parts.push("model said " + v.model_status);
    v.reviews.forEach((r) => parts.push("review: " + r.action + " by " + r.reviewer + " at " + r.at));
    return parts.join(" · ");
  });
  const evidenceLines = a.verdicts.length ? a.verdicts[0].evidence.map((e) =>
    [e.id, e.cited ? "cited" : "not cited", "found by " + e.found_by, e.polarity, e.source_id || e.external_url].join(" · ")) : [];
  const promiseLines = a.promises.map((p) => {
    const pv = p.provenance;
    return [p.id, "chars " + pv.start_char + "–" + pv.end_char + " of " + pv.source_id,
      pv.prompt_version, pv.model, "type " + p.promise_type, "hedge " + p.hedge_level].join(" · ");
  });
  const line = (t) => `<div class="lineage">${esc(t)}</div>`;
  return `<details class="tech"><summary>Technical details</summary>
    ${line("commitment " + a.id + " · " + a.promise_type + " · built from " + a.prompt_version)}
    ${verdictLines.map(line).join("")}
    ${promiseLines.map(line).join("")}
    ${evidenceLines.map(line).join("")}
  </details>`;
}

// Scroll to an evidence item, opening its <details> if it is folded away.
function jump(a) {
  const el = document.getElementById(a.getAttribute("href").slice(1));
  if (!el) return;
  const d = el.closest("details");
  if (d) d.open = true;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.style.background = "var(--accent-wash)";
  setTimeout(() => { el.style.background = ""; }, 1600);
}
