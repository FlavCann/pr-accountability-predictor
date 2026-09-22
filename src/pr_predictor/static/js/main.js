// Entry point: boot, the tab bar and the company chips.

import { api } from "./lib/api.js";
import { $ } from "./lib/dom.js";
import { esc } from "./lib/format.js";
import { readHash, state, writeHash } from "./state.js";
import { selectCompany } from "./views/company.js";
import { renderEvals } from "./views/evals.js";

function setView(view) {
  state.view = view;
  document.querySelectorAll(".tab").forEach((t) =>
    t.setAttribute("aria-selected", String(t.dataset.view === view)));
  $("#companies").hidden = view === "evals";
  if (view === "evals") { writeHash(); renderEvals(); }
  else showCompanies(state.company, state.thread);
}

async function showCompanies(company, thread) {
  const names = state.names;
  if (company && names.includes(company)) await selectCompany(company, thread);
  else if (names.length === 1) await selectCompany(names[0]);
  else $("#main").innerHTML = names.length
    ? `<p class="empty">Choose a company.</p>`
    : `<p class="empty">No sources in the store yet. Run the pipeline first.</p>`;
}

async function boot() {
  document.querySelector(".tabs").addEventListener("click", (e) => {
    const t = e.target.closest("[data-view]");
    if (t && t.dataset.view !== state.view) setView(t.dataset.view);
  });
  const h = readHash();
  let names = [];
  try {
    names = (await api("/companies")).companies;
  } catch (e) {
    if (h.view !== "evals") {
      $("#main").innerHTML = `<p class="empty">Could not reach the API: ${esc(e.message)}</p>`;
      return;
    }
  }
  state.names = names;
  const nav = $("#companies");
  nav.innerHTML = names.map((n) =>
    `<button class="chip" data-company="${esc(n)}" aria-pressed="false">${esc(n)}</button>`).join("");
  nav.addEventListener("click", (e) => {
    const b = e.target.closest("[data-company]");
    if (b) selectCompany(b.dataset.company);
  });

  if (h.view === "evals") setView("evals");
  else await showCompanies(h.company, h.thread);
}

boot();
