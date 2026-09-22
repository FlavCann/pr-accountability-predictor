// Page state, and its reflection in the URL: #company=…&thread=…  or  #view=evals

export const state = {
  view: "companies", names: [], company: null, profile: null, threads: [], thread: null,
  filter: { q: "", status: "", type: "" },
};

export function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  return { view: p.get("view"), company: p.get("company"), thread: p.get("thread") };
}

export function writeHash() {
  const p = new URLSearchParams();
  if (state.view === "evals") p.set("view", "evals");
  else {
    if (state.company) p.set("company", state.company);
    if (state.thread) p.set("thread", state.thread);
  }
  history.replaceState(null, "", "#" + p.toString());
}
