// Formatting and escaping. Every string from the API goes through `esc`.

export const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

export const fmtDate = (iso) => iso ? new Date(iso + "T00:00:00").toLocaleDateString(undefined,
  { year: "numeric", month: "short", day: "numeric" }) : "—";

export const pct = (x) => x == null ? "—" : Math.round(x * 100) + "%";

export const f2 = (x) => x == null ? "—" : Number(x).toFixed(2);

export const fmtTime = (s) => {
  if (s == null) return null;
  s = Math.floor(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return (h ? h + ":" + String(m).padStart(2, "0") : m) + ":" + String(sec).padStart(2, "0");
};

// Timestamps from the store are naive UTC.
const utc = (iso) => new Date(/Z|[+-]\d\d:\d\d$/.test(iso) ? iso : iso + "Z");

export const fmtWhen = (iso) => iso ? utc(iso).toLocaleString(undefined,
  { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—";

export const fmtDay = (iso) => iso ? utc(iso).toLocaleDateString(undefined,
  { month: "short", day: "numeric" }) : "—";
