import { STATUS_LABEL } from "../constants.js";
import { esc } from "../lib/format.js";

// A verdict status: colour swatch plus label, never colour alone.
export function badge(status) {
  const key = status || "unadjudicated";
  return `<span class="badge"><span class="sw" style="background:var(--v-${key})"></span>${esc(STATUS_LABEL[key] || key)}</span>`;
}
