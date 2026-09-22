// The PR-accountability reveal: four images, all dark, until a spin lands on
// the company's level. The rating itself comes from the API (`rating.py`).

import { esc } from "../lib/format.js";

const LAPS = 4;            // full passes over the four images before settling
const FIRST_STEP_MS = 55;  // fastest step, at the start
const LAST_STEP_MS = 520;  // slowest step, the one before it lands

export function ratingCard(rating) {
  const cells = rating.scale.map((s, i) => `
    <figure class="rating-cell" data-level="${esc(s.level)}">
      <img src="img/rating/${esc(s.level)}.png" alt="${esc(s.label)}: clown make-up, stage ${i + 1} of ${rating.scale.length}">
    </figure>`).join("");
  return `
    <div class="card rating" id="rating">
      <h2>IR accountability</h2>
      <div class="rating-reel">${cells}</div>
      <div class="rating-actions">
        <button type="button" class="reveal">Reveal</button>
      </div>
      <p class="rating-result" aria-live="polite"></p>
    </div>`;
}

export function mountRating(root, rating) {
  const cells = Array.from(root.querySelectorAll(".rating-cell"));
  const button = root.querySelector(".reveal");
  const result = root.querySelector(".rating-result");
  const target = rating.scale.findIndex((s) => s.level === rating.level);
  if (target < 0) { button.disabled = true; return; }

  let timer = null;
  const light = (i) => cells.forEach((c, j) => c.classList.toggle("lit", j === i));

  function land() {
    light(target);
    cells[target].classList.add("chosen");
    root.classList.remove("spinning");
    const s = rating.scale[target];
    // The lit image carries the result; say so for screen readers.
    result.innerHTML = `<span class="sr-only">IR accountability: ${esc(s.label)}. </span>` +
      (rating.basis === "placeholder"
        ? `<span class="rating-basis" title="${esc(rating.note)}">Placeholder, not yet derived from the record</span>` : "");
    button.textContent = "Spin again";
    button.disabled = false;
  }

  button.addEventListener("click", () => {
    clearTimeout(timer);
    cells.forEach((c) => c.classList.remove("lit", "chosen"));
    result.textContent = "";
    button.disabled = true;

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return land();

    // The last lit step is (total - 1) % n, which is the target.
    const n = cells.length, total = LAPS * n + target + 1;
    let step = 0;
    root.classList.add("spinning");
    const tick = () => {
      light(step % n);
      step += 1;
      if (step >= total) return land();
      // Ease out: the delay grows with the cube of progress, so the wheel
      // races, then visibly slows over the last lap.
      const t = step / total;
      timer = setTimeout(tick, FIRST_STEP_MS + (LAST_STEP_MS - FIRST_STEP_MS) * t * t * t);
    };
    tick();
  });
}
