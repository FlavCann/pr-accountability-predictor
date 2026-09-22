// DOM helpers shared by every view.

export const $ = (sel) => document.querySelector(sel);

// Hover tooltips for anything carrying a `data-tip` attribute.
export function bindTips(root = document) {
  const tip = $("#tip");
  root.querySelectorAll("[data-tip]").forEach((el) => {
    el.addEventListener("mousemove", (e) => {
      tip.textContent = el.dataset.tip;
      tip.style.display = "block";
      const x = Math.min(e.clientX + 12, window.innerWidth - tip.offsetWidth - 8);
      tip.style.left = x + "px";
      tip.style.top = (e.clientY + 14) + "px";
    });
    el.addEventListener("mouseleave", () => { tip.style.display = "none"; });
  });
}
