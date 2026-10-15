"use strict";
(() => {
  const video = document.getElementById("teaser-video");
  const fallback = document.getElementById("teaser-fallback");
  const motion = window.matchMedia("(prefers-reduced-motion: reduce)");

  function showFigure() {
    video.hidden = true;
    fallback.hidden = false;
  }

  function playAutomatically() {
    if (!motion.matches && !navigator.connection?.saveData) {
      video.muted = true;
      // Native controls remain available if the browser declines autoplay.
      video.play().catch(() => {});
    }
  }

  video.addEventListener("error", showFigure);
  video.querySelector("source").addEventListener("error", showFigure);
  motion.addEventListener("change", () => {
    if (motion.matches) video.pause();
  });
  playAutomatically();
})();
