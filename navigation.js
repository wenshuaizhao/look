"use strict";
(() => {
  const links = [...document.querySelectorAll(".section-nav a")];
  const sections = links.map(link => document.querySelector(link.getAttribute("href")));
  let pending = false;

  function update() {
    let current = 0;
    const offset = document.querySelector(".section-nav").getBoundingClientRect().height + 28;
    sections.forEach((section, index) => {
      if (section.getBoundingClientRect().top <= offset) current = index;
    });
    links.forEach((link, index) => {
      link.classList.toggle("active", index === current);
      if (index === current) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
    pending = false;
  }

  function schedule() {
    if (!pending) {
      pending = true;
      requestAnimationFrame(update);
    }
  }
  window.addEventListener("scroll", schedule, { passive: true });
  window.addEventListener("resize", schedule);
  document.fonts.ready.then(update);
  update();
})();
