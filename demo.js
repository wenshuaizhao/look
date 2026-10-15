"use strict";
(() => {
  const slider = document.getElementById("position");
  const box = document.getElementById("box");
  const windowFrame = document.getElementById("window");
  const crop = document.getElementById("crop");
  const photo = new Image();
  const geometry = { width: 640, height: 480, side: 224, y: 160, step: 32 };

  function update() {
    const { width, height, side, y, step } = geometry;
    const x = Number(slider.value) * step;
    windowFrame.style.left = `${x / width * 100}%`;
    windowFrame.style.top = `${y / height * 100}%`;
    windowFrame.style.width = `${side / width * 100}%`;
    windowFrame.style.height = `${side / height * 100}%`;
    const coords = [x / width, y / height, (x + side) / width, (y + side) / height];
    box.textContent = `Target box: [${coords.map(v => Math.round(v * 1000)).join(", ")}]`;
    if (photo.complete && photo.naturalWidth) {
      crop.getContext("2d").drawImage(photo, x, y, side, side, 0, 0, crop.width, crop.height);
    }
  }

  photo.onload = update;
  photo.src = "assets/sheep.jpg";
  slider.addEventListener("input", update);
  update();
})();
