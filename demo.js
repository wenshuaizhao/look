"use strict";
const slider = document.getElementById("position");
const box = document.getElementById("box");
const windowFrame = document.getElementById("window");
const crop = document.getElementById("crop");
const scene = new Image();
function update() {
  const x = Number(slider.value) * 32;
  windowFrame.style.left = `${x / 512 * 100}%`;
  box.textContent = `Target box: [${Math.round(x / 512 * 1000)}, 250, ${Math.round((x + 256) / 512 * 1000)}, 750]`;
  if (scene.complete && scene.naturalWidth) {
    crop.getContext("2d").drawImage(scene, x, 96, 256, 192, 0, 0, 256, 192);
  }
}
scene.onload = update;
scene.src = "assets/scene.png";
slider.addEventListener("input", update);
update();
