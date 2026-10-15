# Animated teaser

`teaser.mp4` is a 15-second, silent H.264 animation of the paper's Look → Perceive → Understand figure. The photograph is COCO train2014 image 477828, retained in `figures/teaser_sheep.jpg`. The selected crop is `(0, 160, 224, 384)` on its 640 × 480 canvas, giving the normalized box `[0, 333, 350, 800]`.

The photograph, crop, descriptions, and interpretation come from the existing paper illustration. Source Sans Pro is distributed under the license in `fonts/LICENSE.txt`.

To reproduce from the repository root:

```bash
python -m pip install -r scripts/requirements-video.txt
python scripts/render_teaser_video.py
```

The original static figure remains available as `teaser.png` and is used as the video poster and fallback.
