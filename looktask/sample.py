"""Turn any image into a look sample: a crop plus the exact box it came from."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .geometry import Geometry


@dataclass
class LookSample:
    crop: Image.Image
    canon: Image.Image
    gt_px: list            # window in canonical pixels
    gt_norm1000: list      # window as thousandths of the canvas
    offset: tuple          # (block_row, block_col) of the window's top-left
    canvas: tuple          # (RW, RH)
    blocks: tuple          # (BW, BH)
    crop_blocks: tuple     # (CBW, CBH)


def sample_window(blocks_w, blocks_h, rng, min_frac=0.15, max_frac=0.80):
    """Merge-block-aligned rectangle -> (x0, y0, x1, y1) in block coords, end-exclusive.

    Block units, not pixels: one block is one visual token, so the window boundary always falls
    on token boundaries and every token is wholly inside or wholly outside. An arbitrary pixel
    rectangle would half-mask the edge patches and make the label ambiguous.
    """
    bw = max(1, int(round(blocks_w * rng.uniform(min_frac, max_frac))))
    bh = max(1, int(round(blocks_h * rng.uniform(min_frac, max_frac))))
    x0 = int(rng.integers(0, blocks_w - bw + 1))
    y0 = int(rng.integers(0, blocks_h - bh + 1))
    return x0, y0, x0 + bw, y0 + bh


def make_look_sample(image: Image.Image, geo: Geometry, rng, **kw) -> LookSample:
    image = image.convert('RGB')
    rw, rh = geo.canonical_size(*image.size)
    _, _, bw_, bh_ = geo.grids(rw, rh)
    canon = image.resize((rw, rh), Image.BICUBIC)

    x0, y0, x1, y1 = sample_window(bw_, bh_, rng, **kw)
    gt_px = [x0 * geo.cell, y0 * geo.cell, x1 * geo.cell, y1 * geo.cell]
    crop = canon.crop(tuple(gt_px))
    return LookSample(
        crop=crop, canon=canon, gt_px=gt_px,
        gt_norm1000=[round(gt_px[0] / rw * 1000), round(gt_px[1] / rh * 1000),
                     round(gt_px[2] / rw * 1000), round(gt_px[3] / rh * 1000)],
        offset=(y0, x0), canvas=(rw, rh), blocks=(bw_, bh_), crop_blocks=(x1 - x0, y1 - y0))
