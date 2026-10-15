"""Patch/block geometry for the look task.

Everything is expressed in the model's *canonical* space -- the image size the processor would
smart-resize to. We do that resize ourselves and pass `do_resize=False`, which makes a crop's
patch (ph, pw) bit-identical to the full image's patch (ph + by0*merge, pw + bx0*merge).
Crop first and let the processor resize instead, and every patch changes.
"""
from __future__ import annotations



class Geometry:
    """Patch grid for one processor. Read from the processor -- never hard-code.

    Qwen3-VL uses patch=16 (Qwen2.5-VL uses 14), so cell size differs between them and a
    hard-coded value silently misaligns every window.
    """

    def __init__(self, image_processor, max_pixels: int):
        p = image_processor.patch_size
        self.patch = p if isinstance(p, int) else p[0]
        self.merge = image_processor.merge_size
        self.cell = self.patch * self.merge          # one merge block == one LLM visual token
        self.max_pixels = max_pixels

    def canonical_size(self, w: int, h: int) -> tuple[int, int]:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        rh, rw = smart_resize(h, w, factor=self.cell, max_pixels=self.max_pixels)
        return rw, rh

    def grids(self, rw: int, rh: int) -> tuple[int, int, int, int]:
        """-> (grid_w, grid_h, blocks_w, blocks_h) i.e. patches then visual tokens."""
        return rw // self.patch, rh // self.patch, rw // self.cell, rh // self.cell

    def patch_row(self, ph: int, pw: int, grid_w: int) -> int:
        """Row of patch (ph, pw) in `pixel_values`.

        The processor flattens merge-block-major, NOT raster
        (image_processing_qwen2_vl.py, permute(0, 2, 5, 3, 6, 1, 4, 7)).
        """
        m = self.merge
        bh, mh = divmod(ph, m)
        bw, mw = divmod(pw, m)
        return ((bh * (grid_w // m) + bw) * m + mh) * m + mw

    def token_index(self, bh: int, bw: int, blocks_w: int) -> int:
        return bh * blocks_w + bw

    def crop_rows_in_full(self, by0, bx0, crop_gw, crop_gh, full_gw) -> torch.Tensor:
        """For each crop-flatten row, the matching full-image-flatten row."""
        import torch
        m = self.merge
        pairs = [(self.patch_row(ph, pw, crop_gw),
                  self.patch_row(ph + by0 * m, pw + bx0 * m, full_gw))
                 for ph in range(crop_gh) for pw in range(crop_gw)]
        return torch.tensor([f for _, f in sorted(pairs)], dtype=torch.long)

    def verify_flatten_order(self, image_processor, rw: int, rh: int) -> bool:
        """Paint one patch and check which pixel_values row moved. Cheap; catches silent breakage."""
        from PIL import Image, ImageDraw
        gw, gh, _, _ = self.grids(rw, rh)
        base = Image.new('RGB', (rw, rh), (128, 128, 128))
        pv0 = image_processor(images=[base], return_tensors='pt')['pixel_values']
        for (ph, pw) in [(0, 0), (0, 1), (1, 0), (1, 1), (3, 5), (gh - 1, gw - 1)]:
            probe = base.copy()
            ImageDraw.Draw(probe).rectangle(
                [pw * self.patch, ph * self.patch,
                 (pw + 1) * self.patch - 1, (ph + 1) * self.patch - 1], fill=(255, 0, 0))
            pv = image_processor(images=[probe], return_tensors='pt')['pixel_values']
            changed = (pv != pv0).any(dim=-1).nonzero().flatten().tolist()
            if changed != [self.patch_row(ph, pw, gw)]:
                return False
        return True
