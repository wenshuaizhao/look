"""Give a crop its FULL-IMAGE positions, without forking any forward pass.

Every positional quantity the Qwen vision tower uses is poppable from kwargs
(transformers/vision_utils.py:104 for position_ids, :266 for the interpolation tables), so a
forward_pre_hook suffices.

Measured on Qwen3-VL-8B (rel L2 change of the visual tokens vs. an unmodified crop):
    absolute pos_embed override ... 0.676   <- the channel
    RoPE override ................ 0.033   <- no-op; RoPE is relative, translation cancels
    both ......................... 0.677
On Qwen2.5-VL (RoPE only, no absolute embedding) the override is inert: 3a and 3b produce
byte-identical text.
"""
from __future__ import annotations

import torch

# Optional: a 5.x refactor. template.py imports this module, and template.py is re-imported by
# `swift eval` from the adapter's args.json -- in the eval env (transformers 4.57.6). Only the
# override classes need these; find_rope_owner/find_visual do not.
try:
    from transformers.vision_utils import (get_vision_interpolation_indices_and_weights,
                                           get_vision_position_ids)
except ImportError:                        # transformers < 5
    get_vision_interpolation_indices_and_weights = get_vision_position_ids = None


def find_rope_owner(model, max_depth=6):
    """Walk down .model until we find the module owning get_rope_index.

    Depth depends on wrapping: bare  Qwen3VLForConditionalGeneration -> .model (Qwen3VLModel),
    but Swift/PEFT adapters add a level, so a fixed `.model` lands short and raises
    AttributeError: 'Qwen3VLForConditionalGeneration' object has no attribute 'get_rope_index'.
    """
    m = model
    for _ in range(max_depth):
        if hasattr(m, 'get_rope_index'):
            return m
        if hasattr(m, 'model'):
            m = m.model
        else:
            break
    raise AttributeError('no get_rope_index found under %s' % type(model).__name__)


def find_visual(model, max_depth=6):
    """Same idea for the vision tower."""
    m = model
    for _ in range(max_depth):
        if hasattr(m, 'visual'):
            return m.visual
        if hasattr(m, 'model'):
            m = m.model
        else:
            break
    raise AttributeError('no visual found under %s' % type(model).__name__)


class PosOverride:
    """rope=True -> ViT RoPE ids translated; abs_=True -> absolute pos_embed taken from full grid."""

    def __init__(self, visual, offset, merge, full_grid_thw, rows_in_full,
                 num_grid_per_side, rope=True, abs_=True):
        self.visual, self.offset, self.merge = visual, offset, merge
        self.full_grid_thw, self.rows_in_full = full_grid_thw, rows_in_full
        self.num_grid, self.rope, self.abs_ = num_grid_per_side, rope, abs_
        self.h = None

    def __enter__(self):
        by0, bx0 = self.offset

        def pre(mod, args, kwargs):
            gthw = kwargs.get('grid_thw', args[1] if len(args) > 1 else None)
            if self.rope:
                # by0/bx0 are BLOCK units; position_ids are PATCH units -> scale by merge.
                pos = get_vision_position_ids(gthw, self.merge).clone()
                pos[:, 0] += by0 * self.merge
                pos[:, 1] += bx0 * self.merge
                kwargs['position_ids'] = pos
            if self.abs_:
                idx, w = get_vision_interpolation_indices_and_weights(
                    self.full_grid_thw.to(gthw.device), num_grid_per_side=self.num_grid,
                    mode=self.visual.interpolation_mode,
                    align_corners=self.visual.interpolation_align_corners,
                    spatial_merge_size=self.merge)
                r = self.rows_in_full.to(idx.device)
                kwargs['interp_indices'] = idx[r]
                kwargs['interp_weights'] = w[r]
            return args, kwargs

        self.h = self.visual.register_forward_pre_hook(pre, with_kwargs=True)
        return self

    def __exit__(self, *a):
        self.h.remove()


class LlmPosOffset:
    """Shift the image tokens' mrope (h, w) so the crop sits at its true place relative to text.

    Wrapping get_rope_index rather than passing position_ids to generate keeps `rope_deltas`
    consistent across decode steps.
    """

    def __init__(self, model, offset):
        self.model, self.offset, self.orig = model, offset, None

    def __enter__(self):
        by0, bx0 = self.offset
        inner = find_rope_owner(self.model)
        self._inner = inner
        self.orig = inner.get_rope_index

        def patched(input_ids=None, mm_token_type_ids=None, *a, **kw):
            pos, deltas = self.orig(input_ids, mm_token_type_ids, *a, **kw)
            if mm_token_type_ids is not None:
                m = (mm_token_type_ids == 1)       # text=0, image=1, video=2
                pos[1][m] += by0
                pos[2][m] += bx0
            return pos, deltas

        inner.get_rope_index = patched
        return self

    def __exit__(self, *a):
        self._inner.get_rope_index = self.orig


def assert_rows_in_full(geo, full_grid_thw, crop_gw, crop_gh, rows_in_full, offset):
    """Translating the crop's own position ids must equal subsetting the full grid's.

    If these disagree, rows_in_full is wrong -- and the absolute pos_embed override, which
    subsets the interpolation tables by exactly those rows, is wrong in the same silent way.
    """
    by0, bx0 = offset
    crop_thw = torch.tensor([[1, crop_gh, crop_gw]])
    translated = get_vision_position_ids(crop_thw, geo.merge).clone()
    translated[:, 0] += by0 * geo.merge
    translated[:, 1] += bx0 * geo.merge
    subset = get_vision_position_ids(full_grid_thw, geo.merge)[rows_in_full]
    assert torch.equal(translated, subset), 'rows_in_full mapping is wrong'
    return True
