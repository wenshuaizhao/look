"""LookTemplate: applies the 3b-both+llm position overrides during training.

A batch mixes normal-task rows and look rows. Look rows carry
`objects.look_offset = [by0, bx0, full_gh, full_gw]`; normal rows carry nothing and get the
identity, so the two mix freely with no branching in the data pipeline.

The overrides are installed in `forward_context` (swift/template/base.py:536), a context manager
wrapped around the model forward -- exactly the hook point for per-batch model patching.

Batching works because the vision tower receives every image of the batch in ONE call, with
per-image rows CONCATENATED in grid_thw order (verified: grid_thw [[1,4,6],[1,2,4]] -> 32 rows,
img0 = rows 0..23, img1 = rows 24..31). So the override is just concat([per-image tensors]).
"""
from __future__ import annotations

from contextlib import ExitStack, nullcontext
from typing import Any, Dict, List

import torch

# `transformers.vision_utils` is a 5.x refactor; the eval env pins 4.57.6 for vlmeval. The
# adapter's args.json re-imports this file on every `swift eval/infer`, so the module must load
# there too. Evaluation never enters the look branch (no look rows), so the overrides are optional.
try:
    from transformers.vision_utils import (get_vision_interpolation_indices_and_weights,
                                           get_vision_position_ids)
    _HAS_VISION_UTILS = True
except ImportError:                        # transformers < 5
    get_vision_interpolation_indices_and_weights = get_vision_position_ids = None
    _HAS_VISION_UTILS = False

import os as _os, sys as _sys
# --custom_register_path exec's this file as a standalone module (no parent package), so a
# relative import raises "attempted relative import with no known parent package". Make the
# package importable by path instead.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from looktask.positions import find_rope_owner, find_visual
from swift.template import register_template
from swift.template.templates.qwen import Qwen3VLTemplate, QwenTemplateMeta

LOOK_TEMPLATE = 'qwen3_vl_look'
_FIRED = {'n': 0}


# ---------------------------------------------------------------------------------------------
# ms-swift's dataset loader REBUILDS `objects` keeping only ref/bbox/bbox_type/image_id
# (swift/dataset/preprocessor/core.py:178-187), which silently strips `look_offset`. With it gone
# every offset arrives as [0,0,0,0], forward_context early-returns, and the look task trains as
# the 3a control (crop, reset positions) -- unlearnable, and indistinguishable from "look does
# not help". This module is imported via --custom_register_path before any dataset is processed,
# so patching here is early enough. Verified by pushing look.jsonl through swift.dataset.load_dataset.
# ---------------------------------------------------------------------------------------------
from swift.dataset.preprocessor.core import RowPreprocessor as _RP

_orig_check_objects = _RP._check_objects
LOOK_KEY = 'look_offset'


def _check_objects_keep_look(row):
    objects = row.get('objects')
    extra = objects.get(LOOK_KEY) if isinstance(objects, dict) else None
    _orig_check_objects(row)
    if extra is not None and isinstance(row.get('objects'), dict):
        row['objects'][LOOK_KEY] = list(extra)


_RP._check_objects = staticmethod(_check_objects_keep_look)


def _crop_rows_in_full(by0, bx0, crop_gw, crop_gh, full_gw, merge):
    pairs = []
    for ph in range(crop_gh):
        for pw in range(crop_gw):
            def row(p, q, gw):
                bh, mh = divmod(p, merge)
                bw, mw = divmod(q, merge)
                return ((bh * (gw // merge) + bw) * merge + mh) * merge + mw
            pairs.append((row(ph, pw, crop_gw), row(ph + by0 * merge, pw + bx0 * merge, full_gw)))
    return torch.tensor([f for _, f in sorted(pairs)], dtype=torch.long)


class _BatchedVisionOverride:
    """Per-image position ids + absolute pos_embed taps, concatenated over the batch."""

    def __init__(self, visual, offsets: torch.Tensor, merge: int, num_grid: int):
        self.visual, self.offsets, self.merge, self.num_grid = visual, offsets, merge, num_grid
        self.h = None

    def __enter__(self):
        def pre(mod, args, kwargs):
            gthw = kwargs.get('grid_thw', args[1] if len(args) > 1 else None)
            if gthw is None:
                return args, kwargs
            pos_parts, idx_parts, w_parts = [], [], []
            for i, (t, h, w) in enumerate(gthw.tolist()):
                one = torch.tensor([[t, h, w]], device=gthw.device)
                pos = get_vision_position_ids(one, self.merge).clone()
                off = self.offsets[i] if i < len(self.offsets) else torch.zeros(4, dtype=torch.long)
                by0, bx0, full_gh, full_gw = [int(v) for v in off]
                is_look = full_gh > 0 and full_gw > 0
                if is_look:
                    pos[:, 0] += by0 * self.merge
                    pos[:, 1] += bx0 * self.merge
                    full = torch.tensor([[1, full_gh, full_gw]], device=gthw.device)
                    idx, wt = get_vision_interpolation_indices_and_weights(
                        full, num_grid_per_side=self.num_grid, mode=self.visual.interpolation_mode,
                        align_corners=self.visual.interpolation_align_corners,
                        spatial_merge_size=self.merge)
                    rows = _crop_rows_in_full(by0, bx0, w, h, full_gw, self.merge).to(idx.device)
                    # Fail in Python with the numbers, not as an asynchronous CUDA device-side assert
                    # (job 20368495 died on rank 3 with "vectorized gather kernel index out of bounds").
                    n_full = full_gh * full_gw
                    if (len(rows) != h * w or int(rows.max()) >= n_full or int(rows.min()) < 0
                            or by0 * self.merge + h > full_gh or bx0 * self.merge + w > full_gw
                            or idx.shape[0] != n_full):
                        raise RuntimeError(
                            f'[LookTemplate] crop/full grid mismatch: image {i} grid_thw={[t, h, w]} '
                            f'offset(by0,bx0,gh,gw)={[by0, bx0, full_gh, full_gw]} rows={len(rows)} '
                            f'rows.max={int(rows.max())} idx.shape={tuple(idx.shape)} '
                            f'n_images={len(gthw)} n_offsets={len(self.offsets)}')
                    if _os.environ.get('LOOK_DEBUG'):
                        print(f'[LookTemplate/debug] image {i}: grid_thw={[t, h, w]} '
                              f'offset={[by0, bx0, full_gh, full_gw]} rows.max={int(rows.max())}/{n_full} '
                              f'pos.max={pos.max(0).values.tolist()}', flush=True)
                    idx, wt = idx[rows], wt[rows]
                else:
                    idx, wt = get_vision_interpolation_indices_and_weights(
                        one, num_grid_per_side=self.num_grid, mode=self.visual.interpolation_mode,
                        align_corners=self.visual.interpolation_align_corners,
                        spatial_merge_size=self.merge)
                pos_parts.append(pos)
                idx_parts.append(idx)
                w_parts.append(wt)
            kwargs['position_ids'] = torch.cat(pos_parts)
            kwargs['interp_indices'] = torch.cat(idx_parts)
            kwargs['interp_weights'] = torch.cat(w_parts)
            return args, kwargs

        self.h = self.visual.register_forward_pre_hook(pre, with_kwargs=True)
        return self

    def __exit__(self, *a):
        self.h.remove()


class _BatchedLlmOffset:
    """Shift each row's image tokens in the LLM mrope by that row's window offset."""

    def __init__(self, model, offsets: torch.Tensor):
        self.model, self.offsets, self.orig = model, offsets, None

    def __enter__(self):
        inner = find_rope_owner(self.model)
        self._inner = inner
        self.orig = inner.get_rope_index
        offsets = self.offsets

        def patched(input_ids=None, mm_token_type_ids=None, *a, **kw):
            pos, deltas = self.orig(input_ids, mm_token_type_ids, *a, **kw)
            if mm_token_type_ids is not None:
                for i in range(min(len(offsets), mm_token_type_ids.shape[0])):
                    by0, bx0, gh, gw = [int(v) for v in offsets[i]]
                    if gh <= 0:
                        continue
                    m = (mm_token_type_ids[i] == 1)
                    pos[1][i][m] += by0
                    pos[2][i][m] += bx0
            return pos, deltas

        inner.get_rope_index = patched
        return self

    def __exit__(self, *a):
        self._inner.get_rope_index = self.orig


class LookTemplate(Qwen3VLTemplate):

    def _encode(self, inputs) -> Dict[str, Any]:
        # read before super(): normalize_bbox mutates inputs.objects
        off = list((inputs.objects or {}).get('look_offset') or [0, 0, 0, 0])
        encoded = super()._encode(inputs)
        encoded['look_offset'] = off
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to=None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        res['look_offset'] = torch.tensor(
            [list(b.get('look_offset') or [0, 0, 0, 0]) for b in batch], dtype=torch.long)
        return res

    def forward_context(self, model, inputs):
        # pop: the model would reject an unexpected kwarg
        offsets = inputs.pop('look_offset', None)
        if offsets is None or not bool((offsets[:, 2] > 0).any()):
            return super().forward_context(model, inputs)
        if not _HAS_VISION_UTILS:
            raise RuntimeError('look rows present but transformers.vision_utils is unavailable '
                               '(transformers < 5): the position override cannot be applied.')
        # Log the first few activations: arm 2's original run trained with the override silently
        # inactive (look_offset stripped by the loader). Its presence in the log is the proof.
        _FIRED['n'] += 1
        if _FIRED['n'] <= 3:
            n_look = int((offsets[:, 2] > 0).sum())
            print(f'[LookTemplate] override ACTIVE (call {_FIRED["n"]}): {n_look}/{len(offsets)} '
                  f'look rows in batch, first offset={offsets[0].tolist()}', flush=True)
        base = self.get_base_model(model)
        visual = find_visual(base)
        vcfg = base.config.vision_config
        num_grid = int(getattr(vcfg, 'num_position_embeddings', 0) ** 0.5) or visual.num_grid_per_side
        stack = ExitStack()
        stack.enter_context(_BatchedVisionOverride(visual, offsets, self.processor.image_processor.merge_size, num_grid))
        stack.enter_context(_BatchedLlmOffset(base, offsets))
        return stack


register_template(QwenTemplateMeta(LOOK_TEMPLATE, template_cls=LookTemplate, default_system=None,
                                   thinking_prefix='<think>\n'))
