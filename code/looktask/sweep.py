"""Zero-shot look-task sweep: many images x window sizes x prompts.

Establishes the arm-0 baseline and checks the machinery at scale rather than on one example.

Usage:
  python -m looktask.sweep --n 50 --out sweep.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

os.environ.setdefault('WORK', os.path.abspath(os.environ.get('LOOK_WORK', 'work')))
os.environ.setdefault('MODELSCOPE_CACHE', os.environ['WORK'] + '/cache/modelscope')
os.environ.setdefault('HF_HOME', os.environ['WORK'] + '/cache/huggingface')
os.environ.setdefault('MAX_PIXELS', '1003520')
os.environ.setdefault('PYTHONNOUSERSITE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
from contextlib import ExitStack                                      # noqa: E402
from PIL import Image                                                 # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from looktask import (Geometry, LlmPosOffset, PosOverride, PROMPTS,   # noqa: E402
                      assert_rows_in_full, make_look_sample)

from looktask.metrics import BOX_RE, parse_box, iou


def image_paths(n, work):
    """COCO images already cached locally by prepare_data.py."""
    paths, seen = [], set()
    with open(f'{work}/data/refcoco_val.jsonl') as f:
        for line in f:
            p = json.loads(line)['images'][0]
            if p not in seen and os.path.exists(p):
                seen.add(p)
                paths.append(p)
            if len(paths) >= n:
                break
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    ap.add_argument('--adapters', default=None, help='LoRA checkpoint dir to evaluate')
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='sweep.json')
    args = ap.parse_args()
    work = os.environ['WORK']

    from swift import get_model_processor
    model, processor = get_model_processor(args.model, torch_dtype=torch.bfloat16,
                                           device_map='cuda:0')
    if args.adapters:
        from swift.tuners import Swift
        model = Swift.from_pretrained(model, args.adapters)
        print('loaded adapters:', args.adapters, flush=True)
    model.eval()
    from looktask.positions import find_visual
    visual = find_visual(model)
    geo = Geometry(processor.image_processor, int(os.environ['MAX_PIXELS']))
    vcfg = model.config.vision_config
    num_grid = int(getattr(vcfg, 'num_position_embeddings', 0) ** 0.5) or visual.num_grid_per_side
    print(f'{args.model} | patch={geo.patch} merge={geo.merge} cell={geo.cell} '
          f'num_grid_per_side={num_grid}', flush=True)

    @torch.no_grad()
    def run(img, prompt, ctxs, max_new_tokens=48):
        msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors='pt',
                           do_resize=False).to(model.device)
        with ExitStack() as st:
            for cm in ctxs:
                st.enter_context(cm)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        txt = processor.tokenizer.decode(out[0, inputs['input_ids'].shape[1]:],
                                         skip_special_tokens=False)
        for term in ('<|im_end|>', '<|endoftext|>'):      # chat terminators only
            txt = txt.replace(term, '')
        return txt.strip()

    rng = np.random.default_rng(args.seed)
    paths = image_paths(args.n, work)
    print(f'{len(paths)} images', flush=True)
    groups = list(PROMPTS)
    rows, checked = [], False

    for k, path in enumerate(paths):
        try:
            s = make_look_sample(Image.open(path), geo, rng)
        except Exception as e:
            print(f'  skip {os.path.basename(path)}: {type(e).__name__}', flush=True)
            continue
        gw, gh, _, _ = geo.grids(*s.canvas)
        cgw, cgh, _, _ = geo.grids(*s.crop.size)
        rows_in_full = geo.crop_rows_in_full(s.offset[0], s.offset[1], cgw, cgh, gw)
        full_thw = torch.tensor([[1, gh, gw]])
        if not checked:
            assert_rows_in_full(geo, full_thw, cgw, cgh, rows_in_full, s.offset)
            print('rows_in_full assertion passed', flush=True)
            checked = True

        group = groups[k % len(groups)]
        prompt = str(rng.choice(PROMPTS[group]))
        gt = s.gt_norm1000
        # The look task is 3b-both+llm: crop, with every positional quantity set to its
        # full-image value (absolute pos_embed + ViT RoPE + LLM mrope).
        ctxs = [PosOverride(visual, s.offset, geo.merge, full_thw, rows_in_full, num_grid),
                LlmPosOffset(model, s.offset)]
        txt = run(s.crop, prompt, ctxs)
        b = parse_box(txt)
        rows.append(dict(image=os.path.basename(path), cond='3b_look', group=group, prompt=prompt,
                         gt=gt, pred=b, iou=iou(gt, b), parsed=b is not None,
                         win_frac=round(s.crop_blocks[0] * s.crop_blocks[1]
                                        / (s.blocks[0] * s.blocks[1]), 4), text=txt))
        if (k + 1) % 10 == 0:
            print(f'  {k + 1}/{len(paths)}', flush=True)

    json.dump(rows, open(args.out, 'w'), indent=1)

    print('\n=== zero-shot look task (3b-both+llm) ===')
    print(f'n={len(rows):3d}  parse_rate={np.mean([x["parsed"] for x in rows]):.3f}  '
          f'mean IoU={np.mean([x["iou"] for x in rows]):.3f}  '
          f'IoU@0.5={np.mean([x["iou"] >= .5 for x in rows]):.3f}')
    parsed = [x for x in rows if x['parsed']]
    if parsed:
        print(f'IoU among parsed only: {np.mean([x["iou"] for x in parsed]):.3f} (n={len(parsed)})')
    import numpy as _np
    fr = _np.array([x['win_frac'] for x in rows])
    print(f'\ndegenerate "whole canvas" baseline on this sample: mean IoU = {fr.mean():.3f}')
    if len(rows) >= 30:
        q = _np.quantile(fr, [1/3, 2/3])
        print('by window area (IoU | parsed):')
        for lab, sel in [('small', fr <= q[0]), ('medium', (fr > q[0]) & (fr <= q[1])),
                         ('large', fr > q[1])]:
            r = [x for x, k in zip(rows, sel) if k]
            pr = [x['iou'] for x in r if x['parsed']]
            print(f'  {lab:<7} n={len(r):3d} area~{_np.mean([x["win_frac"] for x in r]):.3f} '
                  f'parse={_np.mean([x["parsed"] for x in r]):.2f} '
                  f'IoU|parsed={(_np.mean(pr) if pr else 0):.3f} '
                  f'(degenerate={_np.mean([x["win_frac"] for x in r]):.3f})')
    print('\nby prompt group (3b_look):')
    by = defaultdict(list)
    for x in rows:
        if x['cond'] == '3b_look':
            by[x['group']].append(x)
    for g in sorted(by, key=lambda g: -np.mean([x['iou'] for x in by[g]])):
        r = by[g]
        print(f'  {g:<10} n={len(r):3d} parse={np.mean([x["parsed"] for x in r]):.2f} '
              f'IoU={np.mean([x["iou"] for x in r]):.3f}')
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
