"""Probe comparison at scale: which restriction of the visible region measures visual awareness?

Scales the one-image pilot of notebook 02 to N views. For every view the model is asked for the
bounding box of the region it can see, under seven conditions:

  full           the whole canonical canvas                         -> target: the whole canvas
  black / grey   full canvas, everything outside the window painted  -> target: the window
  zero / mean    full canvas, patch_embed output of every patch outside the window replaced
                 by zeros / by the mean embedding (their position embeddings are kept)
                                                                     -> target: the window
  crop_reset     only the crop, positions from (0, 0)                -> target: the window
  crop_restored  only the crop, every position set to its full-image value (the Look task:
                 absolute pos_embed + ViT RoPE + LLM M-RoPE, exactly as looktask.sweep)
                                                                     -> target: the window

Views, windows and prompts are drawn with the same code and seed as looktask.sweep, so the first
300 views are the paper's Look-task views and `crop_restored` on them must reproduce the sweep.

  python -m looktask.probe run --n 1000 --conds full black grey zero --out DIR [--adapters CKPT]
  python -m looktask.probe score --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault('WORK', os.path.abspath(os.environ.get('LOOK_WORK', 'work')))
os.environ.setdefault('MODELSCOPE_CACHE', os.environ['WORK'] + '/cache/modelscope')
os.environ.setdefault('HF_HOME', os.environ['WORK'] + '/cache/huggingface')
os.environ.setdefault('MAX_PIXELS', '1003520')
os.environ.setdefault('PYTHONNOUSERSITE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np                                                         # noqa: E402
from PIL import Image                                                      # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from looktask import Geometry, PROMPTS, make_look_sample                   # noqa: E402
from looktask.sweep import image_paths, iou, parse_box                     # noqa: E402

CONDS = ['full', 'black', 'grey', 'zero', 'mean', 'crop_reset', 'crop_restored']
FULL_BOX = [0, 0, 1000, 1000]


def views(n, geo, seed=0):
    """Identical sampling to looktask.sweep.main: one rng drives windows and prompt choice."""
    rng = np.random.default_rng(seed)
    groups = list(PROMPTS)
    out = []
    for k, path in enumerate(image_paths(n, os.environ['WORK'])):
        try:
            s = make_look_sample(Image.open(path), geo, rng)
        except Exception as e:                                             # same skip as the sweep
            print(f'  skip {os.path.basename(path)}: {type(e).__name__}', flush=True)
            continue
        group = groups[k % len(groups)]
        prompt = str(rng.choice(PROMPTS[group]))
        out.append((k, path, s, group, prompt))
    return out


def outside_rows(geo, s):
    """pixel_values rows (merge-block-major order) of every patch outside the window."""
    gw, gh, _, _ = geo.grids(*s.canvas)
    by0, bx0 = s.offset
    cbw, cbh = s.crop_blocks
    m = geo.merge
    inside = lambda ph, pw: by0 <= ph // m < by0 + cbh and bx0 <= pw // m < bx0 + cbw
    return sorted(geo.patch_row(ph, pw, gw) for ph in range(gh) for pw in range(gw) if not inside(ph, pw))


def painted(s, fill):
    out = Image.new('RGB', s.canon.size, fill)
    out.paste(s.crop, (s.gt_px[0], s.gt_px[1]))
    return out


class PatchEmbedMask:
    """Replace the patch_embed output rows of the excluded patches (notebook 02, conditions 2a/2b)."""

    def __init__(self, visual, rows, mode):
        self.visual, self.rows, self.mode, self.h = visual, rows, mode, None

    def _hook(self, module, args, output):
        import torch
        out = output.clone()
        idx = torch.tensor(self.rows, device=out.device, dtype=torch.long)
        fill = torch.zeros_like(out[0]) if self.mode == 'zero' else out.mean(0)
        out[idx] = fill.to(out.dtype)
        return out

    def __enter__(self):
        self.h = self.visual.patch_embed.register_forward_hook(self._hook)
        return self

    def __exit__(self, *a):
        self.h.remove()


def run(args):
    import torch
    from contextlib import ExitStack
    from swift import get_model_processor
    from looktask.positions import LlmPosOffset, PosOverride, find_visual

    os.makedirs(args.out, exist_ok=True)
    model, processor = get_model_processor(args.model, torch_dtype=torch.bfloat16, device_map='cuda:0')
    if args.adapters:
        from swift.tuners import Swift
        model = Swift.from_pretrained(model, args.adapters)
        print('loaded adapters:', args.adapters, flush=True)
    model.eval()
    visual = find_visual(model)
    geo = Geometry(processor.image_processor, int(os.environ['MAX_PIXELS']))
    num_grid = int(getattr(model.config.vision_config, 'num_position_embeddings', 0) ** 0.5) or visual.num_grid_per_side

    @torch.no_grad()
    def generate(img, prompt, ctxs):
        msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors='pt', do_resize=False).to(model.device)
        with ExitStack() as st:
            for cm in ctxs:
                st.enter_context(cm)
            out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
        t = processor.tokenizer.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=False)
        for term in ('<|im_end|>', '<|endoftext|>'):
            t = t.replace(term, '')
        return t.strip()

    vs = views(args.n, geo, args.seed)
    print(f'{len(vs)} views | conditions {args.conds}', flush=True)
    for cond in args.conds:
        path = f'{args.out}/probe_{cond}.jsonl'
        done = {json.loads(l)['k'] for l in open(path)} if os.path.exists(path) else set()
        with open(path, 'a') as f:
            for i, (k, p, s, group, prompt) in enumerate(vs):
                if k in done:
                    continue
                gt = FULL_BOX if cond == 'full' else s.gt_norm1000
                if cond == 'full':
                    img, ctxs = s.canon, []
                elif cond in ('black', 'grey'):
                    img, ctxs = painted(s, (0, 0, 0) if cond == 'black' else (128, 128, 128)), []
                elif cond in ('zero', 'mean'):
                    img, ctxs = s.canon, [PatchEmbedMask(visual, outside_rows(geo, s), cond)]
                elif cond == 'crop_reset':
                    img, ctxs = s.crop, []
                else:                                                      # crop_restored == looktask.sweep
                    gw, gh, _, _ = geo.grids(*s.canvas)
                    cgw, cgh, _, _ = geo.grids(*s.crop.size)
                    rows_in_full = geo.crop_rows_in_full(s.offset[0], s.offset[1], cgw, cgh, gw)
                    full_thw = torch.tensor([[1, gh, gw]])
                    img = s.crop
                    ctxs = [PosOverride(visual, s.offset, geo.merge, full_thw, rows_in_full, num_grid),
                            LlmPosOffset(model, s.offset)]
                txt = generate(img, prompt, ctxs)
                b = parse_box(txt)
                f.write(json.dumps(dict(k=k, image=os.path.basename(p), cond=cond, group=group, prompt=prompt,
                                        gt=gt, pred=b, iou=iou(gt, b), parsed=b is not None,
                                        win_frac=round(s.crop_blocks[0] * s.crop_blocks[1] / (s.blocks[0] * s.blocks[1]), 4),
                                        text=txt)) + '\n')
                f.flush()
                if (i + 1) % 100 == 0:
                    print(f'  {cond}: {i + 1}/{len(vs)}', flush=True)
        print(f'done {cond}', flush=True)


def score(args):
    out = {}
    for cond in CONDS:
        path = f'{args.out}/probe_{cond}.jsonl'
        if not os.path.exists(path):
            continue
        rs = [json.loads(l) for l in open(path)]
        p = [r for r in rs if r['parsed']]
        out[cond] = {'n': len(rs), 'parse': len(p) / len(rs),
                     'iou_parsed': float(np.mean([r['iou'] for r in p])) if p else 0.0,
                     'iou_all': float(np.mean([r['iou'] for r in rs])),
                     'success': float(np.mean([r['iou'] >= .5 for r in rs])),
                     'first300_iou_parsed': float(np.mean([r['iou'] for r in p if r['k'] < 300])) if p else 0.0}
        o = out[cond]
        print(f"PROBE {cond:14s} n={o['n']:4d} parse={o['parse']:.3f} IoU|parsed={o['iou_parsed']:.3f} "
              f"IoU all={o['iou_all']:.3f} success@0.5={o['success']:.3f} (first 300: {o['first300_iou_parsed']:.3f})")
    json.dump(out, open(f'{args.out}/probe.json', 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    r.add_argument('--adapters', default=None)
    r.add_argument('--n', type=int, default=1000)
    r.add_argument('--seed', type=int, default=0)
    r.add_argument('--conds', nargs='+', default=CONDS, choices=CONDS)
    r.add_argument('--out', required=True)
    s_ = sub.add_parser('score')
    s_.add_argument('--out', required=True)
    a = ap.parse_args()
    run(a) if a.cmd == 'run' else score(a)
