"""Dump full look-task inputs and replies for manual inspection.

(Was `inspect.py`. That name SHADOWED the stdlib `inspect` module: ms-swift's plugin loader puts
this directory on sys.path, and vlmeval's spawned worker then imported THIS file as `inspect`,
crashing every adapter evaluation before it started -- bug #13. Never name a module after stdlib.)

Writes a readable .md plus the crop image and a full-image overlay for every sample, so the
prompt, what the model actually saw, and its untruncated reply can be checked side by side.

  python -m looktask.dump_samples --n 12       # -> scripts/vlm_grounding/look_inspect/
"""
from __future__ import annotations

import argparse, json, os, re, sys
from contextlib import ExitStack

os.environ.setdefault('WORK', os.path.abspath(os.environ.get('LOOK_WORK', 'work')))
os.environ.setdefault('MODELSCOPE_CACHE', os.environ['WORK'] + '/cache/modelscope')
os.environ.setdefault('HF_HOME', os.environ['WORK'] + '/cache/huggingface')
os.environ.setdefault('MAX_PIXELS', '1003520')
os.environ.setdefault('PYTHONNOUSERSITE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402
from PIL import Image, ImageDraw                                         # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from looktask import (Geometry, LlmPosOffset, PosOverride, PROMPTS,      # noqa: E402
                      assert_rows_in_full, make_look_sample)
from looktask.sweep import image_paths, iou, parse_box                   # noqa: E402


def collapse_image_tokens(text, tok='<|image_pad|>'):
    """Show '<|image_pad|> x N' instead of N copies, so the prompt stays readable."""
    n = text.count(tok)
    return (text.replace(tok * n, f'{tok} x{n}') if n > 1 else text), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--seed', type=int, default=1)
    # default under the repo so results sit next to the code
    ap.add_argument('--out', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'look_inspect'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from swift import get_model_processor
    model, processor = get_model_processor(args.model, torch_dtype=torch.bfloat16, device_map='cuda:0')
    model.eval()
    from looktask.positions import find_visual
    visual = find_visual(model)
    geo = Geometry(processor.image_processor, int(os.environ['MAX_PIXELS']))
    vcfg = model.config.vision_config
    num_grid = int(getattr(vcfg, 'num_position_embeddings', 0) ** 0.5) or visual.num_grid_per_side

    rng = np.random.default_rng(args.seed)
    paths = image_paths(args.n, os.environ['WORK'])
    groups = list(PROMPTS)
    md = [f'# Look-task inspection — {args.model}\n',
          f'Task = **3b-both+llm**: the model receives ONLY the crop, with absolute `pos_embed`, '
          f'ViT RoPE and LLM mrope all set to the crop\'s full-image values.\n']

    for k, path in enumerate(paths):
        s = make_look_sample(Image.open(path), geo, rng)
        gw, gh, _, _ = geo.grids(*s.canvas)
        cgw, cgh, _, _ = geo.grids(*s.crop.size)
        rows_in_full = geo.crop_rows_in_full(s.offset[0], s.offset[1], cgw, cgh, gw)
        full_thw = torch.tensor([[1, gh, gw]])
        if k == 0:
            assert_rows_in_full(geo, full_thw, cgw, cgh, rows_in_full, s.offset)

        group = groups[k % len(groups)]
        prompt = str(rng.choice(PROMPTS[group]))
        msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[s.crop], return_tensors='pt',
                           do_resize=False).to(model.device)
        rendered, n_img = collapse_image_tokens(
            processor.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False))

        with ExitStack() as st:
            st.enter_context(PosOverride(visual, s.offset, geo.merge, full_thw, rows_in_full, num_grid))
            st.enter_context(LlmPosOffset(model, s.offset))
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        reply = processor.tokenizer.decode(out[0, inputs['input_ids'].shape[1]:],
                                           skip_special_tokens=True)
        box = parse_box(reply)

        crop_f, over_f = f'{k:02d}_crop.png', f'{k:02d}_context.png'
        s.crop.save(os.path.join(args.out, crop_f))
        ov = s.canon.copy(); d = ImageDraw.Draw(ov)
        d.rectangle(s.gt_px, outline=(0, 255, 0), width=6)
        ov.resize((s.canvas[0] // 2, s.canvas[1] // 2)).save(os.path.join(args.out, over_f))

        md += [f'\n---\n\n## Sample {k}  ·  `{os.path.basename(path)}`  ·  group `{group}`\n',
               f'- canvas {s.canvas}, blocks {s.blocks}; window blocks offset {s.offset}, '
               f'size {s.crop_blocks}; crop {s.crop.size} = {n_img} visual tokens',
               f'- **GT** px `{s.gt_px}` · norm1000 `{s.gt_norm1000}`',
               f'- **PRED** `{box}` · **IoU {iou(s.gt_norm1000, box):.3f}**\n',
               f'![crop]({crop_f}) ![context]({over_f})\n',
               '### Full input to the model\n```\n' + rendered + '\n```\n',
               '### Full reply\n```\n' + reply + '\n```']
        print(f'[{k}] {group:<10} IoU {iou(s.gt_norm1000, box):.3f}  reply {len(reply)} chars', flush=True)

    open(os.path.join(args.out, 'README.md'), 'w').write('\n'.join(md))
    print('\nwrote', os.path.join(args.out, 'README.md'))


if __name__ == '__main__':
    main()
