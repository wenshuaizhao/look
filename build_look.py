"""Generate look-task samples from the mix images, in ms-swift's native grounding format.

Each sample is: crop image + a look prompt + `<bbox>` as the answer, where `objects.bbox` holds
the window in **norm1 canvas coordinates**.

Why norm1: the image handed to the model is the CROP, but the label is in CANVAS coordinates.
ms-swift's normalize_bbox divides by the image's own size (base.py:309), which would be the crop
-- wrong. With `bbox_type='norm1'` it uses width=height=1 (base.py:318), so the stored 0..1 value
is simply scaled to the model's convention (x1000 for Qwen3-VL). Canvas-relative, crop-independent.

`objects.look_offset` = [by0, bx0, full_gh, full_gw] rides along untouched (normalize_bbox only
consumes bbox/width/height/bbox_type/image_id) and is what LookTemplate uses to install the
position overrides at forward time.

  python build_look.py --src $WORK/data/mix --out $WORK/data/look --n 46000
"""
from __future__ import annotations

import argparse, json, os, sys

os.environ.setdefault('WORK', os.path.abspath(os.environ.get('LOOK_WORK', 'work')))
os.environ.setdefault('MODELSCOPE_CACHE', os.environ['WORK'] + '/cache/modelscope')
os.environ.setdefault('HF_HOME', os.environ['WORK'] + '/cache/huggingface')
os.environ.setdefault('MAX_PIXELS', '1003520')
os.environ.setdefault('PYTHONNOUSERSITE', '1')

import numpy as np                                             # noqa: E402
from PIL import Image                                          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from looktask import Geometry, PROMPTS, make_look_sample       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.environ['WORK'] + '/data/mix')
    ap.add_argument('--out', default=os.environ['WORK'] + '/data/look')
    ap.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    ap.add_argument('--n', type=int, default=46000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--drop-groups', default='', help='comma-separated prompt groups to exclude')
    args = ap.parse_args()

    from swift import get_processor
    geo = Geometry(get_processor(args.model).image_processor, int(os.environ['MAX_PIXELS']))
    print(f'patch={geo.patch} merge={geo.merge} cell={geo.cell}', flush=True)

    drop = {g for g in args.drop_groups.split(',') if g}
    groups = [g for g in PROMPTS if g not in drop]
    prompts = [(g, p) for g in groups for p in PROMPTS[g]]
    print(f'{len(prompts)} prompts across {len(groups)} groups', flush=True)

    srcs = sorted(f for f in os.listdir(args.src) if f.endswith('.jsonl'))
    images = []
    for f in srcs:
        for line in open(os.path.join(args.src, f)):
            for p in json.loads(line)['images']:
                images.append(p)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(images)
    images = images[:args.n]
    print(f'{len(images)} source images from {len(srcs)} datasets', flush=True)

    crop_dir = os.path.join(args.out, 'crops')
    os.makedirs(crop_dir, exist_ok=True)
    kept = skipped = 0
    with open(os.path.join(args.out, 'look.jsonl'), 'w') as out:
        for i, path in enumerate(images):
            try:
                s = make_look_sample(Image.open(path), geo, rng)
            except Exception:
                skipped += 1
                continue
            gw, gh, _, _ = geo.grids(*s.canvas)
            if min(s.crop_blocks) < 1:
                skipped += 1
                continue
            fp = os.path.join(crop_dir, f'{kept:06d}.jpg')
            s.crop.convert('RGB').save(fp, 'JPEG', quality=92)
            rw, rh = s.canvas
            x0, y0, x1, y1 = s.gt_px
            g, prompt = prompts[int(rng.integers(0, len(prompts)))]
            out.write(json.dumps({
                'messages': [{'role': 'user', 'content': prompt},
                             {'role': 'assistant', 'content': '<bbox>'}],
                'images': [fp],
                'objects': {'ref': [], 'bbox': [[x0 / rw, y0 / rh, x1 / rw, y1 / rh]],
                            'bbox_type': 'norm1',
                            'look_offset': [s.offset[0], s.offset[1], gh, gw]},
                'prompt_group': g,
            }, ensure_ascii=False) + '\n')
            kept += 1
            if kept % 2000 == 0:
                print(f'  {kept}/{len(images)}', flush=True)
    print(f'\nkept={kept} skipped={skipped} -> {args.out}/look.jsonl', flush=True)


if __name__ == '__main__':
    main()
