"""Referring-expression grounding: RefCOCO / RefCOCO+ / RefCOCOg test splits, GroundingME, Ref-L4, Ref-Adv-s,
FineCops-Ref (positives + negatives) and SOREC (testA/testB; images from SODA-D must be placed in $WORK/cache/sorec/images).

  python -m bench.ground gen   --sets refcoco_testA refcoco_testB ... [--adapters CKPT] [--n N] --out DIR
  python -m bench.ground score --out DIR

Sets (jxu124/refcoco{,plus,g} parquets in $WORK/cache/refcoco_hf; images = local COCO train2014 cache):
  refcoco_testA (1,975 refs / 5,657 expressions)   refcoco_testB (1,810 / 5,095)
  refcocoplus_testA (1,975 / 5,726)                refcocoplus_testB (1,798 / 4,889)
  refcocog_test (5,023 / 9,602)
  groundingme (1,005; lirang04/GroundingME, images embedded; Discriminative/Spatial/Limited/Rejection)
One generation per *expression* (the standard protocol); metric = Acc@0.5 (IoU >= 0.5 with the GT box in
original pixels) plus mean IoU and parse rate. Boxes are read in Qwen3-VL's norm-1000 convention (what the
look task trained); an 'abs' reading (pixels of the resized input) is stored per sample as a diagnostic.
GroundingME uses the authors' prompt and scoring verbatim (evaluate.py): a rejection sample is correct iff
the answer is {"bbox_2d": null} -- or unparseable, which their parser maps to [0,0,0,0] as well.
"""
from __future__ import annotations
import argparse
import glob
import io
import json
import os
import re
import sys
from collections import defaultdict
import pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.vlm import WORK, append_jsonl, read_jsonl                  # noqa: E402
from looktask.metrics import BOX_RE                                     # noqa: E402

REF_DIR = f'{WORK}/cache/refcoco_hf'
GME_DIR = f'{WORK}/cache/groundingme'
COCO_TRAIN = f'{WORK}/cache/modelscope/media_resources/coco2014/train2014'
REF_SETS = {'refcoco_testA': 'refcoco_test-', 'refcoco_testB': 'refcoco_testB-',
            'refcocoplus_testA': 'refcocoplus_test-', 'refcocoplus_testB': 'refcocoplus_testB-',
            'refcocog_test': 'refcocog_test-'}
REFL4_DIR = f'{WORK}/cache/refl4'
REFADV_PARQUET = f'{WORK}/cache/refadv/data/train-00000-of-00001.parquet'
FINECOPS_DIR = f'{WORK}/cache/finecops'
SOREC_DIR = f'{WORK}/cache/sorec'
EXTRA_SETS = ['refl4_test', 'refadv', 'finecops_pos', 'finecops_neg', 'sorec_testA', 'sorec_testB']
ALL_SETS = list(REF_SETS) + ['groundingme'] + EXTRA_SETS
REF_PROMPT = 'Locate the {expr} in the image and output its bounding box coordinates.'
GME_PROMPT = ("All spatial relationships are defined from the viewer's perspective, where 'front' means closer to "
              "the viewer and 'back' means farther from the viewer. Please provide the bounding box coordinate of "
              "the object the following statement describes:\n{description}\nEnsure that all details mentioned "
              "about the object are accurate. Provide at most one bounding box. If a matching object is found, "
              "provide its bounding box as a JSON in the format {{\"bbox_2d\": [x1, y1, x2, y2]}}. If no matching "
              "object is found, output {{\"bbox_2d\": null}}.")
NULL_RE = re.compile(r'"bbox(?:_2d|es)?"\s*:\s*(?:null|\[\s*\])', re.I)
# Ref-Adv authors' "direct" prompt (configs/qwen.yaml); Qwen3-VL read in norm-1000 as in their config.
REFADV_PROMPT = 'Locate every object that matches the description "{expr}" in the image. Report bbox coordinates in JSON format.'
# FineCops-Ref has no MLLM prompt of its own; its negatives need a refusal option, so both its positives and its
# negatives use the GroundingME-style prompt (one box or {"bbox_2d": null}). Positive acc is NOT comparable to RefCOPT.
FINECOPS_PROMPT = ('Please provide the bounding box coordinate of the object the following statement describes:\n{expr}\n'
                   'Ensure that all details mentioned about the object are accurate. Provide at most one bounding box. '
                   'If a matching object is found, provide its bounding box as a JSON in the format {{"bbox_2d": [x1, y1, x2, y2]}}. '
                   'If no matching object is found, output {{"bbox_2d": null}}.')


def load_set(name, n=None):
    rows = []
    if name == 'groundingme':
        d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{GME_DIR}/test-*.parquet'))])
        for _, r in d.iterrows():
            bb = r['bbox']
            rows.append({'id': f'groundingme/{int(r["id"])}', 'set': name, 'expr': r['description'],
                         'gt': [float(x) for x in bb] if bb is not None and len(bb) == 4 else [0, 0, 0, 0],
                         'l1': r['subtask_l1'], 'l2': r['subtask_l2'], 'w': int(r['width']), 'h': int(r['height']),
                         '_img': r['image']['bytes']})
    elif name in REF_SETS:
        f = glob.glob(f'{REF_DIR}/{REF_SETS[name]}*.parquet')[0]
        d = pd.read_parquet(f)
        for _, r in d.iterrows():
            info = json.loads(r['raw_image_info'])
            path = f'{COCO_TRAIN}/{os.path.basename(r["image_path"])}'
            for s in r['sentences']:
                rows.append({'id': f'{name}/{int(r["ref_id"])}/{int(s["sent_id"])}', 'set': name, 'expr': s['sent'],
                             'gt': [float(x) for x in r['bbox']], 'w': int(info['width']), 'h': int(info['height']),
                             'image': path})
    elif name == 'refl4_test':
        d = pd.read_parquet(f'{REFL4_DIR}/ref-l4-test.parquet')
        for _, r in d.iterrows():
            x, y, w, h = [float(v) for v in re.findall(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?', str(r['bbox']))[:4]]
            rows.append({'id': f'refl4_test/{r["id"]}', 'set': name, 'expr': r['caption'], 'gt': [x, y, x + w, y + h],
                         'w': int(r['width']), 'h': int(r['height']), 'image': f'{REFL4_DIR}/images/{r["file_name"]}',
                         'l1': r['file_name'].split('_')[0], 'area': float(r['bbox_area']), 'rewrite': bool(str(r['is_rewrite']) == 'True')})
    elif name == 'refadv':
        d = pd.read_parquet(REFADV_PARQUET)
        def nums(v):
            return [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?', str(v))]
        for _, r in d.iterrows():
            k = int(nums(r['distractors'])[0])
            rows.append({'id': f'refadv/{int(r["row_idx"])}', 'set': name, 'expr': r['normal_caption'],
                         'gt': nums(r['solution'])[:4], 'w': int(nums(r['width'])[0]), 'h': int(nums(r['height'])[0]),
                         'l1': f'distractors_{"2-3" if k <= 3 else "4-6" if k <= 6 else ">=7"}',
                         'negation': str(r['use_negation']).lower() == 'true', 'human': str(r['human_authored']).lower() == 'true',
                         '_img': r['image']['bytes']})
    elif name == 'finecops_pos':
        c = json.load(open(f'{FINECOPS_DIR}/test_expression_pos_coco_format.json'))
        imgs = {im['id']: im for im in c['images']}
        for a in c['annotations']:
            im = imgs[a['image_id']]
            x, y, w, h = json.loads(a['bbox']) if isinstance(a['bbox'], str) else a['bbox']
            rows.append({'id': f'finecops_pos/{a["id"]}', 'set': name, 'expr': a['caption'], 'gt': [x, y, x + w, y + h],
                         'w': int(im['width']), 'h': int(im['height']), 'image': f'{FINECOPS_DIR}/gqa_images/{im["file_name"]}',
                         'l1': f'level{a["level"]}', 'l2': a['tuple_type']})
    elif name == 'finecops_neg':
        a = json.load(open(f'{FINECOPS_DIR}/test_expression_all.json'))
        for k, x in a.items():
            if not x.get('negative_type'):
                continue
            iid = str(x['image_id'])
            path = (f'{FINECOPS_DIR}/neg_images/final_neg_images/{iid}.jpg' if iid.startswith('neg_')
                    else f'{FINECOPS_DIR}/gqa_images/{iid}.jpg')
            rows.append({'id': f'finecops_neg/{k}', 'set': name, 'expr': x['expression'], 'gt': [0, 0, 0, 0],
                         'w': 0, 'h': 0, 'image': path, 'l1': f'neg_{x["negative_cate"]}', 'l2': x['negative_type']})
    elif name in ('sorec_testA', 'sorec_testB'):
        d = json.load(open(f'{SOREC_DIR}/{name.split("_")[1]}.json'))
        for x in d:
            bx, by, bw, bh = x['bbox']
            rows.append({'id': f'{name}/{x["id"]}', 'set': name, 'expr': x['ref'], 'gt': [bx, by, bx + bw, by + bh],
                         'w': 0, 'h': 0, 'image': f'{SOREC_DIR}/images/{x["image"]}'})
    if n and n < len(rows):
        import random
        random.Random(0).shuffle(rows)
        rows = rows[:n]
    return rows


def iou(a, b):
    if a == [0, 0, 0, 0] or b == [0, 0, 0, 0]:
        return 1.0 if a == b else 0.0
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


BBOXES_RE = re.compile(r'"bboxes"\s*:\s*\[\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')


def parse_box(text):
    if NULL_RE.search(text):
        return 'null'
    for rx in [BBOXES_RE] + list(BOX_RE):
        m = rx.search(text)
        if m:
            return [float(m.group(i)) for i in range(1, 5)]
    return None


def to_pixels(box, w, h, cw, ch, mode):
    """norm1000 -> original pixels; abs -> pixels of the (cw x ch) resized input mapped to original."""
    if mode == 'norm1000':
        return [box[0] / 1000 * w, box[1] / 1000 * h, box[2] / 1000 * w, box[3] / 1000 * h]
    return [box[0] * w / cw, box[1] * h / ch, box[2] * w / cw, box[3] * h / ch]


def gen(args):
    from bench.vlm import canonical, generate, load_model
    os.makedirs(args.out, exist_ok=True)
    path = f'{args.out}/preds.jsonl'
    done = {r['id'] for r in read_jsonl(path)}
    todo = [r for s in args.sets for r in load_set(s, args.n) if r['id'] not in done]
    print(f'{len(done)} cached, {len(todo)} to run over {args.sets}', flush=True)
    if not todo:
        return
    model, processor, geo = load_model(args.model, args.adapters)
    for k, r in enumerate(todo):
        img = Image.open(io.BytesIO(r.pop('_img'))) if '_img' in r else Image.open(r['image'])
        if not r['w']:
            r['w'], r['h'] = img.size
        img = canonical(img, geo)
        if r['set'] == 'groundingme':
            prompt = GME_PROMPT.format(description=r['expr'])
        elif r['set'] == 'refadv':
            prompt = REFADV_PROMPT.format(expr=r['expr'])
        elif r['set'].startswith('finecops'):
            prompt = FINECOPS_PROMPT.format(expr=r['expr'])
        else:
            prompt = REF_PROMPT.format(expr=r['expr'])
        txt = generate(model, processor, img, prompt, args.max_new_tokens)
        r['cw'], r['ch'] = img.size
        r['response'] = txt
        append_jsonl(path, r)
        if k < 3 or k % 500 == 0:
            print(f'[{k}/{len(todo)}] {r["id"]} "{r["expr"][:60]}" gt={[round(x) for x in r["gt"]]} -> {txt[:120]!r}', flush=True)


def score(args):
    rows = read_jsonl(f'{args.out}/preds.jsonl')
    agg = defaultdict(lambda: {'n': 0, 'hit': 0, 'iou': 0.0, 'parsed': 0, 'hit_abs': 0})
    for r in rows:
        box = parse_box(r['response'])
        keys = [r['set']] + ([f"{r['set']}/{r['l1']}"] if r.get('l1') else [])
        if r['set'] == 'refadv' and r.get('negation'):
            keys.append('refadv/negation')
        if box == 'null' or box is None:
            pred = [0, 0, 0, 0]                       # GroundingME rule: null/unparseable == "no object"
            i_norm = i_abs = iou(pred, r['gt'])
            parsed = box == 'null'
        else:
            i_norm = iou(to_pixels(box, r['w'], r['h'], r['cw'], r['ch'], 'norm1000'), r['gt'])
            i_abs = iou(to_pixels(box, r['w'], r['h'], r['cw'], r['ch'], 'abs'), r['gt'])
            parsed = True
        for key in keys:
            a = agg[key]
            a['n'] += 1
            a['hit'] += i_norm >= 0.5
            a['hit_abs'] += i_abs >= 0.5
            a['iou'] += i_norm
            a['parsed'] += parsed
    out = {k: {'n': v['n'], 'acc@0.5': round(v['hit'] / v['n'], 4), 'mean_iou': round(v['iou'] / v['n'], 4),
               'parse_rate': round(v['parsed'] / v['n'], 4), 'acc@0.5_if_abs_pixels': round(v['hit_abs'] / v['n'], 4)}
           for k, v in sorted(agg.items())}
    json.dump(out, open(f'{args.out}/ground.json', 'w'), indent=1)
    for k, v in out.items():
        print(f'GROUND {k:28s} n={v["n"]:5d} acc@0.5={v["acc@0.5"]:.4f} mIoU={v["mean_iou"]:.4f} parse={v["parse_rate"]:.3f} (abs-pixel reading: {v["acc@0.5_if_abs_pixels"]:.4f})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gen')
    g.add_argument('--sets', nargs='+', default=ALL_SETS, choices=ALL_SETS)
    g.add_argument('--n', type=int, default=None, help='random subset per set (default: all)')
    g.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    g.add_argument('--adapters', default=None)
    g.add_argument('--max_new_tokens', type=int, default=64)
    g.add_argument('--out', required=True)
    s = sub.add_parser('score')
    s.add_argument('--out', required=True)
    a = ap.parse_args()
    {'gen': gen, 'score': score}[a.cmd](a)
