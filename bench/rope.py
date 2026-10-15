"""ROPE (Chen et al., NeurIPS 2024, "Multi-Object Hallucination in VLMs"), validation split,
Default multi-object setting: five red numbered boxes are drawn on the image, the model names the
class of each from a 50-class list, per-object exact-match accuracy.

  python -m bench.rope gen   [--adapters CKPT] [--n N] --out DIR   # DIR/preds.jsonl (resumable)
  python -m bench.rope score --out DIR                             # DIR/rope.json, prints metrics

Data: $WORK/cache/rope/validation (sled-umich/ROPE). Prompt, class lists, answer parsing and the
exact-match rule are copied from sled-group/moh (evaluation.py, constant.py); the one liberalisation
is that parsing tolerates newlines/extra spaces around "objK:" (the original crashed on any
deviation and then scored the whole entry as 5 misses -- we count exactly the same way: an
unparseable object slot is a miss). Subsets: Wild (1172), Hom (490), Het (246), Adv AAAAB (334),
Adv BAAAA (334) = 2576 entries x 5 objects. COCO images come from val2017 (a subset of val2014),
so no overlap with the train2014-based training mix.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.vlm import WORK, append_jsonl, read_jsonl                  # noqa: E402

ROPE = f'{WORK}/cache/rope'
FILES = {'AAAAB_data.json': 'adv_AAAAB', 'BAAAA_data.json': 'adv_BAAAA',
         'merged_heterogenous_data.json': 'het', 'merged_homogenous_data.json': 'hom',
         'merged_mixed_data.json': 'wild'}
PROMPT = ("Given the classes: <50 classes>. There are five red bounding boxes in this image. For each "
          "object within the red bounding boxes, identify its class from the list. Provide the class names "
          "in the format: 'obj1: <class1>, obj2: <class2>, obj3: <class3>, obj4: <class4>, obj5: <class5>', "
          "with no additional words or punctuation. For example: obj1: class, obj2: class, obj3: class, "
          "obj4: class, obj5: class. Replace class with the actual names of the classes from your class "
          "list. Ensure that no placeholders or brackets are used around the class names and that no "
          "additional words or punctuation are added to the response.")
VALIDATION_CLASSES = {
    'COCO': ['dining table', 'cow', 'spoon', 'potted plant', 'zebra', 'donut', 'traffic light', 'backpack',
             'vase', 'sports ball', 'person', 'apple', 'chair', 'bicycle', 'pizza', 'remote', 'tennis racket',
             'bench', 'boat', 'motorcycle', 'bottle', 'broccoli', 'suitcase', 'dog', 'orange', 'skis', 'cake',
             'umbrella', 'cell phone', 'bowl', 'wine glass', 'surfboard', 'handbag', 'sheep', 'tie', 'banana',
             'sink', 'kite', 'clock', 'cup', 'giraffe', 'bus', 'knife', 'bird', 'elephant', 'truck', 'book',
             'car', 'tv', 'carrot'],
    'ADE': ['plant', 'window', 'glass', 'windshield', 'vase', 'mirror', 'tree', 'ceiling', 'cabinet', 'rock',
            'person', 'bag', 'chair', 'door', 'light', 'food', 'arm', 'base', 'bottle', 'brand', 'grass', 'box',
            'pole', 'license plate', 'curtain', 'plate', 'mountain', 'table', 'head', 'building', 'balcony',
            'shelf', 'pillow', 'column', 'shutter', 'flowerpot', 'leg', 'apron', 'sign', 'picture', 'cushion',
            'flower', 'drawer', 'wheel', 'roof', 'book', 'price tag', 'car', 'rim', 'handle'],
}
OBJ_RE = re.compile(r'obj\s*([1-5])\s*:\s*([^,\n]*)', re.I)


def entries(n=None, seed=0):
    rows = []
    for f, subset in FILES.items():
        for k, e in enumerate(json.load(open(f'{ROPE}/validation/{f}'))):
            rows.append({'id': f'{subset}/{k}', 'subset': subset, 'source': e['data_source'],
                         'image': ROPE + e['folder'].replace('jpg', 'png'),
                         'gt': [o['name'].lower() for o in e['objects']]})
    if n and n < len(rows):                          # stratified: keep every subset's share
        import random
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:n]
    return rows


def parse(text):
    pred = [''] * 5
    for m in OBJ_RE.finditer(text):
        k = int(m.group(1)) - 1
        if not pred[k]:
            pred[k] = re.sub(r'\s+', ' ', m.group(2).strip().strip("'\"*.").strip()).lower()
    return pred


def gen(args):
    from bench.vlm import canonical, generate, load_model
    os.makedirs(args.out, exist_ok=True)
    path = f'{args.out}/preds.jsonl'
    done = {r['id'] for r in read_jsonl(path)}
    todo = [e for e in entries(args.n, args.seed) if e['id'] not in done]
    print(f'{len(done)} cached, {len(todo)} to run', flush=True)
    if not todo:
        return
    model, processor, geo = load_model(args.model, args.adapters)
    for k, e in enumerate(todo):
        prompt = PROMPT.replace('<50 classes>', ', '.join(VALIDATION_CLASSES[e['source']]))
        txt = generate(model, processor, canonical(Image.open(e['image']), geo), prompt, args.max_new_tokens)
        append_jsonl(path, {**e, 'response': txt, 'pred': parse(txt)})
        if k < 3 or k % 100 == 0:
            print(f'[{k}/{len(todo)}] {e["id"]} gt={e["gt"]} -> {txt[:200]!r}', flush=True)


def score(args):
    rows = read_jsonl(f'{args.out}/preds.jsonl')
    hit, tot = defaultdict(int), defaultdict(int)
    pos_hit, pos_tot = [0] * 5, [0] * 5
    unparsed = 0
    for r in rows:
        pred = parse(r['response'])
        unparsed += not any(pred)
        for i, (p, g) in enumerate(zip(pred, r['gt'])):
            ok = int(p == g)
            for key in ('all', r['subset'], r['source'], f'{r["subset"]}/{r["source"]}'):
                hit[key] += ok
                tot[key] += 1
            pos_hit[i] += ok
            pos_tot[i] += 1
    acc = {k: hit[k] / tot[k] for k in tot}
    metrics = {'n_entries': len(rows), 'n_objects': tot['all'], 'unparsed_entries': unparsed,
               'acc': acc, 'acc_by_position': [h / max(t, 1) for h, t in zip(pos_hit, pos_tot)],
               'prompt': PROMPT}
    json.dump(metrics, open(f'{args.out}/rope.json', 'w'), indent=1)
    print('ROPE ' + json.dumps({'n': len(rows), 'unparsed': unparsed,
                                **{k: round(acc[k], 4) for k in ('all', 'wild', 'hom', 'het', 'adv_AAAAB',
                                                                   'adv_BAAAA', 'COCO', 'ADE') if k in acc},
                                'by_pos': [round(x, 3) for x in metrics['acc_by_position']]}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gen')
    g.add_argument('--n', type=int, default=None, help='stratified random subset (default: all 2576)')
    g.add_argument('--seed', type=int, default=0)
    g.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    g.add_argument('--adapters', default=None)
    g.add_argument('--max_new_tokens', type=int, default=64)
    g.add_argument('--out', required=True)
    s = sub.add_parser('score')
    s.add_argument('--out', required=True)
    a = ap.parse_args()
    {'gen': gen, 'score': score}[a.cmd](a)
