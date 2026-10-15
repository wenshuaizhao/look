"""One table over every evaluated arm/checkpoint under $WORK/runs/eval.

  python -m bench.summary [--tags arm0-base arm1-final ...] [--md]

Columns (arrow = direction of "better"): look parse rate and IoU among parsed ↑ (sweep, n=300; re-scored JSON when present), MMStar ↑ (rescored),
POPE ↑, HallusionBench ↑ (aAcc), MMMU val ↑ (rescored, deterministic extractor), MMMU-Pro ↑ (rescored),
ROPE ↑ (official = unparsed slot counts as miss) with parsed-slot fraction, CHAIRi ↓ and CHAIR recall ↑.
Missing benchmarks print as "-". The MMMU/MMMU-Pro/MMStar columns come from `rescored_<DS>.json`,
written by bench.rescore_mcq (run it first if a new arm landed).
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
import pandas as pd

WORK = os.environ.get('WORK', os.path.abspath(os.environ.get('LOOK_WORK', 'work')))
E = f'{WORK}/runs/eval'


def score_csv(tag, ds, pattern):
    fs = sorted(glob.glob(f'{E}/{tag}/{ds}/vlmeval/*/*/*{pattern}'))
    if not fs:
        return None
    d = pd.read_csv(fs[-1])
    return float(d.iloc[0][d.columns[1]])


def look_iou(tag):
    """(parse rate, IoU among parsed) from the sweep JSON; the re-scored file wins when present."""
    for name in ('look_sweep.rescored.json', 'look_sweep.json'):
        f = f'{E}/{tag}/{name}'
        if os.path.exists(f):
            rows = json.load(open(f))
            parsed = [r for r in rows if r.get('parsed')]
            return (len(parsed) / max(len(rows), 1),
                    sum(r['iou'] for r in parsed) / max(len(parsed), 1))
    return None, None


def rope(tag):
    f = f'{E}/{tag}/rope/preds.jsonl'
    if not os.path.exists(f):
        return None, None
    tot = hit = parsed = 0
    for line in open(f):
        r = json.loads(line)
        for p, g in zip(r['pred'], r['gt']):
            tot += 1
            hit += p == g
            parsed += bool(p)
    return hit / tot, parsed / tot


def chair(tag):
    f = f'{E}/{tag}/chair/chair.json'
    if not os.path.exists(f):
        return None, None
    m = json.load(open(f))['metrics']
    return m['CHAIRi'], m['recall']


def ground(tag):
    """Acc@0.5 per grounding set from bench.ground: RefCOCO/RefCOCO+ = mean of testA/testB, RefCOCOg, GroundingME,
    Ref-L4, Ref-Adv-s, FineCops positives (Acc@0.5) and negatives (rejection rate = 'acc' under the null rule), SOREC mean."""
    f = f'{E}/{tag}/ground/ground.json'
    if not os.path.exists(f):
        return [None] * 9
    g = json.load(open(f))
    def acc(k):
        return g[k]['acc@0.5'] if k in g else None
    def mean2(a, b):
        return None if a is None or b is None else (a + b) / 2
    return [mean2(acc('refcoco_testA'), acc('refcoco_testB')), mean2(acc('refcocoplus_testA'), acc('refcocoplus_testB')),
            acc('refcocog_test'), acc('groundingme'), acc('refl4_test'), acc('refadv'), acc('finecops_pos'),
            acc('finecops_neg'), mean2(acc('sorec_testA'), acc('sorec_testB'))]


def rescored(ds):
    f = f'{E}/rescored_{ds}.json'
    return json.load(open(f)) if os.path.exists(f) else {}


def fmt(x, nd=3):
    return '-' if x is None else (f'{x:.{nd}f}' if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='*', default=None)
    ap.add_argument('--md', action='store_true', help='markdown table')
    a = ap.parse_args()
    tags = a.tags or sorted(d for d in os.listdir(E) if os.path.isdir(f'{E}/{d}'))
    mmmu, mmpro, mmstar = rescored('MMMU_DEV_VAL'), rescored('MMMU_Pro_10c'), rescored('MMStar')
    cols = ['tag', 'look parse', 'look IoU (parsed) ↑', 'MMStar ↑', 'POPE ↑', 'HallB ↑', 'MMMU ↑', 'MMMU-Pro ↑', 'ROPE ↑', 'ROPE parsed', 'CHAIRi ↓', 'CHAIR recall ↑', 'RefCOCO ↑', 'RefCOCO+ ↑', 'RefCOCOg ↑', 'GME ↑', 'Ref-L4 ↑', 'Ref-Adv ↑', 'FineCops+ ↑', 'FineCops− rej ↑', 'SOREC ↑']
    rows = []
    for t in tags:
        ra, rp = rope(t)
        ci, cr = chair(t)
        lp, li = look_iou(t)
        gr = ground(t)
        rows.append([t, fmt(lp), fmt(li), fmt(mmstar.get(t, {}).get('acc')), fmt(score_csv(t, 'POPE', '_score.csv'), 1),
                     fmt(score_csv(t, 'HallusionBench', '_score.csv'), 1), fmt(mmmu.get(t, {}).get('acc')),
                     fmt(mmpro.get(t, {}).get('acc')), fmt(ra), fmt(rp), fmt(ci), fmt(cr)] + [fmt(x) for x in gr])
    if a.md:
        print('| ' + ' | '.join(cols) + ' |')
        print('|' + '---|' * len(cols))
        for r in rows:
            print('| ' + ' | '.join(r) + ' |')
    else:
        w = [max(len(str(x)) for x in col) for col in zip(cols, *rows)]
        for r in [cols] + rows:
            print('  '.join(str(x).ljust(k) for x, k in zip(r, w)))


if __name__ == '__main__':
    main()
