"""Build the 46k training mix as local JSONL + JPEGs, by STREAMING.

Why streaming: `#N` in ms-swift samples *after* the whole split is downloaded
(swift/dataset/dataset_meta.py:120). Multimodal-Fatima/VQAv2_train is 443k rows with embedded
images -- taking 12k the normal way pulled 80 GB before it was stopped. Streaming reads only the
shards it needs.

Splits: train AND validation both feed the mix. Evaluation is on separate benchmarks
(MMMU, MMMU-Pro, MMStar, POPE, CHAIR, ROPE), none of which overlap these sources.

AI2D is absent: no usable train split exists (lmms-lab/ai2d is the 3,088-row benchmark set,
lhndzn/AI2D relabels it "train", tejas21private/ai2d_dataset is image+description, not QA).

  python build_mix.py --out $WORK/data/mix
"""
from __future__ import annotations

import argparse, json, os, sys

os.environ.setdefault('WORK', os.path.abspath(os.environ.get('LOOK_WORK', 'work')))
os.environ.setdefault('HF_HOME', os.environ['WORK'] + '/cache/huggingface')
os.environ.setdefault('PYTHONNOUSERSITE', '1')

import warnings, logging                                      # noqa: E402
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
from datasets import load_dataset                             # noqa: E402
from PIL import Image                                         # noqa: E402


def _msg(q, a):
    return [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': a}]


def vqav2(r):
    a = r.get('multiple_choice_answer')
    if not a:
        ans = r.get('answers') or []
        a = max(set(ans), key=ans.count) if ans else None
    return (_msg(r['question'], str(a)), [r['image']]) if a and r.get('question') else None


def textvqa(r):
    ans = r.get('answers') or []
    if not ans or not r.get('question'):
        return None
    return _msg(r['question'], max(set(ans), key=ans.count)), [r['image']]


def chartqa(r):
    lab = r.get('label')
    lab = lab[0] if isinstance(lab, list) and lab else lab
    return (_msg(r['query'], str(lab)), [r['image']]) if lab and r.get('query') else None


def aokvqa(r):
    ch, i = r.get('choices') or [], r.get('correct_choice_idx')
    if not ch or i is None or not r.get('question'):
        return None
    return _msg(r['question'], str(ch[i])), [r['image']]


def scienceqa(r):
    # ~47% of ScienceQA has no image; the mix only wants the multimodal half.
    if r.get('image') is None:
        return None
    ch, i = r.get('choices') or [], r.get('answer')
    if not ch or i is None:
        return None
    q = r['question'] + (f"\n{r['hint']}" if r.get('hint') else '')
    a = (r.get('solution') or '').strip()
    a = f'{a}\nSo the final answer is: {ch[i]}' if a else str(ch[i])
    return _msg(q, a), [r['image']]


def mm_wrapped(r):
    """nimapourjafar/mm_*: data=[{data,modality,role}], images=[{bytes,path}]."""
    msgs, used = [], 0
    for t in (r.get('data') or []):
        role = 'assistant' if t.get('role') == 'assistant' else 'user'
        if t.get('modality') == 'image':
            piece, used = '<image>', used + 1
        else:
            piece = (t.get('data') or '').strip()
        if not piece:
            continue
        if msgs and msgs[-1]['role'] == role:
            msgs[-1]['content'] = (msgs[-1]['content'] + '\n' + piece).strip()
        else:
            msgs.append({'role': role, 'content': piece})
    if len(msgs) < 2 or msgs[-1]['role'] != 'assistant':
        return None
    imgs = (r.get('images') or [])[:used] or (r.get('images') or [])
    return msgs, imgs


SPECS = [
    dict(name='vqav2',     src='Multimodal-Fatima/VQAv2_train', splits=['train'],              n=12000, fn=vqav2),
    dict(name='textvqa',   src='lmms-lab/textvqa',              splits=['train', 'validation'], n=6000, fn=textvqa),
    dict(name='visual7w',  src='nimapourjafar/mm_visual7w',     splits=['train'],               n=6000, fn=mm_wrapped),
    dict(name='aokvqa',    src='HuggingFaceM4/A-OKVQA',         splits=['train', 'validation'], n=5000, fn=aokvqa),
    dict(name='chartqa',   src='HuggingFaceM4/ChartQA',         splits=['train', 'val'],        n=5000, fn=chartqa),
    dict(name='textcaps',  src='nimapourjafar/mm_textcaps',     splits=['train'],               n=4000, fn=mm_wrapped),
    dict(name='scienceqa', src='derek-thomas/ScienceQA',        splits=['train', 'validation'], n=4000, fn=scienceqa),
    dict(name='geomverse', src='nimapourjafar/mm_geomverse',    splits=['train'],               n=4000, fn=mm_wrapped),
]


def to_pil(x):
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, dict):
        import io
        if x.get('bytes'):
            return Image.open(io.BytesIO(x['bytes']))
        if x.get('path'):
            return Image.open(x['path'])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.environ['WORK'] + '/data/mix')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--buffer', type=int, default=3000)
    ap.add_argument('--only', default=None, help='comma-separated subset of dataset names')
    args = ap.parse_args()
    only = set(args.only.split(',')) if args.only else None
    os.makedirs(args.out, exist_ok=True)

    totals = {}
    for spec in SPECS:
        if only and spec['name'] not in only:
            continue
        name, target = spec['name'], spec['n']
        img_dir = os.path.join(args.out, 'images', name)
        os.makedirs(img_dir, exist_ok=True)
        path = os.path.join(args.out, f'{name}.jsonl')
        kept = skipped = 0
        with open(path, 'w') as out:
            for split in spec['splits']:
                if kept >= target:
                    break
                try:
                    ds = load_dataset(spec['src'], split=split, streaming=True)
                    ds = ds.shuffle(seed=args.seed, buffer_size=args.buffer)
                except Exception as e:
                    print(f'  {name}/{split}: FAIL {type(e).__name__}: {str(e)[:90]}', flush=True)
                    continue
                for row in ds:
                    if kept >= target:
                        break
                    try:
                        got = spec['fn'](row)
                    except Exception:
                        got = None
                    if not got:
                        skipped += 1
                        continue
                    msgs, imgs = got
                    pils = [p for p in (to_pil(i) for i in imgs) if p is not None]
                    if not pils:
                        skipped += 1
                        continue
                    paths = []
                    for j, im in enumerate(pils):
                        fp = os.path.join(img_dir, f'{kept:06d}_{j}.jpg')
                        im.convert('RGB').save(fp, 'JPEG', quality=90)
                        paths.append(fp)
                    out.write(json.dumps({'messages': msgs, 'images': paths},
                                         ensure_ascii=False) + '\n')
                    kept += 1
                    if kept % 1000 == 0:
                        print(f'  {name}: {kept}/{target}', flush=True)
        totals[name] = kept
        print(f'{name:<10} kept={kept:<6} target={target:<6} skipped={skipped} -> {path}', flush=True)

    print('\n=== mix ===')
    for k, v in totals.items():
        print(f'  {k:<10} {v}')
    print(f'  {"TOTAL":<10} {sum(totals.values())}')


if __name__ == '__main__':
    main()
