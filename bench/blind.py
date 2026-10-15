"""Blind-image existence probe: does the model actually read the image, or answer from the text prior?

  python -m bench.blind run [--adapters CKPT] [--n 1000] --out DIR
  python -m bench.blind score --out DIR

Idea (Wenshuai, 2026-09-24). Every RefCOCO expression names an object that IS in its image. Ask the model
whether the image contains it, but feed the *placeholder* in place of the visual features. The model then
has no visual evidence at all, so the honest answer is always "No"; any "Yes" is produced by the language
prior alone.

Three conditions over the same 1,000 (image, expression) pairs, so the blind number is interpretable:

| cond       | image features | expression            | label | reads as                         |
|------------|----------------|-----------------------|-------|----------------------------------|
| `blind`    | placeholder    | the image's own       | No    | hallucination with no evidence   |
| `real`     | real           | the image's own       | Yes   | can it say Yes when it can see   |
| `mismatch` | real           | another image's       | No    | does it verify what it sees      |

`real` is the control that stops a model that just always answers No from looking perfect; the headline
statistic is the **gap** P(Yes | real) - P(Yes | blind). `mismatch` labels are approximate: a referring
expression from another image is usually but not always false of this one, so read it as a trend.

`--full` runs every RefCOCO testA+testB expression (10,752) instead of a 1,000-image sample.

How the blinding works: `get_image_features` is replaced so that it returns the `<|image_pad|>` embedding
repeated once per visual token, and all-zero deepstack features. The scatter into `inputs_embeds` then
writes back exactly the placeholder embedding that was already there, and nothing is added at the deepstack
layers. Sequence length, attention mask, mrope position ids and the feature-count check are all identical
to the real run -- the *only* difference is that no image information enters. The vision tower is skipped,
so blind rows are also cheaper than real ones.

Answer is read from logits, not generated text: one forward pass, then the log-sum-exp of the "Yes"
variants against the "No" variants at the first answer position. That removes any parsing question and
gives a probability instead of a hard label.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
from collections import defaultdict
import torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.ground import load_set                                     # noqa: E402
from bench.vlm import WORK, append_jsonl, canonical, read_jsonl       # noqa: E402


def load_any(model_id, adapters=None):
    """Qwen models are pre-resized to the canonical geometry used in training and passed with
    do_resize=False, exactly as in every other bench here. Other families have their own fixed
    preprocessing (LLaVA 336px, Gemma 896px, ...), so their processor is left to do its own thing."""
    from swift import get_model_processor
    model, processor = get_model_processor(model_id, torch_dtype=torch.bfloat16, device_map='cuda:0')
    if adapters:
        from swift.tuners import Swift
        model = Swift.from_pretrained(model, adapters)
        print('loaded adapters:', adapters, flush=True)
    if 'gemma' in model_id.lower():
        # ms-swift's Gemma forward patch (swift/model/models/gemma.py:296) is stale for transformers 5.x:
        # it calls .to() on get_image_features(...).pooler_output, which Gemma-4 returns as a *tuple* of
        # per-image tensors; stock transformers does torch.cat(...) first. Restore the stock forward, which
        # ms-swift keeps as .origin_forward. Only matters for training-time concerns we do not use here.
        for mod in model.modules():
            if hasattr(mod, 'origin_forward'):
                mod.forward = mod.origin_forward
                print('restored stock forward on', type(mod).__name__, flush=True)
    model.eval()
    geo = None
    if 'qwen' in model_id.lower():
        from looktask.geometry import Geometry
        geo = Geometry(processor.image_processor, int(os.environ.get('MAX_PIXELS', 1003520)))
    return model, processor, geo

PROMPT = 'Does this image contain: {expr}? Answer with Yes or No.'
CONDS = ['blind', 'real', 'mismatch']


def yn_ids(tok):
    """Every single token whose *whole* surface form is yes/no, found by scanning the vocabulary.

    Bug #18: the previous version tokenized a list of variants and took the first id of each. On LLaMA-style
    SentencePiece tokenizers (LLaVA) both ' Yes' and ' no' begin with the bare space marker (id 29871), so that
    id landed in *both* groups; blinded, its logit dominates the log-sum-exp on each side equally and p(Yes)
    was pinned at ~0.499 for every sample -- LLaVA's blind yes_rate of 0.0000 was an artifact, not honesty.
    Scanning the vocabulary is tokenizer-agnostic and cannot produce an overlap."""
    yes, no = [], []
    for s_, i in tok.get_vocab().items():
        try:
            d = tok.convert_tokens_to_string([s_]).strip().lower()
        except Exception:
            continue
        (yes if d == 'yes' else no if d == 'no' else []).append(i)
    assert yes and no and not (set(yes) & set(no)), f'bad yes/no id sets: {len(yes)}/{len(no)}'
    return sorted(yes), sorted(no)


def find_mm_model(model):
    """The Qwen3VLModel instance that owns get_image_features (under PEFT/Swift wrappers)."""
    m = model
    for _ in range(8):
        if hasattr(m, 'get_image_features') and hasattr(m, 'get_placeholder_mask'):
            return m
        if hasattr(m, 'model'):
            m = m.model
        elif hasattr(m, 'base_model'):
            m = m.base_model
        else:
            break
    raise RuntimeError('could not locate the multimodal model owning get_image_features')


def _blank_like(out, pad_vec):
    """Same structure as the vision tower's output, but carrying no image information:
    the <image> placeholder embedding wherever features are scattered into inputs_embeds,
    and zeros for Qwen3-VL's deepstack contributions (a placeholder has no deepstack analogue)."""
    def pad_like(t):
        return pad_vec.to(t.dtype).to(t.device).expand_as(t).contiguous()
    if hasattr(out, 'pooler_output'):                     # Qwen2.5-VL / Qwen3-VL return a ModelOutput
        pooler = out.pooler_output
        new_pooler = ([pad_like(t) for t in pooler] if isinstance(pooler, (list, tuple)) else pad_like(pooler))
        deep = getattr(out, 'deepstack_features', None)
        if deep is not None:                              # Qwen3-VL also injects at layers 8/16/24
            return type('BlindOut', (), {'pooler_output': new_pooler,
                                         'deepstack_features': [torch.zeros_like(t) for t in deep]})()
        out.pooler_output = new_pooler
        return out
    if torch.is_tensor(out):                              # LLaVA, Gemma, InternVL
        return pad_like(out)
    if isinstance(out, (list, tuple)):
        return type(out)(_blank_like(t, pad_vec) for t in out)
    raise TypeError(f'unhandled get_image_features return type: {type(out)}')


class Blind:
    """Feed the image *placeholder* in place of the visual features.

    Every HF VLM we test (Qwen2.5-VL, Qwen3-VL, LLaVA, Gemma, InternVL) builds the multimodal sequence the
    same way: `get_image_features(...)` -> `input_ids == config.image_token_id` -> `masked_scatter`. Replacing
    the features with the placeholder's own embedding makes that scatter a no-op, so sequence length, attention
    mask, position ids and the feature-count check are all untouched and the only thing removed is the image.

    Qwen3-VL has a fast path that skips the vision tower entirely (shapes come from grid_thw). Every other
    architecture calls the real tower and then discards its output, which costs time but needs no per-model
    knowledge of feature shapes.
    """

    def __init__(self, model, pad_token_id):
        self.owner = find_mm_model(model)
        self.model = model
        self.pad_token_id = pad_token_id
        self.orig = None
        vcfg = getattr(model.config, 'vision_config', None)
        self.fast = vcfg is not None and getattr(vcfg, 'deepstack_visual_indexes', None) is not None

    def __enter__(self):
        owner, pad_id = self.owner, self.pad_token_id
        self.orig = orig = owner.get_image_features
        embed = owner.get_input_embeddings()
        pad_vec = embed(torch.tensor([pad_id], device=next(self.model.parameters()).device))[0].detach()

        if self.fast:
            vcfg = self.model.config.vision_config
            n_deep, merge = len(vcfg.deepstack_visual_indexes), vcfg.spatial_merge_size

            def blinded(pixel_values, image_grid_thw, return_dict=True, **kw):
                dev, dt = pixel_values.device, pixel_values.dtype
                pv = pad_vec.to(dev, dt)
                per_image = [int(t * h * w) // (merge * merge) for t, h, w in image_grid_thw.tolist()]
                pooler = [pv.expand(n, pv.shape[-1]).contiguous() for n in per_image]
                deep = [torch.zeros(sum(per_image), pv.shape[-1], device=dev, dtype=dt) for _ in range(n_deep)]
                return type('BlindOut', (), {'pooler_output': pooler, 'deepstack_features': deep})()
        else:
            def blinded(*a, **kw):
                return _blank_like(orig(*a, **kw), pad_vec)

        owner.get_image_features = blinded
        return self

    def __exit__(self, *a):
        self.owner.get_image_features = self.orig


@torch.no_grad()
def p_yes(model, processor, img, prompt, yes_ids, no_ids, blind_ctx=None, native_geo=True):
    msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    kw = {'do_resize': False} if native_geo else {}
    inputs = processor(text=[text], images=[img], return_tensors='pt', **kw).to(model.device)
    if blind_ctx is None:
        logits = model(**inputs).logits[0, -1].float()
    else:
        with blind_ctx:
            logits = model(**inputs).logits[0, -1].float()
    ly = torch.logsumexp(logits[yes_ids], 0)
    ln = torch.logsumexp(logits[no_ids], 0)
    return float(torch.sigmoid(ly - ln)), int(logits.argmax())


def samples(n, seed=0, full=False):
    """Sampled mode: n distinct RefCOCO test images, one expression each.
    Full mode: every expression of RefCOCO testA + testB (10,752), grouped by image so the image cache hits.
    Each row also carries a mismatched expression taken from a row half the list away, always a different image."""
    rows = load_set('refcoco_testA') + load_set('refcoco_testB')
    if full:
        picked = sorted(rows, key=lambda r: (r['image'], r['id']))
    else:
        by_img, order = {}, []
        for r in rows:
            if r['image'] not in by_img:
                by_img[r['image']] = r
                order.append(r['image'])
        rng = random.Random(seed)
        rng.shuffle(order)
        picked = [by_img[p] for p in order[:n]]
    half = len(picked) // 2
    for i, r in enumerate(picked):
        j = (i + half) % len(picked)
        while picked[j]['image'] == r['image']:          # never pair an image with itself
            j = (j + 1) % len(picked)
        r['mismatch_expr'] = picked[j]['expr']
    return picked


def run(args):
    os.makedirs(args.out, exist_ok=True)
    path = f'{args.out}/preds.jsonl'
    done = {(r['id'], r['cond']) for r in read_jsonl(path)}
    rows = samples(args.n, args.seed, full=args.full)
    todo = [(r, c) for r in rows for c in CONDS if (r['id'], c) not in done]
    print(f'{len(done)} cached, {len(todo)} forwards to run', flush=True)
    if not todo:
        return
    model, processor, geo = load_any(args.model, args.adapters)
    tok = processor.tokenizer
    yes_ids, no_ids = yn_ids(tok)
    print('yes ids', yes_ids, [tok.decode([i]) for i in yes_ids], flush=True)
    print('no  ids', no_ids, [tok.decode([i]) for i in no_ids], flush=True)
    cfg = model.config
    pad_id = next(getattr(cfg, k) for k in ('image_token_id', 'image_token_index') if getattr(cfg, k, None) is not None)
    print('image placeholder token:', pad_id, repr(tok.decode([pad_id])), flush=True)
    blind_ctx = Blind(model, pad_id)
    cache = {}
    for k, (r, cond) in enumerate(todo):
        if r['image'] not in cache:
            cache.clear()
            im = Image.open(r['image'])
            cache[r['image']] = canonical(im, geo) if geo is not None else im.convert('RGB')
        img = cache[r['image']]
        expr = r['mismatch_expr'] if cond == 'mismatch' else r['expr']
        pv, top = p_yes(model, processor, img, PROMPT.format(expr=expr), yes_ids, no_ids,
                        blind_ctx if cond == 'blind' else None, native_geo=geo is not None)
        append_jsonl(path, {'id': r['id'], 'image': r['image'], 'cond': cond, 'expr': expr,
                            'label': 'Yes' if cond == 'real' else 'No', 'p_yes': round(pv, 5),
                            'top_token': tok.decode([top])})
        if k < 6 or k % 500 == 0:
            print(f'[{k}/{len(todo)}] {cond:8s} p(Yes)={pv:.3f} top={tok.decode([top])!r} "{expr[:50]}"', flush=True)


def score(args):
    rows = read_jsonl(f'{args.out}/preds.jsonl')
    agg = defaultdict(list)
    for r in rows:
        agg[r['cond']].append(r)
    out = {}
    for cond in CONDS:
        rs = agg.get(cond, [])
        if not rs:
            continue
        yes = [r['p_yes'] > 0.5 for r in rs]
        answered = sum(str(r.get('top_token', '')).strip().lower() in ('yes', 'no') for r in rs)
        out[cond] = {'n': len(rs), 'yes_rate': round(sum(yes) / len(rs), 4),
                     'answered_yn': round(answered / len(rs), 4),
                     'mean_p_yes': round(sum(r['p_yes'] for r in rs) / len(rs), 4),
                     'acc': round(sum((r['p_yes'] > 0.5) == (r['label'] == 'Yes') for r in rs) / len(rs), 4)}
    if 'real' in out and 'blind' in out:
        out['gap_real_minus_blind'] = round(out['real']['yes_rate'] - out['blind']['yes_rate'], 4)
    json.dump(out, open(f'{args.out}/blind.json', 'w'), indent=1)
    print('BLIND ' + json.dumps(out))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('run')
    g.add_argument('--n', type=int, default=1000)
    g.add_argument('--seed', type=int, default=0)
    g.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    g.add_argument('--adapters', default=None)
    g.add_argument('--full', action='store_true', help='every RefCOCO test expression instead of an n-image sample')
    g.add_argument('--out', required=True)
    s = sub.add_parser('score')
    s.add_argument('--out', required=True)
    a = ap.parse_args()
    {'run': run, 'score': score}[a.cmd](a)
