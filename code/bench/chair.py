"""CHAIR (Rohrbach et al. 2018) on COCO val2014: caption N images, count hallucinated objects.

  python -m bench.chair gen   --n 500 [--adapters CKPT] --out DIR     # writes DIR/captions.jsonl (resumable)
  python -m bench.chair score --out DIR                               # writes DIR/chair.json, prints metrics

Protocol: 500 random val2014 images (seed 0, fixed across arms), prompt
"Please describe this image in detail.", greedy, 512 new tokens -- the setting used by OPERA/HA-DPO
and most LVLM hallucination papers. GT object set per image = COCO instance categories +
objects mentioned in the 5 GT captions (original chair.py). Metrics:
  CHAIRs  = fraction of captions with >=1 hallucinated COCO object   (lower is better)
  CHAIRi  = hallucinated object mentions / all COCO object mentions  (lower is better)
  recall  = |mentioned ∩ GT| / |GT|  (coverage, so a model cannot win CHAIR by saying nothing)
  length  = mean words per caption
Port of LisaAnne/Hallucination/utils/chair.py; `pattern.en.singularize` (py2-only) is replaced by
the WordNet noun lemmatizer, everything else (synonym table, double words, qualifiers) is verbatim.
COCO val2014 is disjoint from every training image (mix uses train2014 / OpenImages / etc.).
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
from collections import Counter
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.vlm import WORK, append_jsonl, read_jsonl                  # noqa: E402

COCO = f'{WORK}/cache/coco'
PROMPT = 'Please describe this image in detail.'


# ------------------------------------------------------------------ scorer
class CHAIR:
    def __init__(self, imids):
        import nltk
        self.nltk = nltk
        self.lemma = nltk.stem.WordNetLemmatizer()
        self.imid_to_objects = {imid: set() for imid in imids}
        synonyms = [s.strip().split(', ') for s in open(f'{COCO}/chair/synonyms.txt') if s.strip()]
        self.mscoco_objects = set()
        self.inverse_synonym_dict = {}
        for syn in synonyms:
            self.mscoco_objects.update(syn)
            for s in syn:
                self.inverse_synonym_dict[s] = syn[0]
        coco_double_words = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light',
                             'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter',
                             'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket',
                             'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear', 'hair drier',
                             'potted plant', 'bow tie', 'laptop computer', 'stove top oven', 'home plate',
                             'train track']
        animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                        'animal', 'cub']
        vehicle_words = ['jet', 'train']
        self.double_word_dict = {w: w for w in coco_double_words}
        for a in animal_words:
            self.double_word_dict[f'baby {a}'] = a
            self.double_word_dict[f'adult {a}'] = a
        for v in vehicle_words:
            self.double_word_dict[f'passenger {v}'] = v
        self.double_word_dict['bow tie'] = 'tie'
        self.double_word_dict['toilet seat'] = 'toilet'
        self.double_word_dict['wine glas'] = 'wine glass'

    def singularize(self, w):
        return self.lemma.lemmatize(w, 'n')

    def caption_to_words(self, caption):
        words = [self.singularize(w) for w in self.nltk.word_tokenize(caption.lower())]
        i, double_words, idxs = 0, [], []
        while i < len(words):
            idxs.append(i)
            dw = ' '.join(words[i:i + 2])
            if dw in self.double_word_dict:
                double_words.append(self.double_word_dict[dw]); i += 2
            else:
                double_words.append(words[i]); i += 1
        words = double_words
        if 'toilet' in words and 'seat' in words:
            words = [w for w in words if w != 'seat']
        idxs = [idxs[k] for k, w in enumerate(words) if w in self.mscoco_objects]
        words = [w for w in words if w in self.mscoco_objects]
        node_words = [self.inverse_synonym_dict[w] for w in words]
        return words, node_words, idxs, double_words

    def get_annotations(self):
        inst = json.load(open(f'{COCO}/annotations/instances_val2014.json'))
        id_to_name = {c['id']: c['name'] for c in inst['categories']}
        for a in inst['annotations']:
            if a['image_id'] in self.imid_to_objects:
                self.imid_to_objects[a['image_id']].add(self.inverse_synonym_dict[id_to_name[a['category_id']]])
        caps = json.load(open(f'{COCO}/annotations/captions_val2014.json'))
        for a in caps['annotations']:
            if a['image_id'] in self.imid_to_objects:
                _, node_words, _, _ = self.caption_to_words(a['caption'])
                self.imid_to_objects[a['image_id']].update(node_words)

    def compute(self, rows):
        n_caps = n_halluc_caps = n_halluc_words = n_coco_words = 0
        recall_num = recall_den = 0
        lengths, out = [], []
        for r in rows:
            words, node_words, idxs, raw = self.caption_to_words(r['caption'])
            gt = self.imid_to_objects[r['image_id']]
            halluc = [(w, nw) for w, nw in zip(words, node_words) if nw not in gt]
            n_caps += 1
            n_halluc_caps += bool(halluc)
            n_halluc_words += len(halluc)
            n_coco_words += len(node_words)
            recall_num += len(set(node_words) & gt)
            recall_den += len(gt)
            lengths.append(len(raw))
            out.append({'image_id': r['image_id'], 'caption': r['caption'], 'gt': sorted(gt),
                        'generated': node_words, 'hallucinated': halluc})
        return {'n': n_caps,
                'CHAIRs': n_halluc_caps / max(n_caps, 1),
                'CHAIRi': n_halluc_words / max(n_coco_words, 1),
                'recall': recall_num / max(recall_den, 1),
                'length': sum(lengths) / max(n_caps, 1),
                'objects_per_caption': n_coco_words / max(n_caps, 1)}, out


# ------------------------------------------------------------------ cli
def image_ids(n, seed=0):
    inst = json.load(open(f'{COCO}/annotations/instances_val2014.json'))
    imgs = sorted(inst['images'], key=lambda x: x['id'])
    rng = random.Random(seed)
    rng.shuffle(imgs)
    return [(im['id'], f"{COCO}/val2014/{im['file_name']}") for im in imgs[:n]]


def gen(args):
    from bench.vlm import canonical, generate, load_model
    os.makedirs(args.out, exist_ok=True)
    path = f'{args.out}/captions.jsonl'
    done = {r['image_id'] for r in read_jsonl(path)}
    todo = [(i, p) for i, p in image_ids(args.n, args.seed) if i not in done]
    print(f'{len(done)} cached, {len(todo)} to caption', flush=True)
    if not todo:
        return
    model, processor, geo = load_model(args.model, args.adapters)
    for k, (imid, p) in enumerate(todo):
        cap = generate(model, processor, canonical(Image.open(p), geo), PROMPT, args.max_new_tokens)
        append_jsonl(path, {'image_id': imid, 'file': p, 'caption': cap})
        if k < 2 or k % 50 == 0:
            print(f'[{k}/{len(todo)}] {imid}: {cap[:160]!r}', flush=True)


def score(args):
    rows = read_jsonl(f'{args.out}/captions.jsonl')
    ev = CHAIR([r['image_id'] for r in rows])
    ev.get_annotations()
    metrics, per = ev.compute(rows)
    json.dump({'metrics': metrics, 'prompt': PROMPT, 'sentences': per},
              open(f'{args.out}/chair.json', 'w'), indent=1)
    top = Counter(nw for r in per for _, nw in r['hallucinated']).most_common(10)
    print('CHAIR ' + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}))
    print('most hallucinated:', top)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gen')
    g.add_argument('--n', type=int, default=500)
    g.add_argument('--seed', type=int, default=0)
    g.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    g.add_argument('--adapters', default=None)
    g.add_argument('--max_new_tokens', type=int, default=512)
    g.add_argument('--out', required=True)
    s = sub.add_parser('score')
    s.add_argument('--out', required=True)
    a = ap.parse_args()
    {'gen': gen, 'score': score}[a.cmd](a)
