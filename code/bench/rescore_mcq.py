"""Deterministic re-scoring of vlmeval multiple-choice results (MMMU, MMMU-Pro, MMStar).

Why: vlmeval scores MCQ answers with a rule-based "prefetch" (the option letter must be among the
last 5 tokens and appear exactly once) and otherwise asks a GPT judge. We run without a judge, so
every prefetch failure is counted as WRONG. The base model answers MMMU with ~1,200-char
chain-of-thought ("... **Correct Answer: C. Nucleus**"), and 88% of its answers were unscored
(reported 0.08); the SFT arms end with "So the final answer is: C. 1/4" and lost 29-38%. The arm
differences were therefore mostly differences in answer verbosity, not knowledge.

This script re-reads `*_openai_result.xlsx` (prediction + options + answer) and extracts the
answer with fixed rules, identical for every arm:
  MCQ : last explicit "final/correct answer is/: X" phrase  ->  vlmeval's own can_infer_option on
        the last line  ->  unique option *text* match in the last 300 chars  ->  unscored (=wrong)
  open: text after the last "answer is/:" phrase, normalised (lower, strip $ * spaces, LaTeX),
        exact match against any GT variant or numeric match within 1% relative tolerance
Prints per-arm accuracy per split with the residual unscored rate, so the reader can see how much
is still unresolved (should be a few %, and similar across arms).

  python -m bench.rescore_mcq --runs $WORK/runs/eval --dataset MMMU_DEV_VAL [--split validation]
"""
from __future__ import annotations
import argparse
import ast
import glob
import json
import os
import re
import pandas as pd

LETTERS = 'ABCDEFGHIJ'
FINAL_RE = re.compile(
    r'(?:final\s+answer|correct\s+answer|correct\s+option|answer)\s*(?:is|would\s+be|should\s+be)?\s*[:：]?\s*'
    r'[\*\s]*\(?\s*([A-J])\s*(?:[\).:\*,\s]|$)', re.I)
OPEN_RE = re.compile(r'(?:final\s+answer|answer)\s*(?:is|would\s+be|should\s+be)?\s*[:：]\s*(.+?)\s*$', re.I | re.M)
BOLD_LETTER_RE = re.compile(r'\*\*\s*\(?([A-J])\s*[\).:]', re.I)


def can_infer_option(answer, choices):
    """vlmeval.utils.matching_util.can_infer_option, verbatim logic (no logging)."""
    answer_mod = answer
    for c in '.()[],:;!*#{}':
        answer_mod = answer_mod.replace(c, ' ')
    splits = [x.strip() for x in answer_mod.split()]
    count = sum(1 for c in choices if c in splits)
    if count == 1:
        for ch in choices:
            if 'A' in splits and len(splits) > 3:
                return False
            if ch in splits and splits.index(ch) > (len(splits) - 5):
                return ch
    return False


def options_of(row):
    return {L: str(row[L]).strip() for L in LETTERS if L in row.index and pd.notna(row[L]) and str(row[L]).strip()}


def extract_mcq(pred: str, opts: dict):
    pred = str(pred)
    valid = set(opts) or set('ABCD')
    hits = [m.group(1).upper() for m in FINAL_RE.finditer(pred) if m.group(1).upper() in valid]
    if hits:
        return hits[-1], 'final-phrase'
    hits = [m.group(1).upper() for m in BOLD_LETTER_RE.finditer(pred) if m.group(1).upper() in valid]
    if hits:
        return hits[-1], 'bold-letter'
    lines = [ln for ln in pred.strip().splitlines() if ln.strip()]
    if lines:
        r = can_infer_option(lines[-1], valid)
        if r:
            return r, 'last-line'
    tail = pred[-300:].lower()
    found = [L for L, t in opts.items() if len(t) >= 2 and t.lower() in tail]
    if len(found) == 1:
        return found[0], 'option-text'
    return None, 'unscored'


def norm(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r'\\[a-z]+\{([^}]*)\}', r'\1', s)          # \frac{a}{b} -> ab (crude, symmetric)
    s = s.replace('$', '').replace('**', '').replace('\\', '').replace(',', '')
    s = re.sub(r'\s+', '', s).rstrip('.')
    return s


def numbers(s):
    return [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', str(s).replace(',', ''))]


def extract_open(pred: str, gt):
    pred = str(pred)
    m = list(OPEN_RE.finditer(pred))
    cand = m[-1].group(1) if m else pred.strip().splitlines()[-1] if pred.strip() else ''
    cand = cand.strip().strip('*').strip()
    try:
        gts = ast.literal_eval(str(gt)) if str(gt).startswith('[') else [gt]
    except Exception:
        gts = [gt]
    flat = []
    for g in gts:
        flat.extend(g if isinstance(g, list) else [g])
    nc, ng = norm(cand), [norm(g) for g in flat]
    if nc and any(nc == g or (len(g) >= 3 and g in nc) for g in ng):
        return cand, 1, 'exact'
    pn = numbers(cand)
    for g in flat:
        gn = numbers(g)
        if pn and gn and len(gn) == 1 and abs(pn[-1] - gn[0]) <= 0.01 * max(abs(gn[0]), 1e-9):
            return cand, 1, 'numeric'
        if pn and gn and len(gn) == 1 and pn[0] == gn[0]:
            return cand, 1, 'numeric'
    return cand, 0, 'exact' if m else 'unscored'


def score_file(path):
    x = pd.read_excel(path)
    rows = []
    for _, r in x.iterrows():
        qt = str(r.get('question_type', 'multiple-choice'))
        if qt == 'open':
            cand, hit, how = extract_open(r['prediction'], r['answer'])
        else:
            opts = options_of(r)
            cand, how = extract_mcq(r['prediction'], opts)
            hit = int(cand is not None and cand == str(r['answer']).strip().upper())
        rows.append({'index': r['index'], 'split': r.get('split', 'none'), 'qtype': qt, 'answer': r['answer'],
                     'extracted': cand, 'how': how, 'hit': hit, 'vlmeval_hit': int(r.get('hit', 0))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=os.environ.get('WORK', os.path.abspath(os.environ.get('LOOK_WORK', 'work'))) + '/runs/eval')
    ap.add_argument('--dataset', default='MMMU_DEV_VAL')
    ap.add_argument('--split', default=None, help='report only this split (e.g. validation)')
    a = ap.parse_args()
    out = {}
    newest = {}                                   # one file per arm: the latest vlmeval run dir
    for f in sorted(glob.glob(f'{a.runs}/*/{a.dataset}/vlmeval/*/*/*_openai_result.xlsx')):
        newest[f.split('/')[-6]] = f
    for arm, f in sorted(newest.items()):
        d = score_file(f)
        d.to_csv(f'{a.runs}/{arm}/{a.dataset}/rescored.csv', index=False)
        sub = d if a.split is None else d[d['split'] == a.split]
        mcq = sub[sub.qtype != 'open']
        res = {'n': len(sub), 'acc': round(sub.hit.mean(), 4), 'acc_mcq': round(mcq.hit.mean(), 4) if len(mcq) else None,
               'acc_open': round(sub[sub.qtype == 'open'].hit.mean(), 4) if (sub.qtype == 'open').any() else None,
               'unscored': round((sub.how == 'unscored').mean(), 4), 'vlmeval_acc': round(sub.vlmeval_hit.mean(), 4),
               'how': sub.how.value_counts().to_dict()}
        out[arm] = res
        print(f'{arm:28s} {a.dataset} split={a.split or "all"}: acc={res["acc"]:.4f} (mcq {res["acc_mcq"]}, open {res["acc_open"]}) '
              f'unscored={res["unscored"]:.3f} | vlmeval said {res["vlmeval_acc"]:.4f} | {res["how"]}')
    json.dump(out, open(f'{a.runs}/rescored_{a.dataset}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
