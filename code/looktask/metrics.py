"""Bounding-box parsing and IoU shared by inference and CPU checks."""
import re

# Order matters only for precedence; every form a Qwen3-VL model emits must be here.
# The trained target is <|box_start|>(x,y),(x,y)<|box_end|>; those are SPECIAL tokens, so a
# skip_special_tokens=True decode leaves the bare pair "(x,y),(x,y)". Without the last pattern a
# look-trained model scored parse_rate 0.000 on 300/300 correctly formatted answers (bug #11).
BOX_RE = [re.compile(r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'),
          re.compile(r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'),
          re.compile(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'),
          re.compile(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)'),
          # Bare "x1,y1,x2,y2" with no brackets at all: the mix-only arm answers RefCOCO this way on ~40% of
          # expressions ("598,486,803,974"). Without this pattern they were scored as *unparsed* (bug #17).
          # Anchored to a standalone run of exactly four integers so numbers inside prose do not match.
          re.compile(r'(?<![\d.,])(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})(?![\d,])(?!\.\d)')]


def parse_box(t):
    for r in BOX_RE:
        m = r.search(t)
        if m:
            return [int(v) for v in m.groups()]
    return None


def iou(a, b):
    if b is None:
        return 0.0
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    ar = lambda c: max(0, c[2] - c[0]) * max(0, c[3] - c[1])
    u = ar(a) + ar(b) - i
    return i / u if u else 0.0
