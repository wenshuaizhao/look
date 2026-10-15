"""Run the LOOK crop sampler on the bundled sheep photograph, using only a CPU."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from looktask.geometry import Geometry
from looktask.sample import make_look_sample
from looktask.prompts import ALL


class IllustrationGeometry(Geometry):
    def canonical_size(self, w, h):
        # The illustrative canvas is already aligned. Model runs use Geometry's
        # processor-specific smart_resize, rather than this identity operation.
        if w % self.cell or h % self.cell:
            raise ValueError('Illustration must be merge-block aligned')
        return w, h


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('work/quickstart'))
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    image = Image.open(Path(__file__).with_name('sheep.jpg')).convert('RGB')
    geo = IllustrationGeometry(SimpleNamespace(patch_size=16, merge_size=2), 1003520)
    s = make_look_sample(image, geo, np.random.default_rng(args.seed))
    crop_path = (args.out / 'crop.png').resolve()
    s.crop.save(crop_path)
    context = s.canon.copy()
    ImageDraw.Draw(context).rectangle(s.gt_px, outline='#147d83', width=5)
    context.save(args.out / 'context.png')
    gw, gh, _, _ = geo.grids(*s.canvas)
    rw, rh = s.canvas
    row = {
        'messages': [{'role': 'user', 'content': ALL[0]},
                     {'role': 'assistant', 'content': '<bbox>'}],
        'images': [str(crop_path)],
        'objects': {'ref': [], 'bbox': [[x / (rw if i % 2 == 0 else rh)
                                        for i, x in enumerate(s.gt_px)]],
                    'bbox_type': 'norm1',
                    'look_offset': [*s.offset, gh, gw]},
    }
    (args.out / 'look.jsonl').write_text(json.dumps(row) + '\n')
    print(json.dumps({'canvas': s.canvas, 'crop': s.crop.size,
                      'box_norm1000': s.gt_norm1000, 'output': str(args.out)}, indent=2))
    print('Saved context.png, crop.png, and look.jsonl.')


if __name__ == '__main__':
    main()
