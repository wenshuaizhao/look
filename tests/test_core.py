"""Check source-patch alignment, coordinate conventions, and model reply parsing."""
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from looktask.geometry import Geometry
from looktask.sample import make_look_sample
from looktask.metrics import iou, parse_box


class FixedGeometry(Geometry):
    def canonical_size(self, w, h):
        return w, h


class CoreTests(unittest.TestCase):
    def test_crops_preserve_exact_source_pixels_and_canvas_coordinates(self):
        geo = FixedGeometry(SimpleNamespace(patch_size=16, merge_size=2), 1003520)
        pixels = np.random.default_rng(5).integers(0, 256, (384, 512, 3), dtype=np.uint8)
        for seed in range(20):
            s = make_look_sample(Image.fromarray(pixels), geo, np.random.default_rng(seed))
            x0, y0, x1, y1 = s.gt_px
            np.testing.assert_array_equal(np.asarray(s.crop), pixels[y0:y1, x0:x1])
            self.assertEqual(s.offset, (y0 // 32, x0 // 32))
            self.assertTrue(all(x % 32 == 0 for x in s.gt_px))
            self.assertEqual(s.gt_norm1000, [round(x0/512*1000), round(y0/384*1000),
                                            round(x1/512*1000), round(y1/384*1000)])

    def test_flatten_indices_follow_merge_block_order(self):
        geo = Geometry(SimpleNamespace(patch_size=16, merge_size=2), 1003520)
        expected = [(0, 0), (0, 1), (1, 0), (1, 1),
                    (0, 2), (0, 3), (1, 2), (1, 3),
                    (2, 0), (2, 1), (3, 0), (3, 1),
                    (2, 2), (2, 3), (3, 2), (3, 3)]
        self.assertEqual([geo.patch_row(y, x, 4) for y, x in expected], list(range(16)))

    def test_reply_formats_and_parse_failures(self):
        for reply in ['{"bbox_2d": [10,20,300,400]}', '[10,20,300,400]',
                      '<|box_start|>(10,20),(300,400)<|box_end|>',
                      '(10,20),(300,400)', '10,20,300,400']:
            self.assertEqual(parse_box(reply), [10, 20, 300, 400])
        self.assertIsNone(parse_box('I cannot see an image.'))
        self.assertEqual(iou([0, 0, 1000, 1000], None), 0)
        self.assertEqual(iou([0, 0, 1000, 1000], [0, 0, 500, 1000]), 0.5)


if __name__ == '__main__':
    unittest.main()
