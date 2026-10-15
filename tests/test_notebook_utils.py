"""Small-sample scoring, adapter selection, and sample-integrity checks."""
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

from notebook_utils import (load_samples, summarize_probes, summarize_visabench,
                            find_adapter, default_models, validate_models)

ROOT = Path(__file__).resolve().parents[1]


class NotebookTests(unittest.TestCase):
    def test_bundled_samples_are_distinct_test_items(self):
        rows = load_samples(ROOT)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({r['image'] for r in rows}), 4)
        for row in rows:
            self.assertTrue(row['id'].startswith(('refcoco_testA/', 'refcoco_testB/')))
            self.assertNotEqual(row['expr'], row['mismatch_expr'])

    def test_parse_failures_count_as_zero_without_changing_parsed_mean(self):
        rows = [dict(condition='crop_restored', pred=[0,0,500,1000],
                     gt=[0,0,500,1000], iou=1.),
                dict(condition='crop_restored', pred=None, gt=[0,0,500,1000], iou=0.)]
        score = summarize_probes(rows)[0]
        self.assertEqual(score['parse_rate'], .5)
        self.assertEqual(score['iou_all'], .5)
        self.assertEqual(score['iou_parsed'], 1.)
        self.assertEqual(score['whole_canvas_iou'], .5)

    def test_omitted_blind_scores_keep_raw_forced_values(self):
        rows = [dict(model='Qwen', condition=c, p_yes=p, top_token='Yes' if p>.5 else 'No',
                     reportable=True) for c,p in [('real',.9),('real',.4),('blind',.1),('blind',.2)]]
        rows += [dict(model='LLaVA', condition='blind', p_yes=.9,
                      top_token='The', reportable=False)]
        scores = {(r['model'],r['condition']):r for r in summarize_visabench(rows)}
        self.assertEqual(scores['Qwen','real']['gap'], .5)
        self.assertIsNone(scores['LLaVA','blind']['yes_rate'])
        self.assertIsNone(scores['LLaVA','blind']['gap'])
        self.assertEqual(scores['LLaVA','blind']['forced_yes_rate'], 1.)
        self.assertEqual(scores['LLaVA','blind']['top_token_yes_no'], 0.)

    def test_required_adapter_never_falls_back_to_base(self):
        with self.assertRaisesRegex(ValueError, 'set its adapter path'):
            validate_models([dict(label='SFT', enabled=True, adapter=None, requires_adapter=True)])

    def test_ambiguous_adapter_discovery_requires_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'LOOK_TEST_ADAPTER':''}):
            for version in ['v0','v1']:
                (Path(tmp)/'runs/arm1'/version/'checkpoint-2848').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'Multiple arm1 checkpoints'):
                find_adapter(tmp,'arm1',2848,'LOOK_TEST_ADAPTER')

    def test_registry_covers_requested_models(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'LOOK_SFT_ADAPTER':'','LOOK_BEST_ADAPTER':''}):
            models = default_models(tmp)
        self.assertEqual([m['key'] for m in models], ['qwen3','sft','look','qwen25','llava','gemma4'])
        self.assertTrue(all(m['enabled'] for m in models))
        self.assertIsNone(models[1]['adapter'])
        self.assertIsNone(models[2]['adapter'])


if __name__ == '__main__':
    unittest.main()
