"""Integrity checks for retained measurements, not a fresh model evaluation."""
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class SparsePublicationTests(unittest.TestCase):
    def test_frozen_population_aggregation_and_scope(self):
        d = json.loads((ROOT/'experiments/ds4-q1-sparse18-pre.json').read_text())
        lock = json.loads((ROOT/'configs/balanced64-v1.json').read_text())
        rows = d['per_window']
        self.assertEqual([r['window_id'] for r in rows], [r['window_id'] for r in lock['windows']])
        self.assertEqual(len(rows), 64)
        self.assertEqual([r['ordinal'] for r in rows], list(range(64)))
        self.assertTrue(all(r['positions'] == 1024 for r in rows))
        self.assertTrue(all(math.isfinite(r['kld_mean']) and r['kld_mean'] >= 0 for r in rows))
        self.assertAlmostEqual(math.fsum(r['kld_mean'] for r in rows)/64, d['kld_mean'], places=15)
        self.assertEqual(sum(r['top1_matches'] for r in rows), d['top1_matches'])
        self.assertEqual(d['top1_matches'], 64148)
        self.assertEqual(d['positions'], 65536)
        self.assertFalse(d['uniform_model_quality_claim'])
        self.assertFalse(d['repair_performed'])
        self.assertIsNone(d['threshold_source'])
        self.assertIn('NOT_ADJUDICATED', d['quality_acceptance'])
        for name, group in d['classes'].items():
            members = [r for r in rows if r['source_class'] == name]
            self.assertEqual(len(members), group['windows'])
            self.assertEqual(sum(r['top1_matches'] for r in members), group['top1_matches'])
            self.assertAlmostEqual(math.fsum(r['kld_mean'] for r in members)/len(members), group['kld_mean'], places=15)

if __name__ == '__main__':
    unittest.main()
