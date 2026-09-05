"""CPU-only publication checks; never replay sealed model forwards."""
import importlib.util
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


class Q4PrePublicationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = json.loads((ROOT / 'Evals/results/deepseek-v4-flash-0731-q4-pre-qualified-v1.json').read_text())

    def test_qualification_is_explicit(self):
        for key in ('repair_applied', 'historical_runtime_recovered', 'historical_q3_equivalence_demonstrated', 'ranked_with_historical_rows', 'native_control_subtracted'):
            self.assertIs(self.row[key], False)
        for key in ('mmlu', 'shipping_bytes', 'comparison_bpw'):
            self.assertIsNone(self.row[key])
        self.assertFalse(self.row['expected_identity_provenance']['candidate_result_used'])

    def test_both_complete_ordered_series(self):
        ids = self.row['expected_identity']['window_ids']
        self.assertEqual(len(ids), 64)
        self.assertEqual(len(set(ids)), 64)
        for arm in self.row['rows']:
            series = arm['per_window']
            self.assertEqual([r['window_id'] for r in series], ids)
            self.assertEqual([r['ordinal'] for r in series], list(range(64)))
            self.assertEqual([r['forward_kl'] for r in series], arm['window_values'])
            self.assertEqual(sum(r['positions'] for r in series), 65536)
            self.assertEqual(sum(r['top1_matches'] for r in series), arm['top1_matches'])
            self.assertAlmostEqual(math.fsum(arm['window_values']) / 64, arm['forward_kl'], places=15)
            self.assertEqual(arm['top1_rate'], arm['top1_matches'] / 65536)
            self.assertTrue(all(math.isfinite(v) and v >= 0 for v in arm['window_values']))

    def test_canonical_controller_accepts_published_metric(self):
        spec = importlib.util.spec_from_file_location('q4_publication_controller', ROOT / 'tools/autonomy/controller.py')
        controller = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(controller)
        metric = self.row['controller_metric']
        result = controller.verify_metric(json.dumps(metric).encode(), self.row['expected_identity'])
        q4 = next(r for r in self.row['rows'] if r['mode'] == 'q4')
        self.assertEqual(result['value'], q4['forward_kl'])
        self.assertEqual(metric['window_values'], q4['window_values'])
        bad = dict(metric, value=metric['value'] + 1)
        with self.assertRaises(ValueError):
            controller.verify_metric(json.dumps(bad), self.row['expected_identity'])

    def test_readme_displays_exact_measured_values(self):
        text = (ROOT / 'Evals/README.md').read_text()
        section = text.split('## Runtime-qualified Q4 PRE measurement', 1)[1].split('## EXL', 1)[0]
        self.assertIn('not ranked with the historical comparison', section)
        for arm in self.row['rows']:
            self.assertIn(str(arm['forward_kl']), section)
            self.assertIn(f"{arm['top1_rate'] * 100:.2f}%", section)
            self.assertIn(f"{arm['top1_matches']:,} / 65,536", section)
        self.assertIn('not subtracted', section)


if __name__ == '__main__':
    unittest.main()
