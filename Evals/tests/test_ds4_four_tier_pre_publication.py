"""Existing-row publication validation; never execute model/scorer code."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / 'Evals/results/deepseek-v4-flash-0731-four-tier-pre-qualified-v1.json'

class FourTierPrePublicationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(RESULT.read_text())
        spec = importlib.util.spec_from_file_location('controller', ROOT / 'tools/autonomy/controller.py')
        assert spec is not None and spec.loader is not None
        cls.controller = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.controller)

    def test_four_original_scores_and_order(self):
        expected = [1.3244045625016363, .3235340378881574, .1203225347792691, .047134897943039586]
        self.assertEqual([r['K'] for r in self.data['rows']], [1, 2, 3, 4])
        for r, value in zip(self.data['rows'], expected):
            metric = r['controller_metric']
            self.assertEqual(metric['value'], value)
            self.assertEqual(metric['window_values'], r['per_window_forward_KL'])
            self.assertEqual(len(metric['window_values']), 64)
            self.assertEqual(len(set(r['ordered_window_ids'])), 64)
            self.assertEqual(r['expected_identity']['window_ids'], r['ordered_window_ids'])
            self.assertEqual(r['ordered_window_ids'], self.data['rows'][0]['ordered_window_ids'])
            self.assertAlmostEqual(math.fsum(metric['window_values']) / 64, value, places=15)
            self.assertEqual(r['full_expert_coverage'], 22016)

    def test_controller_readback_and_wrong_identity_rejection(self):
        for r in self.data['rows']:
            raw = json.dumps(r['controller_metric']).encode()
            self.assertAlmostEqual(self.controller.verify_metric(raw, r['expected_identity'])['value'], r['measured_kld'], places=15)
            for key in self.controller.IDENTITY:
                wrong = copy.deepcopy(r['expected_identity']); wrong[key] = 'wrong'
                with self.assertRaises(ValueError): self.controller.verify_metric(raw, wrong)
            wrong = copy.deepcopy(r['controller_metric']); wrong['value'] += 1
            with self.assertRaises(ValueError): self.controller.verify_metric(json.dumps(wrong), r['expected_identity'])

    def test_reference_derived_not_quality_green(self):
        for key in ('quality_approved', 'shipping_package_complete', 'historical_q3_equivalence_demonstrated', 'native_control_subtracted'):
            self.assertIs(self.data[key], False)
        for r in self.data['rows']:
            self.assertIs(r['quality_approved'], False)
            self.assertIsNone(r['shipping_bytes'])
            p = r['expected_identity_provenance']
            if r['K'] < 4: self.assertIs(p['provenance']['candidate_metric_used_to_derive_expected'], False)
            else: self.assertIs(p['candidate_result_used'], False)
        old = json.loads((ROOT / 'Evals/results/deepseek-v4-flash-0731-q4-pre-qualified-v1.json').read_text())
        q4 = self.data['rows'][3]
        self.assertEqual(q4['expected_identity'], old['expected_identity'])
        self.assertEqual(q4['controller_metric']['window_values'], old['controller_metric']['window_values'])
        self.assertEqual(hashlib.sha256((ROOT / 'Evals/results/deepseek-v4-flash-0731-q4-pre-qualified-v1.json').read_bytes()).hexdigest(), 'ca10f933845a89c1b94d940632e6bab36a1d5fdcb519d511b3eac53a692eebc9')

    def test_bytes_are_not_shipping_or_complete_mtp(self):
        n = self.data['native_rest_scope']
        self.assertEqual(n['indexed_native_tensor_count'], 6269)
        self.assertEqual(n['indexed_native_tensor_payload_bytes'], 19708797688)
        self.assertEqual(n['indexed_native_container']['bytes'], 19709482984)
        self.assertEqual(n['indexed_native_container']['bytes'] - n['indexed_native_tensor_payload_bytes'], n['native_container_overhead_bytes'])
        self.assertEqual(len(set(n['absent_historical_correction_tensor_names'])), 10)
        self.assertIs(n['complete_mtp_claimed'], False)
        self.assertIsNone(n['shipping_bytes'])
        for r, routed in zip(self.data['rows'], [35093702144, 69721875968, 104350049792, 138978223616]):
            self.assertEqual(r['exact_artifact_bytes'], routed)
            self.assertEqual(r['indexed_hybrid_weight_container_bytes'], routed + n['indexed_native_container']['bytes'])
            self.assertIn('not a shipped package', r['indexed_hybrid_weight_container_scope'])

    def test_semantic_qualification_and_readme(self):
        s = self.data['scorer_qualification']
        self.assertEqual(len(set(s['scorer_hashes'].values())), 4)
        self.assertIs(s['runtime_equivalence_claimed'], False)
        self.assertIs(s['historical_q3_parity_claimed'], False)
        self.assertIs(s['ordered_teacher_payload_descriptors_match'], True)
        text = (ROOT / 'Evals/README.md').read_text().split('## Qualified four-tier DS4 uniform PRE measurements', 1)[1]
        for r in self.data['rows']: self.assertIn(str(r['measured_kld']), text)
        for phrase in ['not standalone shipping totals', 'not quality GREEN', 'partial-MTP', 'no historical Q3 parity']:
            self.assertIn(phrase, text)

if __name__ == '__main__': unittest.main()
