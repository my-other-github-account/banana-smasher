"""CPU-only admission tests; no CUDA or producer imports."""
import importlib.util
from pathlib import Path
import unittest

MODULE = Path(__file__).parents[1] / 'src/banana_smasher/capacity_advice.py'

class CapacityAdviceTests(unittest.TestCase):
    def test_sufficient_capacity_does_not_visit_historical_outputs(self):
        self.assertTrue(MODULE.exists(), 'capacity-gated advice helper missing')
        spec = importlib.util.spec_from_file_location('capacity_advice', MODULE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        def forbidden():
            self.fail('unnecessary historical output scan')
        samples = iter([17, 18])
        receipt = module.capacity_advice(lambda: next(samples), 12, 4, forbidden)
        self.assertEqual(receipt['action'], 'skipped_sufficient_capacity')
        self.assertEqual(receipt['after_free_bytes'], 18)

    def test_invalid_capacity_values_fail_before_reclaim(self):
        spec = importlib.util.spec_from_file_location('capacity_advice', MODULE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for peak, reserve, free in [(0, 4, 17), (12, -1, 17), (True, 4, 17), (12, 4, float('nan')), (12, 4, -1), (12, 4, True)]:
            with self.subTest(values=(peak, reserve, free)):
                with self.assertRaises(ValueError):
                    module.capacity_advice(lambda: free, peak, reserve, lambda: self.fail('bad input advised'))

    def test_advice_once_and_final_gate_cannot_be_skipped(self):
        spec = importlib.util.spec_from_file_location('capacity_advice', MODULE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for samples, expected_calls, error in [([16, 17], 1, None), ([17, 16], 0, RuntimeError), ([15, 15], 1, RuntimeError), ([17, float('nan')], 0, ValueError)]:
            calls = []
            probe = iter(samples)
            with self.subTest(samples=samples):
                if error:
                    with self.assertRaises(error):
                        module.capacity_advice(lambda: next(probe), 12, 4, lambda: calls.append('advice'))
                else:
                    receipt = module.capacity_advice(lambda: next(probe), 12, 4, lambda: calls.append('advice'))
                    self.assertEqual(receipt['action'], 'advised_once')
                self.assertEqual(len(calls), expected_calls)

if __name__ == '__main__':
    unittest.main()
