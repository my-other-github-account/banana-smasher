"""Regression: keep a statically empty recurrence out of Triton Coalesce."""
import ast
from pathlib import Path
import unittest

SOURCE = Path(__file__).parents[1] / 'src/banana_smasher/qtip_viterbi.py'


class OneStepCompileBoundary(unittest.TestCase):
    def test_recurrence_and_alphabet_are_constexpr_guarded(self):
        module = ast.parse(SOURCE.read_text())
        kernel = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                      and n.name == '_persistent_prefix_viterbi_generic')
        guards = [n for n in kernel.body if isinstance(n, ast.If)
                  and ast.unparse(n.test) == 'STEPS > 1']
        self.assertEqual(len(guards), 1)
        guard = guards[0]
        self.assertEqual(ast.unparse(guard.body[0].test), 'DISTANCE_ALPHABET')
        self.assertEqual(ast.unparse(guard.body[1]), 'step = 1')
        self.assertIsInstance(guard.body[2], ast.While)
        self.assertEqual(ast.unparse(guard.body[2].test), 'step < STEPS')
        self.assertEqual(len(guard.body), 3)
        self.assertFalse(guard.orelse)
        self.assertFalse(any(isinstance(n, ast.While) for n in kernel.body))

    def test_actual_cuda_one_step_overlap(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch unavailable; structural test only')
        if not torch.cuda.is_available():
            self.skipTest('CUDA required for compiler regression')
        from banana_smasher import qtip_viterbi as module
        batch, states, prefixes = 3, 65536, 16384
        torch.manual_seed(8767)
        x = torch.randn((2, batch), device='cuda', dtype=torch.float16)
        alphabet = torch.randn((2, 1024), device='cuda', dtype=torch.float16)
        keys = torch.arange(states, device='cuda', dtype=torch.int64)
        keys = ((keys * (keys + 1)) >> 6) & 1023
        lut = alphabet[:, keys].contiguous()
        overlap = torch.tensor([0, 127, 16383], device='cuda', dtype=torch.int32)
        scratch = torch.empty((2, batch, prefixes), device='cuda')
        pointers = torch.full((1, batch, prefixes), 65535, device='cuda', dtype=torch.int32)
        output = torch.empty((1, batch), device='cuda', dtype=torch.int32)
        module._persistent_prefix_viterbi_generic[(batch,)](
            x, lut, alphabet, overlap, scratch, pointers, output, batch,
            STATES=states, PREFIXES=prefixes, BRANCHES=4, SHIFT=2,
            Q_FACTOR=4096, V=2, STEPS=1, HAS_OVERLAP=True,
            REGISTER_COSTS=True, BRANCH_UNROLL=4, STRUCTURED_GATHER=True,
            BRANCH_POINTERS=False, DISTANCE_ALPHABET=True,
            CONDITIONED_DISTANCE_SUM=True, FUSED_SCHEDULE=False,
            num_warps=16, num_stages=4,
        )
        torch.cuda.synchronize()
        expected = (overlap // 4096) * prefixes + overlap
        self.assertTrue(torch.equal(output[0], expected))


if __name__ == '__main__':
    unittest.main()
