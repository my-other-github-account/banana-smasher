"""Default-off admission and actual LDLQ scope regression tests (CPU only)."""
import ast
from pathlib import Path
import types
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'src' / 'banana_smasher'


class InferenceScopeTests(unittest.TestCase):
    def test_member_admission(self):
        tree = ast.parse((ROOT / 'qtip_batch_controller.py').read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('_common', '_ldlq_inference_scope')]
        ns = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes,
                                type_ignores=[])), '<actual-resolvers>', 'exec'), ns)
        resolve = ns['_ldlq_inference_scope']
        self.assertIs(resolve([{}, {}]), False)
        self.assertIs(resolve([{'ldlq_inference_scope': True}] * 2), True)
        for bad in (0, 1, None, 'true', [], {}):
            for configs in ([{}, {'ldlq_inference_scope': bad}],
                            [{'ldlq_inference_scope': bad}, {}]):
                with self.assertRaises(ValueError):
                    resolve(configs)
        for configs in ([{}, {'ldlq_inference_scope': True}],
                        [{'ldlq_inference_scope': True}, {}], []):
            with self.assertRaises(ValueError):
                resolve(configs)

    def test_scope_and_public_gradient(self):
        import importlib.util
        import torch
        spec = importlib.util.spec_from_file_location('scope_batch', ROOT / 'qtip_batch.py')
        batch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(batch)
        args = types.SimpleNamespace(td_x=16, td_y=16, V=2)

        class Codebook:
            idx_dtype = torch.int32
            def __init__(self, expected):
                self.lut = torch.ones(512)
                self.expected = expected
                self.fail = False
            def quantize(self, value):
                assert self.lut._version == 0
                assert torch.is_inference_mode_enabled() == self.expected
                if self.fail:
                    raise RuntimeError('deliberate scope failure')
                return value.clone(), value[:, ::2].to(torch.int32)

        for context in (torch.enable_grad, torch.no_grad, torch.inference_mode):
            for enabled in (False, True):
                cb = Codebook(enabled or context is torch.inference_mode)
                with context():
                    before = (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
                    w = torch.randn(1, 16, 16, requires_grad=True)
                    lower = torch.zeros(1, 16, 16)
                    fn = batch._ldlq_batch_inference if enabled else batch.ldlq_batch
                    output = fn(w, lower, cb, args, buf_cols=16, for_kernel=False)[0]
                    if context is torch.enable_grad and not enabled:
                        output.sum().backward()
                        self.assertTrue(torch.equal(w.grad, torch.ones_like(w)))
                    self.assertEqual(before, (torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
                    cb.fail = True
                    with self.assertRaisesRegex(RuntimeError, 'deliberate scope failure'):
                        fn(w, lower, cb, args, buf_cols=16, for_kernel=False)
                    self.assertEqual(before, (torch.is_grad_enabled(), torch.is_inference_mode_enabled()))


if __name__ == '__main__':
    unittest.main()
