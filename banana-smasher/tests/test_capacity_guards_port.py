"""CPU-only admission regression; execute real helper/guards with CUDA probes stubbed."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / 'src/banana_smasher'


def load_functions(filename, names, namespace):
    tree = ast.parse((ROOT / filename).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), namespace)
    return namespace


class CapacityGuards(unittest.TestCase):
    def test_owner_capacity_and_both_guards(self):
        ns = load_functions('qtip_rings.py', {'effective_cuda_free_bytes', 'qtip_admission_memory'}, {'Path': Path})
        self.assertIn('qtip_admission_memory', ns, 'owner admission helper missing')
        probe = ns['qtip_admission_memory']
        cuda = SimpleNamespace(mem_get_info=lambda d: (100, 1000), memory_reserved=lambda d: 40,
                               memory_allocated=lambda d: 10,
                               get_device_properties=lambda d: SimpleNamespace(name='NVIDIA GB10'))
        torch = SimpleNamespace(cuda=cuda, int64='int64')
        with patch.dict(ns, _qtip_compute_pids=lambda: [os.getpid()]), patch.object(Path, 'read_text', return_value='MemAvailable: 20 kB\n'):
            self.assertEqual(probe(torch)['available_bytes'], 20480)  # no native-cache double count
        cuda.get_device_properties = lambda d: SimpleNamespace(name='NVIDIA H100')
        self.assertEqual(probe(torch)['available_bytes'], 130)
        cuda.get_device_properties = lambda d: SimpleNamespace(name='NVIDIA GB10')
        for text in ['MemAvailable: -1 kB', 'MemAvailable: 20 MB', 'Missing: 20 kB']:
            with patch.dict(ns, _qtip_compute_pids=lambda: []), patch.object(Path, 'read_text', return_value=text):
                with self.assertRaises((ValueError, KeyError)):
                    probe(torch)
        with patch.dict(ns, _qtip_compute_pids=lambda: [os.getpid() + 100000]):
            with self.assertRaisesRegex(RuntimeError, 'foreign GPU'):
                probe(torch)
        with patch.dict(ns, _qtip_compute_pids=lambda: (_ for _ in ()).throw(OSError('witness unavailable'))):
            with self.assertRaises(OSError):
                probe(torch)
        # Execute the actual alphabet guard. A sentinel tensor call proves admission
        # passed before any allocation rather than mocking the guard itself.
        class Admitted(Exception):
            pass
        torch.tensor = lambda *a, **k: (_ for _ in ()).throw(Admitted())
        cb = SimpleNamespace(decode_mode='quantlut_sym', tlut_bits=9,
                             lut=SimpleNamespace(_version=0, device='cuda'))
        available = [0]
        lutns = load_functions('qtip_viterbi.py', {'_distance_alphabet_lut'},
            {'torch': torch, 'qtip_admission_memory': lambda *a: {'available_bytes': available[0]},
             '_signed_alphabet_representatives': lambda: []})
        threshold = (4 << 30) + (4 << 20)
        for delta in [-1, 0, 1]:
            available[0] = threshold + delta
            with self.assertRaises(RuntimeError if delta < 0 else Admitted):
                lutns['_distance_alphabet_lut'](cb)
        # Bind workspace wiring and unchanged full reserve to actual source AST.
        tree = ast.parse((ROOT / 'qtip_viterbi.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'exact_prefix_viterbi')
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertTrue(any(n.func.id == 'qtip_admission_memory' for n in calls))
        reserve = next(n.value for n in fn.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'reserve' for t in n.targets))
        for alphabet in [False, True]:
            value = eval(compile(ast.Expression(reserve), '', 'eval'), {'distance_alphabet': alphabet})
            self.assertEqual(value, (4 << 30) + ((4 << 20) if alphabet else 0))


if __name__ == '__main__':
    unittest.main()
