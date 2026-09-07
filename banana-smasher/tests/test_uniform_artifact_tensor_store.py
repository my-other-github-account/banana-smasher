"""CPU-only loader routing contract; tensor arithmetic is tested by codec tests."""
import ast
from pathlib import Path
import re
import types
import unittest


class UniformArtifactRoutingTest(unittest.TestCase):
    def test_every_advertised_routed_layer_decodes_candidate(self):
        path = Path(__file__).parents[1] / 'src/banana_smasher/hf_sharded_balanced64_executor.py'
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ArtifactTensorStore')
        source_reads = []
        source = types.SimpleNamespace(tensor=lambda name: source_reads.append(name) or 'native', names=lambda: set())
        geometry = types.SimpleNamespace(tlut_bits=9, V=2)
        ns = {'Mapping': dict, 'Any': object, 'Sequence': list, 'Path': Path,
              'np': types.SimpleNamespace(ndarray=object, int32='i32', uint16='u16', float32='f32', empty=lambda *a, **k: [], asarray=lambda a, **k: a),
              '_LAYER_NAME': re.compile(r'(?:^|\.)layers\.(\d+)\.'),
              '_subject_source': lambda artifact: {}, 'SourceTensorStore': lambda _: source,
              'QtipGeometry': types.SimpleNamespace(from_mapping=lambda _: geometry),
              'EncodedQtip': lambda **kw: kw, 'gaussian_tlut': lambda **kw: 'lut',
              'decode_qtip': lambda encoded, **kw: encoded['packed']}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
        names = [f'model.language_model.layers.{layer}.mlp.experts.0.gate_proj.weight' for layer in range(3, 45)]
        rows = [{'name': name, 'shape': [1, 1], 'wire': {'geometry': {}, 'trellis': {'path': name}, 'scales': {'path': 'scales'}}, 'source_transform': {'output_quantity': 'descaled_weight'}} for name in names]
        store = ns['ArtifactTensorStore']({'artifact_root': '.', 'routed_tensors': rows, 'geometry': {'routed_layer_ids': list(range(3, 45))}})
        store._load_array = lambda binding: binding['path']
        store._torch_from_numpy = lambda array: array
        for name in names:
            self.assertEqual(store.tensor(name), name, f'routed candidate bypass: {name}')
            self.assertFalse(store.requires_source_scale(name))
        self.assertEqual(source_reads, [])
        self.assertEqual(store.model_reads, 0)
        native = 'model.language_model.layers.3.input_layernorm.weight'
        self.assertEqual(store.tensor(native), 'native')
        self.assertTrue(store.requires_source_scale(native))
        self.assertEqual(source_reads, [native])
        self.assertEqual(store.model_reads, 1)


if __name__ == '__main__':
    unittest.main()
