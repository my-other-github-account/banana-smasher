"""Explicit canonical decoder execution; no skipped conformance or fallback."""
import importlib
import torch
import pytest

def test_explicit_eager_uses_same_canonical_integer_decoder():
    from pathlib import Path
    path = Path(__file__).parents[1] / 'src/banana_smasher/qtip_kernel_decompress.py'
    spec = importlib.util.spec_from_file_location('canonical_decoder', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, 'select_decoder'), 'explicit decoder execution selector missing'
    selected = module.select_decoder('eager')
    assert selected.decode_compressed is module.decode_compressed_eager
    assert module.select_decoder('compiled').decode_compressed is module.decode_compressed
    for k in (1, 2, 3, 4):
        packed = torch.arange(k * 32 * 32 // 16, dtype=torch.int32).to(torch.uint16)
        lut = torch.arange(65536 * 2, dtype=torch.float32).reshape(65536, 2)
        actual = selected.decode_compressed(16, 9, k, 1, 32, 32, packed, lut)
        expected = module.decode_compressed._torchdynamo_orig_callable(16, 9, k, 1, 32, 32, packed, lut)
        assert torch.equal(actual, expected)
    with pytest.raises(ValueError, match='decoder execution'):
        module.select_decoder('auto-fallback')


def test_batch_decoder_execution_admission_and_wiring():
    import ast
    from pathlib import Path
    source = (Path(__file__).parents[1] / 'src/banana_smasher/qtip_batch_controller.py').read_text()
    tree = ast.parse(source)
    names = {'_common', '_packed_decode_execution'}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == 2, 'public batch decoder admission missing'
    for node in functions:
        node.returns = None
        for arg in node.args.args: arg.annotation = None
    ns = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<admission>', 'exec'), ns)
    admit = ns['_packed_decode_execution']
    assert admit([{}, {}]) == 'compiled'
    assert admit([{'packed_decode_execution':'eager'}]*2) == 'eager'
    for configs in ([{'packed_decode_execution':True}], [{'packed_decode_execution':[]}],
                    [{'packed_decode_execution':'skip'}], [{}, {'packed_decode_execution':'eager'}]):
        with pytest.raises(ValueError): admit(configs)
    assert 'kernel_decode = kernel_decode.select_decoder(packed_decode_execution)' in source
    assert '"packed_decode_execution": packed_decode_execution' in source
