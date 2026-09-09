"""Reference LDL opt-in: use the authenticated runner, never a new factorizer."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / 'src/banana_smasher'


def load(name, file, extras=None):
    tree = ast.parse((SRC / file).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1, f'missing {name}'
    scope = dict(torch=torch, Any=object)
    scope.update(extras or {})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])), file, 'exec'), scope)
    return scope[name]


def test_public_reference_mode_is_strict_opt_in():
    common = load('_common', 'qtip_batch_controller.py')
    mode = load('_block_ldl_reference', 'qtip_batch_controller.py', {'_common': common})
    assert mode([{}, {}]) is False
    assert mode([{'block_ldl_reference': True}, {'block_ldl_reference': True}]) is True
    for configs in [[{'block_ldl_reference': 1}], [{'block_ldl_reference': 'true'}], [{'block_ldl_reference': True}, {}]]:
        with pytest.raises(ValueError):
            mode(configs)


def test_public_mode_reaches_builder_and_receipt():
    tree = ast.parse((SRC / 'qtip_batch_controller.py').read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'build_qtip_batch']
    assert len(calls) == 1
    assert any(k.arg == 'block_ldl_reference' and isinstance(k.value, ast.Name) and k.value.id == 'block_ldl_reference' for k in calls[0].keywords)
    source = (SRC / 'qtip_batch.py').read_text()
    tree = ast.parse(source)
    builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_qtip_batch')
    args = dict(zip([a.arg for a in builder.args.kwonlyargs], builder.args.kw_defaults))
    assert isinstance(args['block_ldl_reference'], ast.Constant) and args['block_ldl_reference'].value is False
    assert '"block_ldl_reference": block_ldl_reference' in source
    assert '_reference_block_ldl_batch(runner, hessian_batch, 16)' in source


def test_reference_ldl_calls_bound_singleton_math_and_preserves_input():
    calls = []
    def reference(h, block):
        calls.append((h.clone(), block))
        h.add_(3)
        return h, torch.eye(h.shape[0])
    runner = SimpleNamespace(load_official_qtip=lambda: (None, None, SimpleNamespace(block_LDL=reference), None))
    h = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)
    original = h.clone()
    result = load('_reference_block_ldl_batch', 'qtip_batch.py')(runner, h, 2)
    assert torch.equal(h, original)
    assert torch.equal(result, original + 3)
    assert len(calls) == 2 and all(v[1] == 2 and v[0].ndim == 2 for v in calls)


def test_reference_ldl_failure_is_not_a_fallback():
    runner = SimpleNamespace(load_official_qtip=lambda: (None, None, SimpleNamespace(block_LDL=lambda h, b: None), None))
    with pytest.raises(RuntimeError, match='reference block LDL'):
        load('_reference_block_ldl_batch', 'qtip_batch.py')(runner, torch.eye(4)[None], 2)
