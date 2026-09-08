"""Exercise the actual public receipt expression without importing CUDA."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('k', [1, 2, 3, 4])
def test_batch_receipt_reports_actual_prefix_geometry(k):
    source = Path(__file__).parents[1] / 'src/banana_smasher/qtip_batch.py'
    tree = ast.parse(source.read_text())
    build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_qtip_batch')
    receipt = next(n.value.elts[1] for n in ast.walk(build)
                   if isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple))
    assert isinstance(receipt, ast.Dict)
    expression = next(v for key, v in zip(receipt.keys, receipt.values)
                      if isinstance(key, ast.Constant) and key.value == 'solver_geometry')
    codebook = SimpleNamespace(L=16, K=k, V=2)
    actual = eval(compile(ast.Expression(expression), str(source), 'eval'), {'codebook': codebook})
    assert actual == dict(L=16, K=k, V=2, retained_prefix_costs=2 ** (16 - 2*k),
                          branches_per_prefix=2 ** (2*k), branch_sampling='full')
