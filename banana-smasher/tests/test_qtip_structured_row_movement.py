"""K1 row movement contract; target GPU parity remains mandatory."""
import ast
from pathlib import Path

def test_k1_structured_selection_splits_rows_without_reduction():
    source = (Path(__file__).parents[1] / 'src/banana_smasher/qtip_viterbi.py').read_text()
    tree = ast.parse(source)
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_persistent_prefix_viterbi_generic')
    branches = [n for n in ast.walk(kernel) if isinstance(n, ast.If) and ast.unparse(n.test) == 'BRANCHES == 4']
    assert branches, 'K1 row selection lacks split specialization'
    calls = [ast.unparse(n.func) for stmt in branches[0].body for n in ast.walk(stmt) if isinstance(n, ast.Call)]
    assert calls.count('tl.split') == 3
    assert 'tl.sum' not in calls and 'tl.gather' not in calls
    assert 'candidate < best' in ast.unparse(kernel)
