"""Movement-only row selection; real GPU parity is mandatory separately."""
import ast
from pathlib import Path

def test_structured_predecessor_selection_is_movement_only():
    source = (Path(__file__).parents[1] / 'src/banana_smasher/qtip_viterbi.py').read_text()
    tree = ast.parse(source)
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_persistent_prefix_viterbi_generic')
    branch = next(n for n in ast.walk(kernel) if isinstance(n, ast.If) and ast.unparse(n.test) == 'STRUCTURED_GATHER')
    calls = [ast.unparse(n.func) for n in ast.walk(branch) if isinstance(n, ast.Call)]
    assert 'tl.sum' not in calls, 'row selection still reduces a full masked prefix vector'
    assert 'tl.gather' in calls
    assert 'candidate < best' in ast.unparse(kernel)
