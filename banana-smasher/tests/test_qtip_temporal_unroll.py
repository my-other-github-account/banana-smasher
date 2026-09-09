"""Temporal unroll is separate from branch unroll and predecessor layout."""
import ast
from pathlib import Path


def test_generic_temporal_loop_unrolls_only_alphabet():
    src = Path(__file__).parents[1] / 'src/banana_smasher/qtip_viterbi.py'
    node = next(n for n in ast.parse(src.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_persistent_prefix_viterbi_generic')
    loops = [n for n in ast.walk(node) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'step']
    assert len(loops) == 1, 'missing temporal range loop'
    value = next(k.value for k in loops[0].iter.keywords if k.arg == 'loop_unroll_factor')
    expr = compile(ast.Expression(value), str(src), 'eval')
    assert eval(expr, {'DISTANCE_ALPHABET': False}) == 1
    assert eval(expr, {'DISTANCE_ALPHABET': True}) == 2
