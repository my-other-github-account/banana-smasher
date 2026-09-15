"""K4 internal-producer admission; public untrusted operands stay strict."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest

SOURCE = Path(__file__).parents[1] / 'src/banana_smasher/qtip_viterbi.py'

def function(name):
    node = next(n for n in ast.parse(SOURCE.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    env = {'Any': object, 'torch': SimpleNamespace(Tensor=object)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), env)
    return env[name]

def test_k4_resolver_and_producer_reach_metadata_guard():
    assert function('resolve_bounded_overlap')((16, 4, 2), True, False) is True
    cb = SimpleNamespace(L=16, K=4, V=2)
    with pytest.raises(ValueError, match=r'expects CUDA'):
        function('quantize_from_exact_states')(cb, SimpleNamespace(ndim=2, is_cuda=False))

@pytest.mark.parametrize('k', [1, 4])
def test_complete_state_domain_proves_internal_prefix_range(k):
    shift = k * 2
    prefixes = 1 << (16 - shift)
    states = range(1 << 16)
    assert all(0 <= (state >> shift) < prefixes for state in states)
    assert (65535 >> shift) == prefixes - 1

@pytest.mark.parametrize('geometry,value,profile', [
    ((16,2,2), True, False), ((16,3,2), True, False),
    ((15,4,2), True, False), ((16,4,1), True, False),
    ((16,4,2), 1, False), ((16,4,2), 'true', False),
    ((16,4,2), True, True),
])
def test_reject_unqualified_geometry_type_or_profile(geometry, value, profile):
    with pytest.raises(ValueError):
        function('resolve_bounded_overlap')(geometry, value, profile)

def test_public_entry_cannot_opt_out_of_validation():
    nodes = {n.name: n for n in ast.parse(SOURCE.read_text()).body
             if isinstance(n, ast.FunctionDef)}
    public = nodes['exact_prefix_viterbi']
    assert not public.args.kwonlyargs
    assert '_bounded_overlap=False' in ast.unparse(public)
    assert 'raise ValueError' in ast.unparse(nodes['_validate_overlap_prefixes'])
    assert '_assert_async' not in SOURCE.read_text()
