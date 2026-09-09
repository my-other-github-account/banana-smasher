"""Distance algebra is opt-in: default arithmetic and safety stay unchanged."""
from __future__ import annotations
import ast
from pathlib import Path
import pytest

SOURCE = Path(__file__).parents[1] / 'src/banana_smasher/qtip_viterbi.py'

def resolver():
    node = next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'resolve_distance_polynomial')
    env = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), env)
    return env[node.name]

@pytest.mark.parametrize('value', [None, False, True])
def test_admission(value):
    assert resolver()((16, 1, 2), value) is (value is True)

@pytest.mark.parametrize('geometry,value', [((16,3,2),True), ((16,2,2),True), ((16,1,2),1), ((16,1,2),'true')])
def test_reject(geometry, value):
    with pytest.raises(ValueError):
        resolver()(geometry, value)

def test_public_binding_and_memory_budget():
    source = SOURCE.read_text()
    assert 'DISTANCE_POLYNOMIAL=distance_polynomial' in source
    assert 'lut_bytes=cb.lut.numel() * cb.lut.element_size() * (3 if distance_polynomial else 1)' in source
    assert 'viterbi_distance_polynomial' in SOURCE.with_name('solver_qtip_profile.py').read_text()
    tree = ast.parse(SOURCE.with_name('qtip_batch_controller.py').read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='_common' and 'viterbi_distance_polynomial' in ast.unparse(n)]
    assert len(calls) == 1

@pytest.mark.parametrize('steps,batch,zero', [(2,1,False),(8,17,False),(128,256,False),(128,1,True)])
def test_device_bounded_objective(steps,batch,zero):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('physical CUDA required')
    from types import SimpleNamespace
    from banana_smasher.qtip_viterbi import exact_prefix_viterbi
    torch.manual_seed(314159)
    lut = torch.zeros((2,65536),device='cuda') if zero else torch.randn((2,65536),device='cuda')
    x = torch.zeros((steps*2,batch),device='cuda') if zero else torch.randn((steps*2,batch),device='cuda')
    cb = SimpleNamespace(L=16,K=1,V=2,lut=lut,_banana_smasher_structured_gather=True,_banana_smasher_branch_unroll=True,_banana_smasher_viterbi_num_warps=16,_banana_smasher_backpointer_dtype='uint16',_banana_smasher_lut_l1_retention=True)
    def cost(states):
        values=lut[:,states.long()].permute(1,0,2).reshape(steps*2,batch)
        return ((values-x).double().square()).sum(0)
    for overlap in [None,torch.randint(0,16384,(batch,),device='cuda',dtype=torch.int32)]:
        cb._banana_smasher_distance_polynomial=False
        baseline=exact_prefix_viterbi(cb,x,overlap)
        cb._banana_smasher_distance_polynomial=True
        candidate=exact_prefix_viterbi(cb,x,overlap)
        torch.cuda.synchronize()
        assert candidate.shape==baseline.shape and candidate.dtype==torch.int32
        assert bool(((candidate>=0)&(candidate<65536)).all())
        # Not a byte gate: actual full-state sequence objective tolerance.
        assert bool((cost(candidate)<=cost(baseline)*1.0001+1e-9).all())
        if zero: assert torch.equal(baseline,candidate)
