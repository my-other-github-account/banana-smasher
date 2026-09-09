"""Exact K1 row extraction, including unreachable costs; CUDA checks public API."""
import ast
from pathlib import Path
import types
import numpy as np
import pytest

SOURCE = Path(__file__).parents[1] / 'src/banana_smasher/qtip_viterbi.py'

def helper():
    nodes = [n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_select_k1_cost_row']
    assert len(nodes) == 1, 'K1 split-row selector is not implemented'
    node = nodes[0]; node.decorator_list = []
    tl = types.SimpleNamespace(constexpr=int, reshape=np.reshape, trans=np.transpose,
        split=lambda x: (x[..., 0], x[..., 1]), where=np.where)
    env = {'tl': tl}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), env)
    return env['_select_k1_cost_row']

@pytest.mark.parametrize('q', range(4))
@pytest.mark.parametrize('unreachable', [False, True])
def test_selects_every_exact_predecessor(q, unreachable):
    select = helper()
    costs = np.arange(16384, dtype=np.float32)
    if unreachable: costs[::7] = np.inf
    got = select(costs, q, 4096)
    np.testing.assert_array_equal(got, costs.reshape(4, 4096)[q])

@pytest.mark.parametrize('steps,batch', [(2,1),(128,3)])
def test_cuda_public_structured_vs_generic(steps, batch):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available(): pytest.skip('CUDA required')
    from banana_smasher.qtip_viterbi import exact_prefix_viterbi
    torch.manual_seed(1729)
    cb = types.SimpleNamespace(L=16,K=1,V=2,tlut=torch.randn(65536,2,device='cuda',dtype=torch.float16))
    x = torch.randn(steps*2,batch,device='cuda',dtype=torch.float16)
    for overlap in [None,torch.randint(0,16384,(batch,),device='cuda',dtype=torch.int32)]:
        cb._banana_smasher_structured_gather=False
        ref=exact_prefix_viterbi(cb,x,overlap)
        cb._banana_smasher_structured_gather=True
        got=exact_prefix_viterbi(cb,x,overlap)
        torch.cuda.synchronize()
        assert torch.equal(ref,got)
