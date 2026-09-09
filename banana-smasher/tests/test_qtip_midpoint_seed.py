"""First pass consumes only one midpoint seed; later path states stay full."""
from __future__ import annotations
import ast
from pathlib import Path
import pytest
SOURCE=Path(__file__).parents[1]/'src/banana_smasher/qtip_viterbi.py'

def test_midpoint_gate_is_explicit_and_default_off():
    n=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='exact_prefix_viterbi')
    args={a.arg:d for a,d in zip(n.args.kwonlyargs,n.args.kw_defaults)}
    assert 'midpoint_only' in args and isinstance(args['midpoint_only'],ast.Constant) and args['midpoint_only'].value is False
    assert 'MIDPOINT_ONLY=midpoint_only' in SOURCE.read_text()

@pytest.mark.parametrize('batch',[1,17,256])
def test_device_midpoint_equals_full_seed(batch):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('physical CUDA required')
    from types import SimpleNamespace
    from banana_smasher.qtip_viterbi import exact_prefix_viterbi
    torch.manual_seed(314159)
    cb=SimpleNamespace(L=16,K=1,V=2,lut=torch.randn((2,65536),device='cuda'),_banana_smasher_structured_gather=True,_banana_smasher_branch_unroll=True,_banana_smasher_viterbi_num_warps=16,_banana_smasher_backpointer_dtype='uint16',_banana_smasher_lut_l1_retention=True)
    x=torch.randn((256,batch),device='cuda')
    full=exact_prefix_viterbi(cb,x)
    midpoint=exact_prefix_viterbi(cb,x,midpoint_only=True)
    torch.cuda.synchronize()
    assert midpoint.shape==(1,batch)
    assert torch.equal(midpoint,full[64:65])
    with pytest.raises(ValueError):exact_prefix_viterbi(cb,x,torch.zeros(batch,device='cuda',dtype=torch.int32),midpoint_only=True)
