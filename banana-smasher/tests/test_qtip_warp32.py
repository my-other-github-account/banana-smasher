"""Bounded K1 scheduling upper-limit experiment; production default stays 16."""
import ast
from pathlib import Path
import pytest
SOURCE=Path(__file__).parents[1]/'src/banana_smasher/qtip_viterbi.py'
def resolver():
    node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='resolve_viterbi_num_warps')
    ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),ns)
    return ns[node.name]
def test_k1_32_warp_bound():
    r=resolver();assert r((16,1,2),32)==32
    assert r((16,1,2),None)==16
    for geom,arg in [((16,3,2),32),((16,2,2),32),((16,1,2),64),((16,1,2),32.0)]:
        with pytest.raises(ValueError):r(geom,arg)
@pytest.mark.parametrize('batch,steps,ties',[(1,2,False),(17,8,False),(256,128,False),(1,128,True)])
def test_device_warp32_matches_original(batch,steps,ties):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('real CUDA required')
    from types import SimpleNamespace
    from banana_smasher.qtip_viterbi import exact_prefix_viterbi
    torch.manual_seed(8544)
    x=torch.zeros((2*steps,batch),device='cuda',dtype=torch.float16) if ties else torch.randn((2*steps,batch),device='cuda',dtype=torch.float16)
    lut=torch.zeros((2,65536),device='cuda') if ties else torch.randn((2,65536),device='cuda')
    cb=SimpleNamespace(L=16,K=1,V=2,lut=lut,_banana_smasher_structured_gather=True,_banana_smasher_branch_unroll=True,_banana_smasher_lut_l1_retention=True,_banana_smasher_backpointer_dtype='uint16')
    for overlap in [None,torch.randint(0,16384,(batch,),device='cuda',dtype=torch.int32)]:
        cb._banana_smasher_viterbi_num_warps=16;ref=exact_prefix_viterbi(cb,x,overlap)
        cb._banana_smasher_viterbi_num_warps=32;got=exact_prefix_viterbi(cb,x,overlap)
        torch.cuda.synchronize();assert torch.equal(ref,got)
