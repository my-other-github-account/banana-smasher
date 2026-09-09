"""Signed quantlut alphabet admission and actual opt-in CUDA equivalence."""
import ast
from pathlib import Path
from functools import lru_cache
import pytest

SOURCE=Path(__file__).parents[1]/'src/banana_smasher/qtip_viterbi.py'

def mapping():
    nodes=[n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_signed_alphabet_representatives']
    assert len(nodes)==1, 'signed alphabet map missing'
    env={'lru_cache':lru_cache};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),env)
    return env['_signed_alphabet_representatives']()

def test_original_quantlut_sym_mapping_covers_all_states():
    reps=mapping();assert len(reps)==1024
    for s in range(65536):
        key=((s*(s+1))>>6)&1023
        r=reps[key]
        assert ((r*(r+1))>>6)&1023==key

@pytest.mark.parametrize('steps,batch',[(2,1),(128,3)])
def test_cuda_alphabet_public_path(steps,batch):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from types import SimpleNamespace
    from banana_smasher.qtip_viterbi import exact_prefix_viterbi
    torch.manual_seed(1729)
    tlut=torch.randn((512,2),device='cuda')
    s=torch.arange(65536,device='cuda',dtype=torch.int64);h=s*(s+1)
    lut=tlut[(h>>6)&511].clone();lut[:,0]*=1-2*((h>>15)&1)
    cb=SimpleNamespace(L=16,K=1,V=2,decode_mode='quantlut_sym',tlut_bits=9,lut=lut.T.contiguous(),_banana_smasher_structured_gather=True,_banana_smasher_branch_unroll=True)
    x=torch.randn((steps*2,batch),device='cuda',dtype=torch.float16)
    for overlap in [None,torch.randint(0,16384,(batch,),device='cuda',dtype=torch.int32)]:
        cb._banana_smasher_distance_alphabet=False;ref=exact_prefix_viterbi(cb,x,overlap)
        cb._banana_smasher_distance_alphabet=True;got=exact_prefix_viterbi(cb,x,overlap)
        torch.cuda.synchronize();assert torch.equal(ref,got)
    cb.lut[0,0]+=1
    with pytest.raises(ValueError,match='alphabet'):
        exact_prefix_viterbi(cb,x)
