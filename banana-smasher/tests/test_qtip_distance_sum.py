"""Pre-summed alphabet distance: separate numerical and layout gates."""
import ast
from pathlib import Path
import pytest
SOURCE=Path(__file__).parents[1]/'src/banana_smasher/qtip_viterbi.py'
def test_single_distance_gather():
    text=SOURCE.read_text()
    assert 'distance_sum = delta_a * delta_a + delta_b * delta_b' in text
    assert 'candidate = predecessor_cost + tl.gather(distance_sum, alphabet_key, axis=0)' in text

@pytest.mark.parametrize('steps,batch', [(2,1),(128,3)])
def test_cuda_numerical_gate(steps,batch):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available(): pytest.skip('CUDA required')
    from types import SimpleNamespace
    from banana_smasher.qtip_viterbi import exact_prefix_viterbi
    torch.manual_seed(1729)
    tlut=torch.randn((512,2),device='cuda')
    s=torch.arange(65536,device='cuda',dtype=torch.int64);h=s*(s+1)
    lut=tlut[(h>>6)&511].clone();lut[:,0]*=1-2*((h>>15)&1)
    cb=SimpleNamespace(L=16,K=1,V=2,decode_mode='quantlut_sym',tlut_bits=9,lut=lut.T.contiguous(),_banana_smasher_structured_gather=True,_banana_smasher_branch_unroll=True)
    x=torch.randn((steps*2,batch),device='cuda',dtype=torch.float16)
    for overlap in [None,torch.randint(0,16384,(batch,),device='cuda',dtype=torch.int32)]:
        cb._banana_smasher_conditioned_distance_sum=False
        cb._banana_smasher_distance_alphabet=False;ref=exact_prefix_viterbi(cb,x,overlap)
        cb._banana_smasher_distance_alphabet=True
        cb._banana_smasher_conditioned_distance_sum=True
        cb._banana_smasher_projection="down"
        got=exact_prefix_viterbi(cb,x,overlap)
        torch.cuda.synchronize()
        if overlap is None: assert torch.equal(ref,got)
        assert got.shape==ref.shape and got.dtype==ref.dtype
        assert bool(((got>=0)&(got<65536)).all())
        a=lut[ref.long()].reshape(steps,batch,2)
        b=lut[got.long()].reshape(steps,batch,2)
        target=x.reshape(steps,2,batch).permute(0,2,1).float()
        e0=((a-target)**2).sum();e1=((b-target)**2).sum()
        assert bool(torch.isfinite(e1))
        assert e1<=e0*1.0001+1e-6
