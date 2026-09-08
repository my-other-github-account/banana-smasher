"""Real-device paired-pointer wire parity; must not be reported from a skip."""
from types import SimpleNamespace
import pytest
import torch

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('batch,steps,zero', [(1,2,False),(17,8,False),(256,128,False),(1,128,True)])
def test_paired_private_pointers_match_int32_states(batch,steps,zero):
    from banana_smasher.qtip_viterbi import exact_prefix_viterbi
    torch.manual_seed(314159)
    lut = torch.zeros((2,65536),device='cuda') if zero else torch.randn((2,65536),device='cuda')
    cb=SimpleNamespace(L=16,K=1,V=2,lut=lut,_banana_smasher_viterbi_num_warps=16,_banana_smasher_structured_gather=True,_banana_smasher_branch_unroll=True)
    x=torch.zeros((2*steps,batch),device='cuda') if zero else torch.randn((2*steps,batch),device='cuda')
    for overlap in [None, torch.randint(0,16384,(batch,),device='cuda',dtype=torch.int32)]:
        cb._banana_smasher_backpointer_dtype='int32'
        ref=exact_prefix_viterbi(cb,x,overlap)
        cb._banana_smasher_backpointer_dtype='uint16'
        actual=exact_prefix_viterbi(cb,x,overlap)
        torch.cuda.synchronize()
        assert torch.equal(ref,actual)
        assert actual.dtype==torch.int32
        assert actual.shape==(steps,batch)
