"""Research-only loop live-range experiment; same recurrence, no pruning."""
import ast
from pathlib import Path
import pytest
SOURCE=Path(__file__).parents[1]/'src/banana_smasher/qtip_viterbi.py'

def test_research_register_cap_is_only_k1_structured():
    tree=ast.parse(SOURCE.read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func).startswith('_persistent_prefix_viterbi_generic[')]
    assert len(calls)==1
    opts={k.arg:ast.unparse(k.value) for k in calls[0].keywords}
    assert opts.get('maxnreg')=='64 if K == 1 and structured_gather else None'

@pytest.mark.parametrize('batch,steps,ties',[(1,2,False),(17,8,False),(256,128,False),(1,128,True)])
def test_device_register_cap_matches_original_recurrence(batch,steps,ties):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('real CUDA required')
    from banana_smasher.qtip_viterbi import _persistent_prefix_viterbi_generic as kernel
    torch.manual_seed(8544)
    x=torch.zeros((2*steps,batch),device='cuda',dtype=torch.float16) if ties else torch.randn((2*steps,batch),device='cuda',dtype=torch.float16)
    lut=torch.zeros((2,65536),device='cuda') if ties else torch.randn((2,65536),device='cuda')
    overlap=torch.randint(0,16384,(batch,),device='cuda',dtype=torch.int32)
    scratch=torch.empty((1,),device='cuda')
    pointers=torch.empty((steps,batch,16384),device='cuda',dtype=torch.uint16)
    for has_overlap in [False,True]:
        outputs=[]
        for disabled in [False,True]:
            out=torch.empty((steps,batch),device='cuda',dtype=torch.int32)
            kernel[(batch,)](x,lut,overlap,scratch,pointers,out,B=batch,STATES=65536,PREFIXES=16384,BRANCHES=4,SHIFT=2,Q_FACTOR=4096,V=2,STEPS=steps,HAS_OVERLAP=has_overlap,REGISTER_COSTS=True,BRANCH_UNROLL=4,STRUCTURED_GATHER=True,BRANCH_POINTERS=False,maxnreg=64 if disabled else None,LUT_EVICTION='evict_last',num_warps=16,num_stages=1)
            outputs.append(out)
        torch.cuda.synchronize()
        assert torch.equal(*outputs)
