"""Research-only loop live-range experiment; same recurrence, no pruning."""
import ast
from pathlib import Path
import pytest
SOURCE=Path(__file__).parents[1]/'src/banana_smasher/qtip_viterbi.py'

def test_recurrence_can_disable_licm_without_changing_step_domain():
    node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_persistent_prefix_viterbi_generic')
    loops=[n for n in ast.walk(node) if isinstance(n,ast.For) and ast.unparse(n.target)=='step']
    assert len(loops)==1, 'recurrence needs explicit tl.range LICM control'
    call=loops[0].iter
    assert ast.unparse(call.func)=='tl.range'
    assert [ast.unparse(a) for a in call.args]==['1','STEPS']
    assert any(k.arg=='disable_licm' and ast.unparse(k.value)=='DISABLE_LICM' for k in call.keywords)
    assert not any(isinstance(n,ast.AugAssign) and ast.unparse(n.target)=='step' for n in ast.walk(node))

@pytest.mark.parametrize('batch,steps,ties',[(1,2,False),(17,8,False),(256,128,False),(1,128,True)])
def test_device_licm_matches_original_recurrence(batch,steps,ties):
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
            kernel[(batch,)](x,lut,overlap,scratch,pointers,out,B=batch,STATES=65536,PREFIXES=16384,BRANCHES=4,SHIFT=2,Q_FACTOR=4096,V=2,STEPS=steps,HAS_OVERLAP=has_overlap,REGISTER_COSTS=True,BRANCH_UNROLL=4,STRUCTURED_GATHER=True,BRANCH_POINTERS=False,DISABLE_LICM=disabled,LUT_EVICTION='evict_last',num_warps=16,num_stages=1)
            outputs.append(out)
        torch.cuda.synchronize()
        assert torch.equal(*outputs)
