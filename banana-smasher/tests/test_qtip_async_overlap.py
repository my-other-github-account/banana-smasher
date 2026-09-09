"""Same-predicate device-side overlap guard; no unsafe validation bypass."""
from __future__ import annotations
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
SOURCE=Path(__file__).parents[1]/'src/banana_smasher/qtip_viterbi.py'

def nodes():
    return {n.name:n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef)}

def test_async_guard_keeps_predicate_without_host_bool():
    calls=[]
    class Tensor:
        is_cuda=True;device='cuda';ndim=1;dtype='int32'
        def numel(self):return 3
        def __lt__(self,x):calls.append(('lt',x));return self
        def __ge__(self,x):calls.append(('ge',x));return self
        def __or__(self,x):return self
        def any(self):return self
        def __invert__(self):return self
        def __bool__(self):raise AssertionError('host scalar barrier')
    torch=SimpleNamespace(Tensor=Tensor,int32='int32',int64='int64',_assert_async=lambda predicate,msg:calls.append(('assert',msg)))
    env={'torch':torch};node=nodes()['_validate_overlap_prefixes']
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),env)
    env[node.name](Tensor(),x=Tensor(),batch=3,prefixes=16384,async_validation=True)
    assert calls==[('lt',0),('ge',16384),('assert','overlap prefixes must be in [0, 16384)')]

@pytest.mark.parametrize('steps,batch',[(2,1),(8,17),(128,256)])
def test_device_guard_matches_sync(steps,batch):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('physical CUDA required')
    from banana_smasher.qtip_viterbi import exact_prefix_viterbi
    torch.manual_seed(314159)
    cb=SimpleNamespace(L=16,K=1,V=2,lut=torch.randn((2,65536),device='cuda'),_banana_smasher_structured_gather=True,_banana_smasher_branch_unroll=True,_banana_smasher_viterbi_num_warps=16,_banana_smasher_backpointer_dtype='uint16',_banana_smasher_lut_l1_retention=True)
    x=torch.randn((2*steps,batch),device='cuda')
    for overlap in [None,torch.randint(0,16384,(batch,),device='cuda',dtype=torch.int32)]:
        cb._banana_smasher_async_overlap_validation=False;ref=exact_prefix_viterbi(cb,x,overlap)
        cb._banana_smasher_async_overlap_validation=True;got=exact_prefix_viterbi(cb,x,overlap)
        torch.cuda.synchronize();assert torch.equal(ref,got)

def test_invalid_device_guard_is_fatal_in_isolated_process():
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('physical CUDA required')
    import subprocess,sys
    code='import torch; from banana_smasher.qtip_viterbi import _validate_overlap_prefixes; x=torch.zeros((2,1),device="cuda"); p=torch.tensor([16384],device="cuda",dtype=torch.int32); _validate_overlap_prefixes(p,x=x,batch=1,prefixes=16384,async_validation=True); torch.cuda.synchronize()'
    p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=45)
    assert p.returncode!=0
    assert 'device-side assert' in p.stderr or 'overlap prefixes' in p.stderr
