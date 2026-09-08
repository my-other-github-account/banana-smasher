"""Default-preserving LUT policy admission and device checks."""
from __future__ import annotations
import ast
from pathlib import Path
import pytest

SOURCE=Path(__file__).parents[1]/'src/banana_smasher/qtip_viterbi.py'

def resolver():
    tree=ast.parse(SOURCE.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='resolve_lut_l1_retention')
    ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),ns)
    return ns[node.name]

@pytest.mark.parametrize('value',[None,False,True])
def test_qualified_policy(value):
    assert resolver()((16,1,2),value) is (value is True)

@pytest.mark.parametrize('geometry,value',[((16,3,2),True),((16,2,2),True),((16,1,2),1),((16,1,2),'true')])
def test_unqualified_policy_rejected(geometry,value):
    with pytest.raises(ValueError):resolver()(geometry,value)

def test_only_lut_loads_use_policy():
    tree=ast.parse(SOURCE.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_persistent_prefix_viterbi_generic')
    loads=[n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='load']
    lut=[n for n in loads if 'lut_ptr' in ast.unparse(n.args[0])]
    assert len(lut)==5
    for n in lut:assert any(k.arg=='eviction_policy' and ast.unparse(k.value)=='LUT_EVICTION' for k in n.keywords)
    for n in loads:
        if n not in lut:assert not any(k.arg in ('cache_modifier','eviction_policy') for k in n.keywords)
    assert isinstance(node.args.defaults[-1],ast.Constant) and node.args.defaults[-1].value==''

@pytest.mark.parametrize('batch,steps,zero',[(1,2,False),(17,8,False),(256,128,False),(1,128,True)])
def test_device_optin_matches_incumbent(batch,steps,zero):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('CUDA required; skip is not device evidence')
    from types import SimpleNamespace
    from banana_smasher.qtip_viterbi import exact_prefix_viterbi
    torch.manual_seed(314159)
    lut=torch.zeros((2,65536),device='cuda') if zero else torch.randn((2,65536),device='cuda')
    cb=SimpleNamespace(L=16,K=1,V=2,lut=lut,_banana_smasher_structured_gather=True,_banana_smasher_branch_unroll=True)
    x=torch.zeros((2*steps,batch),device='cuda') if zero else torch.randn((2*steps,batch),device='cuda')
    for overlap in [None,torch.randint(0,16384,(batch,),device='cuda',dtype=torch.int32)]:
        cb._banana_smasher_lut_l1_retention=False;ref=exact_prefix_viterbi(cb,x,overlap)
        cb._banana_smasher_lut_l1_retention=True;got=exact_prefix_viterbi(cb,x,overlap)
        torch.cuda.synchronize();assert torch.equal(ref,got)

@pytest.mark.parametrize('enabled',[False,True])
def test_public_installer_binds_retention(monkeypatch,enabled):
    import sys,types
    source=SOURCE.with_name('solver_qtip_profile.py')
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_install_configured_viterbi')
    package=types.ModuleType('banana_smasher');package.__path__=[]
    module=types.ModuleType('banana_smasher.qtip_viterbi')
    module.resolve_viterbi_num_warps=lambda *a:16
    module.resolve_backpointer_dtype=lambda *a:'int32'
    module.resolve_structured_gather=lambda *a:False
    module.resolve_branch_unroll=lambda *a:1
    module.resolve_lut_l1_retention=resolver()
    monkeypatch.setitem(sys.modules,'banana_smasher',package)
    monkeypatch.setitem(sys.modules,'banana_smasher.qtip_viterbi',module)
    env=dict(__name__='banana_smasher.solver_qtip_profile',_ExactTimers=object,Any=object,known_qtip_geometries=lambda:{(16,1,2)},backend_for_geometry=lambda g:'persistent',PERSISTENT_BACKENDS={'persistent'},_install_profiled_exact_viterbi=lambda *a,**k:{})
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),env)
    cb=types.SimpleNamespace(L=16,K=1,V=2)
    identity=env['_install_configured_viterbi'](cb,None,None,{'geometry':{'L':16,'K':1,'V':2},'viterbi_lut_l1_retention':enabled},profile_mode=False)
    assert cb._banana_smasher_lut_l1_retention is enabled
    assert identity.get('viterbi_lut_l1_retention',False) is enabled
    if enabled:assert identity['production_default'] is False

def test_batch_checks_retention_homogeneity():
    tree=ast.parse(SOURCE.with_name('qtip_batch_controller.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='_common' and 'viterbi_lut_l1_retention' in ast.unparse(n)]
    assert len(calls)==1
