"""Actual grouped reduction algebra; physical panel still required."""
import ast
from pathlib import Path
import types
import numpy as np
import pytest

SOURCE=Path(__file__).parents[1]/'src/banana_smasher/qtip_viterbi.py'

def load(name, **env):
    node=next((n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name),None)
    assert node is not None, f'missing {name}'
    node.decorator_list=[]
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),env)
    return env[name]

@pytest.mark.parametrize('fixture',['finite','ties','inf','nan'])
def test_actual_group_min_matches_ordered_strict_updates(fixture):
    rng=np.random.default_rng(813)
    candidates=rng.random((4,1024),dtype=np.float32)
    states=np.arange(4096,dtype=np.int32).reshape(4,1024)
    if fixture=='ties':candidates[:]=1
    if fixture=='inf':candidates[:,::3]=np.inf
    if fixture=='nan':candidates[::2,::3]=np.nan
    tl=types.SimpleNamespace(where=np.where,min=lambda x,axis:np.min(x,axis=axis))
    best,chosen=load('_strict_branch_group_min',tl=tl)(candidates,states)
    reference=np.full(1024,np.inf,dtype=np.float32);indices=np.zeros(1024,dtype=np.int32)
    for q in range(4):
        take=candidates[q]<reference
        reference=np.where(take,candidates[q],reference);indices=np.where(take,states[q],indices)
    # Infinite groups never update the outer incumbent; state is immaterial there.
    assert np.array_equal(best,reference)
    assert np.array_equal(chosen[np.isfinite(best)],indices[np.isfinite(best)])


def test_public_grouped_branches_reaches_specialized_kernel():
    tree=ast.parse(SOURCE.read_text())
    body=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='exact_prefix_viterbi')
    calls=[n for n in ast.walk(body) if isinstance(n,ast.Call) and '_persistent_prefix_viterbi[' in ast.unparse(n.func)]
    assert len(calls)==1
    assert any(k.arg=='GROUP_BRANCHES' and ast.unparse(k.value)=='group_branches' for k in calls[0].keywords)
    kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_persistent_prefix_viterbi')
    assert sum(isinstance(n,ast.Call) and ast.unparse(n.func)=='_strict_branch_group_min' for n in ast.walk(kernel))==2
    install=SOURCE.with_name('solver_qtip_profile.py').read_text()
    batch=SOURCE.with_name('qtip_batch_controller.py').read_text()
    assert 'viterbi_branch_grouped' in install and 'viterbi_branch_grouped' in batch


@pytest.mark.parametrize('value,k,structured,unroll,valid',[
    (True,3,False,1,True),(False,1,False,1,True),
    (1,3,False,1,False),(None,3,False,1,False),
    (True,1,False,1,False),(True,3,True,1,False),(True,3,False,4,False),
])
def test_actual_public_installer_admission(monkeypatch,value,k,structured,unroll,valid):
    import sys
    source=SOURCE.with_name('solver_qtip_profile.py')
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_install_configured_viterbi')
    package=types.ModuleType('banana_smasher');package.__path__=[]
    module=types.ModuleType('banana_smasher.qtip_viterbi')
    module.resolve_viterbi_num_warps=lambda *a:8
    module.resolve_backpointer_dtype=lambda *a:'int32'
    module.resolve_branch_unroll=lambda *a:unroll
    module.resolve_structured_gather=lambda *a:structured
    monkeypatch.setitem(sys.modules,'banana_smasher',package)
    monkeypatch.setitem(sys.modules,'banana_smasher.qtip_viterbi',module)
    env=dict(__name__='banana_smasher.solver_qtip_profile',_ExactTimers=object,Any=object,
        known_qtip_geometries=lambda:{(16,k,2)},backend_for_geometry=lambda g:'persistent',
        PERSISTENT_BACKENDS={'persistent'},_install_profiled_exact_viterbi=lambda *a,**kw:{})
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),env)
    cb=types.SimpleNamespace(L=16,K=k,V=2)
    cfg={'geometry':{'L':16,'K':k,'V':2},'viterbi_branch_grouped':value}
    if not valid:
        with pytest.raises(ValueError,match='viterbi_branch_grouped'):
            env['_install_configured_viterbi'](cb,None,None,cfg,profile_mode=False)
    else:
        result=env['_install_configured_viterbi'](cb,None,None,cfg,profile_mode=False)
        assert cb._banana_smasher_branch_grouped is value
        if value:assert result['viterbi_branch_grouped']==4 and result['production_default'] is False


def test_grouped_variables_do_not_change_overlap_loop_carried_ranks():
    tree=ast.parse(SOURCE.read_text())
    kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_persistent_prefix_viterbi')
    groups=[n for n in ast.walk(kernel) if isinstance(n,ast.If) and ast.unparse(n.test)=='GROUP_BRANCHES']
    assert len(groups)==2
    for node in groups:
        assigned={n.id for stmt in node.body for n in ast.walk(stmt) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
        assert not assigned.intersection({'q','state','lut0','lut1','candidate','predecessor_prefix','predecessor_cost'})
