"""CPU admission for opt-in full-branch unrolling; not GPU acceptance."""
from __future__ import annotations
import ast
from pathlib import Path
import pytest

SOURCE = Path(__file__).parents[1] / 'src/banana_smasher/qtip_viterbi.py'

def resolver():
    nodes: list[ast.stmt] = [n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'resolve_branch_unroll']
    assert len(nodes) == 1, 'missing full-branch unroll admission'
    env = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), env)
    return env['resolve_branch_unroll']

@pytest.mark.parametrize('value,expected', [(None,1),(False,1),(True,4)])
def test_k1_unroll(value, expected):
    assert resolver()((16,1,2), value) == expected

@pytest.mark.parametrize('value', [1,4,'true',[],{}])
def test_refuse_non_boolean(value):
    with pytest.raises(ValueError): resolver()((16,1,2),value)

@pytest.mark.parametrize('value,expected', [(None,1),(False,1),(True,4)])
def test_k3_unroll(value, expected):
    assert resolver()((16,3,2), value) == expected

@pytest.mark.parametrize('geometry', [(16,2,2),(16,4,2),(17,1,2)])
def test_refuse_unmeasured_geometry(geometry):
    with pytest.raises(ValueError): resolver()(geometry,True)


def test_public_installer_binds_unroll(monkeypatch):
    import sys, types
    source=SOURCE.with_name('solver_qtip_profile.py')
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_install_configured_viterbi')
    package=types.ModuleType('banana_smasher'); package.__path__=[]
    module=types.ModuleType('banana_smasher.qtip_viterbi')
    module.resolve_viterbi_num_warps=lambda *a:16
    module.resolve_backpointer_dtype=lambda *a:'int32'
    module.resolve_branch_unroll=resolver()
    module.resolve_lut_l1_retention = lambda *a: False
    module.resolve_structured_gather = lambda *a: False
    monkeypatch.setitem(sys.modules,'banana_smasher',package)
    monkeypatch.setitem(sys.modules,'banana_smasher.qtip_viterbi',module)
    env=dict(__name__='banana_smasher.solver_qtip_profile',_ExactTimers=object,Any=object,known_qtip_geometries=lambda:{(16,1,2)},backend_for_geometry=lambda g:'persistent',PERSISTENT_BACKENDS={'persistent'},_install_profiled_exact_viterbi=lambda *a,**k:{})
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),env)
    cb=types.SimpleNamespace(L=16,K=1,V=2)
    identity=env['_install_configured_viterbi'](cb,None,None,{'geometry':{'L':16,'K':1,'V':2},'viterbi_branch_unroll':True},profile_mode=False)
    assert getattr(cb,'_banana_smasher_branch_unroll',None) is True
    assert identity['viterbi_branch_unroll']==4 and identity['production_default'] is False

@pytest.mark.parametrize('kernel_name', ['_persistent_prefix_viterbi_generic', '_persistent_prefix_viterbi'])
def test_kernel_uses_bounded_full_branch_unroll(kernel_name):
    tree=ast.parse(SOURCE.read_text())
    kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==kernel_name)
    loops=[n for n in ast.walk(kernel) if isinstance(n,ast.For) and ast.unparse(n.target)=='q']
    assert len(loops)==2
    for loop in loops:
        assert any(k.arg=='loop_unroll_factor' and ast.unparse(k.value)=='BRANCH_UNROLL' for k in loop.iter.keywords)
    body=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_exact_prefix_viterbi_impl')
    assert 'BRANCH_UNROLL=branch_unroll' in ast.unparse(body)


def test_batch_refuses_mixed_branch_schedules():
    source=SOURCE.with_name('qtip_batch_controller.py').read_text()
    tree=ast.parse(source)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='_common']
    assert any('viterbi_branch_unroll' in ast.unparse(n) for n in calls)
