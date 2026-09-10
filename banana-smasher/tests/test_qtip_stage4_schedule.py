"""Execute actual admission/installer AST without importing accelerator modules."""
import ast
import copy
import subprocess
import sys
import types
from pathlib import Path
import pytest
R = Path(__file__).resolve().parents[1] / 'src/banana_smasher'

def nodes(file):
    return {n.name: n for n in ast.parse((R/file).read_text()).body if isinstance(n, ast.FunctionDef)}

def load(file, name, env=None):
    env = {} if env is None else env
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), nodes(file)[name]], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), 'actual-source', 'exec'), env)
    return env[name]

def module(monkeypatch):
    m = types.ModuleType('banana_smasher.qtip_viterbi')
    env = {}
    for name in nodes('qtip_viterbi.py'):
        if name.startswith('resolve_'):
            setattr(m, name, load('qtip_viterbi.py', name, env))
    monkeypatch.setitem(sys.modules, m.__name__, m)
    return m

def config(projection='down', value=True):
    return dict(geometry=dict(L=16,K=1,V=2), projection=projection,
                viterbi_distance_alphabet=True, viterbi_num_warps=16 if projection=='down' else 8,
                viterbi_conditioned_distance_sum=projection=='down',
                viterbi_fused_schedule=projection=='fused13', viterbi_stage4_schedule=value)

def batch(monkeypatch):
    module(monkeypatch)
    def common(label, values):
        if any(v != values[0] for v in values):
            raise ValueError(label)
        return values[0]
    return load('qtip_batch_controller.py', '_stage4_schedule', dict(__package__='banana_smasher', _common=common))

@pytest.mark.parametrize('projection',['down','fused13'])
def test_admission_and_null(monkeypatch, projection):
    f = batch(monkeypatch)
    c = config(projection)
    assert f([c,c.copy()]) is True
    if projection=='fused13':
        c['viterbi_conditioned_distance_sum']=None
    else:
        c['viterbi_fused_schedule']=None
    assert f([c,c.copy()]) is True
    for v in (False,None):
        c['viterbi_stage4_schedule']=v
        assert f([c,c.copy()]) is False
    assert f([{},{}]) is False

@pytest.mark.parametrize('values',[(True,1),(1,True),(False,0),(0,False),(True,False),(False,True),('yes',True),(True,[])])
def test_every_member_before_homogeneity(monkeypatch, values):
    with pytest.raises(ValueError):
        batch(monkeypatch)([config(value=v) for v in values])

@pytest.mark.parametrize('change',[
    dict(geometry=dict(L=16,K=3,V=2)),dict(projection='up'),
    dict(viterbi_distance_alphabet=False),dict(viterbi_num_warps=8),
    dict(viterbi_conditioned_distance_sum=False),dict(viterbi_fused_schedule=True)])
def test_reject_unmeasured_down(monkeypatch, change):
    c=config();c.update(change)
    with pytest.raises(ValueError):batch(monkeypatch)([c])

@pytest.mark.parametrize('change',[dict(viterbi_fused_schedule=False),dict(viterbi_num_warps=16),dict(viterbi_conditioned_distance_sum=True)])
def test_reject_unmeasured_fused(monkeypatch,change):
    c=config('fused13');c.update(change)
    with pytest.raises(ValueError):batch(monkeypatch)([c])

@pytest.mark.parametrize('projection',['down','fused13'])
def test_actual_configured_and_public_reset(monkeypatch, projection):
    module(monkeypatch)
    env=dict(__package__='banana_smasher',known_qtip_geometries=lambda:[(16,1,2)],
             backend_for_geometry=lambda g:'persistent', PERSISTENT_BACKENDS={'persistent'},
             _install_profiled_exact_viterbi=lambda *a,**k:{})
    install=load('solver_qtip_profile.py','_install_configured_viterbi',env)
    cb=types.SimpleNamespace(L=16,K=1,V=2)
    c=config(projection)
    identity=install(cb,None,None,c,profile_mode=False)
    assert cb._banana_smasher_stage4_schedule is True
    assert identity['viterbi_num_stages']==4 and identity['production_default'] is False
    for v in (None,False,'absent'):
        c=config(projection)
        if v=='absent':c.pop('viterbi_stage4_schedule')
        else:c['viterbi_stage4_schedule']=v
        cb._banana_smasher_stage4_schedule=True
        identity=install(cb,None,None,c,profile_mode=False)
        assert cb._banana_smasher_stage4_schedule is False
        assert 'viterbi_stage4_schedule' not in identity
    public=load('qtip_viterbi.py','install_exact_prefix_viterbi',dict(_require_triton=lambda:None,types=types,geometry=lambda cb:{}))
    cb._banana_smasher_stage4_schedule=True
    public(cb)
    assert cb._banana_smasher_stage4_schedule is False

def test_launch_and_default_source_identity():
    path='banana-smasher/src/banana_smasher/qtip_viterbi.py'
    old=subprocess.check_output(['git','-C',str(R),'show','feb61b4f4bc55c0280b426f4d665af2cfcbb2eee:'+path],text=True)
    source=ast.parse((R/'qtip_viterbi.py').read_text())
    launches=[n for n in ast.walk(source) if isinstance(n,ast.keyword) and n.arg=='num_stages']
    assert len(launches)==2
    for n in launches:
        assert ast.dump(n.value)==ast.dump(ast.parse('4 if stage4_schedule else 1',mode='eval').body)
    class Disable(ast.NodeTransformer):
        def visit_Assign(self,n):
            if any(isinstance(t,ast.Name) and t.id=='stage4_schedule' or isinstance(t,ast.Attribute) and t.attr=='_banana_smasher_stage4_schedule' for t in n.targets):return None
            return self.generic_visit(n)
        def visit_IfExp(self,n):
            if isinstance(n.test,ast.Name) and n.test.id=='stage4_schedule':return n.orelse
            return self.generic_visit(n)
    source=Disable().visit(source)
    source.body=[n for n in source.body if not isinstance(n,ast.FunctionDef) or n.name!='resolve_stage4_schedule']
    assert ast.dump(source,include_attributes=False)==ast.dump(ast.parse(old),include_attributes=False)

def test_batch_entry_wires_admission():
    main=nodes('qtip_batch_controller.py')['main_batch']
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_stage4_schedule' for n in ast.walk(main))
