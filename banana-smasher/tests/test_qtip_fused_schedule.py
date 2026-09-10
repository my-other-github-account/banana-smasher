"""CPU admission, installer propagation and frozen kernel branch checks."""
import ast
from pathlib import Path
import types
import pytest
ROOT=Path(__file__).resolve().parents[1]/'src/banana_smasher'
def load(name, function, env=None):
 t=ast.parse((ROOT/name).read_text());n=next((n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==function),None)
 assert n is not None, f'missing {function}'
 e={} if env is None else env
 exec(compile(ast.Module(body=[n],type_ignores=[]),name,'exec'),e)
 return e[function]
def resolver():return load('qtip_viterbi.py','resolve_fused_schedule')
def test_default_and_supported():
 r=resolver();assert r((16,1,2),'fused13',True,True,False,8) is True
 assert r((16,3,2),'down',None,False,False,16) is False
 assert r((16,3,2),'down',False,False,False,16) is False
@pytest.mark.parametrize('value',[0,1,'yes',[],{}])
def test_nonboolean(value):
 with pytest.raises(ValueError):resolver()((16,1,2),'fused13',value,True,False,8)
@pytest.mark.parametrize('geometry,projection,alphabet,conditioned,warps',[
 ((16,3,2),'fused13',True,False,8),((16,1,2),'down',True,False,8),
 ((16,1,2),'fused13',False,False,8),((16,1,2),'fused13',True,True,8),
 ((16,1,2),'fused13',True,False,16)])
def test_reject_untested(geometry,projection,alphabet,conditioned,warps):
 with pytest.raises(ValueError):resolver()(geometry,projection,True,alphabet,conditioned,warps)
def test_public_installer_restores_default():
 t=ast.parse((ROOT/'qtip_viterbi.py').read_text());f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='install_exact_prefix_viterbi')
 assert any(isinstance(n,ast.Assign) and any(isinstance(x,ast.Attribute) and x.attr=='_banana_smasher_fused_schedule' for x in n.targets) and isinstance(n.value,ast.Constant) and n.value.value is False for n in f.body)
def test_configured_and_batch_wiring():
 s=(ROOT/'solver_qtip_profile.py').read_text();assert 'cb._banana_smasher_fused_schedule = fused_schedule' in s
 assert 'viterbi_fused_schedule=True' in s
 t=ast.parse((ROOT/'qtip_batch_controller.py').read_text());main=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='main_batch')
 assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_fused_schedule' for n in ast.walk(main))
def test_launch_binds_actual_option():
 t=ast.parse((ROOT/'qtip_viterbi.py').read_text());f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='_exact_prefix_viterbi_impl')
 assert any(isinstance(n,ast.keyword) and n.arg=='FUSED_SCHEDULE' and isinstance(n.value,ast.Name) and n.value.id=='fused_schedule' for n in ast.walk(f))

def test_actual_configured_installer_enables_and_restores():
 from banana_smasher import solver_qtip_profile as sp
 cb=types.SimpleNamespace(L=16,K=1,V=2,quantize_seq=lambda *a:None,quantize=lambda *a:None)
 exact=types.SimpleNamespace(geometry=lambda cb:{'L':16,'K':1,'V':2})
 config=dict(geometry=dict(L=16,K=1,V=2),projection='fused13',viterbi_num_warps=8,viterbi_distance_alphabet=True,viterbi_fused_schedule=True)
 identity=sp._install_configured_viterbi(cb,exact,sp._ExactTimers(),config,profile_mode=False)
 assert cb._banana_smasher_fused_schedule is True
 assert identity['viterbi_fused_schedule'] is True and identity['production_default'] is False
 config.pop('viterbi_fused_schedule')
 identity=sp._install_configured_viterbi(cb,exact,sp._ExactTimers(),config,profile_mode=False)
 assert cb._banana_smasher_fused_schedule is False and 'viterbi_fused_schedule' not in identity
