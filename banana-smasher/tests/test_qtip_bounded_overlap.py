import ast
from pathlib import Path
import pytest
SOURCE=Path(__file__).parents[1]/'src/banana_smasher/qtip_viterbi.py'
def resolver():
 t=ast.parse(SOURCE.read_text());nodes=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='resolve_bounded_overlap']
 assert len(nodes)==1,'missing bounded-overlap opt-in resolver'
 env={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),env);return env['resolve_bounded_overlap']
def test_default_off_geometry_type_profile():
 f=resolver()
 assert f((16,1,2),None,False) is False
 assert f((16,1,2),False,False) is False
 assert f((16,1,2),True,False) is True
 for g,v,p in [((16,2,2),True,False),((16,1,2),1,False),((16,1,2),'true',False),((16,1,2),True,True)]:
  with pytest.raises(ValueError):f(g,v,p)
def test_public_strict_no_device_assert():
 t=ast.parse(SOURCE.read_text());nodes={n.name:n for n in t.body if isinstance(n,ast.FunctionDef)}
 assert '_bounded_overlap=False' in ast.unparse(nodes['exact_prefix_viterbi'])
 assert not nodes['exact_prefix_viterbi'].args.kwonlyargs
 assert '_assert_async' not in SOURCE.read_text()
 assert 'raise ValueError' in ast.unparse(nodes['_validate_overlap_prefixes'])
def test_config_and_batch_source_binding():
 s=SOURCE.with_name('solver_qtip_profile.py').read_text();b=SOURCE.with_name('qtip_batch_controller.py').read_text()
 assert 'bounded_overlap=bounded_overlap' in s
 assert 'viterbi_bounded_overlap=True' in s
 assert '_common("Viterbi bounded overlap"' in b
 assert 'if bounded_overlap:' in s


def test_installer_opt_in_then_default_restores_native():
 import types
 tree=ast.parse(SOURCE.with_name('solver_qtip_profile.py').read_text())
 node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_install_profiled_exact_viterbi')
 env=dict(Any=object,_ExactTimers=object,torch=types.SimpleNamespace(Tensor=object),types=types)
 exec(compile(ast.Module(body=[node],type_ignores=[]),'installer','exec'),env)
 native=lambda x: ('native', x)
 cb=types.SimpleNamespace(L=16,K=1,V=2,quantize=native,quantize_seq=lambda x,overlap=None: x)
 calls=[]
 exact=types.SimpleNamespace(geometry=lambda cb:{},quantize_from_exact_states=lambda cb,x: calls.append(x) or ('bounded', x),exact_prefix_viterbi=lambda cb,x,overlap=None: x)
 timers=types.SimpleNamespace(calls=0,sequences=0)
 f=env['_install_profiled_exact_viterbi']
 f(cb,exact,timers,profile_mode=False)
 assert cb.quantize is native
 f(cb,exact,timers,profile_mode=False,bounded_overlap=True)
 x=types.SimpleNamespace(shape=(8193,256))
 assert cb.quantize(x)==('bounded',x)
 assert timers.calls==66 and timers.sequences==16896
 f(cb,exact,timers,profile_mode=False)
 assert cb.quantize is native and len(calls)==1


@pytest.mark.parametrize('values', [(True,1),(1,True),(False,0),(0,False),(True,False)])
def test_every_batch_flag_is_validated(values):
 from banana_smasher import qtip_batch_controller as batch
 assert hasattr(batch, '_bounded_overlap'), 'missing per-config batch admission'
 configs=[dict(geometry=dict(L=16,K=1,V=2),viterbi_bounded_overlap=v) for v in values]
 with pytest.raises(ValueError):batch._bounded_overlap(configs)

def test_batch_default_and_enabled():
 from banana_smasher.qtip_batch_controller import _bounded_overlap
 assert _bounded_overlap([{},{}]) is False
 assert _bounded_overlap([dict(geometry=dict(L=16,K=1,V=2),viterbi_bounded_overlap=True)]*2) is True
