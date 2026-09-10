"""Execute real AST admission/install functions without importing CUDA."""
import ast,sys,types
from pathlib import Path
import pytest
R=Path(__file__).resolve().parents[1]/'src/banana_smasher'
def nodes(file):return {n.name:n for n in ast.parse((R/file).read_text()).body if isinstance(n,ast.FunctionDef)}
def run(node,env):
 node=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]));exec(compile(node,'actual-source','exec'),env);return env[node.body[0].name]
def common(label,values):
 if any(v != values[0] for v in values):raise ValueError(label)
 return values[0]
def module():
 m=types.ModuleType('banana_smasher.qtip_viterbi');e={};ns=nodes('qtip_viterbi.py')
 m.resolve_fused_schedule=run(ns['resolve_fused_schedule'],e)
 m.resolve_viterbi_num_warps=run(ns['resolve_viterbi_num_warps'],e)
 return m
@pytest.mark.parametrize('values',[(True,1),(1,True),(False,0),(0,False),(True,False),(False,True)])
def test_actual_batch_rejects_all_members(monkeypatch,values):
 m=module();monkeypatch.setitem(sys.modules,'banana_smasher.qtip_viterbi',m)
 f=run(nodes('qtip_batch_controller.py')['_fused_schedule'],{'__package__':'banana_smasher','_common':common})
 configs=[dict(geometry=dict(L=16,K=1,V=2),projection='fused13',viterbi_distance_alphabet=True,viterbi_num_warps=8,viterbi_fused_schedule=v) for v in values]
 with pytest.raises(ValueError):f(configs)
def test_actual_batch_homogeneous(monkeypatch):
 m=module();monkeypatch.setitem(sys.modules,'banana_smasher.qtip_viterbi',m)
 f=run(nodes('qtip_batch_controller.py')['_fused_schedule'],{'__package__':'banana_smasher','_common':common})
 assert f([{},{}]) is False
 c=dict(geometry=dict(L=16,K=1,V=2),projection='fused13',viterbi_distance_alphabet=True,viterbi_num_warps=8,viterbi_fused_schedule=True)
 assert f([c,c.copy()]) is True

def test_kernel_default_body_is_exact_baseline():
 import subprocess,copy
 text=subprocess.check_output(['git','-C',str(R),'show','7e077e151dc2ceaeafe7a9dee7258d4f4fae4a26:banana-smasher/src/banana_smasher/qtip_viterbi.py'],text=True)
 base=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='_persistent_prefix_viterbi_generic')
 candidate=copy.deepcopy(nodes('qtip_viterbi.py')[base.name])
 class Select(ast.NodeTransformer):
  def visit_If(self,n):
   if isinstance(n.test,ast.Name) and n.test.id=='FUSED_SCHEDULE':return [self.visit(x) for x in n.orelse]
   return self.generic_visit(n)
 candidate=Select().visit(candidate);candidate.args=base.args
 assert ast.dump(candidate,include_attributes=False)==ast.dump(base,include_attributes=False)
