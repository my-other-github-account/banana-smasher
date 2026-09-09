"""Projection-constrained numerical opt-in; fused/default remain untouched."""
import ast
from pathlib import Path
import pytest
SOURCE=Path(__file__).parents[1]/'src/banana_smasher/qtip_viterbi.py'
def resolver():
    tree=ast.parse(SOURCE.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='resolve_conditioned_distance_sum']
    assert len(nodes)==1,'missing fail-closed projection admission'
    env={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),env);return env[nodes[0].name]
def test_default_off_and_down_only():
    f=resolver();assert f((16,1,2),'down',None,True) is False
    assert f((16,1,2),'down',True,True) is True
    for g,p,v,a in [((16,1,2),'fused13',True,True),((16,2,2),'down',True,True),((16,1,2),'down',1,True),((16,1,2),'down',True,False)]:
        with pytest.raises(ValueError):f(g,p,v,a)
def test_public_installer_and_batch_bind_flag():
    s=SOURCE.with_name('solver_qtip_profile.py').read_text();b=SOURCE.with_name('qtip_batch_controller.py').read_text()
    assert 'cb._banana_smasher_conditioned_distance_sum = conditioned_distance_sum' in s
    assert 'viterbi_conditioned_distance_sum=True' in s
    assert '_common("Viterbi conditioned distance sum"' in b
