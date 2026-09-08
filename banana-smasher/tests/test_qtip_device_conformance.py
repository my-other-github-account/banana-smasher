"""Exercise actual packed conformance arithmetic; GPU acceptance is separate."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import pytest
import torch

SOURCE=Path(__file__).parents[1]/'src/banana_smasher/qtip_batch.py'

def function() -> Any:
 node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_decode_candidate')
 env: dict[str,Any]=dict(Any=object,torch=torch)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),env)
 return env['_decode_candidate']

@pytest.mark.parametrize('on_device',[False,True])
def test_conformance_retains_exact_and_max_abs_metrics(on_device):
 x=torch.arange(8,dtype=torch.float32).reshape(2,4)/7
 candidate: dict[str,Any]=dict(geometry=dict(L=16,K=3,V=2,tlut_bits=9),shape=[2,4],trellis=torch.zeros(1),Wscale=torch.tensor(1.),SV=torch.ones(2),SU=torch.ones(4),reconstructed_weight=x.half())
 runner=SimpleNamespace(fwht=lambda x:x);cb=SimpleNamespace(lut=torch.zeros(2,2));decoder=SimpleNamespace(decode_compressed=lambda *a:x.clone())
 result=function()(runner,candidate,cb,decoder,torch.device('cpu'),compare_on_device=on_device)
 assert result['fp16_bit_exact'] and result['fp16_bit_equal_fraction']==1
 assert result['max_abs_fp32_vs_stored_fp16']==float((x-x.half().float()).abs().max())
 assert result['conformance_comparison']=='device' if on_device else result['conformance_comparison']=='cpu'
 candidate['reconstructed_weight'][0,0]=100
 with pytest.raises(RuntimeError,match='conformance failed'):
  function()(runner,candidate,cb,decoder,torch.device('cpu'),compare_on_device=on_device)


def test_public_batch_config_pins_and_rejects_mixed_device_conformance():
 path=SOURCE.with_name('qtip_batch_controller.py');tree=ast.parse(path.read_text())
 nodes: list[ast.stmt]=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_common','_packed_conformance_on_device')]
 env: dict[str,Any]=dict(Sequence=list,Mapping=dict,Any=object)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
 resolve=env['_packed_conformance_on_device']
 assert resolve([{}]) is False
 assert resolve([{'packed_conformance_on_device':True}]*2) is True
 for values in [[True,False],[1,1],['yes','yes']]:
  with pytest.raises(ValueError):resolve([{'packed_conformance_on_device':x} for x in values])
 body=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main_batch')
 call=next(n for n in ast.walk(body) if isinstance(n,ast.Call) and ast.unparse(n.func)=='build_qtip_batch')
 assert any(k.arg=='packed_conformance_on_device' for k in call.keywords)
