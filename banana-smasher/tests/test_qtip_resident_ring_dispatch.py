"""Exercise resident dispatch without GPU allocation or runtime imports."""
from __future__ import annotations
import ast
from collections import defaultdict
import hashlib
import json
import os
from typing import Any
from pathlib import Path
import sys
import time
import types
from collections.abc import Sequence
import pytest

SOURCE=Path(__file__).parents[1]/'src/banana_smasher/solver_qtip_profile.py'

@pytest.mark.parametrize('k',[0,1,2,3,4,5])
def test_resident_dispatch_admits_existing_full16_batch_geometries(tmp_path,monkeypatch,k):
    tree=ast.parse(SOURCE.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main_many')
    module=ast.Module(body=[node],type_ignores=[])
    paths=[tmp_path/'E001_fused13.json',tmp_path/'E002_fused13.json']
    calls=[]
    def batch(chunk,root,layer,**kw):
        calls.append(chunk)
        return [dict(status='PASS',layer=layer,expert=i+1,projection='fused13',assignment_sha256=str(i),total_wall_seconds=1.) for i in range(len(chunk))]
    package=types.ModuleType('banana_smasher');package.__path__=[]
    controller=types.ModuleType('banana_smasher.qtip_batch_controller');controller.main_batch=batch
    monkeypatch.setitem(sys.modules,'banana_smasher',package)
    monkeypatch.setitem(sys.modules,'banana_smasher.qtip_batch_controller',controller)
    env: dict[str, Any] = dict(os=os,Path=Path,Any=object,Sequence=Sequence,time=time,json=json,hashlib=hashlib,defaultdict=defaultdict,
        __package__='banana_smasher',_ordered_qtip_configs=lambda *a,**kw:paths,
        _validated_existing_unit=lambda *a,**kw:None,
        _read_qtip_config=lambda p:dict(geometry=dict(L=16,K=k,V=2),projection='fused13'),
        validate_qtip_projection=lambda p:p,_process_receipt=lambda:dict(pid=1),
        _public_receipt=lambda x:x,_atomic_json=lambda *a:None)
    exec(compile(module,str(SOURCE),'exec'),env)
    if k not in (1,2,3,4):
        with pytest.raises(ValueError, match='refuses serial fallback'):
            env['main_many'](tmp_path,tmp_path,5,batch_size=2)
        assert calls == []
        return
    receipt=env['main_many'](tmp_path,tmp_path,5,batch_size=2)
    assert calls==[paths]
    assert receipt['computed_units']==2 and receipt['cross_unit_batch_size']==2
