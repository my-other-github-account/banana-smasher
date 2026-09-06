"""Exercise the real orchestration with only the accelerator boundary stubbed."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
import pytest


@pytest.mark.parametrize("target_layer", [3, 4, 5])
def test_calibrated_cell_uses_public_solver_and_exact_capture_roster(tmp_path, monkeypatch, target_layer):
    def put(path, obj):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(obj))
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    model=tmp_path/'model-index.json';put(model,{'weight_map':{}})
    ledger=tmp_path/'ledger.json';put(ledger,{'rows':[{'item_id':'book:0'},{'item_id':'book:1'}]})
    reference=tmp_path/'reference.json';put(reference,{})
    lut=tmp_path/'lut.json';put(lut,{})
    windows=[]
    for i in range(2):
        path=tmp_path/f'x{i}.json';put(path,{'win':0,'layer':target_layer})
        windows.append({'ordinal':0,'item_id':f'book:{i}','path':str(path),'sha256':sha(path)})
    captures=tmp_path/'captures.json';put(captures,{'status':'PASS_NATIVE_CAPTURE','layer':target_layer,'ledger_sha256':sha(ledger),'windows':windows})
    template=tmp_path/'template.json';put(template,{'geometry':{'K':2},'input_identity':{'model_index':{'path':str(model),'sha256':sha(model)}},'tier':'qtip@2.00','qtip_root':str(tmp_path),'tlut_source':str(lut),'reference_unit':str(reference)})
    output=tmp_path/'output';put(output/'q2_cell/SHARDS.json',{'intended_basis':sha(model)})
    spec=tmp_path/'spec.json';put(spec,{'layer':target_layer,'output':str(output),'canonical_commit':'test-only','calibration_ledger':str(ledger),'calibration_ledger_sha256':sha(ledger),'capture_results':[str(captures)],'windows':2,'template_config':str(template),'intended_basis':sha(model)})
    calls=[]
    def solve(config, root, layer, **kwargs):
        assert layer==target_layer
        config=json.loads(Path(config).read_text());calls.append(config)
        assert config['fit_windows']==2 and config['geometry']['K']==2
        assert config['pack_counts']=={'qtip2':1}
        assert [json.loads(p.read_text())['win'] for p in sorted(Path(config['fit_capture_root']).glob('*.pt'))]==[0,1]
        artifact=root/f"solve/L{target_layer:03d}/E{config.get('expert',0):03d}_fused13/QTIP_UNIT.pt";put(artifact,{'stub':True})
        put(artifact.parent/'QTIP_SOLVE_RECEIPT.json',{'status':'PASS','artifact_sha256':sha(artifact)})
    sp=ModuleType('banana_smasher.solver_qtip_profile');sp._atomic_json=put;sp._atomic_torch=put;sp._canonical_rht_seed=lambda *args:42;sp.main=solve
    runner=ModuleType('banana_smasher.qtip_runner');runner.load_official_qtip=lambda:(None,)*4
    adapter=ModuleType('banana_smasher.glm_qtip_source_adapter');adapter.capture_source_closure=lambda *args:{'sha256':'fixture'}
    package=ModuleType('banana_smasher');package.solver_qtip_profile=sp;package.qtip_runner=runner
    torch=SimpleNamespace(set_num_threads=lambda x:None,manual_seed=lambda x:None,use_deterministic_algorithms=lambda x:None,backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),cudnn=SimpleNamespace(allow_tf32=False)),load=lambda p,**kwargs:json.loads(Path(p).read_text()))
    for name,value in [('torch',torch),('banana_smasher',package),('banana_smasher.solver_qtip_profile',sp),('banana_smasher.qtip_runner',runner),('banana_smasher.glm_qtip_source_adapter',adapter)]:
        monkeypatch.setitem(sys.modules,name,value)
    script=Path(__file__).parents[2]/'tools/glm_calibrated_cell.py'
    monkeypatch.setattr(sys,'argv',[str(script),str(spec)])
    runpy.run_path(str(script),run_name='__main__')
    assert len(calls)==1
    assert json.loads((output/'CORRECTION_RESULT.json').read_text())['status']=='PASS_SINGLE_CELL_CLEAN_CALIBRATION'
    frozen={p.name:sha(p) for p in (output/'q2_cell/fitcaptures').iterdir()}
    resumed=json.loads(spec.read_text());resumed.update(prepared_config=str(output/f'q2_cell/L{target_layer:03d}_E000_fused13.json'),solve_root=str(output/'q2_cell_attempt2'),result_name='CORRECTION_A2_RESULT.json')
    put(spec,resumed);put(output/'q2_cell_attempt2/SHARDS.json',{'intended_basis':sha(model)})
    runpy.run_path(str(script),run_name='__main__')
    assert {p.name:sha(p) for p in (output/'q2_cell/fitcaptures').iterdir()}==frozen
    assert json.loads((output/'CORRECTION_A2_RESULT.json').read_text())['status']=='PASS_SINGLE_CELL_CLEAN_CALIBRATION'
    next_config=json.loads((output/f'q2_cell/L{target_layer:03d}_E000_fused13.json').read_text())
    next_config.update(layer=target_layer,expert=1,projection='fused13')
    put(output/'e1.json',next_config)
    resumed.update(prepared_config=str(output/'e1.json'),solve_root=str(output/'e1'),result_name='E1_RESULT.json',layer=target_layer,expert=1,projection='fused13')
    put(spec,resumed);put(output/'e1/SHARDS.json',{'intended_basis':sha(model)})
    runpy.run_path(str(script),run_name='__main__')
    result=json.loads((output/'E1_RESULT.json').read_text())
    assert result['cell']==f'L{target_layer:03d}/E001_fused13'
    assert '/E001_fused13/' in result['artifact']
    assert {p.name:sha(p) for p in (output/'q2_cell/fitcaptures').iterdir()}==frozen
    rejected=json.loads(spec.read_text())
    rejected.pop('prepared_config')
    rejected.update(expert=0,solve_root=str(output/'wrong-layer'),result_name='WRONG_LAYER.json')
    put(spec,rejected)
    bad_capture=json.loads(captures.read_text());bad_capture['layer']=target_layer+1;put(captures,bad_capture)
    with pytest.raises(AssertionError,match='capture layer mismatch'):
        runpy.run_path(str(script),run_name='__main__')
    assert len(calls)==3
    bad_capture['layer']=target_layer
    tensor=Path(windows[0]['path']);put(tensor,{'win':0,'layer':target_layer+1})
    bad_capture['windows'][0]['sha256']=sha(tensor);put(captures,bad_capture)
    rejected.update(solve_root=str(output/'wrong-tensor'),result_name='WRONG_TENSOR.json');put(spec,rejected)
    with pytest.raises(AssertionError,match='capture tensor layer mismatch'):
        runpy.run_path(str(script),run_name='__main__')
    assert len(calls)==3
