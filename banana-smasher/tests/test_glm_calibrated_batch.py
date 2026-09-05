import importlib.util,json
from pathlib import Path
import pytest

def test_batch_does_not_replay_and_stops_on_failure(tmp_path):
    path=Path(__file__).parents[2]/'tools/glm_calibrated_batch.py'
    assert path.exists(), 'missing production batch orchestrator'
    spec=importlib.util.spec_from_file_location('batch',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    entries=[]
    for expert in (2,3):
        p=tmp_path/f'{expert}.json';p.write_text(json.dumps({'layer':3,'expert':expert,'projection':'fused13','output':str(tmp_path),'result_name':f'{expert}.result.json'}));entries.append(str(p))
    calls=[]
    def execute(argv,check):
        calls.append(argv)
        d=json.loads(Path(argv[-1]).read_text());(tmp_path/d['result_name']).write_text(json.dumps({'status':'PASS_SINGLE_CELL_CLEAN_CALIBRATION','cell':f"L003/E{d['expert']:03d}_fused13"}))
    m.run_batch(entries,tmp_path/'progress.json',execute=execute)
    assert len(calls)==2
    assert json.loads((tmp_path/'progress.json').read_text())['completed']==['L003/E002_fused13','L003/E003_fused13']
    with pytest.raises(ValueError,match='existing terminal'):
        m.run_batch(entries,tmp_path/'other-progress.json',execute=execute)
    assert len(calls)==2
