import importlib.util,json
from pathlib import Path
import pytest
import sys


def test_resident_exec_restores_argv_and_reuses_imports(tmp_path):
    path=Path(__file__).parents[2]/'tools/glm_calibrated_batch.py'
    spec=importlib.util.spec_from_file_location('resident_batch_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    helper=tmp_path/'resident_fixture.py'
    helper.write_text('count=0\n')
    entry=tmp_path/'cell.py'
    entry.write_text('import resident_fixture,sys,json\nfrom pathlib import Path\nresident_fixture.count+=1\nPath(sys.argv[1]).write_text(json.dumps(dict(count=resident_fixture.count,pid=__import__("os").getpid())))\n')
    old=list(sys.argv);sys.path.insert(0,str(tmp_path))
    try:
        for i in range(2):
            output=tmp_path/f'{i}.json'
            m.execute_resident([sys.executable,str(entry),str(output)],check=True)
            d=json.loads(output.read_text())
            assert d['count']==i+1
            assert d['pid']==__import__('os').getpid()
            assert sys.argv==old
        entry.write_text('raise RuntimeError("cell failed")\n')
        with pytest.raises(RuntimeError,match='cell failed'):
            m.execute_resident([sys.executable,str(entry),'unused'],check=True)
        assert sys.argv==old
    finally:
        sys.path.remove(str(tmp_path));sys.modules.pop('resident_fixture',None)

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
