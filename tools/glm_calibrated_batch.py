"""Bounded sequencing only; all encoding remains in the canonical cell solver."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run_batch(spec_paths, progress_path, execute=subprocess.run):
    rows=[]
    for p in spec_paths:
        s=json.loads(Path(p).read_text())
        cell=f"L{s['layer']:03d}/E{s['expert']:03d}_{s['projection']}"
        terminal=Path(s['output'])/s['result_name']
        if terminal.exists():
            raise ValueError(f'existing terminal; consume or exclude, never replay: {terminal}')
        rows.append((p,cell,terminal))
    if len({r[1] for r in rows})!=len(rows):
        raise ValueError('duplicate cell')
    progress_path=Path(progress_path)
    completed=[]
    def save(state,active=None):
        tmp=progress_path.with_suffix('.tmp')
        with tmp.open('w') as f:
            json.dump(dict(state=state,active=active,completed=completed,updated=time.time()),f)
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,progress_path)
    for p,cell,terminal in rows:
        save('RUNNING',cell)
        try:
            execute([sys.executable,str(Path(__file__).with_name('glm_calibrated_cell.py')),str(p)],check=True)
            result=json.loads(terminal.read_text())
            assert result['status']=='PASS_SINGLE_CELL_CLEAN_CALIBRATION' and result['cell']==cell
        except BaseException:
            save('FAIL',cell)
            raise
        completed.append(cell)
        save('CELL_PASS',cell)
    save('PASS')


if __name__=='__main__':
    spec=json.loads(Path(sys.argv[1]).read_text())
    run_batch(spec['cell_specs'],spec['progress_path'])
