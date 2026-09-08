import os,json,time,hashlib
from pathlib import Path
import sys
sys.path.insert(0,"/dev/shm/t_182fbc9d/public_run8524/banana-smasher/src")
import torch
from banana_smasher import solver_qtip_profile as sp
R=Path('/dev/shm/t_182fbc9d');B=Path('/dev/shm/t_ebcba52e_ldlq_run8503');O=Path(os.environ['OUT']);torch.set_num_threads(8)
assert json.loads(Path('/home/dnola/HOST_CLAIM.json').read_text())['pid']==os.getppid()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
runner=Path(sp.__file__).with_name('qtip_runner.py');qv=sp._load_public_qtip_runner(runner,sha(runner));cfg=json.loads((R/'public_build_run8524/public/E242_down/CONFIG.json').read_text());qv.QTIP=Path(cfg['qtip_root']);bits,ldlq,math,kd=qv.load_official_qtip();start=time.perf_counter();rows=[]
for expert,projection in [(242,'down'),(243,'down'),(242,'fused13')]:
 cell=f'E{expert}_{projection}';cfg=json.loads((R/f'public_build_run8524/public/{cell}/CONFIG.json').read_text());captures=sp._load_captures(Path(cfg['fit_capture_root']),4,16);fit,_=sp._prepare_fit_windows(qv,captures,model_root=Path(cfg['model_root']),layer=4,expert=expert,projection=projection,device=torch.device('cuda'));sp._release_capture_bank(Path(cfg['fit_capture_root']),4,16,captures);w,_=sp._load_weight(Path(cfg['model_root']),4,expert,projection)
 control=(B/f'pair_warm_run8512/parallel1/solve/L004/{cell}/QTIP_UNIT.pt') if projection=='down' else R/'fused_r2/control1/warm/solve/L004/E242_fused13/QTIP_UNIT.pt';u=torch.load(control,map_location='cpu',weights_only=True);ref=qv.decode_packed_weight(u,kd,torch.device('cuda')).half().cpu()
 for arm in ['public']:
  a=R/f'public_build_run8524/{arm}/{cell}/build/solve/L004/{cell}/QTIP_UNIT.pt';u=torch.load(a,map_location='cpu',weights_only=True);x=qv.decode_packed_weight(u,kd,torch.device('cuda')).half().cpu();qualified=torch.load(R/f'whole_l1/warm1/{cell}/build/solve/L004/{cell}/QTIP_UNIT.pt',map_location='cpu',weights_only=True);qualified_w=qv.decode_packed_weight(qualified,kd,torch.device('cuda')).half().cpu();assert torch.equal(x,qualified_w);metrics=qv.split_metrics(fit,w,ref,x,torch.device('cuda'));assert metrics['qtip_hyb']['sse_ratio_vs_true_vq']<=1.0001+1e-12;assert torch.isfinite(x).all();rows.append(dict(cell=cell,arm=arm,artifact_sha256=sha(a),reference_sha256=sha(control),decoded_equal=torch.equal(x,ref),max_abs=float((x.float()-ref.float()).abs().max()),metrics=metrics))
  with (O/'QUALITY.json').open('w') as f:json.dump(dict(rows=rows,validation_seconds=time.perf_counter()-start,scope='canonical decoded consumer and authentic clean-fit, not heldout',sse_limit=1.0001,pin='53f05876cb02a95e2accdce8a0bc23efc7edd5f6'),f,indent=2);f.flush();os.fsync(f.fileno())
print('PASS_OUTPUTS',len(rows),flush=True)
