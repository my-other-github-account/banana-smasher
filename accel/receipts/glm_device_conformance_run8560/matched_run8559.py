import os,sys,json,time,hashlib,subprocess,traceback
from pathlib import Path
ROOT=Path('/dev/shm/t_6ac2b6f5');OUT=ROOT/'matched_run8559';CONTROL=Path('/dev/shm/t_1269dc5f/traceback_run8542r3');CLAIM=Path('/home/dnola/HOST_CLAIM.json');CAND='c769d086701769b08fdc85d5191afc2aa321e3e7'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
 with tmp.open('w') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
 fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def identity(pid,expected=None):
 start=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
 for _ in range(200):
  argv=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')[:-1]
  if argv and (expected is None or argv==expected):return dict(pid=pid,pgid=os.getpgid(pid),start_ticks=start,argv=argv)
  time.sleep(.01)
 raise RuntimeError('CHILD_EXEC_HANDSHAKE_FAILED')
def dead(pid):return not Path(f'/proc/{pid}').exists()
name,arm=sys.argv[1:];assert json.loads(CLAIM.read_text())['pid']==os.getppid()
if name=='gpu_tests':
 tests=ROOT/'conformance_code/banana-smasher/tests'
 raise SystemExit(subprocess.call([sys.executable,'-m','pytest',str(tests/'test_qtip_deferred_conformance_copy.py'),str(tests/'test_qtip_device_conformance.py'),*[str(p) for p in tests.glob('test_qtip_batch*.py')],'-q']))

import torch
from banana_smasher import solver_qtip_profile as sp
from banana_smasher.glm_qtip_source_adapter import capture_source_closure
torch.set_num_threads(8);free=torch.cuda.mem_get_info()[0];put(OUT/(name+'_CUDA_ADMISSION.json'),dict(cuda_free=free,peak_estimate=12<<30,reserve=4<<30,reason='Original24GiB estimate was an unmeasured blanket double. Retained same-cell singleton max reserved1514143744B; two-member linear batching plus conservative extra headroom below12GiB; canonical per-geometry/per-Viterbi exact reserve gates remain mandatory. No reserve weakened.'));assert free>(12+4)<<30
stat=os.statvfs(ROOT);assert stat.f_bavail*stat.f_frsize>(1+4)<<30
if name=='quality':
 runner=Path(sp.__file__).with_name('qtip_runner.py');qv=sp._load_public_qtip_runner(runner,sha(runner));cfg=json.loads((CONTROL/'B1/warm/E242_down/CONFIG.json').read_text());qv.QTIP=Path(cfg['qtip_root']);bits,ldlq,math,kd=qv.load_official_qtip();start=time.perf_counter();rows=[]
 for expert,projection in [(242,'down'),(243,'down'),(242,'fused13')]:
  cell=f'E{expert}_{projection}';cfg=json.loads((CONTROL/f'B1/warm/{cell}/CONFIG.json').read_text());captures=sp._load_captures(Path(cfg['fit_capture_root']),4,16);fit,_=sp._prepare_fit_windows(qv,captures,model_root=Path(cfg['model_root']),layer=4,expert=expert,projection=projection,device=torch.device('cuda'));sp._release_capture_bank(Path(cfg['fit_capture_root']),4,16,captures);w,_=sp._load_weight(Path(cfg['model_root']),4,expert,projection)
  control=CONTROL/f'B1/warm/{cell}/build/solve/L004/{cell}/QTIP_UNIT.pt';ref=qv.decode_packed_weight(torch.load(control,map_location='cpu',weights_only=True),kd,torch.device('cuda')).half().cpu()
  for label in ['C1','C2']:
   for phase in ['setup','warm']:
    group='GROUP_DOWN' if projection=='down' else 'GROUP_FUSED';path=OUT/f'{label}/{phase}/{group}/build/solve/L004/{cell}/QTIP_UNIT.pt';x=qv.decode_packed_weight(torch.load(path,map_location='cpu',weights_only=True),kd,torch.device('cuda')).half().cpu();metrics=qv.split_metrics(fit,w,ref,x,torch.device('cuda'));assert torch.isfinite(x).all();rows.append(dict(passed=metrics['qtip_hyb']['sse_ratio_vs_true_vq']<=1.0001+1e-12,cell=cell,arm=label,phase=phase,artifact_sha256=sha(path),reference_sha256=sha(control),decoded_equal=torch.equal(x,ref),max_abs=float((x.float()-ref.float()).abs().max()),metrics=metrics));put(OUT/'QUALITY.json',dict(rows=rows,validation_seconds=time.perf_counter()-start,scope='canonical decode and clean-fit only; not heldout',limit=1.0001))
 assert len(rows)==12
 put(OUT/'GATE.json',dict(status='PASS' if all(r['passed'] for r in rows) else 'REJECT_QUALITY',passed=sum(r['passed'] for r in rows),count=len(rows),limit=1.0001))
 raise SystemExit()
from banana_smasher.qtip_batch_controller import main_batch
rows=[];pin=CAND
for phase in ['setup','warm']:
 for group,cells in [('GROUP_DOWN',[(242,'down'),(243,'down')]),('GROUP_FUSED',[(242,'fused13')])]:
  D=OUT/name/phase/group;D.mkdir(parents=True);paths=[]
  for expert,projection in cells:
   cell=f'E{expert}_{projection}';cd=D/cell;cd.mkdir();cfg=json.loads((CONTROL/f'B1/warm/{cell}/CONFIG.json').read_text());assert sha(cfg['input_identity']['model_index']['path'])==json.loads((ROOT/'SHARDS.json').read_text())['intended_basis'];runner=Path(sp.__file__).with_name('qtip_runner.py');cfg['qtip_runner']=str(runner);cfg['block_ldl_unitwise']=True;cfg['block_ldl_reference']=True;cfg['viterbi_distance_alphabet']=True;cfg['packed_conformance_on_device']=(arm=='candidate')
   manifest=json.loads(Path(cfg['materialization']['run_manifest']).read_text());manifest['canonical_commit']=pin;manifest['tiers'][0]['bindings']['qtip_runner']=dict(path=str(runner),sha256=sha(runner));mp=cd/'MANIFEST.json';put(mp,manifest);cfg['materialization']['run_manifest']=str(mp);cfg['materialization']['run_manifest_sha256']=sha(mp)
   qv=sp._load_public_qtip_runner(runner,sha(runner));qv.QTIP=Path(cfg['qtip_root']);bits,ldlq,math,kd=qv.load_official_qtip();closure=capture_source_closure(qv,dict(bitshift=bits,ldlq=ldlq,math_utils=math,kernel_decompress=kd));put(cd/'IMPORT_CLOSURE.json',closure);cfg['glm_source_closure_sha256']=closure['sha256'];cp=cd/'CONFIG.json';put(cp,cfg);paths.append(cp)
  dest=D/'build';dest.mkdir();put(dest/'SHARDS.json',json.loads((ROOT/'SHARDS.json').read_text()));torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();result=main_batch(paths,dest,4);torch.cuda.synchronize();rows.append(dict(phase=phase,group=group,cells=cells,seconds=time.perf_counter()-t,result=result,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()));put(OUT/name/'RESULT.json',dict(pin=pin,rows=rows,scope='Matched ABBA2 baseline false/candidate true on same source pin, fresh processes with shared warm caches and identical authentic2-down/fused panel. Setup and warm reported separately, not cold JIT or production speed'))
