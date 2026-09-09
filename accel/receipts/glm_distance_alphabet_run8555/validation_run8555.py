import os,sys,json,time,hashlib,subprocess,fcntl,traceback,tarfile
from pathlib import Path
ROOT=Path('/dev/shm/t_1269dc5f');OUT=ROOT/'validation_run8555'; CONTROL=ROOT/'traceback_run8542r3';OLD=Path('/dev/shm/t_182fbc9d');CLAIM=Path('/home/dnola/HOST_CLAIM.json');BASE='2a118a825ef00ff67dd0cbc65d07cc67a2919089';CAND='8406c6a38faf0a43cb85a374fd9ab0f0516c3f58'
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
if len(sys.argv)==1:
 OUT.mkdir(exist_ok=False)
 put(OUT/'AUTHORITY.json',dict(card_comment='1788920954 explicit dedicated bs01 s6 supersedes stale remote allocation',driver_text=Path('/home/dnola/missions/DRIVER_GOALS.md').read_text(),script_sha256=sha(__file__),hypothesis='Factor per-step subtractions over actual1024 signed quantlut values, then gather differences before original recurrence arithmetic; exact full-LUT validation and version-bound preparation; additional4MiB admission; default off, retaining broadcast, full-state pointers, LUT policy, branch arithmetic, full forward and traceback. Frozen clean-fit SSE ratio<=1.0001; no tuning from heldout.'))
 with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);raw=CLAIM.read_bytes();old=json.loads(raw);assert old['owner']==old['task_id']=='t_1269dc5f' and dead(old['pid'])
  prior=Path(old['mission']);assert json.loads((prior/'TERMINAL.json').read_text())['returncode']==0;assert dead(json.loads((prior/'CHILD.json').read_text())['pid'])
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  assert old['pid']==1860235 and old['start_ticks']==8817572
  basis='3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05';assert sha('/dev/shm/t_ebcba52e_ldlq_run8503/selected_pair_run8507/model.safetensors.index.json')==basis
  mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024; st=os.statvfs(ROOT);assert mem>(12+4)<<30 and st.f_bavail*st.f_frsize>(1+4)<<30
  put(OUT/'PREFLIGHT.json',dict(mem_available=mem,peak_estimate=12<<30,reserve=4<<30,disk_available=st.f_bavail*st.f_frsize,prior_basis=old['source_model_index_sha256'],intended_basis=basis))
  with (ROOT/'SHARDS.lock').open('a+') as sl:
   fcntl.flock(sl,fcntl.LOCK_EX);raws=(ROOT/'SHARDS.json').read_bytes();shards=json.loads(raws);assert shards['intended_basis']==old['source_model_index_sha256'] and all(r['owner']=='t_1269dc5f' for r in shards['rows']);put(OUT/'SHARDS_PREIMAGE.json',shards);shards['intended_basis']=basis;shards['rows']=[dict(owner='t_1269dc5f',layer=4,experts=[242,243],projections=['down'],operation='unitwise_grouped',receipt=str(OUT/'TERMINAL.json')),dict(owner='t_1269dc5f',layer=4,experts=[242],projections=['fused13'],operation='unitwise_grouped',receipt=str(OUT/'TERMINAL.json'))];assert (ROOT/'SHARDS.json').read_bytes()==raws;put(ROOT/'SHARDS.json',shards)
  c=dict(old);c.update(**identity(os.getpid()),source_model_index_sha256=basis,mission=str(OUT),receipt=str(OUT/'IDENTITY.json'),stage='unitwise_grouped',expiry_unix=time.time()+7200,exact_cas_from_sha256=hashlib.sha256(raw).hexdigest());c.pop('terminal_at',None);put(OUT/'PREIMAGE.json',old);assert CLAIM.read_bytes()==raw;put(CLAIM,c);assert json.loads(CLAIM.read_text())==c;put(OUT/'IDENTITY.json',c)
 try:
  for tarname,dirname in [('validation_pin_run8555.tar','validation_code_run8555'),('traceback_base_run8542.tar','traceback_base_run8542')]:
   code=ROOT/dirname
   if not code.exists():
    code.mkdir()
    with tarfile.open(ROOT/tarname) as archive:archive.extractall(code,filter='data')
   with tarfile.open(ROOT/tarname) as archive:
    member=archive.extractfile('banana-smasher/src/banana_smasher/qtip_viterbi.py');assert hashlib.sha256(member.read()).hexdigest()==sha(code/'banana-smasher/src/banana_smasher/qtip_viterbi.py')
   put(OUT/(dirname+'_DEPLOY.json'),dict(tar_sha256=sha(ROOT/tarname),module_sha256=sha(code/'banana-smasher/src/banana_smasher/qtip_viterbi.py')))
  jobs=[('gpu_tests','candidate')];rows=[]
  for name,arm in jobs:
   env=dict(os.environ,TRITON_CACHE_DIR='/dev/shm/t_ebcba52e_ldlq_run8503/cache/triton',BANANA_SMASHER_KERNEL_CACHE='/dev/shm/t_ebcba52e_ldlq_run8503/cache/kernels',XDG_CACHE_HOME='/dev/shm/t_ebcba52e_ldlq_run8503/cache/xdg',PYTHONPATH=str(ROOT/'validation_code_run8555/banana-smasher/src'))
   cmd=[sys.executable,'-u',__file__,name,arm or 'none'];t=time.perf_counter()
   with (OUT/f'{name}.log').open('w') as log:
    p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);put(OUT/'CHILD.json',identity(p.pid,cmd));rc=p.wait()
   rows.append(dict(job=name,arm=arm,returncode=rc,seconds=time.perf_counter()-t));put(OUT/'PROGRESS.json',rows);assert rc==0,(name,rc)
  put(OUT/'TERMINAL.json',dict(returncode=0,rows=rows))
 except BaseException:
  put(OUT/'FAILURE.json',dict(error=traceback.format_exc()));put(OUT/'TERMINAL.json',dict(returncode=1));raise
 finally:
  with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);c=json.loads(CLAIM.read_text());assert c['pid']==os.getpid();c.update(stage='unitwise_grouped_terminal_retained',terminal_at=time.time());put(CLAIM,c)
 raise SystemExit()
name,arm=sys.argv[1:];assert json.loads(CLAIM.read_text())['pid']==os.getppid()
if name=='gpu_tests':
 tests=ROOT/'validation_code_run8555/banana-smasher/tests'
 raise SystemExit(subprocess.call([sys.executable,'-m','pytest',str(tests/'test_qtip_distance_alphabet.py'),str(tests/'test_qtip_structured_gather.py'),str(tests/'test_qtip_lut_cache_policy.py'),*[str(p) for p in tests.glob('test_qtip_batch*.py')],'-q']))
sys.path.insert(0,str((ROOT/'validation_code_run8555' if arm=='candidate' else ROOT/'traceback_base_run8542')/'banana-smasher/src'))
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
   cell=f'E{expert}_{projection}';cd=D/cell;cd.mkdir();cfg=json.loads((CONTROL/f'B1/warm/{cell}/CONFIG.json').read_text());assert sha(cfg['input_identity']['model_index']['path'])==json.loads((ROOT/'SHARDS.json').read_text())['intended_basis'];runner=Path(sp.__file__).with_name('qtip_runner.py');cfg['qtip_runner']=str(runner);cfg['block_ldl_unitwise']=True;cfg['block_ldl_reference']=True;cfg['viterbi_distance_alphabet']=True
   manifest=json.loads(Path(cfg['materialization']['run_manifest']).read_text());manifest['canonical_commit']=pin;manifest['tiers'][0]['bindings']['qtip_runner']=dict(path=str(runner),sha256=sha(runner));mp=cd/'MANIFEST.json';put(mp,manifest);cfg['materialization']['run_manifest']=str(mp);cfg['materialization']['run_manifest_sha256']=sha(mp)
   qv=sp._load_public_qtip_runner(runner,sha(runner));qv.QTIP=Path(cfg['qtip_root']);bits,ldlq,math,kd=qv.load_official_qtip();closure=capture_source_closure(qv,dict(bitshift=bits,ldlq=ldlq,math_utils=math,kernel_decompress=kd));put(cd/'IMPORT_CLOSURE.json',closure);cfg['glm_source_closure_sha256']=closure['sha256'];cp=cd/'CONFIG.json';put(cp,cfg);paths.append(cp)
  dest=D/'build';dest.mkdir();put(dest/'SHARDS.json',json.loads((ROOT/'SHARDS.json').read_text()));torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();result=main_batch(paths,dest,4);torch.cuda.synchronize();rows.append(dict(phase=phase,group=group,cells=cells,seconds=time.perf_counter()-t,result=result,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()));put(OUT/name/'RESULT.json',dict(pin=pin,rows=rows,scope='Signed distance alphabet 2-down grouped plus single fused compared with retained reference_ldl_run8545 C1/C2 same groups; historical/shared-cache, no cold or production ratio'))
