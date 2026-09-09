import os,sys,json,time,hashlib,subprocess,fcntl,traceback,tarfile
from pathlib import Path
ROOT=Path('/dev/shm/t_1269dc5f');OUT=ROOT/'grouped_phase_probe_run8545'; PRODUCTS=ROOT/'grouped_run8544r2'; CONTROL=ROOT/'traceback_run8542r3';OLD=Path('/dev/shm/t_182fbc9d');CLAIM=Path('/home/dnola/HOST_CLAIM.json');BASE='2a118a825ef00ff67dd0cbc65d07cc67a2919089';CAND='a2a7df42114ace02c39454963abe05f588eefac5'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
 with tmp.open('w') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
 fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def identity(pid):return dict(pid=pid,pgid=os.getpgid(pid),start_ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]),argv=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')[:-1])
def dead(pid):return not Path(f'/proc/{pid}').exists()
if len(sys.argv)==1:
 OUT.mkdir(exist_ok=False)
 with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);raw=CLAIM.read_bytes();old=json.loads(raw);assert old['owner']==old['task_id']=='t_1269dc5f' and dead(old['pid'])
  prior=Path(old['mission']);assert json.loads((prior/'TERMINAL.json').read_text())['returncode']==0;assert dead(json.loads((prior/'CHILD.json').read_text())['pid'])
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  assert old['pid']==1820686 and old['start_ticks']==8193341
  basis='3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05';assert sha('/dev/shm/t_ebcba52e_ldlq_run8503/selected_pair_run8507/model.safetensors.index.json')==basis
  mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024; st=os.statvfs(ROOT);assert mem>(12+4)<<30 and st.f_bavail*st.f_frsize>(1+4)<<30
  put(OUT/'PREFLIGHT.json',dict(mem_available=mem,peak_estimate=12<<30,reserve=4<<30,disk_available=st.f_bavail*st.f_frsize,prior_basis=old['source_model_index_sha256'],intended_basis=basis))
  with (ROOT/'SHARDS.lock').open('a+') as sl:
   fcntl.flock(sl,fcntl.LOCK_EX);raws=(ROOT/'SHARDS.json').read_bytes();shards=json.loads(raws);assert shards['intended_basis']==old['source_model_index_sha256'] and all(r['owner']=='t_1269dc5f' for r in shards['rows']);put(OUT/'SHARDS_PREIMAGE.json',shards);shards['intended_basis']=basis;shards['rows']=[dict(owner='t_1269dc5f',layer=4,experts=[242,243],projections=['down'],operation='unitwise_grouped',receipt=str(OUT/'QUALITY.json')),dict(owner='t_1269dc5f',layer=4,experts=[242],projections=['fused13'],operation='unitwise_grouped',receipt=str(OUT/'QUALITY.json'))];assert (ROOT/'SHARDS.json').read_bytes()==raws;put(ROOT/'SHARDS.json',shards)
  c=dict(old);c.update(**identity(os.getpid()),source_model_index_sha256=basis,mission=str(OUT),receipt=str(OUT/'IDENTITY.json'),stage='unitwise_grouped',expiry_unix=time.time()+7200,exact_cas_from_sha256=hashlib.sha256(raw).hexdigest());c.pop('terminal_at',None);put(OUT/'PREIMAGE.json',old);assert CLAIM.read_bytes()==raw;put(CLAIM,c);assert json.loads(CLAIM.read_text())==c;put(OUT/'IDENTITY.json',c)
 try:
  for tarname,dirname in [('grouped_pin_run8544.tar','grouped_code_run8544'),('traceback_base_run8542.tar','traceback_base_run8542')]:
   code=ROOT/dirname
   if not code.exists():
    code.mkdir()
    with tarfile.open(ROOT/tarname) as archive:archive.extractall(code,filter='data')
   with tarfile.open(ROOT/tarname) as archive:
    member=archive.extractfile('banana-smasher/src/banana_smasher/qtip_viterbi.py');assert hashlib.sha256(member.read()).hexdigest()==sha(code/'banana-smasher/src/banana_smasher/qtip_viterbi.py')
   put(OUT/(dirname+'_DEPLOY.json'),dict(tar_sha256=sha(ROOT/tarname),module_sha256=sha(code/'banana-smasher/src/banana_smasher/qtip_viterbi.py')))
  jobs=[('quality',None)];rows=[]
  for name,arm in jobs:
   env=dict(os.environ,TRITON_CACHE_DIR='/dev/shm/t_ebcba52e_ldlq_run8503/cache/triton',BANANA_SMASHER_KERNEL_CACHE='/dev/shm/t_ebcba52e_ldlq_run8503/cache/kernels',XDG_CACHE_HOME='/dev/shm/t_ebcba52e_ldlq_run8503/cache/xdg',PYTHONPATH=str(ROOT/'grouped_code_run8544/banana-smasher/src'))
   cmd=[sys.executable,'-u',__file__,name,arm or 'none'];t=time.perf_counter()
   with (OUT/f'{name}.log').open('w') as log:
    p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);put(OUT/'CHILD.json',identity(p.pid));rc=p.wait()
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
 raise SystemExit(subprocess.call([sys.executable,'-m','pytest',str(ROOT/'grouped_code_run8544/banana-smasher/tests/test_qtip_step_unroll2.py'),str(ROOT/'grouped_code_run8544/banana-smasher/tests/test_qtip_lut_cache_policy.py'),'-q']))
sys.path.insert(0,str((ROOT/'grouped_code_run8544' if arm=='candidate' else ROOT/'traceback_base_run8542')/'banana-smasher/src'))
import torch
from banana_smasher import solver_qtip_profile as sp
from banana_smasher.glm_qtip_source_adapter import capture_source_closure
torch.set_num_threads(8);free=torch.cuda.mem_get_info()[0];put(OUT/(name+'_CUDA_ADMISSION.json'),dict(cuda_free=free,peak_estimate=12<<30,reserve=4<<30,reason='Original24GiB estimate was an unmeasured blanket double. Retained same-cell singleton max reserved1514143744B; two-member linear batching plus conservative extra headroom below12GiB; canonical per-geometry/per-Viterbi exact reserve gates remain mandatory. No reserve weakened.'));assert free>(12+4)<<30
stat=os.statvfs(ROOT);assert stat.f_bavail*stat.f_frsize>(1+4)<<30
if name=='quality':
 from banana_smasher import qtip_batch as qb
 import inspect
 runner=Path(sp.__file__).with_name('qtip_runner.py');qv=sp._load_public_qtip_runner(runner,sha(runner));cfg=json.loads((CONTROL/'B1/warm/E243_down/CONFIG.json').read_text());qv.QTIP=Path(cfg['qtip_root']);bits,ldlq,math,kd=qv.load_official_qtip()
 captures=sp._load_captures(Path(cfg['fit_capture_root']),4,16);fit,_=sp._prepare_fit_windows(qv,captures,model_root=Path(cfg['model_root']),layer=4,expert=243,projection='down',device=torch.device('cuda'));sp._release_capture_bank(Path(cfg['fit_capture_root']),4,16,captures)
 w,_=sp._load_weight(Path(cfg['model_root']),4,243,'down');torch.manual_seed(cfg['rht_seed']);su=(torch.randn(w.shape[1],device='cuda').sign()+1e-5).sign().float()
 h,_,_=qv.build_hessian(fit,su,torch.device('cuda'));hs=math.regularize_H(h.clone(),1e-2);hb=h.clone().unsqueeze(0);qb._regularize_hessian_batch(hb,1e-2,unitwise=True)
 def compare(a,b):return dict(equal=torch.equal(a,b),max_abs=float((a-b).abs().max()),different=int((a!=b).sum()))
 l1,_=math.block_LDL(hs,16);l2=qb.block_ldl_batch(hs.unsqueeze(0),16,unitwise=True)[0];l3=qb.block_ldl_batch(hb,16,unitwise=True)[0]
 put(OUT/'PHASE_PROBE.json',dict(cell='E243_down',hessian=compare(hs,hb[0]),ldl_same_h=compare(l1,l2),ldl_original=compare(l1,l3),regularize_source=inspect.getsource(math.regularize_H),ldl_source=inspect.getsource(math.block_LDL),scope='same-input preprocessing only; no encode or speed denominator'))
