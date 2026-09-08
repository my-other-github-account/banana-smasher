"""Matched fresh-process/private-cache DS4 panel; no sealed warm replay."""
import os,sys,json,time,hashlib,fcntl,subprocess,traceback,resource
from pathlib import Path
TASK='t_ebcba52e';PIN='7927793657c7a48e1e02f5cb9e437e1c5705002d'
ROOT=Path('/dev/shm/t_ebcba52e_ds4_k3_fused_cold_run8474');CLAIM=Path('/home/dnola/HOST_CLAIM.json')
PRIOR=Path('/dev/shm/t_ebcba52e_ds4_k3_fused_run8474')
API=Path('/dev/shm/t_ebcba52e_api_79277936/banana-smasher/src')
BASIS='98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp')
 with t.open('w') as f:json.dump(d,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(t,p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def memory():return {k:int(v.split()[0])*1024 for k,v in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())}
def gate():
 for e in [78,79,80]:
  c=json.loads((PRIOR/f'baseline_warm2/E{e}.json').read_text())
  assert sha(Path(c['model_root'])/'model.safetensors.index.json')==BASIS
 assert 32*1024**3<memory()['MemAvailable']-4*1024**3
 v=os.statvfs(ROOT.parent);assert v.f_bavail*v.f_frsize>4*1024**3
if '--child' in sys.argv:
 arm=sys.argv[2];out=ROOT/arm;start=time.perf_counter();gate()
 claim=json.loads(CLAIM.read_text());controller=json.loads((ROOT/'IDENTITY.json').read_text())
 assert claim['task_id']==TASK and claim['pid']==os.getppid()==controller['pid']
 assert claim['source_model_index_sha256']==BASIS and claim['stage']==ROOT.name
 identity=dict(pid=os.getpid(),pgid=os.getpgid(0),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),argv=sys.argv)
 put(out/'IDENTITY.json',identity)
 sys.path.insert(0,str(API))
 import torch,banana_smasher
 from banana_smasher.qtip_batch_controller import main_batch
 torch.set_num_threads(8);runner=Path(banana_smasher.__file__).parent/'qtip_runner.py';paths=[]
 for e in [78,79,80]:
  cfg=json.loads((PRIOR/f'baseline_warm2/E{e}.json').read_text());assert cfg['geometry']==dict(K=3,L=16,V=2)
  cfg['block_ldl_unitwise']=arm.startswith('candidate')
  cp=out/f'configs/E{e}.json';put(cp,cfg);paths.append(cp)
 import_seconds=time.perf_counter()-start;phases=[]
 for phase in ['cold','warm']:
  gate();dest=out/phase;dest.mkdir();put(dest/'SHARDS.json',json.loads((ROOT/'SHARDS.json').read_text()))
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();result=[]
  if arm.startswith('candidate'):result=main_batch(paths,dest,26)
  else:
   for cp in paths:result.extend(main_batch([cp],dest,26))
  torch.cuda.synchronize()
  receipt=dict(phase=phase,wall=time.perf_counter()-t,result=result,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,mem_available=memory()['MemAvailable'])
  put(out/(phase+'.json'),receipt);phases.append(receipt)
 put(out/'RESULT.json',dict(identity=identity,import_config_seconds=import_seconds,child_total_seconds=time.perf_counter()-start,phases=phases,module=banana_smasher.__file__))
 raise SystemExit()
ROOT.mkdir(exist_ok=False);gate()
identity=dict(task_id=TASK,pid=os.getpid(),pgid=os.getpgid(0),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),argv=sys.argv,canonical_git_pin=PIN,receipt=str(ROOT/'IDENTITY.json'),script_sha256=sha(__file__))
put(ROOT/'IDENTITY.json',identity)
with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);raw=CLAIM.read_bytes();c=json.loads(raw)
 assert hashlib.sha256(raw).hexdigest()==sys.argv[1] and c['task_id']==TASK
 assert c['stage']=='t_ebcba52e_ds4_k3_fused_verify_run8474_terminal_retained'
 assert not Path('/proc/'+str(c['pid'])).exists()
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 put(ROOT/'CLAIM_PREIMAGE.json',c)
 put(ROOT/'SHARDS.json',dict(intended_basis=BASIS,rows=[dict(owner=TASK,layer=26,experts=[78,79,80],projections=['fused13'],K=3,operation='new_private_cache_cold_comparison',receipt=str(ROOT/'IDENTITY.json'))]))
 claim=dict(identity,schema='banana-smasher.host-claim.v1',host='spark-6',owner=TASK,workload_pid=os.getpid(),state='CLAIMED',status='CLAIMED',stage=ROOT.name,claimed_at=time.time(),expiry_unix=time.time()+7200,mission=str(ROOT),source_model_index_sha256=BASIS,claim_preimage_sha256=hashlib.sha256(raw).hexdigest())
 assert CLAIM.read_bytes()==raw;put(CLAIM,claim);put(ROOT/'CLAIM.json',claim)
try:
 put(ROOT/'SCIENCE.json',dict(basis=BASIS,panel='authentic currentK3 L026 E078/E079/E080 fused13',purpose='new matched private-cache cold condition; not replay of shared-cache baseline',order=['baseline1','candidate1','candidate2','baseline2'],quality='frozen limits from sealed distinct baseline; independent decode validation required, no heldout claim',baseline_limits_sha256=sha(PRIOR/'FROZEN_LIMITS.json'),incumbent_result_sha256=sha('/dev/shm/t_ebcba52e_ds4_k3_fused_run8474/RESULT.json'),cache_scope='private canonical/Triton/Inductor/CUDA/XDG per child; source/prebuilt assets and OS page cache SHARED',production_promotion=False))
 for arm in ['baseline1','candidate1','candidate2','baseline2']:
  gate();out=ROOT/arm;out.mkdir();env=dict(os.environ)
  for name,sub in [('BANANA_SMASHER_KERNEL_CACHE','canonical'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','cuda'),('XDG_CACHE_HOME','xdg')]:env[name]=str(out/'cache'/sub)
  put(out/'CACHE_ENV.json',{k:env[k] for k in ['BANANA_SMASHER_KERNEL_CACHE','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','CUDA_CACHE_PATH','XDG_CACHE_HOME']});t=time.perf_counter()
  with (out/'child.log').open('w') as log:
   p=subprocess.Popen([sys.executable,'-u',__file__,'--child',arm],env=env,stdout=log,stderr=subprocess.STDOUT)
   ticks=int(Path(f'/proc/{p.pid}/stat').read_text().split()[21]);put(ROOT/'PROGRESS.json',dict(arm=arm,child_pid=p.pid,child_start_ticks=ticks,pgid=os.getpgid(p.pid),argv=[sys.executable,'-u',__file__,'--child',arm]));rc=p.wait()
  assert rc==0,(arm,rc,str(out/'child.log'))
  put(out/'PARENT_WALL.json',dict(seconds=time.perf_counter()-t,returncode=rc))
 put(ROOT/'RESULT.json',dict(state='SEALED_COLD_PROCESS_PANEL',identity=identity,arms={a:json.loads((ROOT/a/'RESULT.json').read_text()) for a in ['baseline1','candidate1','candidate2','baseline2']},production_promotion=False,ended=time.time()))
except BaseException as e:
 put(ROOT/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
finally:
 with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);c=json.loads(CLAIM.read_text());assert c['task_id']==TASK and c['pid']==os.getpid();c.update(stage=ROOT.name+'_terminal_retained',terminal_at=time.time());put(CLAIM,c)
