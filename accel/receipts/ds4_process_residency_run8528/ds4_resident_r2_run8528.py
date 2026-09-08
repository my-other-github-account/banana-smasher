import os,sys,json,time,hashlib,subprocess,fcntl,traceback
from pathlib import Path
ROOT=Path('/dev/shm/t_1269dc5f');OUT=ROOT/'ds4_resident_r2_run8528';OLD=Path('/dev/shm/t_182fbc9d');CLAIM=Path('/home/dnola/HOST_CLAIM.json');BASE='6a40f916a6def348cb6a65bcd648006ab3242211';CAND='7391e41b2e7dad4cd0ebeda7dc6e95e05cae92e0'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
 with tmp.open('w') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def identity(pid):return dict(pid=pid,pgid=os.getpgid(pid),start_ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]),argv=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')[:-1])
def dead(pid):return not Path(f'/proc/{pid}').exists()
if len(sys.argv)==1:
 OUT.mkdir(exist_ok=False)
 with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);raw=CLAIM.read_bytes();old=json.loads(raw);assert old['owner']==old['task_id']=='t_1269dc5f' and dead(old['pid'])
  prior=Path(old['mission']);assert prior==ROOT/'ds4_resident_run8528' and json.loads((prior/'TERMINAL.json').read_text())['returncode']==1 and (prior/'FAILURE.json').exists();assert dead(json.loads((prior/'CHILD.json').read_text())['pid'])
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  basis='98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b';assert sha('/home/dnola/models/hf/DeepSeek-V4-Flash-0731/model.safetensors.index.json')==basis
  with (ROOT/'SHARDS.lock').open('a+') as sl:
   fcntl.flock(sl,fcntl.LOCK_EX);raws=(ROOT/'SHARDS.json').read_bytes();shards=json.loads(raws);assert shards['intended_basis']==old['source_model_index_sha256'] and all(r['owner']=='t_1269dc5f' for r in shards['rows']);put(OUT/'SHARDS_PREIMAGE.json',shards);shards['intended_basis']=basis;shards['rows']=[dict(owner='t_1269dc5f',layer=26,experts=[78,79,80],projections=['fused13'],operation='resident_process_amortization_not_production')];assert (ROOT/'SHARDS.json').read_bytes()==raws;put(ROOT/'SHARDS.json',shards)
  c=dict(old);c.update(**identity(os.getpid()),mission=str(OUT),receipt=str(OUT/'IDENTITY.json'),stage='ds4_resident_abba',source_model_index_sha256=basis,expiry_unix=time.time()+7200,exact_cas_from_sha256=hashlib.sha256(raw).hexdigest());c.pop('terminal_at',None);put(OUT/'PREIMAGE.json',old);assert CLAIM.read_bytes()==raw;put(CLAIM,c);assert json.loads(CLAIM.read_text())==c;put(OUT/'IDENTITY.json',c)
 try:
  rows=[]
  for arm in ['B1','C1','C2','B2']:
   t=time.perf_counter();commands=([['0'],['1']] if arm.startswith('B') else [['0','1']])
   for phases in commands:
    env=dict(os.environ,TRITON_CACHE_DIR='/dev/shm/t_ebcba52e_ldlq_run8503/cache/triton',BANANA_SMASHER_KERNEL_CACHE='/dev/shm/t_ebcba52e_ldlq_run8503/cache/kernels',XDG_CACHE_HOME='/dev/shm/t_ebcba52e_ldlq_run8503/cache/xdg')
    cmd=[sys.executable,'-u',__file__,arm,*phases]
    with (OUT/f'{arm}_{phases[0]}.log').open('w') as log:
     p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);put(OUT/'CHILD.json',identity(p.pid));rc=p.wait()
    assert rc==0,(arm,phases,rc)
   rows.append(dict(arm=arm,process_inclusive_seconds=time.perf_counter()-t));put(OUT/'PROGRESS.json',rows)
  put(OUT/'TERMINAL.json',dict(returncode=0,rows=rows))
 except BaseException:
  put(OUT/'FAILURE.json',dict(error=traceback.format_exc()));put(OUT/'TERMINAL.json',dict(returncode=1));raise
 finally:
  with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);c=json.loads(CLAIM.read_text());assert c['pid']==os.getpid();c.update(stage='ds4_resident_terminal_retained',terminal_at=time.time());put(CLAIM,c)
 raise SystemExit()
assert json.loads(CLAIM.read_text())['pid']==os.getppid()
arm=sys.argv[1];phases=sys.argv[2:];API=OLD/'main_run8524/banana-smasher/src';sys.path.insert(0,str(API))
import torch
from banana_smasher import solver_qtip_profile as sp
from banana_smasher.qtip_batch_controller import main_batch
torch.set_num_threads(8);free=torch.cuda.mem_get_info()[0];put(OUT/(arm+'_'+phases[0]+'_PREFLIGHT.json'),dict(cuda_free=free,peak_estimate=12<<30,reserve=4<<30,retained_same_panel_peak_allocated=1626499584,retained_same_panel_peak_reserved=2086666240));assert free>(12+4)<<30
stat=os.statvfs(ROOT);assert stat.f_bavail*stat.f_frsize>(1+4)<<30
for phase in phases:
 out=OUT/arm/phase;out.mkdir(parents=True);put(out/'SHARDS.json',json.loads((ROOT/'SHARDS.json').read_text()));paths=[]
 for expert in [78,79,80]:
  source=Path(f'/dev/shm/t_ebcba52e_ds4_k3_fused_run8474/candidate_warm1/E{expert}.json');cfg=json.loads(source.read_text());assert sha(cfg['input_identity']['model_index']['path'])==json.loads((out/'SHARDS.json').read_text())['intended_basis'];runner=Path(sp.__file__).with_name('qtip_runner.py');cfg['qtip_runner']=str(runner);cfg['exact_solver']='banana_smasher.qtip_viterbi@'+BASE
  manifest=json.loads(Path(cfg['materialization']['run_manifest']).read_text());manifest['canonical_git_pin']=BASE;manifest['tiers'][0]['bindings']['qtip_runner']=dict(path=str(runner),sha256=sha(runner),bytes=runner.stat().st_size);mp=out/f'MANIFEST_E{expert}.json';put(mp,manifest);cfg['materialization']['run_manifest']=str(mp);cfg['materialization']['run_manifest_sha256']=sha(mp);cp=out/f'E{expert}.json';put(cp,cfg);paths.append(cp)
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();results=main_batch(paths,out,26);torch.cuda.synchronize();wall=time.perf_counter()-t;put(out/'RESULT.json',dict(pin=BASE,main_batch_seconds=wall,results=results,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),scope='two identical representative K3 triples per arm; fresh processes vs one resident process, shared caches; no production or heldout claim',config_source_sha256={str(p):sha(p) for p in paths}))
