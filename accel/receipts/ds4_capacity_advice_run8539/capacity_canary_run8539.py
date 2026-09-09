import os,sys,json,time,hashlib,subprocess,fcntl,traceback,importlib.util,tarfile
from pathlib import Path
ROOT=Path('/dev/shm/t_1269dc5f'); OUT=ROOT/'capacity_canary_run8539'; CLAIM=Path('/home/dnola/HOST_CLAIM.json'); PIN='212a5dd0be04220d218ba5e0a7dae9758f2ddb07'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
 os.chmod(p,0o444)
 with p.open('rb') as f:os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def replace(p,x):
 tmp=p.with_suffix('.8539.tmp')
 with tmp.open('w') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
 fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def ident(pid):return dict(pid=pid,pgid=os.getpgid(pid),start_ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]),argv=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')[:-1])
if len(sys.argv)==1:
 OUT.mkdir(exist_ok=False)
 with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);raw=CLAIM.read_bytes();old=json.loads(raw);assert old['owner']==old['task_id']=='t_1269dc5f' and not Path(f'/proc/{old["pid"]}').exists()
  prior=Path(old['mission']);assert json.loads((prior/'TERMINAL.json').read_text())['returncode']==0
  assert not Path(f'/proc/{json.loads((prior/"CHILD.json").read_text())["pid"]}').exists()
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  basis=sha('/home/dnola/models/hf/DeepSeek-V4-Flash-0731/model.safetensors.index.json');assert basis==old['source_model_index_sha256']
  mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024;disk=os.statvfs(ROOT);assert mem>(12+4)<<30 and disk.f_bavail*disk.f_frsize>(1+4)<<30
  with (ROOT/'SHARDS.lock').open('a+') as sl:
   fcntl.flock(sl,fcntl.LOCK_EX);sr=(ROOT/'SHARDS.json').read_bytes();s=json.loads(sr);assert s['intended_basis']==basis and all(r['owner']=='t_1269dc5f' for r in s['rows']);put(OUT/'SHARDS_PREIMAGE.json',s)
   s['rows']=[dict(owner='t_1269dc5f',layer=26,experts=[78,79,80],projections=['fused13'],operation='capacity_advice_canary_existing_units_no_solve',receipt=str(OUT/'RESULT.json'))];assert (ROOT/'SHARDS.json').read_bytes()==sr;replace(ROOT/'SHARDS.json',s)
  c=dict(old);c.update(**ident(os.getpid()),mission=str(OUT),receipt=str(OUT/'IDENTITY.json'),stage='capacity_advice_canary',expiry_unix=time.time()+7200,exact_cas_from_sha256=hashlib.sha256(raw).hexdigest());put(OUT/'PREIMAGE.json',old);assert CLAIM.read_bytes()==raw;replace(CLAIM,c);assert json.loads(CLAIM.read_text())==c;put(OUT/'IDENTITY.json',c);put(OUT/'PREFLIGHT.json',dict(mem_available=mem,peak_estimate=12<<30,reserve=4<<30,disk_available=disk.f_bavail*disk.f_frsize,basis=basis))
 try:
  with (OUT/'WORKER.log').open('x') as log:
   p=subprocess.Popen([sys.executable,'-u',__file__,'child'],stdout=log,stderr=subprocess.STDOUT);put(OUT/'CHILD.json',ident(p.pid));rc=p.wait()
  put(OUT/'TERMINAL.json',dict(returncode=rc));assert rc==0
 except BaseException:
  put(OUT/'FAILURE.json',dict(error=traceback.format_exc()));raise
 finally:
  with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);c=json.loads(CLAIM.read_text());assert c['pid']==os.getpid();c.update(stage='capacity_canary_terminal_retained',terminal_at=time.time());replace(CLAIM,c)
 raise SystemExit()
assert json.loads(CLAIM.read_text())['pid']==os.getppid()
code=OUT/'code';code.mkdir()
with tarfile.open(ROOT/'capacity_pin_run8539.tar') as t:t.extractall(code,filter='data')
module=code/'banana-smasher/src/banana_smasher/capacity_advice.py';spec=importlib.util.spec_from_file_location('capacity_advice',module);api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
cp=subprocess.run([sys.executable,str(code/'banana-smasher/tests/test_capacity_advice.py')],capture_output=True,text=True);put(OUT/'TESTS.json',dict(returncode=cp.returncode,stdout=cp.stdout,stderr=cp.stderr));assert cp.returncode==0
import torch
torch.set_num_threads(8)
def probe():return torch.cuda.mem_get_info()[0]
assert probe()>(12+4)<<30
paths=sorted((ROOT/'ds4_distinct_run8528').glob('C1/*/solve/L026/*/QTIP_UNIT.pt'));assert len(paths)==3
expected=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in paths];put(OUT/'INPUTS.json',expected)
def advise():
 for d in expected:
  p=Path(d['path']);assert sha(p)==d['sha256'] and p.stat().st_size==d['bytes']
  fd=os.open(p,os.O_RDONLY)
  try:os.fsync(fd);os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
  finally:os.close(fd)
rows=[]
for arm in ['B1','C1','C2','B2']:
 t=time.perf_counter()
 if arm[0]=='B':
  before=probe();advise();after=probe();assert after>(12+4)<<30;r=dict(action='unconditional_historical_advice',before_free_bytes=before,after_free_bytes=after)
 else:r=api.capacity_advice(probe,12<<30,4<<30,advise)
 r.update(arm=arm,wall_seconds=time.perf_counter()-t);rows.append(r);put(OUT/(arm+'.json'),r)
assert all(sha(d['path'])==d['sha256'] for d in expected)
put(OUT/'RESULT.json',dict(pin=PIN,module_sha256=sha(module),tar_sha256=sha(ROOT/'capacity_pin_run8539.tar'),rows=rows,all_three_original_artifacts_unchanged=True,scope='capacity helper mechanical canary on three authentic retained units; no solve/no heldout/no production speed denominator; tmpfs advice not claimed as reclamation',quality='no numerical implementation or scientific byte changed; original numerical gate retained'))
