import os,sys,json,time,hashlib,subprocess,fcntl,traceback,tarfile
from pathlib import Path
TASK='t_6ac2b6f5'; ROOT=Path('/dev/shm')/TASK
PLAN=Path(sys.argv[1]); spec=json.loads(PLAN.read_text());OUT=ROOT/spec['attempt']
DURABLE=Path('/home/dnola/missions')/TASK/spec['attempt']
CLAIM=Path('/home/dnola/HOST_CLAIM.json');SHARDS=Path('/dev/shm/t_1269dc5f/SHARDS.json')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.with_suffix('.tmp').open('w') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(p.with_suffix('.tmp'),p)
 fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def receipt(name,x):
 put(OUT/name,x);put(DURABLE/name,x)
def identity(pid):
 start=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
 for _ in range(200):
  argv=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')[:-1]
  if argv:return dict(pid=pid,pgid=os.getpgid(pid),start_ticks=start,argv=argv)
  time.sleep(.01)
 raise RuntimeError('exec handshake missing argv')
def dead(pid):return not Path(f'/proc/{pid}').exists()
if spec.get('wait_for_predecessor'):
 put(ROOT/(spec['attempt']+'_WAITER.json'),dict(**identity(os.getpid()),expected_pid=spec['expected_pid'],expected_start_ticks=spec['expected_start_ticks'],plan_sha256=sha(PLAN)))
 deadline=time.monotonic()+1200
 while not dead(spec['expected_pid']):
  observed=identity(spec['expected_pid']);assert observed['start_ticks']==spec['expected_start_ticks']
  assert time.monotonic()<deadline,'predecessor wait deadline; no signal sent'
  time.sleep(2)
OUT.mkdir(exist_ok=False);DURABLE.mkdir(parents=True,exist_ok=False)
with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);raw=CLAIM.read_bytes();old=json.loads(raw)
 assert old['owner']==spec['expected_owner'] and old['task_id']==spec['expected_owner']
 assert old['pid']==spec['expected_pid'] and old['start_ticks']==spec['expected_start_ticks'] and dead(old['pid'])
 prior=Path(old['mission']);terminal=json.loads((prior/'TERMINAL.json').read_text());child=json.loads((prior/'CHILD.json').read_text());assert dead(child['pid'])
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 basis='3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05'
 assert old['source_model_index_sha256']==basis==sha('/dev/shm/t_ebcba52e_ldlq_run8503/selected_pair_run8507/model.safetensors.index.json')
 mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
 st=os.statvfs(ROOT);sd=os.statvfs(DURABLE)
 assert mem>(12+4)<<30 and st.f_bavail*st.f_frsize>(1+4)<<30 and sd.f_bavail*sd.f_frsize>(1<<30)+(64<<20)
 receipt('PREFLIGHT.json',dict(mem_available=mem,peak_estimate=12<<30,reserve=4<<30,tmpfs_free=st.f_bavail*st.f_frsize,durable_free=sd.f_bavail*sd.f_frsize,basis=basis,prior_terminal=terminal,prior_child=child,pgrep=subprocess.run(['pgrep','-af','python|t_6ac2b6f5'],capture_output=True,text=True).stdout))
 receipt('AUTHORITY.json',dict(card=TASK,allocation='explicit dedicated spark-6 continuation from t_1269dc5f',driver_text=Path('/home/dnola/missions/DRIVER_GOALS.md').read_text(),plan=spec,plan_sha256=sha(PLAN),script_sha256=sha(__file__)))
 with SHARDS.with_suffix('.lock').open('a+') as sl:
  fcntl.flock(sl,fcntl.LOCK_EX);raws=SHARDS.read_bytes();s=json.loads(raws)
  assert s['intended_basis']==basis and all(r['owner']==spec['expected_owner'] for r in s['rows'])
  receipt('SHARDS_PREIMAGE.json',s)
  for r in s['rows']:r.update(owner=TASK,operation='column_bmm_factorization',receipt=str(OUT/'TERMINAL.json'))
  assert SHARDS.read_bytes()==raws;put(SHARDS,s);put(ROOT/'SHARDS.json',s);receipt('SHARDS.json',s)
 c=dict(old);c.update(**identity(os.getpid()),owner=TASK,task_id=TASK,mission=str(OUT),receipt=str(OUT/'IDENTITY.json'),stage=spec['attempt'],expiry_unix=time.time()+7200,exact_cas_from_sha256=hashlib.sha256(raw).hexdigest());c.pop('terminal_at',None)
 receipt('PREIMAGE.json',old);assert CLAIM.read_bytes()==raw;put(CLAIM,c);assert json.loads(CLAIM.read_text())==c;receipt('IDENTITY.json',c)
try:
 code=ROOT/spec['code_dir'];archive=ROOT/spec['archive']
 assert sha(archive)==spec['archive_sha256'];code.mkdir(exist_ok=False)
 with tarfile.open(archive) as tf:tf.extractall(code,filter='data')
 receipt('DEPLOY.json',dict(pin=spec['pin'],archive_sha256=sha(archive),code_dir=str(code)))
 rows=[]
 for job in spec['jobs']:
  env=dict(os.environ,PYTHONPATH=str(code/'banana-smasher/src'),TRITON_CACHE_DIR='/dev/shm/t_ebcba52e_ldlq_run8503/cache/triton',BANANA_SMASHER_KERNEL_CACHE='/dev/shm/t_ebcba52e_ldlq_run8503/cache/kernels',XDG_CACHE_HOME='/dev/shm/t_ebcba52e_ldlq_run8503/cache/xdg')
  start=time.perf_counter()
  with (OUT/(job['name']+'.log')).open('w') as log:
   p=subprocess.Popen(job['cmd'],env=env,stdout=log,stderr=subprocess.STDOUT);ident=identity(p.pid)
   receipt('CHILD.json',ident);receipt(job['name']+'_CHILD.json',ident)
   rc=p.wait()
  row=dict(name=job['name'],returncode=rc,seconds=time.perf_counter()-start,identity=ident);rows.append(row);receipt('PROGRESS.json',rows)
  receipt(job['name']+'_LOG.json',dict(text=(OUT/(job['name']+'.log')).read_text()))
  assert rc==job.get('expected_returncode',0),row
 receipt('TERMINAL.json',dict(returncode=0,rows=rows))
except BaseException:
 receipt('FAILURE.json',dict(error=traceback.format_exc()));receipt('TERMINAL.json',dict(returncode=1));raise
finally:
 with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);c=json.loads(CLAIM.read_text());assert c['pid']==os.getpid();c.update(stage=spec['attempt']+'_terminal_retained',terminal_at=time.time());put(CLAIM,c);receipt('CLAIM_TERMINAL.json',c)
