import os,sys,json,time,fcntl,subprocess,hashlib,traceback
from pathlib import Path
BASE=Path('/dev/shm/t_ebcba52e_ldlq_run8503');C=Path('/home/dnola/HOST_CLAIM.json');R=BASE/sys.argv[1];worker=BASE/sys.argv[2];pre=sys.argv[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));os.replace(q,p)
R.mkdir(exist_ok=False)
with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as f:
 fcntl.flock(f,fcntl.LOCK_EX);raw=C.read_bytes();old=json.loads(raw)
 assert sha(C)==pre and old['task_id']==old['owner']=='t_ebcba52e' and old['host']==os.uname().nodename=='spark-6'
 assert old['expiry_unix']>time.time() and old['stage'].endswith('_terminal_retained')
 p=Path('/proc')/str(old['pid']);assert not p.exists() or p.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]=='Z'
 oldroot=Path(old['mission']);assert (oldroot/'TERMINAL.json').exists()
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 assert next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))>36<<30
 assert os.statvfs(BASE).f_bavail*os.statvfs(BASE).f_frsize>4<<30
 shards=json.loads((oldroot/'SHARDS.json').read_text());assert shards['intended_basis']==old['source_model_index_sha256']=='3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05'
 c=dict(old);c.update(pid=os.getpid(),pgid=os.getpgid(0),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),stage=R.name,mission=str(R),receipt=str(R/'IDENTITY.json'),script_sha256=sha(__file__),argv=sys.argv,expiry_unix=time.time()+7200,exact_cas_from_sha256=pre);c.pop('terminal_at',None)
 changed={k for k in old.keys()|c.keys() if old.get(k)!=c.get(k)};assert changed<={'pid','pgid','start_ticks','stage','mission','receipt','script_sha256','argv','expiry_unix','exact_cas_from_sha256','terminal_at'}
 put(R/'CLAIM_PREIMAGE.json',old);put(R/'CLAIM_DRY_RENDER.json',dict(candidate=c,changed=sorted(changed),previous_terminal_sha256=sha(oldroot/'TERMINAL.json'),previous_shards_sha256=sha(oldroot/'SHARDS.json'),worker_sha256=sha(worker)))
 assert C.read_bytes()==raw;put(C,c);assert json.loads(C.read_text())==c;put(R/'IDENTITY.json',c);put(R/'CLAIM_POSTIMAGE.json',dict(claim=c,sha256=sha(C)));put(R/'SHARDS.json',shards)
try:
 env=dict(os.environ,PYTHONPATH=str(BASE/'canonical/banana-smasher/src'),TRITON_CACHE_DIR=str(BASE/'cache/triton'),BANANA_SMASHER_KERNEL_CACHE=str(BASE/'cache/kernels'),XDG_CACHE_HOME=str(BASE/'cache/xdg'),OUT=str(R))
 t=time.perf_counter()
 with (R/'WORKER.log').open('w') as f:
  p=subprocess.Popen([sys.executable,'-u',str(worker)],env=env,stdout=f,stderr=subprocess.STDOUT);put(R/'CHILD.json',dict(pid=p.pid,startticks=int(Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19])));rc=p.wait()
 put(R/'TERMINAL.json',dict(returncode=rc,whole_seconds=time.perf_counter()-t));assert rc==0
finally:
 with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);c=json.loads(C.read_text());assert c['pid']==os.getpid();c.update(stage=R.name+'_terminal_retained',terminal_at=time.time());put(C,c);assert json.loads(C.read_text())==c;put(R/'CLAIM_RETAINED.json',dict(claim=c,sha256=sha(C)))
