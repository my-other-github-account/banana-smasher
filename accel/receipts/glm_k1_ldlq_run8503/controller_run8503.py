import os,sys,time,json,hashlib,fcntl,subprocess,tarfile,traceback
from pathlib import Path
R=Path('/dev/shm/t_ebcba52e_ldlq_run8503');C=Path('/home/dnola/HOST_CLAIM.json');TASK='t_ebcba52e'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));os.replace(q,p)
def main():
 assert os.uname().nodename=='spark-6'
 assert sha(R/'closure.tar')=='a08ab89be8af74c98b1a17110eafbd5a0b40269916f706475b53bd2bbd8a8e34'
 for name,dest in [('closure.tar','closure'),('canonical_run8503.tar','canonical')]:
  with tarfile.open(R/name) as t:t.extractall(R/dest,filter='data')
 m=json.loads((R/'closure/MANIFEST.json').read_text())
 for row in m['files']+m['components']:
  p=R/'closure'/row['archive_path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
 put(R/'INPUT_VERIFIED.json',dict(files=len(m['files']),components=len(m['components']),archive_sha256=sha(R/'closure.tar'),canonical_archive_sha256=sha(R/'canonical_run8503.tar')))
 pre='21ecb4e7ca12802e2f5f0828ac54f534f27f33ff0cce6ca9c9e03d55c24b85ca'
 with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);raw=C.read_bytes();old=json.loads(raw)
  assert hashlib.sha256(raw).hexdigest()==pre and old['task_id']==TASK and old['owner']==TASK and old['host']=='spark-6'
  assert old['pid']==1130497 and old['start_ticks']==5639476 and not Path('/proc/1130497').exists()
  assert old['expiry_unix']>time.time()
  historic=Path(old['mission'])
  assert sha(historic/'SHARDS.json')=='9145a3b466012c698f46a8b918b2e0dfdcee740f78f009b55491ae92c5189217'
  assert sha(historic/'RESULT.json')=='4a245dd2700883e320bc4c8abfa54e85baae4783793afa94e3a0c512879074d9'
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  mem=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'));assert mem>36<<30
  v=os.statvfs(R);assert v.f_bavail*v.f_frsize>8<<30
  c=dict(old);c.update(pid=os.getpid(),pgid=os.getpgid(0),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),argv=sys.argv,receipt=str(R/'IDENTITY.json'),script_sha256=sha(__file__),stage=R.name,mission=str(R),expiry_unix=time.time()+7200,source_model_index_sha256=m['intended_basis'],exact_cas_from_sha256=pre);c.pop('terminal_at',None)
  changed={k for k in c.keys()|old.keys() if c.get(k)!=old.get(k)}
  assert changed<={'pid','pgid','start_ticks','argv','receipt','script_sha256','stage','mission','expiry_unix','source_model_index_sha256','exact_cas_from_sha256','terminal_at'}
  put(R/'CLAIM_PREIMAGE.json',old);put(R/'CLAIM_DRY_RENDER.json',dict(candidate=c,changed=sorted(changed),preimage_sha256=pre))
  assert C.read_bytes()==raw;put(C,c);assert json.loads(C.read_text())==c
  put(R/'CLAIM_POSTIMAGE.json',dict(claim=c,sha256=sha(C)));put(R/'IDENTITY.json',c)
  put(R/'SHARDS.json',dict(intended_basis=m['intended_basis'],rows=[dict(owner=TASK,layer=3,experts=[145],projections=['down'],K=1,operation='representative_acceleration_diagnostic_not_production')]))
 try:
  env=dict(os.environ,PYTHONPATH=str(R/'canonical/banana-smasher/src'),TRITON_CACHE_DIR=str(R/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(R/'cache/inductor'),BANANA_SMASHER_KERNEL_CACHE=str(R/'cache/kernels'),CUDA_CACHE_PATH=str(R/'cache/cuda'),XDG_CACHE_HOME=str(R/'cache/xdg'))
  cmd=[sys.executable,'-u',str(R/'micro_run8503.py')];t=time.perf_counter()
  with (R/'MICRO.log').open('w') as f:
   p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT);put(R/'CHILD.json',dict(pid=p.pid,startticks=int(Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19])));rc=p.wait()
  put(R/'TERMINAL.json',dict(returncode=rc,whole_seconds=time.perf_counter()-t));assert rc==0
 except BaseException as e:put(R/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
 finally:
  with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as f:
   fcntl.flock(f,fcntl.LOCK_EX);c=json.loads(C.read_text());assert c['pid']==os.getpid() and c['task_id']==TASK
   c.update(stage=R.name+'_terminal_retained',terminal_at=time.time());put(C,c);assert json.loads(C.read_text())==c;put(R/'CLAIM_RETAINED.json',dict(claim=c,sha256=sha(C)))
if __name__=='__main__':main()
