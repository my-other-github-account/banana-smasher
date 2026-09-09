import os,sys,json,time,hashlib,subprocess,fcntl,traceback,tarfile
from pathlib import Path
ROOT=Path('/dev/shm/t_1269dc5f');OUT=ROOT/'reference_tests_run8545'; CONTROL=ROOT/'traceback_run8542r3';OLD=Path('/dev/shm/t_182fbc9d');CLAIM=Path('/home/dnola/HOST_CLAIM.json');BASE='2a118a825ef00ff67dd0cbc65d07cc67a2919089';CAND='2972151b8d2f64d6f0cb5730cd641fcb63ca8b1c'
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
  assert old['pid']==1822897 and old['start_ticks']==8216815
  basis='3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05';assert sha('/dev/shm/t_ebcba52e_ldlq_run8503/selected_pair_run8507/model.safetensors.index.json')==basis
  mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024; st=os.statvfs(ROOT);assert mem>(12+4)<<30 and st.f_bavail*st.f_frsize>(1+4)<<30
  put(OUT/'PREFLIGHT.json',dict(mem_available=mem,peak_estimate=12<<30,reserve=4<<30,disk_available=st.f_bavail*st.f_frsize,prior_basis=old['source_model_index_sha256'],intended_basis=basis))
  with (ROOT/'SHARDS.lock').open('a+') as sl:
   fcntl.flock(sl,fcntl.LOCK_EX);raws=(ROOT/'SHARDS.json').read_bytes();shards=json.loads(raws);assert shards['intended_basis']==old['source_model_index_sha256'] and all(r['owner']=='t_1269dc5f' for r in shards['rows']);put(OUT/'SHARDS_PREIMAGE.json',shards);shards['intended_basis']=basis;shards['rows']=[dict(owner='t_1269dc5f',layer=4,experts=[242,243],projections=['down'],operation='unitwise_grouped',receipt=str(OUT/'QUALITY.json')),dict(owner='t_1269dc5f',layer=4,experts=[242],projections=['fused13'],operation='unitwise_grouped',receipt=str(OUT/'QUALITY.json'))];assert (ROOT/'SHARDS.json').read_bytes()==raws;put(ROOT/'SHARDS.json',shards)
  c=dict(old);c.update(**identity(os.getpid()),source_model_index_sha256=basis,mission=str(OUT),receipt=str(OUT/'IDENTITY.json'),stage='unitwise_grouped',expiry_unix=time.time()+7200,exact_cas_from_sha256=hashlib.sha256(raw).hexdigest());c.pop('terminal_at',None);put(OUT/'PREIMAGE.json',old);assert CLAIM.read_bytes()==raw;put(CLAIM,c);assert json.loads(CLAIM.read_text())==c;put(OUT/'IDENTITY.json',c)
 try:
  for tarname,dirname in [('reference_ldl_pin_run8545.tar','reference_ldl_code_run8545'),('traceback_base_run8542.tar','traceback_base_run8542')]:
   code=ROOT/dirname
   if not code.exists():
    code.mkdir()
    with tarfile.open(ROOT/tarname) as archive:archive.extractall(code,filter='data')
   with tarfile.open(ROOT/tarname) as archive:
    member=archive.extractfile('banana-smasher/src/banana_smasher/qtip_viterbi.py');assert hashlib.sha256(member.read()).hexdigest()==sha(code/'banana-smasher/src/banana_smasher/qtip_viterbi.py')
   put(OUT/(dirname+'_DEPLOY.json'),dict(tar_sha256=sha(ROOT/tarname),module_sha256=sha(code/'banana-smasher/src/banana_smasher/qtip_viterbi.py')))
  jobs=[('gpu_tests','candidate')];rows=[]
  for name,arm in jobs:
   env=dict(os.environ,TRITON_CACHE_DIR='/dev/shm/t_ebcba52e_ldlq_run8503/cache/triton',BANANA_SMASHER_KERNEL_CACHE='/dev/shm/t_ebcba52e_ldlq_run8503/cache/kernels',XDG_CACHE_HOME='/dev/shm/t_ebcba52e_ldlq_run8503/cache/xdg',PYTHONPATH=str(ROOT/'reference_ldl_code_run8545/banana-smasher/src'))
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
 tests=ROOT/'reference_ldl_code_run8545/banana-smasher/tests'
 raise SystemExit(subprocess.call([sys.executable,'-m','pytest',str(tests/'test_qtip_reference_ldl.py'),str(tests/'test_qtip_unitwise_ldl.py'),*[str(p) for p in tests.glob('test_qtip_batch*.py')],'-q']))
raise RuntimeError('tests only')
