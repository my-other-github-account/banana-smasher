from pathlib import Path
import os,sys,json,time,fcntl,hashlib,subprocess,cProfile,pstats,traceback
T='t_5ade4a57'; B='3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05'
R=Path('/run/t_5ade4a57'); O=Path('/dev/t_5ade4a57/fork8873'); C=Path('/home/dnola/HOST_CLAIM.json'); S=Path('/dev/shm/t_1269dc5f/SHARDS.json')
def put(p,x):
 p=Path(p);tmp=p.with_suffix('.tmp')
 with tmp.open('w') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(R));import storage_k4tracesixteen8821speed as storage
storage.ALLOWANCES['/dev']=2238552455+(2056<<20)
storage.ALLOWANCES['/run'] += 4<<20 # run8843 recovery authority: bounded mechanics; original PRESTAGE/reserve retained
with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX); raw=C.read_bytes();c=json.loads(raw)
 assert (c['owner'],c['pid'],c['start_ticks'])==(T,1255316,75823398)
 assert not Path('/proc/1255316').exists()
 assert json.loads(Path(c['mission'],'TERMINAL.json').read_text())['returncode']==1
 scan=subprocess.run(['pgrep','-af','fork_supervisor8873|fork_arm8873'],capture_output=True,text=True).stdout
 assert all(int(line.split()[0])==os.getpid() for line in scan.splitlines())
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 a=json.loads(Path('/dev/shm/t_5ade4a57/ALLOCATION.json').read_text());assert a['task']==T and a['host']=='spark-6' and a['intended_basis']==B
 sys.path.insert(0,str(R/'bounded8846_code'));from bounded_cap8858 import admit as cap_admit
 cap=storage.evaluate(json.loads((R/'k4tracesixteen8821speed_PRESTAGE.json').read_text()),storage.sample());assert cap['/run']['remaining']>65536 and cap['/dev']['remaining']>(112<<20)
 mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024;assert mem>((46<<30)+(768<<20))
 M=Path('/dev/t_5ade4a57/selected_down8828/model');assert sha(M/'model.safetensors.index.json')==B==c['source_model_index_sha256']
 cap_admit(mem,os.statvfs('/dev').f_bavail*os.statvfs('/dev').f_frsize,56<<20,R/'bounded8846_code/WORKING_SET.json','0dcfe243630fe7324ce73af2f5c925cb8eafedcec6fb6dea282802b64bef0b5d')
 with S.with_suffix('.lock').open('a+') as sl:
  fcntl.flock(sl,fcntl.LOCK_EX);sr=S.read_bytes();s=json.loads(sr);assert s['intended_basis']==B
  assert all((x['owner'],x['pid'],x['startticks'])==(T,1255316,75823398) for x in s['rows'])
  O.mkdir();i=dict(pid=os.getpid(),pgid=os.getpgid(0),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),argv=Path('/proc/self/cmdline').read_bytes().decode().split('\0')[:-1])
  put(O/'PREFLIGHT.json',dict(capacity=cap,mem_available=mem,peak_estimate=38<<30,reserve=8<<30,allocation=a,pgrep=scan,prior_claim=c))
  c.update(**i,state='CLAIMED',status='CLAIMED',execution_claim_released=False,mission=str(O),receipt=str(O/'IDENTITY.json'),stage='fork8873',expiry_unix=time.time()+3600);c.pop('terminal_at',None)
  s['rows']=[dict(owner=T,layer=40,experts=[47],projections=['down'],pid=i['pid'],startticks=i['start_ticks'],operation='original_pin_serialization_ABBA_PRE8',receipt=str(O/'IDENTITY.json'))]
  s['rows'].append(dict(s['rows'][0],layer=20,experts=[0],projections=['fused13']))
  s['rows'].append(dict(s['rows'][0],layer=26,experts=[96,97],projections=['fused13']))
  assert C.read_bytes()==raw and S.read_bytes()==sr;put(S,s);put('/dev/shm/t_5ade4a57/SHARDS.json',s);put(C,c);put(O/'IDENTITY.json',c);put(O/'SHARDS.json',s)
try:
 import tarfile
 p=Path('/dev/t_5ade4a57/cyclic_gc8850.tar');assert sha(p)=='c672c6ea045d8c30502f794d88435062b1c5905071bd5ddeb62ec262805e9d67'
 with tarfile.open(p) as tf:tf.extractall(O/'code',filter='data')
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',SUPERVISOR_PID=str(os.getpid()),PYTHONPATH=str(R)+':'+str(R/'bounded8846_code')+':'+str(O/'code/banana-smasher/src/banana_smasher')+':/dev/t_5ade4a57/originalpin8846/code/banana-smasher/src',TRITON_CACHE_DIR='/dev/t_5ade4a57/triton8680',BANANA_SMASHER_KERNEL_CACHE='/dev/t_5ade4a57/cache8680/kernels',XDG_CACHE_HOME='/run/user/1000/t_5ade4a57/cache8603/xdg')
 rows=[]
 assert sha(R/'fork_arm8873.py')=='44dc0c32d091c47e05530482d60783bf435c25036abfdfb4dc0e243fd9d1f26e'
 assert sha(R/'fork_pair8873.py')=='71a720c4d4f3e69fc0b7ea38338a0ce065ae7371ed80ea843764d52f91d3e433'
 assert sha(R/'pinned_cpu_scope.py')=='95143663a604a72ba125cf2e8d2abe63b9084548626409e52aa70c812187060e'
 for label in ['C1']:
  start=time.perf_counter();parts=['forked_pairs']
  for part in parts:
   cmd=['/home/dnola/humming_env/bin/python','-u',str(R/'fork_pair8873.py')]
   with (O/(label+'_'+part+'.log')).open('x') as log:
    child=subprocess.Popen(cmd,env=dict(env,RESIDENT_PHASES=part,TORCHDYNAMO_DISABLE='0',CUDA_DEVICE_MAX_CONNECTIONS='8',CUDA_MODULE_LOADING='LAZY',QUALITY_PHASE=part,PYTORCH_CUDA_ALLOC_CONF='expandable_segments:False'),stdout=log,stderr=subprocess.STDOUT)
    put(O/(label+'_'+part+'_CHILD.json'),dict(pid=child.pid,start_ticks=int(Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]),argv=cmd,pgid=os.getpgid(child.pid)))
    rc=child.wait()
   assert rc==0,(label,part,rc)
  rows.append(dict(label=label,enclosed_seconds=time.perf_counter()-start));put(O/'PROGRESS.json',rows)
 put(O/'TERMINAL.json',dict(returncode=0,scope='CPU-only forkserver, max2 maintained, end-to-end includes preload; historical PRE unchanged'))
except BaseException:
 put(O/'TERMINAL.json',dict(returncode=1,error=traceback.format_exc()));raise
finally:
 with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);c=json.loads(C.read_text());assert c['pid']==os.getpid();c.update(stage='fork8873_terminal_retained',terminal_at=time.time());put(C,c)
