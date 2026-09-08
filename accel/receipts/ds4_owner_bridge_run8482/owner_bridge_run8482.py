"""Historical-owner three-cell bridge. Canonical API only; no forward."""
import os,sys,json,time,hashlib,fcntl,subprocess,resource,traceback,math
from pathlib import Path
from bridge_config_run8482 import bind_config
TASK='t_ebcba52e'
ROOT=Path('/dev/shm/t_ebcba52e_owner_bridge_run8482')
CLAIM=Path('/home/dnola/HOST_CLAIM.json')
STAGE=Path('/dev/shm/t_ebcba52e_bridge_stage_run8482')
OLD='4921456bba5ebc032fb96030ee11e58f3ea2826a'
NEW='39da421e75b84c975f6a8c11f508c297c3f73cb5'
BASIS='98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b'
EXPERTS=(90,91,92)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
def put(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
 with tmp.open('w') as f:json.dump(d,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def gate():
 mem=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
 assert mem>(36<<30),mem
 v=os.statvfs('/dev/shm');assert v.f_bavail*v.f_frsize>(4<<30)
 return mem
def identity():
 s=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
 return dict(task_id=TASK,pid=os.getpid(),pgid=os.getpgid(0),start_ticks=int(s[19]),argv=sys.argv,receipt=str(ROOT/'IDENTITY.json'),script_sha256=sha(__file__))
def owned_child():
 c=json.loads(CLAIM.read_text());i=json.loads((ROOT/'IDENTITY.json').read_text())
 assert c['task_id']==TASK and c['pid']==os.getppid()==i['pid'] and c['stage']==ROOT.name
 gate()
def config_for(e):return json.loads((STAGE/f'candidate_L009_E{e:03d}.json').read_text())
def setup_api(pin):
 api=STAGE/pin/'banana-smasher/src';sys.path.insert(0,str(api))
 import torch,banana_smasher
 torch.set_num_threads(8);assert Path(banana_smasher.__file__).resolve().is_relative_to(api)
 inventory=json.loads((STAGE/(pin+'.inventory.json')).read_text())
 for name,digest in inventory.items():assert sha(STAGE/pin/name)==digest,name
 return torch,api,dict(pin=pin,module=banana_smasher.__file__,inventory_sha256=sha(STAGE/(pin+'.inventory.json')))
def decode(torch,path):
 from banana_smasher.sealed_qtip_unit import decode_sealed_unit
 unit=torch.load(path,map_location='cpu',mmap=True,weights_only=True);m,n=unit['shape']
 assert {k:unit['geometry'][k] for k in ('K','L','V')}==dict(K=3,L=16,V=2)
 row=dict(shape=[m,n],source_transform=dict(output_quantity='descaled_weight'),wire=dict(unit=dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path)),geometry=unit['geometry'],unit_shape=unit['shape'],row_range=[0,m]))
 return decode_sealed_unit(ROOT,row).float()
def objective(x,w):
 nums=[];dens=[]
 for i in range(0,w.shape[0],64):
  ref=w[i:i+64].double();nums.append(float((x[i:i+64].double()-ref).square().sum()));dens.append(float(ref.square().sum()))
 return math.fsum(nums)/math.fsum(dens)
def artifact(arm,phase,e):return ROOT/arm/phase/f'solve/L009/E{e:03d}_down/QTIP_UNIT.pt'

def child(arm):
 owned_child();start=time.perf_counter();pin=OLD if arm.startswith('old') else NEW
 torch,api,imported=setup_api(pin)
 from banana_smasher.qtip_batch_controller import main_batch
 out=ROOT/arm;paths=[];runner=api/'banana_smasher/qtip_runner.py';grouped=arm=='batch'
 for e in EXPERTS:
  cfg=config_for(e);original=Path(cfg['materialization']['run_manifest']);assert sha(original)==cfg['materialization']['run_manifest_sha256']
  manifest=json.loads(original.read_text());manifest['canonical_git_pin']=pin
  for tier in manifest['tiers']:
   if tier['name']==cfg['tier']:tier['bindings']['qtip_runner']=dict(path=str(runner),sha256=sha(runner),bytes=runner.stat().st_size)
  mp=out/f'MANIFEST_E{e}.json';put(mp,manifest)
  cfg=bind_config(cfg,pin,str(runner),str(mp),sha(mp),grouped)
  cp=out/f'E{e}.json';put(cp,cfg);paths.append(cp)
 put(out/'IMPORT.json',dict(imported,runner_sha256=sha(runner),setup_seconds=time.perf_counter()-start,identity=identity()))
 phases=[]
 for phase in ('cold','warm'):
  before=gate();dest=out/phase;dest.mkdir();put(dest/'SHARDS.json',json.loads((ROOT/'SHARDS.json').read_text()))
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
  if grouped:rows=main_batch(paths,dest,9)
  else:
   rows=[]
   for cp in paths:rows.extend(main_batch([cp],dest,9))
  torch.cuda.synchronize();wall=time.perf_counter()-t
  assert len(rows)==3
  seals=[]
  for row,e in zip(rows,EXPERTS):
   p=artifact(arm,phase,e);rp=p.parent/'QTIP_SOLVE_RECEIPT.json'
   assert row['status']=='PASS' and row['rht_seed']==config_for(e)['rht_seed'] and row['artifact_sha256']==sha(p)
   seals.append(dict(expert=e,artifact_sha256=sha(p),receipt_sha256=sha(rp),config_sha256=sha(out/f'E{e}.json')))
  result=dict(phase=phase,wall=wall,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,mem_available_before=before,mem_available_after=gate(),seals=seals)
  put(out/(phase+'.json'),result);phases.append(result)
 put(out/'RESULT.json',dict(imported=imported,phases=phases,child_total_seconds=time.perf_counter()-start))

def validate(freeze):
 owned_child();start=time.perf_counter();torch,api,imported=setup_api(NEW)
 from banana_smasher.solver_qtip_profile import _load_weight
 rows=[];limits={};arms=('old1','old2') if freeze else ('singleton','batch')
 existing={} if freeze else json.loads((ROOT/'FROZEN_LIMITS.json').read_text())['limits']
 for e in EXPERTS:
  gate();w,_=_load_weight(Path(config_for(e)['model_root']),9,e,'down')
  reference=decode(torch,artifact('old1','warm',e))
  for arm in arms:
   for phase in ('cold','warm'):
    p=artifact(arm,phase,e);x=decode(torch,p);value=objective(x,w)
    assert math.isfinite(value)
    rows.append(dict(expert=e,arm=arm,phase=phase,nmse=value,artifact_sha256=sha(p),max_abs_vs_owner=float((x-reference).abs().max()),pass_numerical=None if freeze else value<=existing[str(e)]))
    del x
  if freeze:
   values=[r['nmse'] for r in rows if r['expert']==e and r['phase']=='warm'];spread=abs(values[0]-values[1]);assert spread<=max(values)*1e-4
   limits[e]=max(values)+max(5*spread,1e-6*max(values))
  del w,reference
 name='FROZEN_LIMITS.json' if freeze else 'VALIDATION.json'
 put(ROOT/name,dict(limits=limits if freeze else existing,rows=rows,all_pass=True if freeze else all(r['pass_numerical'] for r in rows),candidate_not_executed=freeze,validation_seconds=time.perf_counter()-start,imported=imported,heldout_replayed=False,production_adoption=False))

def controller(preimage):
 gate();ROOT.mkdir(exist_ok=False);ident=identity();put(ROOT/'IDENTITY.json',ident)
 with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);raw=CLAIM.read_bytes();old=json.loads(raw)
  assert hashlib.sha256(raw).hexdigest()==preimage
  assert old['task_id']==TASK and old['host']=='spark-6' and old['stage']=='t_ebcba52e_ds4_k3_sixcell_verify_run8474_terminal_retained'
  assert old['pid']==1107075 and old['start_ticks']==5550581 and not Path('/proc/1107075').exists() and not Path('/proc/1068698').exists()
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  claim=dict(old);claim.update(ident,stage=ROOT.name,mission=str(ROOT),expiry_unix=time.time()+7200,exact_cas_from_sha256=preimage);claim.pop('terminal_at',None)
  assert claim['owner']==TASK and claim['source_model_index_sha256']==BASIS
  allowed={'pid','pgid','start_ticks','argv','receipt','script_sha256','stage','mission','expiry_unix','exact_cas_from_sha256','terminal_at'}
  changed={k for k in old.keys()|claim.keys() if old.get(k)!=claim.get(k)};assert changed<=allowed
  put(ROOT/'CLAIM_PREIMAGE.json',old);put(ROOT/'CLAIM_DRY_RENDER.json',dict(preimage_sha256=preimage,candidate=claim,changed_keys=sorted(changed),status='PASS'))
  assert CLAIM.read_bytes()==raw;put(CLAIM,claim);assert json.loads(CLAIM.read_text())==claim
  put(ROOT/'CLAIM_POSTIMAGE.json',dict(claim=claim,sha256=sha(CLAIM)))
 try:
  shards=dict(intended_basis=BASIS,rows=[dict(owner=TASK,layer=9,experts=list(EXPERTS),projections=['down'],K=3,operation='historical_owner_compatibility_diagnostic_not_production',receipt=str(ROOT/'IDENTITY.json'))]);put(ROOT/'SHARDS.json',shards)
  put(ROOT/'SCIENCE.json',dict(scope='Historical owner compatibility diagnostic only; six existing cells never recounted. No forward.',old_pin=OLD,new_pin=NEW,order=['old1','old2','freeze','singleton','batch','validate'],cold='private process/compiler caches; OS/prebuilt assets shared',tolerance='per-cell max(owner warm NMSE)+max(5*repeat delta,1e-6*max NMSE); abort relative repeat spread>1e-4; frozen before proposed arms',production_adoption=False,mem_available_floor_bytes=36<<30))
  for arm in ('old1','old2','freeze','singleton','batch','validate'):
   gate();out=ROOT/arm;out.mkdir();env=dict(os.environ)
   for name,sub in [('BANANA_SMASHER_KERNEL_CACHE','canonical'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','cuda'),('XDG_CACHE_HOME','xdg')]:env[name]=str(out/'cache'/sub)
   argv=[sys.executable,'-u',__file__,'--child',arm];t=time.perf_counter()
   with (out/'child.log').open('w') as log:
    p=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
    put(ROOT/'PROGRESS.json',dict(arm=arm,pid=p.pid,start_ticks=int(Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19]),argv=argv))
    rc=p.wait()
   put(out/'PARENT.json',dict(returncode=rc,seconds=time.perf_counter()-t));assert rc==0,(arm,rc)
  put(ROOT/'RESULT.json',dict(state='SEALED_HISTORICAL_OWNER_BRIDGE',identity=ident,arms={a:json.loads((ROOT/a/'RESULT.json').read_text()) for a in ('old1','old2','singleton','batch')},validation=json.loads((ROOT/'VALIDATION.json').read_text()),production_adoption=False))
 except BaseException as e:
  put(ROOT/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
 finally:
  with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);c=json.loads(CLAIM.read_text());assert c['pid']==os.getpid() and c['task_id']==TASK
   c.update(stage=ROOT.name+'_terminal_retained',terminal_at=time.time());put(CLAIM,c);assert json.loads(CLAIM.read_text())==c
if __name__=='__main__':
 if sys.argv[1]=='--child':
  arm=sys.argv[2]
  if arm in ('freeze','validate'):validate(arm=='freeze')
  else:child(arm)
 else:controller(sys.argv[1])
