import os,sys,json,time,hashlib,subprocess,fcntl,traceback
from pathlib import Path
ROOT=Path('/dev/shm/t_1269dc5f');OUT=ROOT/'ds4_distinct_quality_run8528';OLD=Path('/dev/shm/t_182fbc9d');CLAIM=Path('/home/dnola/HOST_CLAIM.json');BASE='6a40f916a6def348cb6a65bcd648006ab3242211';CAND='7391e41b2e7dad4cd0ebeda7dc6e95e05cae92e0'
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
  prior=Path(old['mission']);assert prior==ROOT/'ds4_distinct_run8528' and json.loads((prior/'TERMINAL.json').read_text())['returncode']==0;assert dead(json.loads((prior/'CHILD.json').read_text())['pid'])
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  basis='98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b';assert sha('/home/dnola/models/hf/DeepSeek-V4-Flash-0731/model.safetensors.index.json')==basis
  with (ROOT/'SHARDS.lock').open('a+') as sl:
   fcntl.flock(sl,fcntl.LOCK_EX);raws=(ROOT/'SHARDS.json').read_bytes();shards=json.loads(raws);assert shards['intended_basis']==old['source_model_index_sha256'] and all(r['owner']=='t_1269dc5f' for r in shards['rows']);put(OUT/'SHARDS_PREIMAGE.json',shards);shards['intended_basis']=basis;shards['rows']=[dict(owner='t_1269dc5f',layer=26,experts=[78,79,80],projections=['fused13'],operation='resident_process_amortization_not_production')];assert (ROOT/'SHARDS.json').read_bytes()==raws;put(ROOT/'SHARDS.json',shards)
  c=dict(old);c.update(**identity(os.getpid()),mission=str(OUT),receipt=str(OUT/'IDENTITY.json'),stage='ds4_quality',source_model_index_sha256=basis,expiry_unix=time.time()+7200,exact_cas_from_sha256=hashlib.sha256(raw).hexdigest());c.pop('terminal_at',None);put(OUT/'PREIMAGE.json',old);assert CLAIM.read_bytes()==raw;put(CLAIM,c);assert json.loads(CLAIM.read_text())==c;put(OUT/'IDENTITY.json',c)
 try:
  with (OUT/'QUALITY.log').open('w') as log:
   p=subprocess.Popen([sys.executable,'-u',__file__,'--child'],stdout=log,stderr=subprocess.STDOUT);put(OUT/'CHILD.json',identity(p.pid));rc=p.wait()
  put(OUT/'TERMINAL.json',dict(returncode=rc));assert rc==0
 except BaseException:
  put(OUT/'FAILURE.json',dict(error=traceback.format_exc()));raise
 finally:
  with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);c=json.loads(CLAIM.read_text());assert c['pid']==os.getpid();c.update(stage='ds4_quality_terminal_retained',terminal_at=time.time());put(CLAIM,c)
 raise SystemExit()
assert json.loads(CLAIM.read_text())['pid']==os.getppid()
sys.path.insert(0,str(OLD/'main_run8524/banana-smasher/src'))
import torch,math
from banana_smasher.solver_qtip_profile import _load_weight
from banana_smasher.sealed_qtip_unit import decode_sealed_unit
torch.set_num_threads(8)
mem=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'));assert mem>(12+4)<<30
R=ROOT/'ds4_distinct_run8528';B=Path('/dev/shm/t_ebcba52e_ds4_k3_fused_run8474');MODEL=Path('/home/dnola/models/hf/DeepSeek-V4-Flash-0731')
assert sha(MODEL/'model.safetensors.index.json')==json.loads((ROOT/'SHARDS.json').read_text())['intended_basis']
limits={int(k):v for k,v in json.loads((B/'FROZEN_LIMITS.json').read_text())['limits'].items()}
put(OUT/'SCIENCE.json',dict(limits_sha256=sha(B/'FROZEN_LIMITS.json'),prior_limits_path=str(B/'FROZEN_LIMITS.json'),basis=sha(MODEL/'model.safetensors.index.json'),pin=BASE,quality_scope='independent canonical decoded weight and original-source NMSE; frozen prior baseline limits, not new heldout'))
def decode(root,rel,e):
 p=root/rel/f'solve/L026/E{e:03d}_fused13/QTIP_UNIT.pt';unit=torch.load(p,map_location='cpu',mmap=True,weights_only=True);m,n=unit['shape'];assert {k:unit['geometry'][k] for k in ['K','L','V']}==dict(K=3,L=16,V=2)
 wire=dict(shape=[m,n],source_transform=dict(output_quantity='descaled_weight'),wire=dict(unit=dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)),geometry=unit['geometry'],unit_shape=unit['shape'],row_range=[0,m]))
 return decode_sealed_unit(root,wire).float(),sha(p)
def objective(x,w):
 num=[];den=[]
 for i in range(0,w.shape[0],64):
  ref=w[i:i+64].double();d=x[i:i+64].double()-ref;num.append(float(d.square().sum()));den.append(float(ref.square().sum()))
 return math.fsum(num)/math.fsum(den)
rows=[];t=time.perf_counter()
for e in [78,79,80]:
 w,_=_load_weight(MODEL,26,e,'fused13');ref,digest=decode(B,'candidate_warm1',e)
 for arm in ['B1','C1','C2','B2']:
  for phase in (['0'] if e==78 else ['1']):
   x,h=decode(R,arm+'/'+phase,e);nmse=objective(x,w);assert torch.isfinite(x).all();assert nmse<=limits[e];rows.append(dict(arm=arm,phase=phase,expert=e,artifact_sha256=h,reference_sha256=digest,nmse=nmse,limit=limits[e],max_abs=float((x-ref).abs().max()),decoded_equal=torch.equal(x,ref),pass_numerical=True));put(OUT/'RESULT.json',dict(rows=rows,validation_seconds=time.perf_counter()-t,scope='original-source NMSE and canonical decode, not new heldout'));del x
 del w,ref
assert len(rows)==12
