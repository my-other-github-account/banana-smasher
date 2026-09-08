"""Bounded authentic current-K3 producer schedule gate; not heldout equivalence."""
import os,sys,json,time,hashlib,fcntl,subprocess,traceback
from pathlib import Path
TASK='t_ebcba52e';ROOT=Path('/dev/shm/t_ebcba52e_ds4_k3_verify_run8471_a2');CLAIM=Path('/home/dnola/HOST_CLAIM.json');INPUT=Path('/dev/shm/t_ebcba52e_ds4_k3_input_run8465');STAGE=INPUT/'source';PIN='eec3bfc578c2fac4678373331f779de4e2abb766';API=Path('/dev/shm/t_ebcba52e_api_eec3bfc5/banana-smasher/src');BASIS='98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b';MODEL=Path('/home/dnola/models/hf/DeepSeek-V4-Flash-0731')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def put(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp')
 with t.open('w') as f:json.dump(d,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(t,p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def gate():
 mem=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'));v=os.statvfs(ROOT.parent)
 assert mem>(32<<30)+(4<<30) and v.f_bavail*v.f_frsize>(1<<30)+(4<<30)
 assert sha(MODEL/'model.safetensors.index.json')==BASIS
 return mem
ROOT.mkdir(exist_ok=False);identity=dict(task_id=TASK,pid=os.getpid(),pgid=os.getpgid(0),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),argv=sys.argv,receipt=str(ROOT/'IDENTITY.json'),script_sha256=sha(__file__),canonical_git_pin=PIN);put(ROOT/'IDENTITY.json',identity)
with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);raw=CLAIM.read_bytes();c=json.loads(raw)
 assert hashlib.sha256(raw).hexdigest()==sys.argv[1] and c['task_id']==TASK and c['pid']==1046502 and c['stage']=='t_ebcba52e_ds4_k3_verify_run8471_terminal_retained'
 assert not Path('/proc/1046502').exists() and (INPUT/'RESULT.json').exists()
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();gate()
 put(ROOT/'CLAIM_PREIMAGE.json',c);shards=dict(intended_basis=BASIS,rows=[dict(owner=TASK,layer=9,expert=e,projection='down',K=3,operation='distinct_current_K3_singleton_grouped',receipt=str(ROOT/'IDENTITY.json')) for e in [90,91,92]]);put(ROOT/'SHARDS.json',shards)
 claim=dict(identity,owner=TASK,state='CLAIMED',status='CLAIMED',host='spark-6',stage=ROOT.name,mission=str(ROOT),expiry_unix=time.time()+7200,source_model_index_sha256=BASIS);assert CLAIM.read_bytes()==raw;put(CLAIM,claim)
try:
 put(ROOT/'RECOVERY.json',dict(predecessor_failure_sha256=sha('/dev/shm/t_ebcba52e_ds4_k3_verify_run8471/FAILURE.json'),kind='harness geometry includes additional wire fields; preserve required K/L/V assertion',forward_replayed=False,build_replayed=False))
 sys.path.insert(0,str(API));import torch,banana_smasher,math
 from banana_smasher.solver_qtip_profile import _load_weight
 from banana_smasher.sealed_qtip_unit import decode_sealed_unit
 torch.set_num_threads(8);assert Path(banana_smasher.__file__).is_relative_to(API)
 B=Path('/dev/shm/t_ebcba52e_ds4_k3_distinct_run8471');C=Path('/dev/shm/t_ebcba52e_ds4_k3_cold_run8471');U=Path('/dev/shm/t_ebcba52e_ds4_k3_unitwise_run8471')
 assert (C/'RESULT.json').exists();cold=json.loads((C/'RESULT.json').read_text());limits={int(k):v for k,v in json.loads((B/'FROZEN_LIMITS.json').read_text())['limits'].items()}
 put(ROOT/'SCIENCE.json',dict(baseline_sha256=sha(B/'RESULT.json'),cold_sha256=sha(C/'RESULT.json'),unitwise_sha256=sha(U/'RESULT.json'),limits_sha256=sha(B/'FROZEN_LIMITS.json'),basis=BASIS,independent_decode=True,heldout_kld=None))
 def decode(root,rel,e):
  p=root/rel/('solve/L009/E%03d_down/QTIP_UNIT.pt'%e);unit=torch.load(p,map_location='cpu',mmap=True,weights_only=True);m,n=unit['shape'];assert {k:unit['geometry'][k] for k in ['K','L','V']}==dict(K=3,L=16,V=2)
  wire=dict(shape=[m,n],source_transform=dict(output_quantity='descaled_weight'),wire=dict(unit=dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)),geometry=unit['geometry'],unit_shape=unit['shape'],row_range=[0,m]))
  return decode_sealed_unit(root,wire).float(),sha(p)
 def objective(x,w):
  numer=[];denom=[]
  for i in range(0,w.shape[0],64):
   ref=w[i:i+64].double();diff=x[i:i+64].double()-ref;numer.append(float(diff.square().sum()));denom.append(float(ref.square().sum()))
  return math.fsum(numer)/math.fsum(denom)
 rows=[];started=time.perf_counter()
 for e in [90,91,92]:
  gate();w,_=_load_weight(MODEL,9,e,'down');ref,_=decode(B,'baseline_warm2',e)
  for root,rel in [(U,a) for a in ['candidate_setup','candidate_warm1','candidate_warm2']]+[(C,a+'/'+phase) for a in ['baseline1','candidate1','candidate2','baseline2'] for phase in ['cold','warm']]:
   x,digest=decode(root,rel,e);nmse=objective(x,w);row=dict(root=str(root),arm=rel,expert=e,artifact_sha256=digest,nmse=nmse,limit=limits[e],pass_numerical=nmse<=limits[e],max_abs_vs_singleton=float((x-ref).abs().max()),decode_pass=True);rows.append(row);del x;put(ROOT/'PROGRESS.json',dict(rows=rows,elapsed=time.perf_counter()-started))
  del w,ref
 speed={phase:sum(next(x['wall'] for x in cold['arms'][a]['phases'] if x['phase']==phase) for a in ['baseline1','baseline2'])/sum(next(x['wall'] for x in cold['arms'][a]['phases'] if x['phase']==phase) for a in ['candidate1','candidate2']) for phase in ['cold','warm']}
 put(ROOT/'RESULT.json',dict(state='SEALED_INDEPENDENT_NUMERICAL_COLD_GATE',identity=identity,rows=rows,count=len(rows),all_pass=all(r['pass_numerical'] for r in rows),speedups=speed,validation_wall=time.perf_counter()-started,heldout_kld=None,production_adoption=False,quality_scope='three down cells independent numerical/decode; no fused or currentK3 heldout frontier'))
except BaseException as e:put(ROOT/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
finally:
 with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);c=json.loads(CLAIM.read_text());assert c['pid']==os.getpid();c.update(stage=ROOT.name+'_terminal_retained',terminal_at=time.time());put(CLAIM,c)
