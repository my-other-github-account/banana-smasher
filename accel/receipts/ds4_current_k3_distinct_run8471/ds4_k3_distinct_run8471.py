"""Bounded authentic current-K3 producer schedule gate; not heldout equivalence."""
import os,sys,json,time,hashlib,fcntl,subprocess,traceback
from pathlib import Path
TASK='t_ebcba52e';ROOT=Path('/dev/shm/t_ebcba52e_ds4_k3_distinct_run8471');CLAIM=Path('/home/dnola/HOST_CLAIM.json');INPUT=Path('/dev/shm/t_ebcba52e_ds4_k3_input_run8465');STAGE=INPUT/'source';PIN='eec3bfc578c2fac4678373331f779de4e2abb766';API=Path('/dev/shm/t_ebcba52e_api_eec3bfc5/banana-smasher/src');BASIS='98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b';MODEL=Path('/home/dnola/models/hf/DeepSeek-V4-Flash-0731')
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
 assert hashlib.sha256(raw).hexdigest()==sys.argv[1] and c['task_id']==TASK and c['pid']==1038054 and c['stage']=='t_ebcba52e_ds4_k3_distinct_input_run8471_terminal_retained'
 assert not Path('/proc/1038054').exists() and (INPUT/'RESULT.json').exists()
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();gate()
 put(ROOT/'CLAIM_PREIMAGE.json',c);shards=dict(intended_basis=BASIS,rows=[dict(owner=TASK,layer=9,expert=e,projection='down',K=3,operation='distinct_current_K3_singleton_grouped',receipt=str(ROOT/'IDENTITY.json')) for e in [90,91,92]]);put(ROOT/'SHARDS.json',shards)
 claim=dict(identity,owner=TASK,state='CLAIMED',status='CLAIMED',host='spark-6',stage=ROOT.name,mission=str(ROOT),expiry_unix=time.time()+7200,source_model_index_sha256=BASIS);assert CLAIM.read_bytes()==raw;put(CLAIM,claim)
try:
 import copy
 OLD=Path('/dev/shm/t_ebcba52e_ds4_k3_benchmark_run8465'); NEW=Path('/dev/shm/t_ebcba52e_ds4_k3_distinct_input_run8471')
 specpath=Path('/dev/shm/BS01_CURRENT_K3_E090_E092_BINDINGS_run8467.json');assert sha(specpath)=='607cb23306f2106fe8a0f2bc560b86095b539c39f8b1fb968538ddb3f26c632e'
 spec=json.loads(specpath.read_text()); cfg0=json.loads((OLD/'BASE_CONFIG.json').read_text()); original0=json.loads((STAGE/json.loads(Path('/dev/shm/BS01_CURRENT_K3_CELL_BINDING_run8467.json').read_text())['representative_cell']['config']['path'].lstrip('/')).read_text())
 sys.path.insert(0,str(API));import torch,banana_smasher
 from banana_smasher.qtip_batch_controller import main_batch
 from banana_smasher.solver_qtip_profile import _load_weight
 from banana_smasher.sealed_qtip_unit import decode_sealed_unit
 assert Path(banana_smasher.__file__).is_relative_to(API)
 torch.set_num_threads(8);runner=Path(banana_smasher.__file__).parent/'qtip_runner.py';configs=[]
 for row in spec['rows']:
  src=NEW/'source'/row['config']['path'].lstrip('/');assert sha(src)==row['config']['sha256'];orig=json.loads(src.read_text())
  assert {k:v for k,v in orig.items() if k not in ['expert','materialization','rht_seed']}=={k:v for k,v in original0.items() if k not in ['expert','materialization','rht_seed']}
  assert orig['materialization']['run_manifest_sha256']==original0['materialization']['run_manifest_sha256']
  cfg=copy.deepcopy(cfg0);cfg['expert']=orig['expert'];cfg['rht_seed']=orig['rht_seed'];cfg['materialization']['source_config_sha256']=orig['materialization']['source_config_sha256']
  m=json.loads(Path(cfg['materialization']['run_manifest']).read_text());m['canonical_git_pin']=PIN;m['tiers'][0]['bindings']['qtip_runner']=dict(path=str(runner),sha256=sha(runner),bytes=runner.stat().st_size)
  mp=ROOT/('MANIFEST_E'+str(cfg['expert'])+'.json');put(mp,m);cfg['materialization']['run_manifest']=str(mp);cfg['materialization']['run_manifest_sha256']=sha(mp);cfg['qtip_runner']=str(runner);cfg['exact_solver']='banana_smasher.qtip_viterbi@'+PIN;configs.append(cfg)
 put(ROOT/'SCIENCE.json',dict(basis=BASIS,owner_binding_sha256=sha(specpath),canonical_pin=PIN,cells=[r['cell'] for r in spec['rows']],candidate='main_batch three distinct configs, all default schedule knobs unchanged',baseline='main_batch singleton serial three distinct configs; two new baseline repeats only',timing='shared-cache setup and two warm repeats, not matched process-cold',quality_rule='per-cell NMSE <= max(baseline repeats)+max(5*spread,1e-6*max); frozen before candidates',heldout_kld=None,production_adoption=False))
 def decoded(path,root):
  unit=torch.load(path,map_location='cpu',mmap=True,weights_only=True);m,n=unit['shape'];row=dict(shape=[m,n],source_transform=dict(output_quantity='descaled_weight'),wire=dict(unit=dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=sha(path)),geometry=unit['geometry'],unit_shape=unit['shape'],row_range=[0,m]));return decode_sealed_unit(root,row).float()
 weights={e:_load_weight(MODEL,9,e,'down')[0] for e in [90,91,92]}
 def nmse(x,e):
  w=weights[e];num=den=0.
  for i in range(0,w.shape[0],128):
   ref=w[i:i+128].double();delta=x[i:i+128].double()-ref;num+=float((delta*delta).sum());den+=float((ref*ref).sum())
  return num/den
 owners={}
 for r in spec['rows']:
  path=NEW/'source'/r['artifact']['path'].lstrip('/');assert sha(path)==r['artifact']['sha256'];x=decoded(path,NEW);owners[r['cell'][1]]=nmse(x,r['cell'][1]);del x
 put(ROOT/'OWNER_DECODE.json',owners);rows=[];limits={}
 for arm in ['baseline_setup','baseline_warm1','baseline_warm2','candidate_setup','candidate_warm1','candidate_warm2']:
  available=gate();out=ROOT/arm;out.mkdir();put(out/'SHARDS.json',shards);paths=[]
  for cfg in configs:
   cp=out/('E'+str(cfg['expert'])+'.json');put(cp,cfg);paths.append(cp)
  put(ROOT/'PROGRESS.json',dict(stage='BUILD',arm=arm));torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
  results=[]
  if arm.startswith('baseline'):
   for cp in paths:results.extend(main_batch([cp],out,9))
  else:results=main_batch(paths,out,9)
  torch.cuda.synchronize();wall=time.perf_counter()-t;peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved();vt=time.perf_counter();cells=[]
  for cfg in configs:
   e=cfg['expert'];p=out/('solve/L009/E%03d_down/QTIP_UNIT.pt'%e);x=decoded(p,ROOT);v=nmse(x,e);cells.append(dict(expert=e,nmse=v,decode_pass=True,artifact_sha256=sha(p),reconstruction_pass=v<=limits[e] if limits else None));del x
  row=dict(arm=arm,wall_seconds=wall,peak_allocated=peak,peak_reserved=reserved,mem_available=available,cells=cells,validation_seconds=time.perf_counter()-vt,results=results);rows.append(row);put(ROOT/(arm+'.json'),row)
  if arm=='baseline_warm2':
   for e in weights:
    values=[next(c['nmse'] for c in r['cells'] if c['expert']==e) for r in rows if r['arm'].startswith('baseline_warm')];limits[e]=max(values)+max(5*abs(values[0]-values[1]),1e-6*max(values))
   put(ROOT/'FROZEN_LIMITS.json',dict(limits=limits,candidate_not_executed=True))
 b=sum(r['wall_seconds'] for r in rows if r['arm'].startswith('baseline_warm'));c=sum(r['wall_seconds'] for r in rows if r['arm'].startswith('candidate_warm'))
 put(ROOT/'RESULT.json',dict(state='SEALED_DISTINCT_CURRENT_K3_BATCHING',identity=identity,rows=rows,warm_speedup=b/c,reconstruction_pass=all(c['reconstruction_pass'] for r in rows if r['arm'].startswith('candidate') for c in r['cells']),limits=limits,heldout_kld=None,production_adoption=False,scope='three distinct current-K3 L009 down cells; shared-cache warm only; no fused/fullmodel/adoption'))
except BaseException as e:put(ROOT/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
finally:
 with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);c=json.loads(CLAIM.read_text());assert c['pid']==os.getpid();c.update(stage=ROOT.name+'_terminal_retained',terminal_at=time.time());put(CLAIM,c)
