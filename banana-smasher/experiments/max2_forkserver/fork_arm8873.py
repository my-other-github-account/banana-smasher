import os,sys,json,time,hashlib,gc,ctypes
from pathlib import Path
sys.path.insert(0,'/dev/shm/t_5ade4a57')
from scope8568 import admit
sys.path.insert(0,'/run/t_5ade4a57')
import storage_k4tracesixteen8821speed as storage
storage.ALLOWANCES['/dev']=2238552455+(2056<<20)
storage.ALLOWANCES['/run'] += 4<<20 # run8843 recovery authority: bounded mechanics; original PRESTAGE/reserve retained
def storage_gate(out):
 return storage.evaluate(json.loads(Path('/run/t_5ade4a57/k4tracesixteen8821speed_PRESTAGE.json').read_text()),storage.sample())
ROOT=Path('/dev/shm/t_5ade4a57');OUT=Path('/dev/t_5ade4a57/fork8873');ASSETS=Path('/dev/t_5ade4a57/k4assets8776');BASE=Path('/dev/t_5ade4a57/selected_down8828/input')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.with_suffix('.tmp').open('w') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(p.with_suffix('.tmp'),p)
name,arm=sys.argv[1:];assert name in ['B1','C1','C2','B2','quality'];assert arm==('candidate' if name.startswith('C') else 'baseline')
claim=json.loads(Path('/home/dnola/HOST_CLAIM.json').read_text());assert claim['pid']==int(os.environ['SUPERVISOR_PID']) and Path('/proc/'+str(claim['pid'])).exists() and claim['owner']=='t_5ade4a57'
shards=json.loads((ROOT/'SHARDS.json').read_text());admit(shards,claim['owner'],claim['source_model_index_sha256'],[(40,47,'down'),(20,0,'fused13'),(26,96,'fused13'),(26,97,'fused13')])
cfg0=json.loads((BASE/'CONFIG.json').read_text());assert cfg0['geometry']==dict(L=16,K=4,V=2);assert sha(cfg0['input_identity']['model_index']['path'])==shards['intended_basis']
import torch
import resource
from bounded_cap8858 import enforce
enforced=enforce(resource,torch.cuda)
from banana_smasher import solver_qtip_profile as sp
from banana_smasher.qtip_batch_controller import main_batch
from headroom8630 import admit_headroom,host_available
trim=ctypes.CDLL('libc.so.6').malloc_trim;trim.argtypes=[ctypes.c_size_t];trim.restype=ctypes.c_int
torch.set_num_threads(8)
def memory_gate(label):
 gc.collect();trim(0);torch.cuda.empty_cache()
 # Conservative 12GiB bound for singleton K4 rather than transferred K1 peak evidence.
 witness=ASSETS/'owner_b10/solve/L040/batches/cb25e9b152355b7c/QTIP_CROSS_UNIT_BATCH_RECEIPT.json'
 assert sha(witness)=='776e80e8a18edb657bb63e164a42023387c1ec2ab39ce5e6554df8c70d26e1a6'
 sample=admit_headroom(host_available(),torch.cuda.mem_get_info()[0],json.loads(witness.read_text())['peak_cuda_reserved_bytes'])
 sample['scope']='2x supplied original same-work K4 peak reserved; unchanged host/device reserves'
 put(OUT/(label+'_MEMORY.json'),sample)
memory_gate(name);storage_gate(OUT)
if name=='quality':
 rows=[]
 for phase in [os.environ['QUALITY_PHASE']]:
  cfg=json.loads((OUT/'B2'/phase/'CONFIG.json').read_text());l=cfg['layer'];e=cfg['expert'];proj=cfg['projection']
  qv=sp._load_public_qtip_runner(Path(cfg['qtip_runner']),sha(cfg['qtip_runner']));qv.QTIP=Path(cfg['qtip_root']);bits,ldlq,math,kd=qv.load_official_qtip()
  captures=sp._load_captures(Path(cfg['fit_capture_root']),l,16);fit,_=sp._prepare_fit_windows(qv,captures,model_root=Path(cfg['model_root']),layer=l,expert=e,projection=proj,device=torch.device('cuda'));sp._release_capture_bank(Path(cfg['fit_capture_root']),l,16,captures)
  w,_=sp._load_weight(Path(cfg['model_root']),l,e,proj);rp=(ASSETS/'owner_b10/solve/L040/E047_down/QTIP_UNIT.pt' if l==40 else Path('/dev/t_5ade4a57/input_additions8825/owner_QTIP_UNIT.pt')) if l!=26 else Path('/dev/t_5ade4a57/stage8870/inputs/original/home/dnola/missions/CLEAN_t_bd7914e6/k1_batch_s7_run8682_y152/solve/L026')/f'E{e:03d}_fused13/QTIP_UNIT.pt'
  ref=qv.decode_packed_weight(torch.load(rp,map_location='cpu',weights_only=True),kd,torch.device('cuda')).half().cpu()
  for label in ['C2','B2']:
   p=OUT/label/phase/f'build/solve/L{l:03d}/E{e:03d}_{proj}/QTIP_UNIT.pt';x=qv.decode_packed_weight(torch.load(p,map_location='cpu',weights_only=True),kd,torch.device('cuda')).half().cpu();metrics=qv.split_metrics(fit,w,ref,x,torch.device('cuda'));assert torch.isfinite(x).all()
   rows.append(dict(arm=label,phase=phase,layer=l,expert=e,projection=proj,artifact_sha256=sha(p),reference_sha256=sha(rp),metrics=metrics,passed=metrics['qtip_hyb']['sse_ratio_vs_true_vq']<=1.0001+1e-12));put(OUT/('QUALITY_'+os.environ['QUALITY_PHASE']+'.json'),dict(rows=rows))
 assert len(rows)==2 and all(r['passed'] for r in rows);put(OUT/('GATE_'+os.environ['QUALITY_PHASE']+'.json'),dict(status='PASS',count=2,passed=2));raise SystemExit()

from banana_smasher.glm_qtip_source_adapter import capture_source_closure
runner=Path(sp.__file__).with_name('qtip_runner.py');qv=sp._load_public_qtip_runner(runner,sha(runner));qv.QTIP=Path(cfg0['qtip_root']);bits,ldlq,math,kd=qv.load_official_qtip();closure=capture_source_closure(qv,dict(bitshift=bits,ldlq=ldlq,math_utils=math,kernel_decompress=kd));cfg0['glm_source_closure_sha256']=closure['sha256'];assert closure['files']['qtip_batch_controller']['sha256']=='6cfd5b41bdf1f95dc0bddef6eecfc4e11e02fd2f1b66734fecb3f7cfe912ab6a';put(OUT/(name+'_IMPORT_CLOSURE.json'),closure)
def original_config(phase,D):
 if phase=='setup':return dict(cfg0)
 if phase=='warm':return json.loads(Path('/dev/t_5ade4a57/representative8825r1/CONFIG.json').read_text())
 assert phase in ['e096','e097'];e=int(phase[1:]);base=Path('/dev/t_5ade4a57/stage8870/inputs');orig=base/'original'
 source=orig/'home/dnola/missions/CLEAN_t_bd7914e6/k1_prepare_s7_run8682_y151'/f'L026_E{e:03d}_fused13_K1.json';cfg=json.loads(source.read_text())
 hm=json.loads((orig/str(cfg['hessian_layer_manifest']).lstrip('/')).read_text())
 def rebase(x):
  if isinstance(x,dict):return {k:rebase(v) for k,v in x.items()}
  if isinstance(x,list):return [rebase(v) for v in x]
  if isinstance(x,str) and x.startswith('/') and (orig/x.lstrip('/')).exists():return str(orig/x.lstrip('/'))
  return x
 hm=rebase(hm);put(D/'HESSIAN.json',hm);cfg['hessian_layer_manifest']=str(D/'HESSIAN.json');cfg['hessian_layer_manifest_sha256']=sha(D/'HESSIAN.json')
 cfg['selected_source_manifest_sha256']=sha(Path('/dev/t_5ade4a57/four_r3_8870/model/SELECTED_TENSORS.json'));cfg['fit_capture_root']=str(orig/cfg['fit_capture_root'].lstrip('/'));cfg['model_root']=str(Path('/dev/t_5ade4a57/four_r3_8870/model'));cfg['input_identity']['model_index']['path']=str(Path('/dev/t_5ade4a57/four_r3_8870/model/model.safetensors.index.json'))
 cfg['reference_unit']=str(orig/cfg['reference_unit'].lstrip('/'));cfg['tlut_source']=str(orig/cfg['tlut_source'].lstrip('/'));cfg['qtip_root']=cfg0['qtip_root']
 cfg['materialization']['run_manifest']=str(orig/cfg['materialization']['run_manifest'].lstrip('/'))
 return cfg
rows=[]
for phase in os.environ.get('RESIDENT_PHASES','setup,warm,e096,e097').split(','):

 storage_gate(OUT);memory_gate(name+'_'+phase);D=OUT/name/phase;D.mkdir(parents=True);cfg=original_config(phase,D);assert sha(cfg['input_identity']['model_index']['path'])==shards['intended_basis'];cfg['glm_source_closure_sha256']=closure['sha256'];cfg['viterbi_bounded_overlap']=False;cfg['packed_decode_execution']='eager';cfg['source_hash_prefetch']=False;cfg['selected_index_cache']=False;cfg['qtip_runner']=str(Path(sp.__file__).with_name('qtip_runner.py'));manifest=json.loads(Path(cfg['materialization']['run_manifest']).read_text());manifest['canonical_commit']='aee895c2abeda737081501a197c493794391b131';manifest['tiers'][0]['bindings']['qtip_runner']=dict(path=cfg['qtip_runner'],sha256=sha(cfg['qtip_runner']));put(D/'MANIFEST.json',manifest);cfg['materialization']=dict(cfg['materialization'],run_manifest=str(D/'MANIFEST.json'),run_manifest_sha256=sha(D/'MANIFEST.json'));put(D/'CONFIG.json',cfg);dest=D/'build';dest.mkdir();put(dest/'SHARDS.json',shards)
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
 from pinned_cpu_scope import pinned_cpu_scope
 with pinned_cpu_scope(torch,True) as transfers:
  put(D/'RESIDENCY_REACH.json',dict(pid=os.getpid(),phase=phase,arm=arm))
  put(D/'BEFORE_MEMORY.json',dict(proc=Path('/proc/self/status').read_text(),allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),free=torch.cuda.mem_get_info()[0]))
  try:
   result=main_batch([D/'CONFIG.json'],dest,cfg['layer'])
  finally:
   put(D/'AFTER_MEMORY.json',dict(proc=Path('/proc/self/status').read_text(),allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),free=torch.cuda.mem_get_info()[0]))
 put(D/'PINNED_REACH.json',dict(events=transfers))
 torch.cuda.synchronize();elapsed=time.perf_counter()-t

 rows.append(dict(phase=phase,seconds=elapsed,result=result,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()));put(OUT/name/(phase+'_RESULT.json'),dict(pin='aee895c2abeda737081501a197c493794391b131',arm=arm,rows=rows,scope='Original aee895c2 synchronous pinned DtoH vs default; adopted eager decoder fixed; no production-rate inference'))
put(OUT/(name+'_WORKING_SET.json'),dict(enforced=enforced,ru_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,proc_status=Path('/proc/self/status').read_text(),cuda_peak_reserved=torch.cuda.max_memory_reserved()))
