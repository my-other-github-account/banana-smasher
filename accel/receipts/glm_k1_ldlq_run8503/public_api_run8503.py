import os,sys,json,time,hashlib,importlib.util,subprocess,statistics
from pathlib import Path
BASE=Path('/dev/shm/t_ebcba52e_ldlq_run8503');OUT=Path(os.environ['OUT'])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):Path(p).write_text(json.dumps(x,indent=2))
def setup():
 import torch
 torch.set_num_threads(8)
 from banana_smasher import solver_qtip_profile as sp,qtip_viterbi as new
 m=json.loads((BASE/'closure/MANIFEST.json').read_text())
 def locate(s):
  rows=[r for r in m['files'] if r['archive_path'].endswith(s)];assert len(rows)==1,(s,len(rows));p=BASE/'closure'/rows[0]['archive_path'];assert sha(p)==rows[0]['sha256'];return p
 runner=Path(sp.__file__).with_name('qtip_runner.py');qv=sp._load_public_qtip_runner(runner,sha(runner));qv.QTIP=locate('lib/algo/ldlq.py').parents[2]
 bits,ldlq,math,kd=qv.load_official_qtip()
 cfg=json.loads(locate('/L003_E145_down_K1.json').read_text());lut=sp._load_tlut(locate('P641_QTIP_TLUT_SOURCE.pt'))
 captures=sp._load_captures(locate('xmoe_L003_win0000.pt').parent,3,16)
 model=BASE/'selected_run8503';fit,fit_source=sp._prepare_fit_windows(qv,captures,model_root=model,layer=3,expert=145,projection='down',device=torch.device('cuda'))
 w,source=sp._load_weight(model,3,145,'down')
 assert list(w.shape)==[4096,2048] and source['index_sha256']==m['intended_basis']
 return torch,sp,new,qv,bits,ldlq,math,kd,cfg,lut,fit,w,source,locate

torch,sp,new,qv,bits,ldlq,math,kd,cfg,lut,fit,w,source,locate=setup()
from banana_smasher.glm_qtip_source_adapter import capture_source_closure
closure=capture_source_closure(qv,dict(bitshift=bits,ldlq=ldlq,math_utils=math,kernel_decompress=kd));put(OUT/'IMPORT_CLOSURE.json',closure)
manifest=json.loads(locate('/K1_L003_E145_down_MANIFEST.json').read_text());runner=Path(qv.__file__)
manifest['canonical_commit']='eb3b97314416440bd835fb8327d60b52b0826bdb'
manifest['tiers'][0]['bindings']['qtip_runner']=dict(path=str(runner),sha256=sha(runner))
mp=OUT/'RUN_MANIFEST.json';put(mp,manifest)
hp=locate('/L003_HESSIAN_MANIFEST.json');h=json.loads(hp.read_text())
def relocate(x):
 if isinstance(x,dict):return {k:relocate(v) for k,v in x.items()}
 if isinstance(x,list):return [relocate(v) for v in x]
 if isinstance(x,str) and x.startswith('/'):
  q=BASE/'closure/files'/x.lstrip('/')
  if q.exists():return str(q)
 return x
h=relocate(h);newhp=OUT/'HESSIAN_MANIFEST.json';put(newhp,h)
cfg=relocate(cfg);cfg.update(model_root=str(BASE/'selected_run8503'),qtip_runner=str(runner),qtip_root=str(qv.QTIP),viterbi_num_warps=8,viterbi_branch_unroll=True,glm_source_closure_sha256=closure['sha256'],selected_source_manifest_sha256=sha(BASE/'selected_run8503/SELECTED_TENSORS.json'),hessian_layer_manifest=str(newhp),hessian_layer_manifest_sha256=sha(newhp))
cfg['input_identity']['model_index']['path']=str(BASE/'selected_run8503/model.safetensors.index.json')
cfg['materialization'].update(run_manifest=str(mp),run_manifest_sha256=sha(mp))
cp=OUT/'CONFIG.json';put(cp,cfg)
torch.cuda.synchronize();t=time.perf_counter()
from banana_smasher.qtip_batch_controller import main_batch
rows=main_batch([cp],OUT,3)
torch.cuda.synchronize();put(OUT/'API_RESULT.json',dict(rows=rows,main_batch_seconds=time.perf_counter()-t,config_sha256=sha(cp),scope='diagnostic single-cell public API, not production adoption'))
