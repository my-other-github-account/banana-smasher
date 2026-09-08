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

def child(arm):
 started=time.perf_counter();torch,sp,new,qv,bits,ldlq,math,kd,cfg,lut,fit,w,source,locate=setup()
 from banana_smasher.qtip_matrix_lifetime import build_qtip_bounded
 old=None
 if arm.startswith('old'):
  for name,suffix in [('banana_smasher.qtip_memory','/qtip_memory_run8248.py'),('banana_smasher.owner_viterbi','/qtip_viterbi_run8248.py')]:
   spec=importlib.util.spec_from_file_location(name,locate(suffix));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);sys.modules[name]=mod
  old=mod
 put(OUT/f'{arm}_INPUT.json',dict(source=source,config_sha256=sha(locate('/L003_E145_down_K1.json')),fit_rows=sum(len(x['x']) for x in fit),fit_mass=sum(float(x['weight'].sum()) for x in fit),rht_seed=cfg['rht_seed'],runner_sha256=sha(Path(qv.__file__)),viterbi_sha256=sha(Path((old or new).__file__)),scope='canonical bounded builder, only Viterbi implementation/schedule differs'))
 rows=[]
 for phase in ('cold','warm'):
  torch.cuda.synchronize();t=time.perf_counter()
  cb=bits.bitshift_codebook(L=16,K=1,V=2,tlut_bits=9,decode_mode='quantlut_sym',tlut=lut.cuda()).cuda()
  timers=sp._ExactTimers()
  if old:solver=sp._install_profiled_exact_viterbi(cb,old,timers,profile_mode=False)
  else:
   cfg.update(viterbi_num_warps=8,viterbi_branch_unroll=True)
   solver=sp._install_configured_viterbi(cb,new,timers,cfg,profile_mode=False)
  sp._bind_public_runner_pack_contract(cb,cfg,w);sp._bind_builder_memory_contract(cb,w)
  torch.cuda.reset_peak_memory_stats()
  candidate,build=build_qtip_bounded(qv,w,fit,cb,ldlq,math,kd,torch.device('cuda'),cfg['rht_seed'])
  torch.cuda.synchronize();build_wall=time.perf_counter()-t
  sp._verify_builder_memory_contract(cb)
  artifact=OUT/f'{arm}_{phase}.pt';torch.save(candidate,artifact)
  row=dict(arm=arm,phase=phase,build_wall=build_wall,whole_iteration_seconds=time.perf_counter()-t,build=build,solver=solver,calls=timers.calls,sequences=timers.sequences,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),artifact_sha256=sha(artifact),bytes=artifact.stat().st_size)
  rows.append(row);print(json.dumps(row),flush=True);put(OUT/f'{arm}_RESULT.json',dict(rows=rows,child_whole_seconds=time.perf_counter()-started));del candidate,cb;torch.cuda.empty_cache()

def validate():
 torch,sp,new,qv,bits,ldlq,math,kd,cfg,lut,fit,w,source,locate=setup();rows=[]
 ref=torch.load(OUT/'old1_warm.pt',map_location='cpu',weights_only=True)['reconstructed_weight']
 for arm in ('old1','new1','new2','old2'):
  for phase in ('cold','warm'):
   u=torch.load(OUT/f'{arm}_{phase}.pt',map_location='cpu',weights_only=True);x=u['reconstructed_weight'];metric=qv.split_metrics(fit,w,ref,x,torch.device('cuda'))
   nmse=float((x.double()-w.double()).square().sum()/w.double().square().sum())
   row=dict(arm=arm,phase=phase,nmse=nmse,max_abs_vs_baseline=float((x.float()-ref.float()).abs().max()),same_decoded=torch.equal(x,ref),metrics=metric);rows.append(row);print(json.dumps(row),flush=True)
 put(OUT/'QUALITY.json',dict(rows=rows,tolerance='candidate NMSE and clean-fit output weighted NMSE <= maximum baseline *1.0001 +1e-12; established before candidate',scope='authentic clean-fit 16-window diagnostic output, not frozen evaluation or full-model quality',frozen_evaluation_touched=False))

torch,sp,new,qv,bits,ldlq,math,kd,cfg,lut,fit,w,source,locate=setup()
assert sha(BASE/'OWNER_ORIGINAL_run8503.pt')=='a202fc9d67cd75e7db8d1f5db95d2de502ef0e67f9bf0a69c6ddd1a6e753ee0a'
u=torch.load(BASE/'OWNER_ORIGINAL_run8503.pt',map_location='cpu',weights_only=True)
ref=qv.decode_packed_weight(u,kd,torch.device('cuda')).half().cpu()
rows=[]
for arm in ('old1','new1','new2','old2'):
 for phase in ('cold','warm'):
  f=BASE/'whole'/f'{arm}_{phase}.pt';v=torch.load(f,map_location='cpu',weights_only=True);x=v['reconstructed_weight']
  row=dict(arm=arm,phase=phase,artifact_sha256=sha(f),original_artifact_sha256=sha(BASE/'OWNER_ORIGINAL_run8503.pt'),max_abs_vs_original=float((x.float()-ref.float()).abs().max()),same_decoded=torch.equal(x,ref),metrics=qv.split_metrics(fit,w,ref,x,torch.device('cuda')),nmse=float((x.double()-w.double()).square().sum()/w.double().square().sum()),original_nmse=float((ref.double()-w.double()).square().sum()/w.double().square().sum()))
  rows.append(row);print(json.dumps(row),flush=True)
put(OUT/'ORIGINAL_QUALITY.json',dict(rows=rows,scope='original owner artifact vs diagnostic arms on authentic original clean-fit windows; not heldout or full-model evaluation',frozen_evaluation_touched=False))
