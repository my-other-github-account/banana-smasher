"""Bounded native L003-input sparse panel paired continuation, no prefix replay."""
import os,sys,json,time,hashlib,subprocess,gc,shutil
from pathlib import Path
R=Path('/dev/shm/t_182fbc9d');O=Path(os.environ['OUT']);I=R/'heldout_input_run8520';S=O/'source';S.mkdir()
sys.path[:0]=[str(R/'hf_runtime'),str(I),str(R/'candidate_lutl1/banana-smasher/src')]
BASIS='3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05'
REMOTE='dnola@192.168.200.3:/home/dnola/missions/GLM_Q2_SCALE_REPAIR_t_44e9a9a7_s3/source_glm/'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def put(p,x):
 p=Path(p);q=p.with_suffix('.tmp')
 with q.open('w') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def memory():return next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
claim=json.loads(Path('/home/dnola/HOST_CLAIM.json').read_text());assert claim['pid']==os.getppid() and claim['source_model_index_sha256']==BASIS
for x in json.loads((I/'ACK.json').read_text())['files']:assert sha(I/x['file'])==x['sha256']
side=json.loads((I/'L003_000.json').read_text());identity=json.loads((I/'IDENTITY.json').read_text());assert side['binding']==sha(I/'IDENTITY.json') and side['sha256']==sha(I/'L003_000.payload') and side['stage']=='L003' and side['slot']==0;assert identity['intended_basis']==BASIS and identity['role']=='teacher'
ledger=json.loads((I/'token_ledger.json').read_text());teacher=json.loads((I/'teacher_capture.json').read_text());row=ledger['rows'][0];assert row['window_id']==28 and row['ordinal']==0 and row['token_ids']==identity['rows'][0]['token_ids']
from banana_smasher.hf_sharded_balanced64_executor import LayerStreamedHFSession as Session,SourceTensorStore,require_hf_runtime
import torch,numpy as np
_,transformers=require_hf_runtime();torch.set_num_threads(8)
from stream_materialize_run8234 import materialize
from readout_run8242 import top_support
from banana_smasher import solver_qtip_profile as sp
members={x['path']:x for x in teacher['source']['binding']['members']}
for name in ['config.json','model.safetensors.index.json']:
 subprocess.run(['rsync','-a',REMOTE+name,str(S/name)],check=True)
 expected=BASIS if name.endswith('index.json') else teacher['source']['config_sha256'];assert sha(S/name)==expected
s=Session.__new__(Session);s.torch=torch;s.transformers=transformers;s.source=dict(teacher['source'],model_root=str(S));s.device='cuda';s._working_set_loads=0;s.store=SourceTensorStore(s.source);s._model=s._meta_model();language=s._language_model();assert len(language.layers)==45
# Prepare canonical decoded sparse replacements; decoder arithmetic stays unchanged.
runner=Path(sp.__file__).with_name('qtip_runner.py');qv=sp._load_public_qtip_runner(runner,sha(runner));cfg=json.loads((R/'whole_l1/warm1/E242_down/CONFIG.json').read_text());qv.QTIP=Path(cfg['qtip_root']);bits,ldlq,math,kd=qv.load_official_qtip();replacements={};bindings=[]
for arm in ['baseline1','baseline2','candidate']:
 replacements[arm]={}
 for expert,projection in [(242,'down'),(243,'down'),(242,'fused13')]:
  cell=f'E{expert}_{projection}'
  if arm=='candidate':p=R/f'whole_l1/warm1/{cell}/build/solve/L004/{cell}/QTIP_UNIT.pt'
  elif projection=='down':p=Path(f'/dev/shm/t_ebcba52e_ldlq_run8503/pair_warm_run8512/parallel1/solve/L004/{cell}/QTIP_UNIT.pt')
  else:p=R/'fused_r2/control1/warm/solve/L004/E242_fused13/QTIP_UNIT.pt'
  u=torch.load(p,map_location='cpu',weights_only=True);w=qv.decode_packed_weight(u,kd,torch.device('cuda')).half().cpu();replacements[arm][(expert,projection)]=w;bindings.append(dict(arm=arm,cell=cell,path=str(p),sha256=sha(p)))
put(O/'BINDING.json',dict(source_basis=BASIS,prefix_sha256=side['sha256'],teacher_sha256=sha(I/'teacher_capture.json'),ledger_sha256=sha(I/'token_ledger.json'),replacements=bindings,delta_limit=1e-8,scope='window28 native-prefix/native-rest sparse L004 K1 panel, not uniform full-model',runtime_transformers=transformers.__version__))
values={a:torch.load(I/'L003_000.payload',map_location='cpu',weights_only=True) for a in replacements};start=time.perf_counter();staging=0.;forward=0.;readout=0.;resident=set()
def stage(module):
 global resident,staging
 if not module.state_dict():return s._prefix_for(module)
 t=time.perf_counter();prefix=s._prefix_for(module);keys=[k for k in s.store.weight_map if k.startswith(prefix)];needed={s.store.weight_map[k] for k in keys}
 assert needed,('empty source dependency',prefix)
 for n in resident-needed:
  with (S/n).open('rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
  (S/n).unlink()
 resident&=needed
 peak=sum(x.numel()*x.element_size() for x in module.state_dict().values())+16*(1<<30);new=sum(members[n]['bytes'] for n in needed-resident)
 assert memory()>peak+new+4*(1<<30),('MEMORY_GATE',prefix,peak,new,memory())
 assert shutil.disk_usage(S).free>new+4*(1<<30)
 for n in sorted(needed-resident):
  prior=Path(os.environ['PREDECESSOR_ROOT'])/'source'/n
  if prior.exists():
   assert sha(prior)==members[n]['sha256'];os.link(prior,S/n)
  else:subprocess.run(['rsync','-a','--partial',REMOTE+n,str(S/n)],check=True)
  assert sha(S/n)==members[n]['sha256'];resident.add(n)
 materialize(s,module);staging+=time.perf_counter()-t
 return prefix
resume=Path(os.environ['PREDECESSOR_ROOT']) if os.environ.get('PREDECESSOR_ROOT') else None
if resume is not None:
 assert not (resume/'RESULT.json').exists(), 'sealed output gate must not be replayed'
 old=json.loads((resume/'BINDING.json').read_text());new=json.loads((O/'BINDING.json').read_text());assert old==new, 'resume scientific binding drift'
with torch.no_grad():
 for li in range(4,45):
  resumed=set()
  if resume is not None:
   for arm in values:
    name=f'L{li:03d}_{arm}';p=resume/(name+'.pt');j=resume/(name+'.json')
    if j.exists():
     meta=json.loads(j.read_text());assert p.stat().st_size==meta['bytes'] and sha(p)==meta['sha256'];assert meta['stage']==li and meta['arm']==arm
     values[arm]=torch.load(p,map_location='cpu',weights_only=True);os.link(p,O/p.name);put(O/j.name,meta);resumed.add(arm)
  if len(resumed)==len(values):continue
  layer=language.layers[li];put(O/'PROGRESS.json',dict(phase='stage',layer=li,elapsed=time.perf_counter()-start));prefix=stage(layer)
  targets={}
  if li==4:
   for name,p in layer.named_parameters():
    if name.endswith('experts.down_proj'):targets['down']=p
    if name.endswith('experts.gate_up_proj'):targets['fused13']=p
   assert set(targets)=={'down','fused13'},list(targets)
  t=time.perf_counter()
  for arm,v in values.items():
   if arm in resumed:continue
   if li==4:
    for (expert,proj),w in replacements[arm].items():assert targets[proj][expert].shape==w.shape;targets[proj][expert].copy_(w.to(targets[proj].dtype).to('cuda'))
   hidden=v['activation'].to('cuda');prior=v['topk'];positions=torch.arange(hidden.shape[1],device='cuda').unsqueeze(0);mask=torch.ones((1,hidden.shape[1]),dtype=torch.bool,device='cuda');ids=torch.tensor(row['token_ids'][:1024],device='cuda',dtype=torch.long).unsqueeze(0)
   y,top=layer(hidden,attention_mask=mask,position_ids=positions,position_embeddings=None,input_ids=ids,past_key_values=None,prev_topk_indices=None if prior is None else prior.to('cuda'),use_cache=False);assert torch.isfinite(y).all();values[arm]=dict(activation=y.cpu(),topk=None if top is None else top.cpu());p=O/f'L{li:03d}_{arm}.pt';torch.save(values[arm],p)
   with p.open('rb') as f:os.fsync(f.fileno())
   put(O/f'L{li:03d}_{arm}.json',dict(stage=li,arm=arm,sha256=sha(p),bytes=p.stat().st_size));del y,top,hidden,prior
  torch.cuda.synchronize();forward+=time.perf_counter()-t;s._dematerialize(layer);gc.collect();torch.cuda.empty_cache();put(O/'PROGRESS.json',dict(phase='sealed',layer=li,elapsed=time.perf_counter()-start,staging_seconds=staging,forward_seconds=forward))
 terminal=[language.hc_head,language.norm,s._model.lm_head]
 for module in terminal:stage(module)
 with np.load(I/'teacher_row_ordinal0.npz',allow_pickle=False) as z:bank={k:z[k] for k in z.files}
 assert np.array_equal(bank['position_map'],np.arange(1024));results=[];t=time.perf_counter()
 for arm,v in values.items():
  hidden=language.norm(language.hc_head(v['activation'].to('cuda'))).squeeze(0);out=top_support(hidden,s._model.lm_head.weight,support_token_ids=bank['support_token_ids']);np.savez(O/f'{arm}.npz',**out)
  a=torch.from_numpy(bank['support_logits']).double().log_softmax(-1);b=torch.from_numpy(out['support_logits']).double().log_softmax(-1);kl=(a.exp()*(a-b)).sum(-1);assert torch.isfinite(kl).all();results.append(dict(arm=arm,kld=float(kl.mean()),top1_matches=int((out['top1_token_ids']==bank['top1_token_ids']).sum()),row_sha256=sha(O/f'{arm}.npz')));del hidden,out,a,b,kl
 readout=time.perf_counter()-t
 for module in reversed(terminal):s._dematerialize(module)
d={x['arm']:x['kld'] for x in results};passed=abs(d['baseline1']-d['baseline2'])<=1e-8 and d['candidate']-d['baseline1']<=1e-8
put(O/'RESULT.json',dict(status='PASS' if passed else 'RED',rows=results,delta_limit=1e-8,staging_seconds=staging,forward_seconds=forward,readout_seconds=readout,whole_seconds=time.perf_counter()-start,resumed_from=None if resume is None else str(resume),scope='sparse L004 K1 on native prefix/rest, window28 teacher8192 conditional KL; not uniform-model proof'));assert passed
