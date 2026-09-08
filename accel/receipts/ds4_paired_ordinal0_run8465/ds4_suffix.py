"""Authorized DS4 ordinal0/window28 paired L013..L042 and readout only.
No original forward.py import/main, no full64, no fitting/prefix replay.
Uses owner's native layer builder and canonical packed decoder unchanged.
"""
import os,sys,json,time,hashlib,fcntl,subprocess,traceback,importlib.util,math
from pathlib import Path
TASK='t_ebcba52e';PIN='5a5d1f03f84e533f0d00dbe38302496e92842811'
ROOT=Path('/dev/shm/t_ebcba52e_ds4_suffix_run8461_a2');CLAIM=Path('/home/dnola/HOST_CLAIM.json')
SOURCE=Path('/dev/shm/t_ebcba52e_ds4_ordinal0_inputs_run8456/source')
MODEL=Path('/home/dnola/models/hf/DeepSeek-V4-Flash-0731')
BASIS='98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b'
G=1<<30

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def put(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp')
 with t.open('w') as f:json.dump(d,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(t,p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def mem():return next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
def free(p):v=os.statvfs(p);return v.f_bavail*v.f_frsize
def progress(stage,**kw):put(ROOT/'PROGRESS.json',dict(stage=stage,time=time.time(),**kw));print(stage,kw,flush=True)
def command(argv,stage):
 child=subprocess.Popen(argv);progress(stage,child_pid=child.pid,startticks=int(Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21]),argv=argv);assert child.wait()==0,(stage,argv)
def transfer(remote,target,digest,size):
 target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():
  assert min(free(ROOT),mem()-40*G)>size+4*G
  command(['rsync','-a','--partial','-e','ssh -o BatchMode=yes -o ConnectTimeout=10','dnola@192.168.200.9:'+remote,str(target)],'INPUT_TRANSFER')
 assert target.stat().st_size==size and sha(target)==digest,(remote,str(target))
def gate():
 c=json.loads(CLAIM.read_text());assert c['task_id']==TASK and c['pid']==os.getpid() and c['start_ticks']==TICKS and c['expiry_unix']>time.time()
 assert sha(MODEL/'model.safetensors.index.json')==BASIS
 assert mem()>40*G+4*G,('memory_peak_estimate_40GiB_plus_reserve',mem())
 assert free(ROOT)>4*G
 return c
assert sha(MODEL/'model.safetensors.index.json')==BASIS
assert sha(SOURCE/'selection.json')=='cb7882a831086a460209f9c56e430072bd9a3f961138d70541d7d03318a9e8ed'
assert sha(SOURCE/'teacher_bank_sharded.py')=='70c9c206cdf64bbe357312d9a12b381990714686f47c995f35e6d8f873a30ae4'
ROOT.mkdir(exist_ok=False);TICKS=int(Path('/proc/self/stat').read_text().split()[21])
identity=dict(task_id=TASK,pid=os.getpid(),pgid=os.getpgid(0),start_ticks=TICKS,argv=sys.argv,receipt=str(ROOT/'IDENTITY.json'),script_sha256=sha(__file__),canonical_git_pin=PIN)
put(ROOT/'IDENTITY.json',identity)
with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);raw=CLAIM.read_bytes();old=json.loads(raw)
 assert hashlib.sha256(raw).hexdigest()==sys.argv[1] and old['task_id']==TASK and old['stage']=='t_ebcba52e_ds4_suffix_run8461_terminal_retained'
 assert not Path('/proc/'+str(old['pid'])).exists()
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 assert json.loads(Path('/home/dnola/missions/CLEAN_t_bd7914e6/QUEUE_AUTHORITY_spark-6_g42471.json').read_text())['state']=='CANCELLED'
 assert not Path('/proc/1623251').exists()
 rows=json.loads((SOURCE/'selection.json').read_text());selected=[r for r in rows if 13<=r['layer']<=42]
 assert len(selected)==30*512 and len({(r['layer'],r['expert'],r['projection']) for r in selected})==len(selected)
 needed=max(sum(r['artifact_bytes'] for r in selected if r['layer']==L) for L in range(13,43))+8*G
 assert free(ROOT)>needed+4*G and mem()>needed+40*G+4*G,(needed,free(ROOT),mem())
 put(ROOT/'PREFLIGHT.json',dict(memory_peak_estimate_bytes=40*G,staging_frontier_estimate_bytes=needed,memavailable_bytes=mem(),tmpfs_free_bytes=free(ROOT),reserve_bytes=4*G,allocation='controller comment1788883083 dedicated s6; stale remote DRIVER_GOALS read and superseded',previous_production_successor='CANCELLED'))
 put(ROOT/'CLAIM_PREIMAGE.json',old);put(ROOT/'SHARDS.json',dict(intended_basis=BASIS,state='CLAIMED',rows=[dict(owner=TASK,ordinal=0,window=28,positions=1024,layers=[13,42],embedding=False,operation='paired_B1_B2_C_sparse_L013E084E085_K1_fourcell_ordinal0_output',receipt=str(ROOT/'IDENTITY.json'))]))
 c=dict(identity,owner=TASK,state='CLAIMED',status='CLAIMED',host='spark-6',stage=ROOT.name,source_model_index_sha256=BASIS,expiry_unix=time.time()+24*3600,mission=str(ROOT))
 assert CLAIM.read_bytes()==raw;put(CLAIM,c);put(ROOT/'CLAIM.json',c)
try:
 gate();started=time.perf_counter()
 spec=json.loads(Path('/dev/shm/DS4_FROZEN_BINDING_run8461.json').read_text())
 import shutil
 PREFIX=Path('/dev/shm/t_ebcba52e_ds4_prefix_run8461_a2')
 prefix_result=json.loads((PREFIX/'RESULT.json').read_text());assert prefix_result['state']=='SEALED_BOUNDED_DS4_K1_PREFIX'
 for tag in ['EMBED','L012']:
  rec=json.loads((PREFIX/'frontier'/f'{tag}.json').read_text());assert sha(rec['path'])==rec['sha256'] and rec['ordinal']==0 and rec['positions']==1024
 shutil.copytree(PREFIX/'frozen',ROOT/'frozen',copy_function=os.link)
 put(ROOT/'PREFIX_BINDING.json',dict(result_sha256=sha(PREFIX/'RESULT.json'),science_sha256=sha(PREFIX/'SCIENCE.json'),frontier=prefix_result['frontier'],frontier_sha256=prefix_result['frontier_sha256']))
 put(ROOT/'RECOVERY.json',dict(predecessor_failure_sha256=sha('/dev/shm/t_ebcba52e_ds4_suffix_run8461/FAILURE.json'),reason='BF16 hash uses uint8 view instead of unsupported NumPy BF16; no suffix forward slots existed'))
 for r in spec['files']:transfer(r['path'],ROOT/'frozen'/Path(r['path']).name,r['sha256'],r['bytes'])
 runtime=PREFIX/'runtime'
 assert sha(PREFIX/'RUNTIME.json') and runtime.exists()
 sys.path.insert(0,str(runtime));sys.path.insert(0,'/dev/shm/t_ebcba52e_api_5a5d1f03/banana-smasher/src')
 import torch
 from banana_smasher import qtip_runner as runner
 import transformers,importlib.metadata
 assert transformers.__version__=='5.12.1' and importlib.metadata.version('tokenizers')=='0.22.2'
 torch.set_num_threads(8)
 from transformers import AutoConfig,AutoModelForCausalLM
 from transformers.cache_utils import DynamicCache
 from transformers.masking_utils import create_sliding_window_causal_mask
 from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding
 from safetensors import safe_open
 loader=importlib.util.spec_from_file_location('ds4_owner_teacher_helper',SOURCE/'teacher_bank_sharded.py');builder=importlib.util.module_from_spec(loader);loader.loader.exec_module(builder)
 runner.QTIP=Path('/dev/shm/t_ebcba52e_ds4_stage_run8451/source/home/dnola/missions/QTIP1_CHAMPION_t_ds4q1_spark_8/inputs/qtip_runtime/qtip-canonical')
 _,_,_,decoder=runner.load_official_qtip()
 put(ROOT/'RUNTIME.json',dict(transformers=transformers.__version__,torch=torch.__version__,runner=dict(path=runner.__file__,sha256=sha(runner.__file__)),decoder=dict(path=decoder.__file__,sha256=sha(decoder.__file__)),helper_sha256=sha(SOURCE/'teacher_bank_sharded.py'),canonical_git_pin=PIN,packages=[dict(path=str(p),sha256=sha(p)) for p in sorted(runtime.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]))
 corpus=json.loads((ROOT/'frozen/windows_ds4_eval.json').read_text());tokens=corpus[28]['token_ids'][:1024];assert len(tokens)==1024
 config=AutoConfig.from_pretrained(MODEL);wm=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map'];assert config.num_hidden_layers==43
 science=dict(intended_basis=BASIS,ordinal=0,window=28,positions=1024,suffix_layers=[13,42],semantics='historical_K1_routed_native_rest_plus_paired_fourcell_K1_replacements',selection_sha256=sha(SOURCE/'selection.json'),frozen_binding_sha256=sha('/dev/shm/DS4_FROZEN_BINDING_run8461.json'),token_ids=tokens,canonical_git_pin=PIN,no_full64=True,no_fit=True,no_readout=False,arms=['baseline1','baseline2','candidate1'],limits_rule=dict(conditional_teacher_support_kl_delta='max(1e-8,5*abs(B1_KL-B2_KL))',max_allowed_repeat_tolerance=1e-5,one_window_only=True,freeze_before_candidate=True,top1_rule='candidate_matches>=min(baseline1_matches,baseline2_matches)',top1_semantics='teacher_support_firstmax_token_vs_candidate_fullvocab_argmax'))
 put(ROOT/'SCIENCE.json',science)
 panel=Path('/dev/shm/t_ebcba52e_ds4_cold_run8456');summary=json.loads(Path('/dev/shm/DS4_COLD_SUMMARY_run8456.json').read_text())
 replacement={}
 for arm in science['arms']:
  replacement[arm]={}
  for cell in summary['arms'][arm]['warm']['cells']:
   key=(cell['expert'],cell['projection']);p=panel/arm/'warm/solve/L013'/f'E{key[0]:03d}_{key[1]}'/'QTIP_UNIT.pt'
   assert sha(p)==cell['artifact_sha256'];replacement[arm][key]=dict(path=str(p),sha256=cell['artifact_sha256'])
  assert set(replacement[arm])=={(e,p) for e in [84,85] for p in ['down','fused13']}
 put(ROOT/'REPLACEMENTS.json',dict(summary_sha256=sha('/dev/shm/DS4_COLD_SUMMARY_run8456.json'),rows={a:[dict(expert=k[0],projection=k[1],**v) for k,v in rows.items()] for a,rows in replacement.items()},producer_timing='already sealed matched cold/warm public builds; all suffix timing is validator cost'))
 with torch.device('meta'):model=AutoModelForCausalLM.from_config(config,attn_implementation='eager')
 model.eval();handles={}
 def close():
  while handles:_,h=handles.popitem();h.__exit__(None,None,None)
 def get_tensor(name):
  p=MODEL/wm[name]
  if str(p) not in handles:close();handles[str(p)]=safe_open(p,framework='pt')
  return handles[str(p)].get_tensor(name)
 def save(tag,hidden,**kw):
  path=ROOT/'frontier'/(tag+'.pt');path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
  tmp=path.with_suffix('.tmp');torch.save(dict(activation=hidden.detach().cpu()),tmp)
  with tmp.open('rb') as f:os.fsync(f.fileno())
  os.replace(tmp,path);put(path.with_suffix('.json'),dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,ordinal=0,window=28,positions=1024,stage=tag,science_sha256=sha(ROOT/'SCIENCE.json'),**kw));return path
 ids=torch.tensor(tokens,device='cuda',dtype=torch.long).unsqueeze(0);pos=torch.arange(1024,device='cuda').unsqueeze(0)
 gate()
 with torch.no_grad():
  embed_payload=torch.load(PREFIX/'frontier/EMBED.pt',map_location='cpu',weights_only=True)
  embeds=embed_payload['activation'][:,:,0,:].to('cuda');del embed_payload
  rotary=DeepseekV4RotaryEmbedding(config).to('cuda');pe={k:rotary(embeds,position_ids=pos,layer_type=t) for k,t in [('main','main'),('compress','compress')]}
  mask=create_sliding_window_causal_mask(config=config,inputs_embeds=embeds,attention_mask=None,past_key_values=DynamicCache(config=config),position_ids=pos);del embeds
  frontiers={arm:PREFIX/'frontier/L012.pt' for arm in science['arms']}
  for L in range(13,43):
   gate();layer_start=time.perf_counter();rr=[r for r in selected if r['layer']==L]
   stage=ROOT/'units'/f'L{L:03d}';stage.mkdir(parents=True)
   paths={r['artifact_path']:r for r in rr};listing=ROOT/f'L{L:03d}_files.txt';listing.write_text('\n'.join(p.lstrip('/') for p in paths)+'\n')
   if L==13:
    shutil.copytree('/dev/shm/t_ebcba52e_ds4_suffix_run8461/units/L013',stage,copy_function=os.link,dirs_exist_ok=True)
   else:
    command(['rsync','-a','--partial','--files-from='+str(listing),'-e','ssh -o BatchMode=yes -o ConnectTimeout=10','dnola@192.168.200.9:/',str(stage)+'/'],f'L{L:03d}_UNITS_TRANSFER')
   for p,r in paths.items():q=stage/p.lstrip('/');assert q.stat().st_size==r['artifact_bytes'] and sha(q)==r['artifact_sha256'],p
   put(ROOT/f'L{L:03d}_INPUTS.json',dict(rows=rr,root=str(stage),selection_sha256=science['selection_sha256']))
   gate();progress('NATIVE_LAYER_BUILD',layer=L)
   sd=builder.build_layer_sd(L,wm,get_tensor,'bf16',None);close()
   injections=[]
   for i,r in enumerate(rr):
    p=stage/r['artifact_path'].lstrip('/');unit=torch.load(p,map_location='cpu',mmap=True,weights_only=True);assert unit['geometry']['K']==1
    decoded=runner.decode_packed_weight(unit,decoder,torch.device('cuda'));torch.cuda.synchronize();half=decoded.half()
    assert runner.tensor_sha256(half)==r['decoded_fp16_sha256'],('historical_fp16_mismatch',L,r['expert'],r['projection'])
    key='mlp.experts.down_proj' if r['projection']=='down' else 'mlp.experts.gate_up_proj';sd[key][r['expert']].copy_(half.to(torch.bfloat16))
    bfsha=hashlib.sha256(sd[key][r['expert']].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest();assert bfsha==r['injected_bf16_sha256']
    injections.append(dict(layer=L,expert=r['expert'],projection=r['projection'],artifact_sha256=r['artifact_sha256'],injected_bf16_sha256=bfsha));del unit,decoded,half
    if i%32==0:progress('PACKED_MATERIALIZE',layer=L,injected=i+1,total=512)
   put(ROOT/f'L{L:03d}_INJECTIONS.json',injections)
   lay=builder.materialize_layer(model,L,sd,config);del sd
   for arm in science['arms']:
    gate()
    consumed=[]
    if L==13:
     for (expert,projection),binding in replacement[arm].items():
      assert sha(binding['path'])==binding['sha256'];unit=torch.load(binding['path'],map_location='cpu',mmap=True,weights_only=True)
      assert unit['geometry']['K']==1;decoded=runner.decode_packed_weight(unit,decoder,torch.device('cuda'))
      target=lay.mlp.experts.down_proj if projection=='down' else lay.mlp.experts.gate_up_proj
      target[expert].copy_(decoded.half().to(torch.bfloat16));consumed.append(dict(expert=expert,projection=projection,artifact_sha256=binding['sha256'],injected_bf16_sha256=runner.tensor_sha256(target[expert].view(torch.uint8))))
      del unit,decoded
     put(ROOT/(arm+'_CONSUMED.json'),consumed)
    progress('FORWARD',layer=L,arm=arm)
    payload=torch.load(frontiers[arm],map_location='cpu',weights_only=True);hidden=payload['activation'].to('cuda');del payload
    hidden=lay(hidden,position_embeddings=pe,position_ids=pos,attention_mask=mask,input_ids=ids,past_key_values=DynamicCache(config=config))
    torch.cuda.synchronize();assert torch.isfinite(hidden).all()
    frontiers[arm]=save(f'{arm}/L{L:03d}',hidden,injections_sha256=sha(ROOT/f'L{L:03d}_INJECTIONS.json'),consumed_sha256=sha(ROOT/(arm+'_CONSUMED.json')),wall_seconds=time.perf_counter()-layer_start)
    del hidden
   builder.dematerialize_layer(model,L);progress('LAYER_SEALED',layer=L,frontiers={a:str(p) for a,p in frontiers.items()},wall_seconds=time.perf_counter()-layer_start)
   # Only authenticated task-owned transfer copies, never original model/units/frontiers.
   for original,r in paths.items():
    copy=stage/original.lstrip('/');assert copy.is_relative_to(stage) and sha(copy)==r['artifact_sha256'];copy.unlink()
   put(ROOT/f'L{L:03d}_STAGING_RELEASED.json',dict(original_source_preserved=True,frontiers_preserved=True,unit_copies=len(paths)))
  gate();progress('READOUT_MATERIALIZE')
  model.lm_head.weight=torch.nn.Parameter(get_tensor('head.weight').to('cuda').to(torch.bfloat16),requires_grad=False)
  model.model.norm.weight=torch.nn.Parameter(get_tensor('norm.weight').to('cuda').to(torch.bfloat16),requires_grad=False)
  for dest,src in [('hc_fn','hc_head_fn'),('hc_base','hc_head_base'),('hc_scale','hc_head_scale')]:setattr(model.model.hc_head,dest,torch.nn.Parameter(get_tensor(src).to('cuda').to(torch.float32),requires_grad=False))
  close();teacher_path=ROOT/'frozen/t8192_win28.pt';teacher=torch.load(teacher_path,map_location='cpu',weights_only=True)
  idx=teacher['idx'][:1024].long();logp=teacher['logprob'][:1024].double()
  # Canonical frozen convention: teacher and student renormalized on exact8192 support, binary64 reduction.
  pnorm=logp-torch.logsumexp(logp,dim=-1,keepdim=True);prob=pnorm.exp();metrics={}
  for arm in science['arms']:
   progress('READOUT',arm=arm);payload=torch.load(frontiers[arm],map_location='cpu',weights_only=True);hidden=payload['activation'].to('cuda');del payload
   h=model.model.norm(model.model.hc_head(hidden));del hidden
   logits=model.lm_head(h[0].to(torch.bfloat16)).float();lp=torch.log_softmax(logits,dim=-1)
   at=lp.gather(1,idx.to('cuda')).to(torch.float16).cpu();argmax=lp.argmax(-1).to(torch.int32).cpu();del logits,lp,h
   qnorm=at.double()-torch.logsumexp(at.double(),dim=-1,keepdim=True)
   kl=(prob*(pnorm-qnorm)).sum(-1);assert torch.isfinite(kl).all() and (kl>=0).all()
   teacher_top=idx.gather(1,pnorm.argmax(-1).unsqueeze(1)).squeeze(1);top1=int((teacher_top==argmax).sum())
   path=ROOT/(arm+'_READOUT.pt');tmp=path.with_suffix('.tmp');torch.save(dict(q_lp_at_ref=at,q_argmax=argmax,idx=idx,kl_per_position=kl,metadata=dict(ordinal=0,window=28,positions=1024,support=8192,teacher_sha256=sha(teacher_path),science_sha256=sha(ROOT/'SCIENCE.json'),frontier_sha256=sha(frontiers[arm]),top1_semantics='teacher_support_firstmax_token_vs_candidate_fullvocab_argmax',normalization='owner full_vocab_fp32_logsoftmax,gather,fp16_wire;binary64 support renorm')),tmp)
   with tmp.open('rb') as f:os.fsync(f.fileno())
   os.replace(tmp,path)
   metrics[arm]=dict(kl=math.fsum(kl.tolist())/1024,top1_matches=top1,top1_denominator=1024,path=str(path),sha256=sha(path),positions=1024,support=8192)
   put(ROOT/(arm+'_METRIC.json'),metrics[arm])
   if arm=='baseline2':
    repeat=abs(metrics['baseline1']['kl']-metrics['baseline2']['kl']);tolerance=max(1e-8,5*repeat)
    put(ROOT/'FROZEN_LIMITS.json',dict(baseline_repeat_delta=repeat,tolerance=tolerance,max_tolerance=1e-5,baseline_readout_sha256=[metrics[a]['sha256'] for a in ['baseline1','baseline2']],candidate_readout_not_executed=True))
    assert tolerance<=1e-5,'baseline reproducibility admission failed'
  delta=metrics['candidate1']['kl']-sum(metrics[a]['kl'] for a in ['baseline1','baseline2'])/2
  result=dict(state='SEALED_BOUNDED_DS4_PAIRED_ORDINAL0_OUTPUT',identity=identity,science_sha256=sha(ROOT/'SCIENCE.json'),prefix_result_sha256=sha(PREFIX/'RESULT.json'),metrics=metrics,delta=delta,tolerance=tolerance,quality_pass=delta<=tolerance and metrics['candidate1']['top1_matches']>=min(metrics[a]['top1_matches'] for a in ['baseline1','baseline2']),scope='one frozen ordinal0; sparse4cell K1 replacements on historical uniformK1 routed/native_rest; not uniform/currentK3/fullmodel acceptance',producer_timing_summary_sha256=sha('/dev/shm/DS4_COLD_SUMMARY_run8456.json'),wall_seconds=time.perf_counter()-started,readout_executed=True,full64_replayed=False,production_promotion=False,cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved())
 put(ROOT/'RESULT.json',result)
except BaseException as e:
 put(ROOT/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),identity=identity));raise
finally:
 with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);c=json.loads(CLAIM.read_text());assert c['pid']==os.getpid();c.update(stage=ROOT.name+'_terminal_retained',terminal_at=time.time());put(CLAIM,c)
