import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import pathlib as p,sys,json,time,hashlib,math,traceback,gc
spec=json.loads(p.Path(sys.argv[1]).read_text());out=p.Path(spec['output']);src=p.Path(__file__).resolve().parents[1]/'banana-smasher/src';sys.path.insert(0,str(src))
import numpy as np,torch
from banana_smasher.hf_sharded_balanced64_executor import PackageHFShardedExecutor,top_support
from banana_smasher.glm_qtip_source_adapter import load_glm_fp8_weight,capture_source_closure
from banana_smasher import qtip_runner as runner
h=lambda b:hashlib.sha256(b).hexdigest();raw=out/'model_output';raw.mkdir(exist_ok=True)
def save(n,d):
 q=out/n;t=q.with_name('.'+q.name+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');os.replace(t,q)
def th(t):return h(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
report={'schema':'single-corrected-cell-native-rest-output-v1','status':'RUNNING','started':time.time(),'canonical_commit':spec['canonical_commit'],'cell':'L003/E000_fused13','arms':['corrected_K2_native_rest'],'evaluation_ids':[28,56,68,71],'positions_per_window':1024,'support':8192,'metric':'KL(teacher||candidate), both distributions renormalized on identical teacher-selected top8192 sorted descending teacher logits; per-position binary64, ordered math.fsum over4096 positions divided once','negative_policy':'reject every negative or non-finite per-position KL; no clamp','clean_fit_admission':True,'uniform_full_model_pre':False,'label':'single clean-calibrated K2 cell/native-rest; all main layers+terminal head executed; not all288 corrected layer','outputs':{},'layer_execution':{}}
try:
 baseline_path=p.Path(spec['native_baseline_receipt']);assert h(baseline_path.read_bytes())==spec['native_baseline_receipt_sha256'];baseline=json.loads(baseline_path.read_text());assert baseline['native_control_exact'] is True
 assert h((src/'banana_smasher/hf_sharded_balanced64_executor.py').read_bytes())==spec['native_executor_sha256']
 assert torch.__version__==baseline['settings']['torch'] and torch.version.cuda==baseline['settings']['cuda']
 for binding in baseline['outputs']['native']:assert h(p.Path(binding['path']).read_bytes())==binding['sha256']
 torch.set_num_threads(4);torch.manual_seed(0);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 caproot=p.Path(spec['evaluation_root']);lp=caproot/'glm-ledger.json';lockp=caproot/'glm-lock.json';assert h(lp.read_bytes())=='ca0e794a6296075f140c01ff4a2539b66002493d01b56e71c83645c79b5ae505';ledger=json.loads(lp.read_text());lock=json.loads(lockp.read_text());assert lock['suite_lock_sha256']=='90eb646254312e9ad360eabcd0d576e14df262968f9979c520ee9ec8322f0696';byid={int(r['item_id']):r for r in ledger['rows']};rows=[byid[i] for i in report['evaluation_ids']]
 cor=json.loads((out/'CORRECTION_RESULT.json').read_text());assert cor['status']=='PASS_SINGLE_CELL_CLEAN_CALIBRATION';artifact=p.Path(cor['artifact']);assert h(artifact.read_bytes())==cor['artifact_sha256'];model=p.Path(spec['model_root']);assert h((model/'model.safetensors.index.json').read_bytes())==spec['intended_basis'];native,_=load_glm_fp8_weight(model,3,0,'fused13');runner.QTIP=p.Path(spec['qtip_root']);mods=runner.load_official_qtip();payload=torch.load(artifact,map_location='cpu',weights_only=True,mmap=True);payload['reconstructed_weight']=native.half();decoded,decode_receipt=runner.decode_packed(payload,mods[3],torch.device('cuda'));decoded=decoded.half().cpu();assert torch.isfinite(decoded).all();del payload;torch.cuda.empty_cache()
 report.update(artifact=str(artifact),artifact_read_sha256=h(artifact.read_bytes()),decoded_fp16_sha256=th(decoded),training_ledger_sha256=cor['training_ledger_sha256'],evaluation_ledger_sha256=h(lp.read_bytes()),settings={'seed':0,'deterministic':True,'tf32':False,'threads':4,'native_working_dtype':'BF16 from independent FP8 source descaling','candidate_working_dtype':'decoded FP16 converted to BF16 at one exact cell','torch':torch.__version__,'cuda':torch.version.cuda,'executable':sys.executable},canonical_closure=capture_source_closure(runner,dict(zip(('bitshift','ldlq','math_utils','kernel_decompress'),mods))))
 session=PackageHFShardedExecutor(subject={'model_root':str(model)},role='teacher',suite_lock=lock,corpus_rows=rows);lang=session._language_model();dev=session.device;arms=report['arms'];tokens=[torch.tensor(r['token_ids'][:1024],dtype=torch.long,device=dev).unsqueeze(0) for r in rows];assert all(t.shape==(1,1024) for t in tokens)
 session._materialize(lang.embed_tokens)
 with torch.no_grad():a0=[lang.embed_tokens(ids).unsqueeze(2).expand(-1,-1,int(lang.config.hc_mult),-1).contiguous().cpu() for ids in tokens]
 session._dematerialize(lang.embed_tokens);activations={a:[x.clone() for x in a0] for a in arms};prev={a:[None]*4 for a in arms};del a0
 route_rows={a:[] for a in arms}
 for li,layer in enumerate(lang.layers[:lang.config.num_hidden_layers]):
  save('OUTPUT_PROGRESS.json',{'stage':'MATERIALIZE','layer':li,'time':time.time()});session._materialize(layer)
  if li==3:
   target=layer.mlp.experts.gate_up_proj;assert tuple(target.shape)==(288,4096,4096);assert torch.equal(target[0].cpu(),native.to(target.dtype));report['native_cell_installed_sha256']=th(target[0])
  for arm in arms:
   if li==3 and arm=='corrected_K2_native_rest':
    with torch.no_grad():target[0].copy_(decoded.to(device=dev,dtype=target.dtype))
    assert torch.equal(target[0].cpu(),decoded.to(target.dtype));report['candidate_cell_installed_sha256']=th(target[0]);report['candidate_cell_dtype']=str(target.dtype)
   nexta=[];nextp=[]
   with torch.no_grad():
    for wi,(ids,a,prior) in enumerate(zip(tokens,activations[arm],prev[arm],strict=True)):
     routed={};handle=None
     if li==3:handle=layer.mlp.gate.register_forward_hook(lambda mod,inp,res:routed.update(topk=res[2].detach().cpu()))
     hidden=a.to(dev);pos=torch.arange(1024,device=dev).unsqueeze(0);mask=torch.ones((1,1024),dtype=torch.bool,device=dev);y,k=layer(hidden,attention_mask=mask,position_ids=pos,position_embeddings=None,input_ids=ids,past_key_values=None,prev_topk_indices=None if prior is None else prior.to(dev),use_cache=False)
     if handle:handle.remove();ki=routed['topk'].reshape(1024,-1).long();route_rows[arm].append({'item_id':report['evaluation_ids'][wi],'e000_rows':int((ki==0).any(1).sum()),'suffix256_287_rows':int((ki>=256).any(1).sum()),'expert_route_counts':torch.bincount(ki.flatten(),minlength=288).tolist()})
     assert torch.isfinite(y).all(),f'nonfinite hidden {arm} L{li}';nexta.append(y.cpu());nextp.append(None if k is None else k.cpu())
   activations[arm]=nexta;prev[arm]=nextp;report['layer_execution'].setdefault(arm,[]).append(li);save('OUTPUT_PROGRESS.json',{'stage':'FORWARD','layer':li,'arm':arm,'windows':4,'time':time.time()})
  session._dematerialize(layer);save('OUTPUT_RESULT.json',report)
 for m in [lang.hc_head,lang.norm,session._model.lm_head]:session._materialize(m)
 baseline_path=p.Path(spec['native_baseline_receipt']);assert h(baseline_path.read_bytes())==spec['native_baseline_receipt_sha256'];baseline=json.loads(baseline_path.read_text());assert baseline['native_control_exact'] is True;assert baseline['evaluation_ledger_sha256']==report['evaluation_ledger_sha256'];assert baseline['evaluation_ids']==report['evaluation_ids']
 outputs={a:[] for a in arms};outputs['native']=[]
 for binding in baseline['outputs']['native']:
  q=p.Path(binding['path']);assert h(q.read_bytes())==binding['sha256'];value=np.load(q);outputs['native'].append({k:value[k] for k in value.files})
 report['native_baseline_receipt_sha256']=spec['native_baseline_receipt_sha256'];report['native_baseline_reused_no_replay']=True
 with torch.no_grad():
  for arm in arms:
   for wi,a in enumerate(activations[arm]):
    hidden=lang.norm(lang.hc_head(a.to(dev))).squeeze(0);support=None if arm=='native' else outputs['native'][wi]['support_token_ids'];value=top_support(hidden,session._model.lm_head.weight,support_token_ids=support,support=8192);assert np.isfinite(value['support_logits']).all();q=raw/f'{arm}_win{report["evaluation_ids"][wi]}.npz';np.savez(q,**value);outputs[arm].append(value);report['outputs'].setdefault(arm,[]).append({'item_id':report['evaluation_ids'][wi],'path':str(q),'sha256':h(q.read_bytes()),'shape':list(value['support_logits'].shape)});save('OUTPUT_RESULT.json',report)
 metrics={}
 for arm in arms:
  values=[];window_rows=[];support_correct=0;full_correct=0;neg=0;nonfinite=0
  for wi,(t,c) in enumerate(zip(outputs['native'],outputs[arm],strict=True)):
   assert np.array_equal(t['support_token_ids'],c['support_token_ids']);tl=torch.from_numpy(t['support_logits']).double();cl=torch.from_numpy(c['support_logits']).double();logp=torch.log_softmax(tl,dim=-1);logq=torch.log_softmax(cl,dim=-1);kl=(logp.exp()*(logp-logq)).sum(dim=-1).numpy();nonfinite+=int((~np.isfinite(kl)).sum());neg+=int((kl<0).sum());values.extend(kl.tolist());su=int((tl.argmax(-1)==cl.argmax(-1)).sum());fu=int((t['top1_token_ids']==c['top1_token_ids']).sum());support_correct+=su;full_correct+=fu;np.save(raw/f'{arm}_win{report["evaluation_ids"][wi]}.kl_fp64.npy',kl);window_rows.append({'item_id':report['evaluation_ids'][wi],'kl_sum_fsum':math.fsum(kl.tolist()) if np.isfinite(kl).all() else None,'support_top1_matches':su,'full_vocab_top1_matches':fu,'logits_bit_exact':bool(np.array_equal(t['support_logits'].view(np.uint32),c['support_logits'].view(np.uint32)))})
  metrics[arm]={'positions':len(values),'negative_positions':neg,'nonfinite_positions':nonfinite,'kl_valid_under_frozen_policy':neg==0 and nonfinite==0,'kl_teacher_to_candidate_support8192':math.fsum(values)/len(values) if neg==0 and nonfinite==0 else None,'support_top1_agreement':support_correct/len(values),'full_vocab_top1_agreement':full_correct/len(values),'windows':window_rows}
 report.update(metrics=metrics,routing=route_rows,main_layer_count=int(lang.config.num_hidden_layers),terminal_head_executed=True,status='MEASURED_CLEAN_CALIBRATED_K2_NATIVE_REST',ended=time.time());assert all(v==list(range(int(lang.config.num_hidden_layers))) for v in report['layer_execution'].values());assert h(artifact.read_bytes())==report['artifact_read_sha256'];assert h(lp.read_bytes())==report['evaluation_ledger_sha256'];assert len(report['outputs'])==1 and all(len(v)==4 for v in report['outputs'].values())
 if not all(m['kl_valid_under_frozen_policy'] for m in metrics.values()):report['status']='FAILED_NUMERIC_GATE'
 report['native_control_reused_exact']=baseline['native_control_exact']
except BaseException as e:report.update(status='FAIL_EXECUTION',error=repr(e),traceback=traceback.format_exc(),ended=time.time());save('OUTPUT_RESULT.json',report);raise
finally:save('OUTPUT_RESULT.json',report)
print(json.dumps({k:v for k,v in report.items() if k not in ['canonical_closure','layer_execution','outputs','routing']},indent=2))
