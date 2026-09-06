import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import pathlib as p,sys,json,time,hashlib,platform,traceback
spec=json.loads(p.Path(sys.argv[1]).read_text());root=p.Path(spec['output']);root.mkdir(exist_ok=True,parents=True);src=p.Path(__file__).resolve().parents[1]/'banana-smasher/src';sys.path.insert(0,str(src))
target_layer=int(spec.get('layer',3));assert 3<=target_layer<45, 'capture layer outside routed scope'
h=lambda b:hashlib.sha256(b).hexdigest()
def save(n,d):
 q=root/n;q.parent.mkdir(exist_ok=True,parents=True);t=q.with_name('.'+q.name+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');f=t.open('rb');os.fsync(f.fileno());f.close();os.replace(t,q)
report={'schema':'native-frozen-train-layer-capture-v2','layer':target_layer,'status':'RUNNING','started':time.time(),'canonical_commit':spec['canonical_commit'],'scope':f'external-document calibration, 1024 tokens, native layers0..{target_layer} only; no solver or evaluation forward','clean_fit_admission':True,'reason':'independent external document; overlap adjudication bound in spec','windows':[]}
try:
 import torch,numpy,transformers
 from banana_smasher.hf_sharded_balanced64_executor import PackageHFShardedExecutor
 model=p.Path(spec['model_root']);ledger=p.Path(spec['calibration_ledger']);train=json.loads(ledger.read_text());start=int(spec.get('start_window',0));rows=train['rows'][start:start+spec['windows']];N=len(rows)
 assert N==spec['windows'] and N>0
 assert h((model/'model.safetensors.index.json').read_bytes())==spec['intended_basis']
 assert h(ledger.read_bytes())==spec['calibration_ledger_sha256']
 assert h(p.Path(spec['separation_receipt']).read_bytes())==spec['separation_receipt_sha256']
 assert spec['clean_fit_admission'] is True
 torch.set_num_threads(4);torch.manual_seed(0);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 report.update(ledger_sha256=h(ledger.read_bytes()),environment={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'transformers':transformers.__version__,'numpy':numpy.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'platform':platform.platform(),'seed':0,'deterministic':True,'tf32':False,'threads':4},source_hashes={str(q):h(q.read_bytes()) for q in [src/'banana_smasher/hf_sharded_balanced64_executor.py',model/'config.json',model/'model.safetensors.index.json',model/'tokenizer.json']})
 # Executor uses suite model semantics only; training ledger never passed through eval roster reader.
 lock=json.loads(p.Path(spec['suite_lock']).read_text());session=PackageHFShardedExecutor(subject={'model_root':str(model)},role='teacher',suite_lock=lock,corpus_rows=rows)
 language=session._language_model();device=session.device;tokens=[torch.tensor(r['token_ids'][:1024],dtype=torch.long,device=device).unsqueeze(0) for r in rows];assert all(t.shape==(1,1024) for t in tokens)
 session._materialize(language.embed_tokens)
 with torch.no_grad():activations=[language.embed_tokens(ids).unsqueeze(2).expand(-1,-1,int(language.config.hc_mult),-1).contiguous().cpu() for ids in tokens]
 session._dematerialize(language.embed_tokens); previous=[None]*N
 for li,layer in enumerate(language.layers[:target_layer+1]):
  save('CAPTURE_PROGRESS.json',{'stage':'MATERIALIZE','layer':li,'time':time.time()});session._materialize(layer);nexta=[];nextp=[]
  with torch.no_grad():
   for ordinal,(row,ids,a,prior) in enumerate(zip(rows,tokens,activations,previous,strict=True)):
    captured={};handles=[]
    if li==target_layer:
     gate=layer.mlp.gate;handles=[gate.register_forward_pre_hook(lambda mod,inp:captured.update(x=inp[0].detach().cpu())),gate.register_forward_hook(lambda mod,inp,output:captured.update(w=output[1].detach().cpu(),topk=output[2].detach().cpu()))]
    hidden=a.to(device);positions=torch.arange(hidden.shape[1],device=device).unsqueeze(0);mask=torch.ones((1,hidden.shape[1]),dtype=torch.bool,device=device)
    output,topk=layer(hidden,attention_mask=mask,position_ids=positions,position_embeddings=None,input_ids=ids,past_key_values=None,prev_topk_indices=None if prior is None else prior.to(device),use_cache=False)
    for handle in handles:handle.remove()
    assert torch.isfinite(output).all(), 'nonfinite native hidden'
    if li==target_layer:
     x=captured['x'].reshape(-1,captured['x'].shape[-1]).to(torch.bfloat16).contiguous();ki=captured['topk'].reshape(1024,-1).to(torch.int16).contiguous();w=captured['w'].reshape(1024,-1).to(torch.bfloat16).contiguous();assert torch.isfinite(x).all() and torch.isfinite(w).all()
     q=root/'captures'/f'xmoe_L{target_layer:03d}_win{ordinal:04d}.pt';q.parent.mkdir(exist_ok=True);tmp=q.with_name('.'+q.name+'.tmp');torch.save({'x':x,'topk':ki,'w':w,'item_id':row['item_id'],'ordinal':ordinal,'RL':1024,'layer':target_layer,'split':'external-document-calibration','win':ordinal,'ledger_sha256':report['ledger_sha256']},tmp);os.replace(tmp,q)
     rec={'item_id':row['item_id'],'ordinal':ordinal,'path':str(q),'sha256':h(q.read_bytes()),'rows':1024,'e000_routed_rows':int((ki==0).any(dim=1).sum()),'source_tokens_sha256':h(json.dumps(row['token_ids'][:1024],separators=(',',':')).encode())};report['windows'].append(rec);save('CAPTURE_RESULT.json',report)
    nexta.append(output.cpu());nextp.append(None if topk is None else topk.cpu());save('CAPTURE_PROGRESS.json',{'stage':'FORWARD','layer':li,'completed_windows':ordinal+1,'time':time.time()})
  activations,previous=nexta,nextp;session._dematerialize(layer)
 report.update(status='PASS_NATIVE_CAPTURE',completed=time.time(),captured_windows=len(report['windows']),e000_routed_rows=sum(r['e000_routed_rows'] for r in report['windows']))
 assert len(report['windows'])==N
except BaseException as e:
 report.update(status='FAILED_NATIVE_CAPTURE',error=repr(e),traceback=traceback.format_exc(),ended=time.time());save('CAPTURE_RESULT.json',report);raise
finally:save('CAPTURE_RESULT.json',report)
print(json.dumps({k:v for k,v in report.items() if k!='windows'},indent=2))
