import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import pathlib as p,sys,json,hashlib,time,traceback
spec=json.loads(p.Path(sys.argv[1]).read_text());out=p.Path(spec['output']);src=p.Path(__file__).resolve().parents[1]/'banana-smasher/src';sys.path.insert(0,str(src))
import torch
from banana_smasher import solver_qtip_profile as sp,qtip_runner as runner
from banana_smasher.glm_qtip_source_adapter import capture_source_closure
root=p.Path(spec.get('solve_root',out/'q2_cell'));root.mkdir(exist_ok=True);result_path=out/spec.get('result_name','CORRECTION_RESULT.json');h=lambda b:hashlib.sha256(b).hexdigest()
def save(q,d):sp._atomic_json(q,d)
layer=int(spec.get('layer',3));expert=int(spec.get('expert',0));projection=spec.get('projection','fused13');cell=f'L{layer:03d}/E{expert:03d}_{projection}'
assert 3<=layer<45 and 0<=expert<288 and projection in ('fused13','down')
assert cell=='L003/E000_fused13' or spec.get('prepared_config'), 'non-default cell requires prepared config'
report={'schema':'single-cell-training-correction-v1','status':'RUNNING','started':time.time(),'cell':cell,'tier':'K2','canonical_commit':spec['canonical_commit'],'training_ledger_sha256':spec['calibration_ledger_sha256'],'clean_fit_admission':True,'label':'external-document clean-calibrated K2 one-cell native-rest; not uniform PRE'}
try:
 torch.set_num_threads(4);torch.manual_seed(0);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert h(p.Path(spec['calibration_ledger']).read_bytes())==report['training_ledger_sha256']
 if spec.get('prepared_config'):
  cp=p.Path(spec['prepared_config']);config=json.loads(cp.read_text());assert config['training_ledger_sha256']==report['training_ledger_sha256'];assert config['input_identity']['model_index']['sha256']==spec['intended_basis'];assert h(p.Path(config['input_identity']['model_index']['path']).read_bytes())==spec['intended_basis'];closure={'sha256':config['glm_source_closure_sha256']}
 else:
  windows=[]
  for capfile in spec['capture_results']:
   part=json.loads(p.Path(capfile).read_text());assert part['status']=='PASS_NATIVE_CAPTURE' and part['ledger_sha256']==report['training_ledger_sha256'];windows.extend(part['windows'])
  assert len(windows)==spec['windows'] and len({r['item_id'] for r in windows})==len(windows)
  expected={r['item_id'] for r in json.loads(p.Path(spec['calibration_ledger']).read_text())['rows']};assert {r['item_id'] for r in windows}==expected
  cap={'windows':[dict(r,ordinal=i) for i,r in enumerate(windows)]};members=[];cr=root/'fitcaptures';cr.mkdir(exist_ok=True)
  for row in cap['windows']:
   original=p.Path(row['path']);assert h(original.read_bytes())==row['sha256'];d=torch.load(original,map_location='cpu',weights_only=True);d['win']=row['ordinal'];q=cr/f'xmoe_L003_win{row["ordinal"]:04d}.pt';assert not q.exists();sp._atomic_torch(q,d);done=q.with_suffix('.pt.DONE.json');save(done,{'status':'PASS','meaning':'physical bytes verified, not clean admission','md5':hashlib.md5(q.read_bytes()).hexdigest(),'layer':3,'win':row['ordinal'],'item_id':row['item_id'],'original_capture_sha256':row['sha256'],'ledger_sha256':report['training_ledger_sha256'],'clean_fit_admission':True});members.append({'window':row['ordinal'],'item_id':row['item_id'],'capture':{'path':str(q),'bytes':q.stat().st_size,'sha256':h(q.read_bytes())},'capture_done':{'path':str(done),'bytes':done.stat().st_size,'sha256':h(done.read_bytes())}})
  assert len(members)==spec['windows']
  hm=root/'L003_HESSIAN_MANIFEST.json';save(hm,{'schema':'banana-smasher-hessian-layer-manifest-v1','status':'PASS','meaning':'authenticated training-only capture inventory; not scientific clean admission','layer':3,'windows':spec['windows'],'capture_root':str(cr),'members':members,'capture_inventory_seal_sha256':h(json.dumps(cap,sort_keys=True).encode()),'clean_fit_admission':True})
  old=p.Path(spec['template_config']);config=json.loads(old.read_text());assert config['geometry']['K']==2;assert config['input_identity']['model_index']['sha256']==spec['intended_basis'];assert h(p.Path(config['input_identity']['model_index']['path']).read_bytes())==spec['intended_basis'];qp=src/'banana_smasher/qtip_runner.py';manifest=root/'QTIP_RUN_MANIFEST.json';save(manifest,{'schema':'bounded-training-correction-run-manifest-v1','canonical_commit':report['canonical_commit'],'cells':['L003/E000_fused13'],'clean_fit_admission':True,'tiers':[{'name':config['tier'],'bindings':{'qtip_runner':{'path':str(qp),'sha256':h(qp.read_bytes())}}}],'training_ledger_sha256':report['training_ledger_sha256'],'source_canary_sha256':h(old.read_bytes())})
  seed_material=h(json.dumps(cap,sort_keys=True).encode());config.update(fit_windows=spec['windows'],fit_capture_root=str(cr),hessian_layer_manifest=str(hm),hessian_layer_manifest_sha256=h(hm.read_bytes()),qtip_runner=str(qp),layer_census={'qtip2':1},pack_counts={'qtip2':1},materialization={'run_manifest':str(manifest),'run_manifest_sha256':h(manifest.read_bytes()),'source_config_sha256':h(hm.read_bytes())},rht_seed_material=seed_material,rht_seed=sp._canonical_rht_seed(seed_material,3,0,'fused13'),clean_fit_admission=True,training_ledger_sha256=report['training_ledger_sha256'])
  runner.QTIP=p.Path(config['qtip_root']);mods=runner.load_official_qtip();closure=capture_source_closure(runner,dict(zip(('bitshift','ldlq','math_utils','kernel_decompress'),mods)));save(root/'SOURCE_CLOSURE.json',closure);config['glm_source_closure_sha256']=closure['sha256'];cp=root/'L003_E000_fused13.json';save(cp,config);assert json.loads((root/'SHARDS.json').read_text())['intended_basis']==spec['intended_basis']
 assert (config.get('layer',3),config.get('expert',0),config.get('projection','fused13'))==(layer,expert,projection), 'prepared cell mismatch'
 report.update(config_sha256=h(cp.read_bytes()),source_closure_sha256=closure['sha256'],rht_seed=config['rht_seed'],tlut_source_sha256=h(p.Path(config['tlut_source']).read_bytes()),reference_sha256=h(p.Path(config['reference_unit']).read_bytes()));save(result_path,report)
 result=sp.main(cp,root,layer,profile_mode=False,kernel_cache_root=root/'kernel-cache');receipt=root/'solve'/cell/'QTIP_SOLVE_RECEIPT.json';artifact=root/'solve'/cell/'QTIP_UNIT.pt';r=json.loads(receipt.read_text());assert r['status']=='PASS' and h(artifact.read_bytes())==r['artifact_sha256'];report.update(status='PASS_SINGLE_CELL_CLEAN_CALIBRATION',artifact=str(artifact),artifact_sha256=h(artifact.read_bytes()),receipt=str(receipt),receipt_sha256=h(receipt.read_bytes()),ended=time.time())
except BaseException as e:report.update(status='FAIL',error=repr(e),traceback=traceback.format_exc(),ended=time.time());save(result_path,report);raise
finally:save(result_path,report)
print(json.dumps(report,indent=2))
