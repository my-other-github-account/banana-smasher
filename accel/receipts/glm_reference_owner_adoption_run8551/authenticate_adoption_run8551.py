"""Read-only authentication of actual owner outputs; never import a GPU payload."""
from pathlib import Path
import hashlib,json,os,shlex,subprocess,time,tarfile
W=Path(__file__).parent; O=W.parent/'t_bd7914e6'
R=O/'durable_k1_batch_s5_run8550_z51'
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,d):
 with p.open('w') as f:json.dump(d,f,indent=2);f.flush();os.fsync(f.fileno())
def remote(host,code):
 p=subprocess.run(['ssh','-o','BatchMode=yes',host,'python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=60)
 assert p.returncode==0,(host,p.stderr)
 return json.loads(p.stdout)
a=json.loads((O/'receipts/REFERENCE_ACTUAL_PRODUCT_run8550.json').read_text())
s=json.loads((O/'k1_batch_s5_run8550_z51.json').read_text())
assert a['pin']==s['canonical_commit']=='aa455253fa0cd8944a4148112d44c537938d40c4'
files=dict(a['files'])
for row in a['rows']:
 files[str(Path(row['artifact']).relative_to(R))]=dict(sha256=row['artifact_sha256'],bytes=row['artifact_bytes'])
for rel,v in files.items():assert H(R/rel)==v['sha256'] and (R/rel).stat().st_size==v['bytes'],rel
code='''from pathlib import Path
import json,hashlib,time
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=Path(%r);files=%r;s=%r
out={'observed':time.time(),'files':{k:dict(sha256=H(r/k),bytes=(r/k).stat().st_size) for k in files},'claim':json.loads(Path('/home/dnola/HOST_CLAIM.json').read_text()),'processes':{str(p):{'exists':Path('/proc/'+str(p)).exists(),'stat':Path('/proc/'+str(p)+'/stat').read_text() if Path('/proc/'+str(p)).exists() else None} for p in [1361352,1362343]},'model_index_sha256':H(Path(s['model_root'])/'model.safetensors.index.json')}
p=Path(s['prepared_result']);out['prepared_result']=json.loads(p.read_text());out['prepared_result_sha256']=H(p)
out['originals']={x['cell']:{'sha256':H(x['path']),'config':json.loads(Path(x['path']).read_text())} for x in s['source_rows']}
out['prepared_configs']={x['cell']:{'sha256':H(x['config']),'config':json.loads(Path(x['config']).read_text()),'reference_sha256':H(x['reference'])} for x in out['prepared_result']['rows']}
out['prepared_manifests']={k:json.loads(Path(v['config']['materialization']['run_manifest']).read_text()) for k,v in out['prepared_configs'].items()}
out['closures']={x['cell']:{k:{'sha256':H(v['path']),'path':v['path']} for k,v in json.loads((r/(x['cell'].replace('/','_')+'_SOURCE_CLOSURE.json')).read_text())['files'].items()} for x in out['prepared_result']['rows']}
out['external_identity']={k:H(s[k]) for k in ['calibration_ledger','separation_receipt']}
print(json.dumps(out))
'''%(a['source'],list(files),s)
b=remote('spark-5-work',code);atomic(W/'OWNER_REMOTE_READBACK_run8551.json',b)
assert b['files']==files
assert b['model_index_sha256']==s['intended_basis']
assert b['prepared_result_sha256']==s['prepared_result_sha256']
assert b['external_identity']['calibration_ledger']==s['calibration_ledger_sha256']
assert b['external_identity']['separation_receipt']==s['separation_receipt_sha256']
assert all(not p['exists'] for p in b['processes'].values())
archive=W/'adoption_aa455253_run8548.tar'
assert H(archive)=='f7dfc145ce77f9a7240e9a417ee8ddecf8e5ec7961da19995896b51c1b9e628e'
with tarfile.open(archive) as t:
 archive_hash={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
ns={};exec(compile((O/'canonical_reference_rebind_run8550.py').read_text(),'reviewed_rebind','exec'),ns)
rows=[]
for x in a['rows']:
 cell=x['cell'];cfg=json.loads(Path(x['config']).read_text());m=json.loads(Path(x['manifest']).read_text());d=json.loads(Path(x['receipt']).read_text());cl=json.loads((R/(cell.replace('/','_')+'_SOURCE_CLOSURE.json')).read_text())
 prep=b['prepared_configs'][cell];original=b['originals'][cell];pr=next(z for z in b['prepared_result']['rows'] if z['cell']==cell);sr=next(z for z in s['source_rows'] if z['cell']==cell)
 assert original['sha256']==sr['sha256']==cfg['materialization']['source_config_sha256']
 assert prep['sha256']==pr['config_sha256'] and prep['reference_sha256']==pr['reference_sha256']
 expected,em=ns['rebind'](prep['config'],b['prepared_manifests'][cell],runner=cfg['qtip_runner'],runner_sha=m['tiers'][0]['bindings']['qtip_runner']['sha256'],closure_sha=cl['sha256'],manifest_path=cfg['materialization']['run_manifest'],selected_sha=cfg.get('selected_source_manifest_sha256'))
 expected['materialization']['run_manifest_sha256']=H(x['manifest'])
 assert expected==cfg and em==m,'unexplained config mutation'
 assert d['config_sha256']==H(x['config']) and d['artifact_sha256']==H(x['artifact'])
 assert d['status']=='PASS' and d['basis_gate']['intended_basis']==s['intended_basis']==d['basis_gate']['index_sha256']
 assert cfg['fit_windows']==d['fit_windows']==16 and cfg['rht_seed']==d['rht_seed']==prep['config']['rht_seed']
 assert cfg['input_identity']==prep['config']['input_identity'] and cfg['training_ledger_sha256']==s['calibration_ledger_sha256']
 for key,val in cl['files'].items():
  assert b['closures'][cell][key]['sha256']==val['sha256']==s['expected_import_files'][key],key
  assert d['glm_source_closure']['files'][key]['sha256']==val['sha256']
  if not key.startswith('runtime.'):
   name=key if key.endswith('.json') else key+'.py'
   assert archive_hash['banana-smasher/src/banana_smasher/'+name]==val['sha256'],key
 assert d['build']['packed_decode']['fp16_bit_exact'] and d['build']['canonical_pack']['canonical_pack_roundtrip_exact']
 rows.append(dict(cell=cell,artifact_sha256=H(x['artifact']),artifact_bytes=Path(x['artifact']).stat().st_size,fit_windows=cfg['fit_windows'],fit_rows=d['build']['fit_rows'],rht_seed=cfg['rht_seed'],config_exact_rebind=True,canonical_and_external_imports_verified=len(cl['files']),packed_decode_owner_receipt_pass=True))
# Unit-level dispatch review without production code or GPU execution.
ns={};exec(compile((O/'pair_dispatch_run8550.py').read_text(),'reviewed_dispatch','exec'),ns)
calls=[];ns['run_pair'](['a','b'],'root',19,lambda *a:calls.append(a));assert calls==[(['a','b'],'root',19)]
for bad in [[],['a'],['a','a'],['a','b','c']]:
 try:ns['run_pair'](bad,'root',19,lambda *a:calls.append(a))
 except AssertionError:pass
 else:raise AssertionError('invalid pair admitted')
assert len(calls)==1
out=dict(status='PASS_INDEPENDENT_ACTUAL_OWNER_OUTPUT_AUTHENTICATION',observed=time.time(),runtime_pin=a['pin'],owner='t_bd7914e6',remote_root='spark-5-work:'+a['source'],files_verified=files,files_count=len(files),rows=rows,remote_receipt='OWNER_REMOTE_READBACK_run8551.json',remote_receipt_sha256=H(W/'OWNER_REMOTE_READBACK_run8551.json'),owner_receipt_sha256=H(O/'receipts/REFERENCE_ACTUAL_PRODUCT_run8550.json'),reviewed_scripts={n:H(O/n) for n in ['batch_k1_reference_run8550.py','canonical_reference_rebind_run8550.py','pair_dispatch_run8550.py','prepare_z51_run8550.py']},dispatch_checks=5,product_wall_seconds=a['product_wall_seconds'],public_batch_wall_seconds=a['batch_wall_seconds'],speedup_claim=False,quality_scope='Independently authenticated production decode/roundtrip receipts, not independently rerun numerical decode or new heldout/full-model score. Prior bounded numerical experiment gate remains separate.',new_outputs=2,no_replay=True,no_foreign_mutation=True)
atomic(W/'PRODUCTION_ADOPTION_READBACK_run8551.json',out)
print(json.dumps({k:v for k,v in out.items() if k not in ['files_verified','rows','reviewed_scripts']},indent=2))
