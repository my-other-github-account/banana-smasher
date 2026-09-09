"""Independent read-only actual-product authentication, no GPU imports/replay."""
from pathlib import Path
import hashlib,json,os,subprocess,shlex,sys,tarfile
W=Path(__file__).parent; O=W.parent/'t_bd7914e6'; R=W/'owner_z71_run8560'
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(name,obj):
 p=W/name
 with p.open('x') as f:json.dump(obj,f,indent=2);f.flush();os.fsync(f.fileno())
ackp=O/'receipts/K1_BATCH_DURABLE_ACK_s5_run8561_z71.json';specp=O/'k1_batch_s5_run8561_z71.json'
a=json.loads(ackp.read_text());s=json.loads(specp.read_text());pin='c769d086701769b08fdc85d5191afc2aa321e3e7'
assert a['status']=='PASS_DURABLE_K1_BATCH' and a['count']==a['new_solves']==2 and a['recomputed_cells']==0
assert a['cells']==['L021/E193_fused13','L021/E194_fused13']
assert s['canonical_commit']==pin==a['source_terminal']['canonical_commit']
assert H(R/'RESULT.json')==a['terminal_sha256']
assert a['claim']['state']==a['claim']['status']=='RELEASED' and a['claim']['pid']==1529726 and a['claim']['start_ticks']==9260591
files={str(p.relative_to(R)):dict(sha256=H(p),bytes=p.stat().st_size) for p in R.rglob('*') if p.is_file()}
code='''from pathlib import Path
import json,hashlib,time
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=%r;r=Path(s['output']);files=%r
out=dict(observed=time.time(),files={k:dict(sha256=H(r/k),bytes=(r/k).stat().st_size) for k in files},claim=json.loads(Path('/home/dnola/HOST_CLAIM.json').read_text()),processes={str(p):dict(exists=Path('/proc/'+str(p)).exists(),stat=Path('/proc/'+str(p)+'/stat').read_text() if Path('/proc/'+str(p)).exists() else None) for p in [1529726,1530716]},model_index_sha256=H(Path(s['model_root'])/'model.safetensors.index.json'))
p=Path(s['prepared_result']);out['prepared_result']=json.loads(p.read_text());out['prepared_result_sha256']=H(p)
out['originals']={x['cell']:dict(sha256=H(x['path']),config=json.loads(Path(x['path']).read_text())) for x in s['source_rows']}
out['prepared_configs']={x['cell']:dict(sha256=H(x['config']),config=json.loads(Path(x['config']).read_text()),reference_sha256=H(x['reference'])) for x in out['prepared_result']['rows']}
out['prepared_manifests']={k:json.loads(Path(v['config']['materialization']['run_manifest']).read_text()) for k,v in out['prepared_configs'].items()}
out['closures']={x['cell']:{k:dict(sha256=H(v['path']),path=v['path']) for k,v in json.loads((r/(x['cell'].replace('/','_')+'_SOURCE_CLOSURE.json')).read_text())['files'].items()} for x in out['prepared_result']['rows']}
out['external_identity']={k:H(s[k]) for k in ['calibration_ledger','separation_receipt']}
print(json.dumps(out))
'''%(s,list(files))
p=subprocess.run(['ssh','spark-5-work','python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=90);assert p.returncode==0,p.stderr
b=json.loads(p.stdout);put('OWNER_REMOTE_READBACK_run8560.json',b)
assert b['files']==files and all(not p['exists'] for p in b['processes'].values())
assert b['model_index_sha256']==s['intended_basis'] and b['prepared_result_sha256']==s['prepared_result_sha256']
for k in ['calibration_ledger','separation_receipt']:assert b['external_identity'][k]==s[k+'_sha256']
archive=W/'conformance_pin.tar';assert H(archive)=='6970da84f48d3550768f0b8aed99eaac25f27efe79b8f093a4c3aa1e2d0aa116'
with tarfile.open(archive) as t:archive_hash={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
expected_imports=json.loads((W/'source/accel/receipts/glm_device_conformance_run8560/EXPECTED_IMPORTS.json').read_text())['files']
sys.path.insert(0,str(O))
from canonical_device_rebind_run8561 import rebind
rows=[]
for cell in a['cells']:
 stem=cell.replace('/','_');cfgp=R/(stem+'_K1.json');mp=R/('K1_'+stem+'_MANIFEST.json');cl=json.loads((R/(stem+'_SOURCE_CLOSURE.json')).read_text());cfg=json.loads(cfgp.read_text());m=json.loads(mp.read_text());rp=R/'solve'/cell/'QTIP_SOLVE_RECEIPT.json';d=json.loads(rp.read_text());ap=rp.with_name('QTIP_UNIT.pt')
 prep=b['prepared_configs'][cell];pr=next(x for x in b['prepared_result']['rows'] if x['cell']==cell);sr=next(x for x in s['source_rows'] if x['cell']==cell)
 assert b['originals'][cell]['sha256']==sr['sha256']==cfg['materialization']['source_config_sha256']
 assert prep['sha256']==pr['config_sha256'] and prep['reference_sha256']==pr['reference_sha256']
 ec,em=rebind(prep['config'],b['prepared_manifests'][cell],runner=cfg['qtip_runner'],runner_sha=m['tiers'][0]['bindings']['qtip_runner']['sha256'],closure_sha=cl['sha256'],manifest_path=cfg['materialization']['run_manifest'],selected_sha=cfg.get('selected_source_manifest_sha256'))
 ec['materialization']['run_manifest_sha256']=H(mp)
 assert ec==cfg and em==m,'unexplained config/manifest mutation'
 assert cfg['packed_conformance_on_device'] is True and m['canonical_commit']==pin
 assert d['config_sha256']==H(cfgp) and d['artifact_sha256']==H(ap)
 assert d['basis_gate']['index_sha256']==d['basis_gate']['intended_basis']==s['intended_basis']
 assert cfg['fit_windows']==d['fit_windows']==16 and cfg['rht_seed']==d['rht_seed']==prep['config']['rht_seed']
 assert cfg['input_identity']==prep['config']['input_identity'] and cfg['training_ledger_sha256']==s['calibration_ledger_sha256']
 for k,v in cl['files'].items():
  assert b['closures'][cell][k]['sha256']==v['sha256']==expected_imports[k]==s['expected_import_files'][k]
  assert d['glm_source_closure']['files'][k]['sha256']==v['sha256']
  if not k.startswith('runtime.') or k=='runtime.kernel_decompress':assert archive_hash['banana-smasher/src/banana_smasher/'+Path(v['path']).name]==v['sha256']
 assert d['build']['packed_decode']['fp16_bit_exact'] and d['build']['packed_decode']['conformance_comparison']=='device' and d['build']['canonical_pack']['canonical_pack_roundtrip_exact']
 for rel in [str(ap.relative_to(R)),str(rp.relative_to(R))]:assert H(Path(a['destination'])/rel)==files[rel]['sha256']
 rows.append(dict(cell=cell,artifact_sha256=H(ap),artifact_bytes=ap.stat().st_size,fit_windows=16,fit_rows=d['build']['fit_rows'],rht_seed=d['rht_seed'],config_exact_rebind=True,physical_imports_verified=len(cl['files']),packed_decode_owner_receipt_pass=True))
put('PRODUCTION_ADOPTION_READBACK_run8560.json',dict(status='PASS_INDEPENDENT_ACTUAL_OWNER_OUTPUT_AUTHENTICATION',runtime_pin=pin,rows=rows,files_count=len(files),files_verified=files,owner_ack_sha256=H(ackp),owner_spec_sha256=H(specp),remote_readback_sha256=H(W/'OWNER_REMOTE_READBACK_run8560.json'),no_replay=True,production_speedup_claim=False,full_model_quality_claim=False))
print(json.dumps(dict(status='PASS',rows=rows,files=len(files)),indent=2))
