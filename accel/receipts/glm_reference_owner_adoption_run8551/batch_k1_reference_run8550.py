"""Persistent K1 missing-only eight-cell production; sealed bindings reused."""
from pathlib import Path
import os,sys,json,hashlib,time
process_started=time.time()
from canonical_reference_rebind_run8550 import rebind
from pair_dispatch_run8550 import run_pair
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
s=json.loads(Path(sys.argv[1]).read_text());r=Path(s['output']);sys.path.insert(0,str(Path(s['repo_root'])/'banana-smasher/src'))
import torch
from banana_smasher import solver_qtip_profile as sp
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert H(Path(__file__).with_name('qtip_rings.json'))==s['source_ring_asset_sha256']
c=json.loads(Path('/home/dnola/HOST_CLAIM.json').read_text());assert c['stage']==r.name and c['task_id']==s['task_id'] and c['state']=='CLAIMED' and c['expiry_unix']>time.time()
assert H(Path(s['model_root'])/'model.safetensors.index.json')==s['intended_basis']
assert H(s['prepared_result'])==s['prepared_result_sha256'];prepared=json.loads(Path(s['prepared_result']).read_text());assert prepared['status']=='PASS_K1_MISSING_BATCH_PREPARATION' and prepared['K']==1
assert [x['cell'] for x in prepared['rows']]==s['prepared_cells']
prepared['rows']=[x for x in prepared['rows'] if x['cell'] in s['cells']]
assert [x['cell'] for x in prepared['rows']]==s['cells']
torch.set_num_threads(4);torch.manual_seed(0);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
import shutil
rows=[];paths=[];started=time.time()
assert len(prepared['rows'])==2 and s['canonical_commit']=='aa455253fa0cd8944a4148112d44c537938d40c4'
for row in prepared['rows']:
 cp=Path(row['config']);assert H(cp)==row['config_sha256'] and H(row['reference'])==row['reference_sha256'];cfg=json.loads(cp.read_text());assert cfg['geometry']['K']==1 and f"L{cfg['layer']:03d}/E{cfg['expert']:03d}_{cfg['projection']}"==row['cell']
 # Republish only the code-bound config/manifest; preserve source, fit and reference identities.
 from banana_smasher.glm_qtip_source_adapter import capture_source_closure
 qpath=Path(sp.__file__).with_name('qtip_runner.py');qsha=H(qpath);qv=sp._load_public_qtip_runner(qpath,qsha);qv.QTIP=sp._config_path(cfg,'qtip_root')
 bitshift,ldlq,math_utils,kernel_decode=qv.load_official_qtip()
 closure=capture_source_closure(qv,dict(bitshift=bitshift,ldlq=ldlq,math_utils=math_utils,kernel_decompress=kernel_decode))
 for name,expected in s['expected_import_files'].items():
  assert closure['files'][name]['sha256']==expected,('IMPORT_DRIFT',name)
  if not name.startswith('runtime.'):
   assert Path(closure['files'][name]['path']).is_relative_to(Path(s['repo_root'])/'banana-smasher/src')
 manifest_path=Path(cfg['materialization']['run_manifest']);assert H(manifest_path)==cfg['materialization']['run_manifest_sha256']
 manifest=json.loads(manifest_path.read_text());fresh_manifest=r/manifest_path.name
 selected=Path(cfg['model_root'])/'SELECTED_TENSORS.json'
 cfg,manifest=rebind(cfg,manifest,runner=str(qpath),runner_sha=qsha,closure_sha=closure['sha256'],manifest_path=str(fresh_manifest),selected_sha=H(selected) if selected.exists() else None)
 sp._atomic_json(fresh_manifest,manifest);cfg['materialization']['run_manifest_sha256']=H(fresh_manifest)
 cp=r/cp.name;sp._atomic_json(cp,cfg);sp._atomic_json(r/(row['cell'].replace('/','_')+'_SOURCE_CLOSURE.json'),closure)
 sp._verify_basis(cfg,r);cell=row['cell'];dest=r/'solve'/cell/'QTIP_SOLVE_RECEIPT.json' 
 assert not dest.exists(),'fresh missing-only batch refuses existing cell; resume by receipts instead'
 paths.append(cp)
from banana_smasher.qtip_batch_controller import main_batch
assert len(set(json.loads(p.read_text())['layer'] for p in paths))==1
start=time.perf_counter();batch_result=run_pair(paths,r,cfg['layer'],main_batch);torch.cuda.synchronize();batch_wall=time.perf_counter()-start
sp._atomic_json(r/'PAIR_MEASUREMENT.json',dict(batch_wall_seconds=batch_wall,timing_scope='two-cell public main_batch wall; not individual-cell latency',result=batch_result,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
for cp in paths:
 cfg=json.loads(cp.read_text());cell=f"L{cfg['layer']:03d}/E{cfg['expert']:03d}_{cfg['projection']}";dest=r/'solve'/cell/'QTIP_SOLVE_RECEIPT.json'
 wall=None
 d=json.loads(dest.read_text());artifact=dest.with_name('QTIP_UNIT.pt');assert d['status']=='PASS' and H(artifact)==d['artifact_sha256']
 rows.append(dict(cell=cell,wall_seconds=wall,receipt=str(dest),receipt_sha256=H(dest),artifact=str(artifact),artifact_sha256=H(artifact),phase_seconds=d.get('build',{}).get('phase_seconds')))
 sp._atomic_json(r/'PROGRESS.json',dict(status='PASS_CELL',rows=rows,accepted=len(rows),expected=len(prepared['rows']),elapsed=time.time()-started,observed=time.time()))
sp._atomic_json(r/'RESULT.json',dict(status='PASS_K1_BOUNDED_BATCH',K=1,rows=rows,accepted=len(rows),elapsed=time.time()-started,not_uniform_pre=True,source_basis=s['intended_basis'],persistent_process=True,ended=time.time(),process_wall_seconds=time.time()-process_started,canonical_commit=s['canonical_commit']))
