import pathlib,json,hashlib,time
P=pathlib.Path
R=P('/home/dnola/missions/DS4_EMPTY_FIT_t_68e5892b/s8-K3-L012DOWN46-resident-run8531')
def load(p):return json.loads(P(p).read_text())
def sha(p):return hashlib.sha256(P(p).read_bytes()).hexdigest()
rows=[json.loads(l) for l in (R/'ADMISSIONS.jsonl').read_text().splitlines()]
modules={};science={};checks=[]
def verify(p,d):
 assert sha(p)==d,(p,d)
 science[str(p)]=d
for row in rows:
 c=load(row['config']['path']);s=load(row['receipt']['path'])
 assert row['cell']==[c['layer'],c['expert'],c['projection']]==[s['layer'],s['expert'],s['projection']]
 assert s['status']==s['build']['status']=='PASS' and s['build']['packed_decode']['runtime_check_performed'] and s['build']['packed_decode']['fp16_bit_exact']
 assert s['basis_gate']['index_sha256']==s['basis_gate']['intended_basis']=='98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b'
 idx=P(c['model_root'])/'model.safetensors.index.json'
 if str(idx) not in science:verify(idx,s['basis_gate']['intended_basis'])
 assert s['rht_seed']==c['rht_seed']
 for key in ['reusable_capture_binding','reusable_population']:
  z=row['source_input'][key]
  if z['path'] not in science:verify(z['path'],z['sha256'])
 for p,d in [(c['hessian_layer_manifest'],c['hessian_layer_manifest_sha256']),(c['materialization']['run_manifest'],c['materialization']['run_manifest_sha256'])]:
  if p not in science:verify(p,d)
 imports=load(row['imports']['path'])
 for name,z in imports.items():
  if name in modules:assert modules[name]==z,(name,z,modules[name])
  else:verify(z['path'],z['sha256']);modules[name]=z
 checks.append(dict(cell=row['cell'],status=s['status'],fit_windows=s['fit_windows'],producer_packed_conformance=True,seed=s['rht_seed']))
for p in R.glob('PRE_SOLVE_IMPORTS_*.json'):
 pre=load(p);post=load(R/p.name.replace('PRE_','POST_'))
 assert all(post.get(n)==z for n,z in pre.items())
print(json.dumps(dict(root=str(R),observed=time.time(),checks=checks,modules=modules,scientific_file_hashes=science,scope='producer receipt and physical science/import binding authentication, not independent decode or heldout scoring')))
