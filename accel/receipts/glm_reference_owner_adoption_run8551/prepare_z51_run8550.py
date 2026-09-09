from gate_run8550 import gate,ssh,W
from native_rest_import_run8226 import atomic
import json,ast,importlib,hashlib

gate()
t=(W/'materialize_product_run8523.py').read_text()
probe=ast.literal_eval(next(n.value for n in ast.parse(t).body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='probe' for x in n.targets)))
prep='k1_prepare_s5_run8550_z50';stage='k1_batch_s5_run8550_z51'
d=ssh('spark-5-work','STAGE='+repr(prep)+'\n'+probe)
assert len(d['result']['rows'])==2
atomic(W/'receipts/K1_PREPARE_CONSUMED_run8550_z50.json',d)
a=json.loads((W/(prep+'.json')).read_text());old=json.loads((W/'k1_batch_s5_run8527_z11.json').read_text());s=dict(a)
for k in ['canonical_commit','estimated_peak_bytes','repo_root','planned_write_bytes','planned_output_bytes','expected_import_files','expected_archive_files']:s[k]=old[k]
s.update(run_id=8550,output='/dev/shm/'+stage,prepared_result='/home/dnola/missions/CLEAN_t_bd7914e6/'+prep+'/RESULT.json',prepared_result_sha256=d['result_sha256'],prepared_cells=[r['cell'] for r in d['result']['rows']])

source=json.loads((W/'receipts/CANDIDATE_SOURCE_STAGED_run8546.json').read_text())
assert source['canonical_commit']=='aa455253fa0cd8944a4148112d44c537938d40c4'
s['canonical_commit']=source['canonical_commit'];s['repo_root']=source['root']
s['expected_archive_files']=[dict(path=k,**v) for k,v in source['manifest'].items() if k.startswith('banana-smasher/src/')]
for key in s['expected_import_files']:
 if not key.startswith('runtime.'):
  name=key if key.endswith('.json') else key+'.py'
  s['expected_import_files'][key]=source['manifest']['banana-smasher/src/banana_smasher/'+name]['sha256']

assert s['prepared_cells']==s['cells'];assert not (W/(stage+'.json')).exists();atomic(W/(stage+'.json'),s)
for name in ['claim_k1_batch_s5_run8527_z11.py','deploy_k1_batch_z11_run8527.py','mirror_k1_durable_run8527_z11.py']:
 text=(W/name).read_text().replace('k1_prepare_s5_run8527_z10','PRESERVED_PREP')
 text=text.replace('run8527','run8550').replace('z11','z51').replace('Z11','Z51').replace("'run_id':8527","'run_id':8550").replace("['run_id']==8527","['run_id']==8550").replace('len(accepted)==48','len(accepted)==2')
 text=text.replace('PRESERVED_PREP',prep)
 if name.startswith('claim_'):text=text.replace('batch_k1_l1_run8523.py','batch_k1_reference_run8550.py')
 if name.startswith('deploy_'):
  text=text.replace("names=[stage+'.json',", "names=['batch_k1_reference_run8550.py','canonical_reference_rebind_run8550.py','pair_dispatch_run8550.py',stage+'.json',")
 p=W/name.replace('run8527','run8550').replace('z11','z51');assert not p.exists();ast.parse(text);p.write_text(text)
t=(W/'authority_k1_z50_run8550.py').read_text().replace(repr([prep]),repr([stage]))
t=t.replace('current_gate();',"current_gate();assert (W/'receipts/K1_PREPARE_CONSUMED_run8550_z50.json').exists();")
p=W/'authority_k1_z51_run8550.py';assert not p.exists();ast.parse(t);p.write_text(t)
auth=importlib.import_module('authority_k1_z51_run8550').gate(stage,'spark-5-work',s['intended_basis']);atomic(W/('ALLOCATION_'+stage+'.json'),auth)
print('READY',stage,len(s['cells']),hashlib.sha256((W/(stage+'.json')).read_bytes()).hexdigest())
