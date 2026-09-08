import json,pathlib,base64,hashlib,shutil
R=pathlib.Path('closure_run8503');m=json.loads((R/'MANIFEST.json').read_text())
p=pathlib.Path('/Users/macmini/.hermes/kanban/boards/banana-smasher/workspaces/t_bd7914e6/receipts/ACCEL_ORIGINAL_METADATA_run8502.json')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='9c209cb4d8bacac17278dc99c77c9b4ed3db4e0a6dba83ad4c005ba2d31c8114'
a=json.loads(p.read_text());out=pathlib.Path('selected_run8503');out.mkdir(exist_ok=True)
def sha(b):return hashlib.sha256(b).hexdigest()
def put(n,x):(out/n).write_text(json.dumps(x,indent=2))
for f in a['files']:
 b=base64.b64decode(f['data']);assert sha(b)==f['sha256'];(out/pathlib.Path(f['path']).name).write_bytes(b)
index=next(r for r in m['files'] if r['source_path'].endswith('model.safetensors.index.json'));shutil.copyfile(R/index['archive_path'],out/'model.safetensors.index.json')
parents={}
for s in a['shards']:
 b=base64.b64decode(s['header_data']);assert sha(b)==s['header_sha256'];name=pathlib.Path(s['source_path']).name; hn=name+'.header';(out/hn).write_bytes(b)
 parents[name]=dict(header_path=hn,header_sha256=sha(b[8:]),bytes=s['parent_total_bytes'])
rows=[];payloads={}
for c in m['components']:
 name=pathlib.Path(c['source_path']).name;parents[name]
 rows.append(dict(key=c['tensor'],parent=c['source_path'],parent_bytes=parents[name]['bytes'],header_sha256=parents[name]['header_sha256'],offset=c['offset'],bytes=c['bytes'],dtype=c['dtype'],shape=c['shape'],data_sha256=c['sha256']))
 payloads[c['tensor']]=pathlib.Path(c['archive_path']).name;shutil.copyfile(R/c['archive_path'],out/payloads[c['tensor']])
put('DESCRIPTOR.json',dict(source_basis=m['intended_basis'],rows=rows,provenance=dict(archive_sha256='a08ab89be8af74c98b1a17110eafbd5a0b40269916f706475b53bd2bbd8a8e34',metadata_sha256=sha(p.read_bytes()))))
put('SELECTED_TENSORS.json',dict(schema='banana-smasher.selected-tensor-source.v1',index_sha256=m['intended_basis'],config_sha256=sha((out/'config.json').read_bytes()),descriptor_path='DESCRIPTOR.json',descriptor_sha256=sha((out/'DESCRIPTOR.json').read_bytes()),payloads=payloads,parents=parents))
import sys
sys.path.insert(0,'canonical/banana-smasher/src')
from banana_smasher.selected_tensor_source import SelectedTensorSource
s=SelectedTensorSource(out)
for c in m['components']:
 row=dict(name=c['tensor'],dtype=c['dtype'],shape=c['shape']);assert sha(s.read(s.root/pathlib.Path(c['source_path']).name,row))==c['sha256']
print('PASS authenticated selected source six raw components',s.receipt)
