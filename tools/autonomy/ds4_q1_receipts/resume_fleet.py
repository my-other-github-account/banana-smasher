#!/usr/bin/env python3
"""Missing-only receipt verification across the fenced s8 cursor transition and four live hosts."""
import concurrent.futures,hashlib,json,os,shlex,subprocess,time
from pathlib import Path
D=Path(__file__).resolve().parent;STATE=D/'FLEET_CURSOR.json'
PROGRAM=r'''import json,sys,os,time,hashlib
from pathlib import Path
def verify_imports(mods, closure):
 required=('qtip_validation_bitshift','qtip_validation_ldlq','qtip_validation_math_utils','banana_smasher.qtip_kernel_decompress')
 if any(n not in mods for n in required):raise ValueError('IMPORT_MISSING_ALIAS')
 checked={}
 for name,row in mods.items():
  path=Path(row['path'])
  if not path.is_absolute() or '..' in path.parts:raise ValueError('IMPORT_PATH_INVALID: '+name)
  if name in closure['external']:
   ref=closure['external'][name];suffix=ref['suffix'];digest=ref['sha256']
  elif name=='banana_smasher' or name.startswith('banana_smasher.'):
   stem='banana-smasher/src/'+name.replace('.','/')
   candidates=[stem+'.py',stem+'/__init__.py']
   matches=[v for v in candidates if v in closure['canonical'] and str(path).endswith('/'+v)]
   if len(matches)!=1:raise ValueError('IMPORT_CANONICAL_PATH_MISMATCH: '+name)
   suffix=matches[0];digest=closure['canonical'][suffix]
  else:raise ValueError('IMPORT_UNBOUND_MODULE: '+name)
  if not str(path).endswith('/'+suffix) or row['sha256']!=digest:raise ValueError('IMPORT_IDENTITY_MISMATCH: '+name)
  checked[name]=digest
 return checked
x=json.loads(sys.argv[1]);C=Path('/home/dnola/HOST_CLAIM.json');c=json.loads(C.read_text());root=Path(x['root']);s=json.loads((root/'SHARDS.json').read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert c['task_id']==s['task_id']=='t_0640a5bf'
pid=c['controller_pid'];proc=Path('/proc',str(pid),'stat')
if c['status']=='RELEASED':
 assert c['released'] and not proc.exists()
 tp=Path(c['terminal_path']);assert sha(tp)==c['terminal_sha256']
 terminal=json.loads(tp.read_text());assert terminal['status']=='PASS' and terminal['error'] is None and terminal['prior_receipts_preserved']
 assert tp.parent==Path(x['sources'][-1]['mission'])
 ticks=c['controller_startticks']
else:
 assert c['status']=='CLAIMED' and c['expires_unix']>time.time()
 ticks=int(proc.read_text().rsplit(')',1)[1].split()[19])
assert [pid,ticks]==x['supervisor']
assert s['controller_pid']==pid and s['controller_startticks']==ticks
assert c['intended_basis']==s['intended_basis']==x['basis']==sha('/home/dnola/models/hf/DeepSeek-V4-Flash-0731/model.safetensors.index.json')
checked=[];last=None
for source in x['sources']:
 d=Path(source['mission']);p=d/'PROGRESS.json';raw=p.read_bytes();progress=json.loads(raw);accepted=progress['accepted'];assert len(accepted)>=source['start']
 if source.get('final') or c['status']=='RELEASED':
  terminal=json.loads((d/'TERMINAL.json').read_text());assert terminal['prior_receipts_preserved']
  if c['status']=='RELEASED':assert terminal['current_count']==progress['completed_cells'] and terminal['accepted']==accepted
 else:assert not (d/'TERMINAL.json').exists()
 for a in accepted[source['start']:]:
  rp=Path(a['receipt']);assert sha(rp)==a['sha256'];r=json.loads(rp.read_text());assert r['status']=='PASS' and sha(rp.parent/r['artifact'])==r['artifact_sha256']
  l,e,proj=a['cell'];assert l in s['layers'];name=f'L{l:03d}_E{e:03d}_{proj}'
  config=Path(a.get('config',str(d/(name+'.json'))));imports=Path(a.get('imports',str(d/(name+'.IMPORTS.json'))))
  assert sha(config)==r['config_sha256']
  assert r['build']['packed_decode']['fp16_bit_exact'] and r['build']['packed_decode']['runtime_check_performed'] and r['build']['canonical_pack']['canonical_pack_roundtrip_exact']
  mods=json.loads(imports.read_text());verified_imports=verify_imports(mods,x['import_closure'])
  checked.append(dict(a,config=str(config),imports=str(imports),artifact_sha256=r['artifact_sha256'],imports_sha256=sha(imports),verified_imports=verified_imports,canonical_pin=x['import_closure']['pin']))
 last=dict(mission=str(d),start=len(accepted));count=progress['completed_cells']
print(json.dumps(dict(host=x['host'],claim=c,shards=s,supervisor=[pid,ticks],checked=checked,committed_count=count,next_source=last,observed_unix=time.time())))
'''
def capture(x):
 closure_raw=(D/'IMPORT_CLOSURE.json').read_bytes()
 if hashlib.sha256(closure_raw).hexdigest()!='c64bfd8ef8933e294457f3cf088d642bcc066c7afbac5c0e485e3a11342b65b0':raise ValueError('IMPORT_CLOSURE_DIGEST_MISMATCH')
 closure=json.loads(closure_raw)
 if closure['pin']!='dcfd22711a8357e43ad6914680cb11073ecd14a0':raise ValueError('IMPORT_CLOSURE_PIN_MISMATCH')
 x=dict(x,import_closure=closure)
 p=subprocess.run(['ssh','-o','BatchMode=yes',x['host'],'python3 -c '+shlex.quote(PROGRAM)+' '+shlex.quote(json.dumps(x))],capture_output=True,text=True,timeout=180)
 assert p.returncode==0,(x['host'],p.stderr);return json.loads(p.stdout)
def save(p,obj):
 raw=(json.dumps(obj,sort_keys=True,indent=2)+'\n').encode();tmp=p.with_name(p.name+'.tmp')
 with tmp.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p);return hashlib.sha256(raw).hexdigest()
raw=STATE.read_bytes();state=json.loads(raw)
with concurrent.futures.ThreadPoolExecutor() as pool:rows=list(pool.map(capture,state['hosts']))
path=D/f'FLEET_READBACK_{time.time_ns()}.json';digest=save(path,dict(previous_cursor_sha256=hashlib.sha256(raw).hexdigest(),hosts=rows))
assert STATE.read_bytes()==raw,'CONCURRENT_CURSOR_CHANGE'
for x,r in zip(state['hosts'],rows):x['sources']=[r['next_source']]
state.update(latest_readback=str(path),latest_readback_sha256=digest,counts={r['host']:r['committed_count'] for r in rows});save(STATE,state)
hp=D/'HANDOFF_STATE.json';h=json.loads(hp.read_text());h.update(fleet_cursor=str(STATE),latest_fleet_readback=str(path),latest_fleet_readback_sha256=digest,verified_frontiers=state['counts'],verified_committed_counts=state['counts'],accepted_since_original_5665=sum(state['counts'].values())-5665,all_ranges=state['all_ranges'],hosts=rows,next_actions=['python3 '+str(Path(__file__).resolve()),'Inspect actual failing host mission TERMINAL.json and named log only on real failure; no duplicate producer','Continue eight-row UNIFORM_TIER_LEDGER.json; no sparse-as-uniform or premature ownership release'],waiting_condition='Active versus terminal producer states are in latest fleet receipt claim.status; current missions and PID identities in FLEET_CURSOR.json; never restart completed shards');save(hp,h)
print(json.dumps(dict(path=str(path),sha256=digest,new_cells_verified=sum(len(r['checked']) for r in rows),counts=state['counts'],accepted_since_original_5665=h['accepted_since_original_5665'],ownership=[dict(host=r['host'],task=r['claim']['task_id'],claim_status=r['claim']['status'],supervisor=r['supervisor'],layers=r['shards']['layers'],expires=r['claim']['expires_unix']) for r in rows]),indent=2))
