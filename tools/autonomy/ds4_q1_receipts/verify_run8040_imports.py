#!/usr/bin/env python3
"""Close only the existing 28-cell import receipts; no replay or cursor mutation."""
import ast, concurrent.futures, hashlib, json, os, shlex, subprocess
from pathlib import Path
D=Path(__file__).resolve().parent
src=ast.parse((D/'resume_fleet.py').read_text())
program=next(ast.literal_eval(n.value) for n in src.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROGRAM' for t in n.targets))
closure_raw=(D/'IMPORT_CLOSURE.json').read_bytes()
assert hashlib.sha256(closure_raw).hexdigest()=='c64bfd8ef8933e294457f3cf088d642bcc066c7afbac5c0e485e3a11342b65b0'
closure=json.loads(closure_raw)
receipt=D/'FLEET_READBACK_1788685240573178000.json'
assert hashlib.sha256(receipt.read_bytes()).hexdigest()=='f8ce23983a8e7f0f0f5da15e494dd5b785bbe93e0427023e90cb0dc0d8902361'
old=json.loads(receipt.read_text());state=json.loads((D/'FLEET_CURSOR.json').read_text())
remote=program.split('checked=[];last=None')[0]+'''
checked=[]
for a in x['checked']:
 p=Path(a['imports']);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==a['imports_sha256']
 assert sha(a['receipt'])==a['sha256']
 r=json.loads(Path(a['receipt']).read_text());assert sha(a['config'])==r['config_sha256']
 mods=json.loads(raw);verified=verify_imports(mods,x['import_closure'])
 assert r['build']['packed_decode']['source_sha256']==verified['banana_smasher.qtip_kernel_decompress']
 checked.append(dict(cell=a['cell'],receipt=a['receipt'],receipt_sha256=a['sha256'],imports=str(p),imports_sha256=a['imports_sha256'],modules=mods,verified_imports=verified))
print(json.dumps(dict(host=x['host'],claim=c,shards=s,supervisor=[pid,ticks],checked=checked,observed_unix=time.time())))
'''
def capture(pair):
 host,row=pair;x=dict(host,checked=row['checked'],import_closure=closure)
 p=subprocess.run(['ssh','-o','BatchMode=yes',x['host'],'python3 -c '+shlex.quote(remote)+' '+shlex.quote(json.dumps(x))],capture_output=True,text=True,timeout=120)
 if p.returncode:raise RuntimeError((x['host'],p.stderr))
 return json.loads(p.stdout)
with concurrent.futures.ThreadPoolExecutor() as pool:
 rows=list(pool.map(capture,[(h,next(r for r in old['hosts'] if r['host']==h['host'])) for h in state['hosts']]))
count=sum(len(r['checked']) for r in rows);assert count==28
result=dict(status='PASS',closed_existing_cells=count,newly_solved_cells=0,cursor_advanced=False,canonical_production_pin=closure['pin'],closure_sha256=hashlib.sha256(closure_raw).hexdigest(),prior_receipt=str(receipt),hosts=rows)
p=D/'IMPORT_CORRECTION_run8042.json';raw=(json.dumps(result,sort_keys=True,indent=2)+'\n').encode()
with p.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),status='PASS',closed_existing_cells=count,host_counts={r['host']:len(r['checked']) for r in rows}),indent=2))
