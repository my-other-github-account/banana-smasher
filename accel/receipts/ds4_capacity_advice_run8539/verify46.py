import pathlib,json,hashlib,time
r=pathlib.Path('/home/dnola/missions/DS4_EMPTY_FIT_t_68e5892b/s8-K3-L012DOWN46-resident-run8531')
def load(p):return json.loads(p.read_text())
def bind(p):
 b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
rows=[json.loads(l) for l in (r/'ADMISSIONS.jsonl').read_text().splitlines()];assert len(rows)==46 and len({tuple(x['cell']) for x in rows})==46
checks=[];receipts=[]
for x in rows:
 for k in ['artifact','receipt','config','imports']:
  d=x[k];actual=bind(pathlib.Path(d['path']));assert actual==d,(k,d,actual);checks.append(actual)
 receipts.append(load(pathlib.Path(x['receipt']['path'])))
result={'root':str(r),'observed':time.time(),'admissions':rows,'bindings':checks,'receipts':receipts,'terminal':load(r/'TERMINAL.json'),'whole_work':{p.name:load(p) for p in r.glob('RESIDENT_WHOLE_WORK_*.json')},'batch_progress':{p.name:load(p) for p in r.glob('BATCH_PROGRESS_*.json')},'claim':load(r/'CLAIM_READBACK.json'),'shards':load(r/'SHARDS.json'),'dead':{str(pid):not pathlib.Path(f'/proc/{pid}').exists() for pid in [1013593,1015063]},'artifact_bytes':sum(x['artifact']['bytes'] for x in rows)}
assert result['terminal']['status']=='PASS' and all(result['dead'].values())
print(json.dumps(result))
