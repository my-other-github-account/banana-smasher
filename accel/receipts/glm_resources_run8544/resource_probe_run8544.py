import json,os,hashlib,time
from pathlib import Path
R=Path('/dev/shm/t_1269dc5f'); O=R/'resources_run8544'
C=json.loads(Path('/home/dnola/HOST_CLAIM.json').read_text());assert C['pid']==os.getppid() and C['task_id']=='t_1269dc5f'
rows=[]
for root in [Path('/dev/shm/t_ebcba52e_ldlq_run8503/cache/kernels'),Path('/dev/shm/t_ebcba52e_ldlq_run8503/cache/triton')]:
 for p in root.rglob('*.json'):
  if p.name!='_persistent_prefix_viterbi_generic.json':continue
  d=json.loads(p.read_text()); ttgir=p.with_suffix('.ttgir'); cubin=p.with_suffix('.cubin')
  rows.append(dict(path=str(p),metadata=d,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),ttgir_header=ttgir.read_text()[:600] if ttgir.exists() else None,cubin_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest() if cubin.exists() else None))
d=dict(rows=rows,scope='cached compiler specializations; not all established as executed; no solver replay',epoch=time.time())
with (O/'RESOURCES.json').open('x') as f:json.dump(d,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(d))
