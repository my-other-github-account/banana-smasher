import pathlib,json,hashlib,os,shutil,subprocess,sys
root=pathlib.Path('/dev/shm/t_012f1c90');name=sys.argv[1]
claim=json.loads(pathlib.Path('/home/dnola/HOST_CLAIM.json').read_text());r=root/name
assert claim['owner']=='t_012f1c90' and claim['mission']==str(r)
assert (r/'TERMINAL.json').exists() and not pathlib.Path('/proc',str(claim['pid'])).exists()
child=json.loads((r/'CHILD.json').read_text());assert not pathlib.Path('/proc',str(child['pid'])).exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
dest=pathlib.Path('/home/dnola/missions/t_012f1c90')/(name+'_seal');assert not dest.exists()
files=[p for p in r.rglob('*') if p.is_file()];files += [root/n for n in sys.argv[2:]]
needed=sum(p.stat().st_size for p in files);free=shutil.disk_usage(dest.parent).free
assert free-needed>(1<<30)+(64<<20),(free,needed)
dest.mkdir();rows=[]
for p in files:
 rel=p.relative_to(root);q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True);data=p.read_bytes();sha=hashlib.sha256(data).hexdigest()
 with q.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 assert hashlib.sha256(q.read_bytes()).hexdigest()==sha
 rows.append(dict(path=str(rel),bytes=len(data),sha256=sha))
summary={}
if (r/'QUALITY.json').exists():
 quality=json.loads((r/'QUALITY.json').read_text())['rows'];passed=[x['metrics']['qtip_hyb']['sse_ratio_vs_true_vq']<=1.0001+1e-12 for x in quality]
 assert len(quality)==12
 times={}
 for a in ['B1','B2','C1','C2']:
  base=r
  rr=json.loads((base/a/'RESULT.json').read_text())['rows'];times[a]={phase:sum(x['seconds'] for x in rr if x['phase']==phase) for phase in ['setup','warm']}
 summary=dict(passed=sum(passed),count=len(passed),limit=1.0001,quality=[dict(cell=x['cell'],arm=x['arm'],phase=x['phase'],ratio=x['metrics']['qtip_hyb']['sse_ratio_vs_true_vq']) for x in quality],times=times,speed_ratio={phase:sum(times[a][phase] for a in ['B1','B2'])/sum(times[a][phase] for a in ['C1','C2']) for phase in ['setup','warm']},timing_scope='Same-pin flagFalse to down-only flagTrue interleaved ABBA; included full public build with shared warm caches. Not production/cold-JIT timing.')
manifest=dict(files=rows,count=len(rows),bytes=needed,pre_free=free,post_free=shutil.disk_usage(dest).free,claim=claim,controller_dead=True,last_child_dead=True,cuda_compute_pids=[],results=summary,completion=False,adoption=False)
with (dest/'PRESERVATION.json').open('x') as f:json.dump(manifest,f,indent=2);f.flush();os.fsync(f.fileno())
fd=os.open(dest,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2));print('SHA256',hashlib.sha256((dest/'PRESERVATION.json').read_bytes()).hexdigest())
