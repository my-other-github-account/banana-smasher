import json,subprocess,pathlib,shlex,hashlib,os,time,shutil
W=pathlib.Path(__file__).parent
runs=['grouped_run8544r2','grouped_quality_run8545','grouped_phase_probe_run8545','reference_ldl_run8545']
for attempt in range(60):
 cmd="from pathlib import Path; p=Path('/dev/shm/t_1269dc5f/reference_ldl_run8545/TERMINAL.json'); print(p.read_text() if p.exists() else 'LIVE')"
 t=subprocess.check_output(['ssh','spark-6','python3 -c '+shlex.quote(cmd)],text=True,timeout=30).strip()
 if t!='LIVE': break
 time.sleep(10)
else: raise RuntimeError('finite collector timeout, remote executor untouched')
for run in runs:
 remote='/dev/shm/t_1269dc5f/'+run;local=W/'receipts'/run
 cmd="import pathlib,hashlib,json; r=pathlib.Path(%r); files=[p for p in r.rglob('*') if p.is_file() and p.suffix in ('.json','.pt','.log') and 'kernels' not in p.parts]; print(json.dumps([dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]))"%remote
 rows=json.loads(subprocess.check_output(['ssh','spark-6','python3 -c '+shlex.quote(cmd)],text=True,timeout=60))
 size=sum(r['bytes'] for r in rows);assert size<512<<20;assert shutil.disk_usage(W).free>size+(4<<30)
 local.mkdir(parents=True,exist_ok=True)
 names=W/(run+'_files.txt');names.write_text('\n'.join(r['path'] for r in rows)+'\n')
 subprocess.run(['rsync','-a','--files-from='+str(names),'spark-6:'+remote+'/',str(local)+'/'],check=True)
 for row in rows:
  p=local/row['path'];assert p.stat().st_size==row['bytes'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
 ack=W/'receipts'/(run+'_PHYSICAL_ACK.json')
 with ack.open('w') as f:json.dump(dict(remote=remote,files=rows,total_bytes=size,verified=True),f,indent=2);f.flush();os.fsync(f.fileno())
 print(run,len(rows),size,'authenticated',flush=True)
