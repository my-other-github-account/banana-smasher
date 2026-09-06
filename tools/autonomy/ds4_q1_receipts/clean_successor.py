#!/usr/bin/env python3
"""Missing clean-compatible L030-L034 Q1 continuation from a frozen plan."""
import fcntl,hashlib,json,os,pwd,shutil,subprocess,sys,time,traceback
from pathlib import Path
D=Path(__file__).resolve().parent
C=Path('/home/dnola/HOST_CLAIM.json')
SOURCE=Path('/home/dnola/missions/QTIP1_CHAMPION_t_ds4q1_spark_8')
DEPLOY=Path('/home/dnola/missions/DS4_Q1_RECOVERY_t_0640a5bf/run8004-narrowed')
CLEAN=Path('/home/dnola/missions/DS4_EMPTY_FIT_t_46b61159/run8048-clean-inputs-r2')
BASIS='98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b'
PIN='dcfd22711a8357e43ad6914680cb11073ecd14a0'
TASK='t_abba7b95'
PREV=Path('/home/dnola/missions/DS4_EMPTY_FIT_t_46b61159/run8050-clean-L029-r2')
POPULATION='b8e512c966acd58db2e3bbdda477a2520ed35dc5086dd8309878c420174fbcbd'
def boundary_gate(claim, source, predecessor, terminal, ticks):
 assert claim['task_id']==source['task_id']=='t_0640a5bf'
 assert predecessor['task_id']=='t_46b61159'
 assert all(x['intended_basis']==BASIS for x in (claim,source,predecessor)), 'BASIS_MISMATCH'
 assert all(x['status']=='RELEASED' for x in (claim,source,predecessor)), 'NOT_RELEASED'
 assert claim['released'] and terminal['status']=='PASS', 'NO_POSITIVE_TERMINAL'
 assert (terminal['pid'],terminal['startticks'])==(predecessor['controller_pid'],predecessor['controller_startticks'])
 assert all(ticks(x['controller_pid']) is None for x in (claim,source,predecessor)), 'PREDECESSOR_LIVE'


def plan_cells(plan):
 assert plan['production_pin']==PIN and plan['clean_population_sha256']==POPULATION
 cells=plan['cells']; unique=set(map(tuple,cells))
 assert cells and len(unique)==len(cells), 'DUPLICATE_OR_EMPTY'
 assert all(30<=l<=34 and 0<=e<=255 and p in ('fused13','down') for l,e,p in cells), 'OUT_OF_SCOPE'
 assert not unique.intersection(map(tuple,plan['retained_only_qualified_cells'])), 'QUALIFIED_NOT_MISSING'
 assert not plan['empty_cells'], 'EMPTY_FIT_REQUIRES_SEPARATE_FENCE'
 return cells

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def raw(d):return (json.dumps(d,sort_keys=True,indent=2)+'\n').encode()
def save(p,d):
 p=Path(p);b=raw(d);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def ticks(pid):
 p=Path('/proc')/str(pid)/'stat'
 return int(p.read_text().rsplit(')',1)[1].split()[19]) if p.exists() else None
def cas(p,old,new):
 with Path(str(p)+'.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);assert p.read_bytes()==old,'CAS_PREIMAGE_CHANGED'
  with p.open('r+b') as f:f.write(new);f.truncate();f.flush();os.fsync(f.fileno())
 assert p.read_bytes()==new

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

def worker():
 sys.path.insert(0,str(DEPLOY/'canonical/banana-smasher/src'))
 import importlib
 from banana_smasher.solver_qtip_profile import main
 cfg=Path(sys.argv[2]);conf=json.loads(cfg.read_text())
 claim=json.loads(C.read_text());assert claim['task_id']==TASK and claim['mission']==str(D) and claim['status']=='CLAIMED'
 assert sha(Path(conf['model_root'])/'model.safetensors.index.json')==BASIS
 try:main(cfg,D,int(cfg.stem[1:4]),profile_mode=False,kernel_cache_root=SOURCE/'kernel-cache')
 finally:
  imports={n:dict(path=str(Path(m.__file__).resolve()),sha256=sha(m.__file__)) for n,m in tuple(sys.modules.items()) if getattr(m,'__file__',None) and Path(m.__file__).is_file() and (n.startswith('banana_smasher') or n.startswith('qtip_validation'))}
  save(D/(cfg.stem+'.IMPORTS.json'),imports)

def supervisor():
 assert os.getuid()==0
 with (D/'RUN.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (D/'TERMINAL.json').exists() and not (D/'SHARDS.json').exists(),'NEVER_REPLAY'
  old=C.read_bytes();prior=json.loads(old);sraw=(SOURCE/'SHARDS.json').read_bytes();s=json.loads(sraw)
  prev_raw=(PREV/'SHARDS.json').read_bytes();prev=json.loads(prev_raw)
  term_raw=(PREV/'TERMINAL.json').read_bytes();term=json.loads(term_raw)
  boundary_gate(prior,s,prev,term,ticks)
  assert prev['controller_pid']==3328236 and prev['controller_startticks']==57635976
  assert sha(prior['terminal_path'])==prior['terminal_sha256']
  assert json.loads(Path(prior['terminal_path']).read_text())['status']=='PASS'
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'FOREIGN_GPU'
  census=subprocess.run(['pgrep','-af','CHAMPION_SHARD_RUNNER|solver_qtip_profile|clean_q1_bounded.py|clean_q1_l029.py|clean_successor.py'],capture_output=True,text=True).stdout
  collisions=[x for x in census.splitlines() if int(x.split()[0]) not in [os.getpid(),os.getppid()]]
  assert not collisions,collisions
  assert sha('/home/dnola/models/hf/DeepSeek-V4-Flash-0731/model.safetensors.index.json')==BASIS==prior['intended_basis']==s['intended_basis']
  plan=json.loads((D/'NEXT_CLEAN_PLAN.json').read_text());targets=plan_cells(plan)
  auth=json.loads((D/'AUTHORIZATION.json').read_text())
  assert sha(D/'NEXT_CLEAN_PLAN.json')==auth['plan_sha256']
  assert sha(Path(__file__))==auth['controller_sha256']
  assert all(not (D/f'solve/L{l:03d}/E{e:03d}_{p}').exists() for l,e,p in targets), 'NEVER_REPLAY'
  save(D/'ADOPTION_PREIMAGE.json',dict(claim=prior,source_shards=s,claim_sha256=hashlib.sha256(old).hexdigest(),source_shards_sha256=hashlib.sha256(sraw).hexdigest(),fence='original and L029 supervisors dead and SHARDS RELEASED; all outputs preserved',predecessor_shards=prev,predecessor_terminal_sha256=hashlib.sha256(term_raw).hexdigest(),authorization=auth,census=census))
  claim=dict(task_id=TASK,owner_task_id=TASK,status='CLAIMED',state='CLAIMED',released=False,controller_pid=os.getpid(),controller_startticks=ticks(os.getpid()),expires_unix=time.time()+7200,intended_basis=BASIS,canonical_git_pin=PIN,mission=str(D),receipt_path=str(D/'CLAIM_READBACK.json'),scope='frozen NEXT_CLEAN_PLAN L030-L034 only; same K1 tier; all prior output preserved')
  current=raw(claim);cas(C,old,current);save(D/'CLAIM_READBACK.json',json.loads(C.read_text()))
  shard=dict(task_id=TASK,owner_task_id=TASK,intended_basis=BASIS,layers=sorted(set(l for l,e,p in targets)),cells=targets,controller_pid=os.getpid(),controller_startticks=ticks(os.getpid()),status='CLAIMED',source_released_shards_sha256=hashlib.sha256(sraw).hexdigest(),receipt_path=str(D/'SHARDS_READBACK.json'))
  with (D/'SHARDS.json').open('xb') as f:f.write(raw(shard));f.flush();os.fsync(f.fileno())
  save(D/'SHARDS_READBACK.json',json.loads((D/'SHARDS.json').read_text()))
  print(json.dumps(dict(pid=os.getpid(),startticks=ticks(os.getpid()),claim=str(D/'CLAIM_READBACK.json'))),flush=True)
  accepted=[];error=None;child=None
  try:
   assert sha(D/'IMPORT_CLOSURE.json')=='c64bfd8ef8933e294457f3cf088d642bcc066c7afbac5c0e485e3a11342b65b0'
   dep=json.loads((DEPLOY/'DEPLOY.json').read_text());assert dep['pin']==PIN
   for name,h in dep['files'].items():assert sha(DEPLOY/'canonical'/name)==h,name
   manifests={}
   for layer in sorted(set(l for l,e,p in targets)):
    manifest=CLEAN/f'L{layer:03d}.CAPTURE_BINDING.json';m=json.loads(manifest.read_text())
    assert sha(manifest)==auth['capture_bindings'][str(layer)]
    assert m['basis_sha256']==BASIS and m['windows']==22 and m['population_manifest_sha256']==POPULATION
    manifests[layer]=(manifest,m)
   assert sha('/home/dnola/missions/DS4_EMPTY_FIT_t_46b61159/run8047-clean/CLEAN_FIT_MANIFEST.json')==POPULATION
   save(D/'MISSING.json',dict(cells=targets,plan_sha256=sha(D/'NEXT_CLEAN_PLAN.json'),no_accepted_replay=True))
   user=pwd.getpwnam('dnola');os.chown(D,user.pw_uid,user.pw_gid)
   for layer,expert,proj in targets:
    manifest,m=manifests[layer]
    assert C.read_bytes()==current and (SOURCE/'SHARDS.json').read_bytes()==sraw and (PREV/'SHARDS.json').read_bytes()==prev_raw
    assert json.loads((D/'SHARDS.json').read_text())==shard
    claim['expires_unix']=time.time()+7200;next_claim=raw(claim);cas(C,current,next_claim);current=next_claim
    mem=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
    free=shutil.disk_usage(D).free
    assert mem-4*(1<<30)>20*(1<<30) and free>8*(1<<30), ('PREFLIGHT',mem,free)
    save(D/'PREFLIGHT.json',dict(memory_available=mem,peak_estimate=20*(1<<30),reserve=4*(1<<30),storage_free=free,source_pin=PIN,cell=[layer,expert,proj]))
    original=SOURCE/f'inputs/configs/L{layer:03d}/E{expert:03d}_{proj}.json';cfg=json.loads(original.read_text());name=f'L{layer:03d}_E{expert:03d}_{proj}'
    prior_receipt=SOURCE/f'solve/L{layer:03d}/E{expert:03d}_{proj}/QTIP_SOLVE_RECEIPT.json';prior_digest=sha(prior_receipt)
    old_manifest=Path(cfg['materialization']['run_manifest']);assert sha(old_manifest)==cfg['materialization']['run_manifest_sha256']
    run=json.loads(old_manifest.read_text());runner=DEPLOY/'canonical/banana-smasher/src/banana_smasher/qtip_runner.py'
    run['canonical_git_pin']=PIN;run['tiers'][0]['bindings']['qtip_runner'].update(path=str(runner),bytes=runner.stat().st_size,sha256=sha(runner))
    mp=D/(name+'.MANIFEST.json');save(mp,run)
    cfg.update(qtip_runner=str(runner),exact_solver='banana_smasher.qtip_viterbi@'+PIN,fit_capture_root=str(CLEAN/'captures'),fit_windows=22,hessian_layer_manifest=str(manifest),hessian_layer_manifest_sha256=sha(manifest))
    cfg['materialization'].update(run_manifest=str(mp),run_manifest_sha256=sha(mp),source_config_sha256=sha(original))
    cfg['fit_population_manifest']=dict(path='/home/dnola/missions/DS4_EMPTY_FIT_t_46b61159/run8047-clean/CLEAN_FIT_MANIFEST.json',sha256=m['population_manifest_sha256'])
    cfg['fit_qualification']='Clean22 whole-window exclusion; ordinary routed-weight fit, no old Hessian reuse; not complete uniform artifact or score.'
    cp=D/(name+'.json');save(cp,cfg)
    assert not (D/f'solve/L{layer:03d}/E{expert:03d}_{proj}').exists()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',HOME=user.pw_dir,USER='dnola',LOGNAME='dnola',PYTHONPATH=str(DEPLOY/'canonical/banana-smasher/src'))
    argv=['/home/dnola/humming_env/bin/python','-u',str(Path(__file__).resolve()),'worker',str(cp)]
    with (D/(name+'.log')).open('w') as log:
     child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,user=user.pw_uid,group=user.pw_gid,cwd=D)
     save(D/(name+'.LAUNCH.json'),dict(pid=child.pid,startticks=ticks(child.pid),argv=argv))
     rc=child.wait(timeout=1800)
    assert rc==0,('CHILD_EXIT',rc,str(D/(name+'.log')))
    rp=D/f'solve/L{layer:03d}/E{expert:03d}_{proj}/QTIP_SOLVE_RECEIPT.json';r=json.loads(rp.read_text())
    assert r['status']=='PASS' and sha(rp.parent/r['artifact'])==r['artifact_sha256'] and r['config_sha256']==sha(cp)
    assert r['build']['packed_decode']['fp16_bit_exact'] and r['build']['packed_decode']['runtime_check_performed'] and r['build']['canonical_pack']['canonical_pack_roundtrip_exact']
    mods=json.loads((D/(name+'.IMPORTS.json')).read_text());verify_imports(mods,json.loads((D/'IMPORT_CLOSURE.json').read_text()))
    assert sha(prior_receipt)==prior_digest
    accepted.append(dict(cell=[layer,expert,proj],receipt=str(rp),sha256=sha(rp),artifact_sha256=r['artifact_sha256'],original_receipt=str(prior_receipt),original_sha256=prior_digest))
    save(D/'PROGRESS.json',dict(accepted=accepted,pid=os.getpid(),startticks=ticks(os.getpid())))
  except BaseException:error=traceback.format_exc()
  finally:
   if child is not None and child.poll() is None:
    save(D/'LIVE_CHILD_ERROR.json',dict(error=error,pid=child.pid,startticks=ticks(child.pid)));raise RuntimeError('LIVE_CHILD_PRESERVED_NO_RELEASE')
   result=dict(status='PASS' if not error and len(accepted)==len(targets) else 'FAIL',error=error,accepted=accepted,pid=os.getpid(),startticks=ticks(os.getpid()),pin=PIN,scope='L030-L034 missing clean-fit replacements only; incomplete uniform tier',finished=time.time())
   save(D/'TERMINAL.json',result)
   shard['status']='RELEASED';save(D/'SHARDS.json',shard)
   assert (SOURCE/'SHARDS.json').read_bytes()==sraw
   cas(C,current,old);save(D/'RELEASE_READBACK.json',dict(restored_exactly=True,claim_sha256=sha(C),source_shards_sha256=hashlib.sha256(sraw).hexdigest()))
   print(json.dumps(result),flush=True)
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='worker':worker()
 else:supervisor()
