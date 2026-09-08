import os,sys,json,time,pathlib,hashlib,subprocess,importlib.util,shutil,traceback,pwd
P=pathlib.Path
B=P('/home/dnola/missions/DS4_EMPTY_FIT_t_68e5892b'); R=B/'s8-K3-L011DOWN3-run8495'; CAN=B/'canonical-39da421e-run8495'; C=P('/home/dnola/HOST_CLAIM.json'); PLAN=B/'PLAN_TRIAL_run8495.json'
spec=importlib.util.spec_from_file_location('coord','/home/dnola/missions/DS4_EMPTY_FIT_t_abba7b95/run8067-clean-L030-L034/clean_successor.py');coord=importlib.util.module_from_spec(spec);spec.loader.exec_module(coord)
load=lambda p:json.loads(P(p).read_text());sha=coord.sha;save=coord.save
plan=load(PLAN);TASK=plan['task'];BASIS=plan['basis'];PIN=plan['pin'];cells=plan['cells']
def ref(p):return dict(path=str(p),sha256=sha(p),bytes=P(p).stat().st_size)
def gate():
 c=load(C);s=load(R/'SHARDS.json');assert c['task_id']==TASK and c['mission']==str(R) and not c['released'] and c['expires_unix']>time.time();assert s['cells']==cells and s['K']==3 and s['plan_sha256']==sha(PLAN)
 assert sha('/home/dnola/models/hf/DeepSeek-V4-Flash-0731/model.safetensors.index.json')==BASIS==c['intended_basis']==s['intended_basis']
def mods():return {n:dict(path=str(P(m.__file__).resolve()),sha256=sha(m.__file__)) for n,m in tuple(sys.modules.items()) if getattr(m,'__file__',None) and P(m.__file__).is_file() and (n.startswith('banana_smasher') or n.startswith('qtip_validation'))}
def worker():
 gate();sys.path.insert(0,str(CAN/'banana-smasher/src'));from banana_smasher import solver_qtip_profile as solver
 from banana_smasher.qtip_batch_controller import main_batch
 from banana_smasher.qtip_rings import resolve_qtip_ring,qtip_ring_manifest
 ring=qtip_ring_manifest(resolve_qtip_ring('3.00'));assert ring['components'][0]['geometry']==dict(L=16,K=3,V=2)
 cps=[];configs=[]
 for row in plan['source_rows']:
  z=row['config'];assert sha(z['path'])==z['sha256'];cfg=load(z['path']);assert [cfg['layer'],cfg['expert'],cfg['projection']]==row['cell'] and row['cell'] in cells
  for key in ['reusable_capture_binding','reusable_population']:
   z=row['source_input'][key];assert sha(z['path'])==z['sha256']
  manifest=load(cfg['hessian_layer_manifest']);assert manifest['basis_sha256']==BASIS and manifest['windows']==22
  for member in manifest['members']:
   for key in ['capture','capture_done']:assert sha(member[key]['path'])==member[key]['sha256']
  mp0=cfg['materialization']['run_manifest'];assert sha(mp0)==cfg['materialization']['run_manifest_sha256'];run=load(mp0)
  runner=CAN/'banana-smasher/src/banana_smasher/qtip_runner.py';run['canonical_git_pin']=PIN;run['tiers'][0].update(name=ring['tier'],ring=ring);run['tiers'][0]['bindings']['qtip_runner'].update(ref(runner))
  l,e,p=row['cell'];mp=R/f'L{l:03d}_E{e:03d}_{p}.MANIFEST.json';save(mp,run)
  cfg.update(geometry=ring['components'][0]['geometry'],backend=ring['components'][0]['backend'],bpw=ring['bpw'],tier=ring['tier'],codebook=ring['codebook'],aot=ring['aot'],block_ldl_unitwise=True,qtip_runner=str(runner),exact_solver='banana_smasher.qtip_viterbi@'+PIN)
  cfg['materialization'].update(run_manifest=str(mp),run_manifest_sha256=sha(mp),source_config_sha256=row['config']['sha256'],qtip_ring_bpw=ring['bpw'])
  cp=R/f'L{l:03d}_E{e:03d}_{p}.json';save(cp,cfg);cps.append(cp);configs.append(cfg)
 closure=load(B/'canonical-4921456b-run8384/IMPORT_CLOSURE.json');closure['canonical']=plan['canonical'];g=solver._read_qtip_config(cps[0]);path,digest=solver._declared_public_qtip_runner(g);runner=solver._load_public_qtip_runner(path,digest);runner.QTIP=solver._config_path(g,'qtip_root');runner.load_official_qtip();coord.verify_imports(mods(),closure);save(R/'PRE_SOLVE_IMPORTS.json',mods())
 import torch
 f,t=torch.cuda.mem_get_info();save(R/'CUDA_PREFLIGHT.json',dict(free_bytes=f,total_bytes=t,reserve_bytes=4<<30,conservative_batch_estimate_bytes=8<<30));assert f-(4<<30)>(8<<30),'BATCH_CAPACITY_REFUSAL'
 gate();begin=time.time();results=main_batch(cps,R,11,kernel_cache_root=B/'s8-K3-kernel-cache');end=time.time();coord.verify_imports(mods(),closure);save(R/'POST_SOLVE_IMPORTS.json',mods());accepted=[]
 for row,cfg,cp,r in zip(plan['source_rows'],configs,cps,results,strict=True):
  l,e,p=row['cell'];rp=R/f'solve/L{l:03d}/E{e:03d}_{p}/QTIP_SOLVE_RECEIPT.json';a=rp.parent/r['artifact'];assert r['status']=='PASS' and sha(a)==r['artifact_sha256'] and sha(cp)==r['config_sha256'];assert r['build']['packed_decode']['fp16_bit_exact'] and r['build']['packed_decode']['runtime_check_performed'] and r['build']['canonical_pack']['canonical_pack_roundtrip_exact']
  accepted.append(dict(K=3,cell=row['cell'],artifact=ref(a),receipt=ref(rp),config=ref(cp),imports=ref(R/'POST_SOLVE_IMPORTS.json'),source_input=row['source_input'],started=begin,accepted=time.time(),qualification=cfg['fit_qualification'],producer_pin=PIN,producer_identity=dict(pin=PIN,publication=plan['publication']),scope='bounded-three-cell-trial'))
 with (R/'ADMISSIONS.jsonl').open('x') as f:
  for a in accepted:f.write(json.dumps(a)+'\n')
  f.flush();os.fsync(f.fileno())
 save(R/'PROGRESS.json',dict(completed_cells=len(accepted),target_cells=3,K=3,started=begin,observed=time.time(),main_batch_wall_seconds=end-begin,accepted_artifact_bytes=sum(a['artifact']['bytes'] for a in accepted),last_cell=cells[-1]))
def supervisor():
 old=C.read_bytes();prior=load(C);assert prior==plan['prior_claim'] and prior['released'] and prior['task_id']==TASK and coord.ticks(prior['controller_pid']) is None
 prev=P(prior['mission']);assert load(prev/'SHARDS.json')==plan['prior_shards'];assert sha(prior['terminal_path'])==prior['terminal_sha256'];assert load(prior['terminal_path'])['status']=='PASS';aw=load(prev/'ACTIVE_WORKER.json');assert coord.ticks(aw['pid']) is None
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'FOREIGN_GPU'
 assert not R.exists(),'RESUME_EXISTING_ATTEMPT_NOT_REPLAY'
 assert len(cells)==len(set(map(tuple,cells)))==3
 for p in B.glob('*K3*/ADMISSIONS.jsonl'):
  assert not {tuple(json.loads(l)['cell']) for l in p.read_text().splitlines()}.intersection(map(tuple,cells)),'ALREADY_ACCEPTED'
 for name,d in plan['canonical'].items():assert sha(CAN/name)==d,name
 assert sha('/home/dnola/models/hf/DeepSeek-V4-Flash-0731/model.safetensors.index.json')==BASIS
 mem=int(next(l.split()[1] for l in P('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024;free=shutil.disk_usage(B).free;assert mem-(4<<30)>(96<<30) and free>(6<<30),('CAPACITY',mem,free)
 R.mkdir();user=pwd.getpwnam('dnola');os.chown(R,user.pw_uid,user.pw_gid);begin=time.time();save(R/'PREFLIGHT.json',dict(started=begin,mem_available_bytes=mem,peak_estimate_bytes=96<<30,reserve_bytes=4<<30,disk_free_bytes=free,storage_estimate_bytes=2<<30,prior_claim=prior,plan=ref(PLAN),no_replay=True))
 c=dict(task_id=TASK,owner_task_id=TASK,status='CLAIMED',state='CLAIMED',released=False,mission=str(R),intended_basis=BASIS,controller_pid=os.getpid(),controller_startticks=coord.ticks(os.getpid()),expires_unix=time.time()+86400,receipt_path=str(R/'CLAIM_READBACK.json'));current=coord.raw(c);coord.cas(C,old,current);save(R/'CLAIM_READBACK.json',load(C));s=dict(c,K=3,cells=cells,plan_sha256=sha(PLAN),execution_owner='macmini.local',logical_owner='bs09',operation='bounded-missing-main_batch-trial')
 with (R/'SHARDS.json').open('xb') as f:f.write(coord.raw(s));f.flush();os.fsync(f.fileno())
 save(R/'SHARDS_READBACK.json',load(R/'SHARDS.json'));child=None;error=None;code=None
 try:
  gate();env=dict(os.environ,HOME=user.pw_dir,USER='dnola',LOGNAME='dnola',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PATH='/home/dnola/humming_env/bin:'+os.environ['PATH'])
  with (R/'SOLVE.log').open('x') as log:
   child=subprocess.Popen(['/home/dnola/humming_env/bin/python','-u',__file__,'worker'],stdout=log,stderr=subprocess.STDOUT,user=user.pw_uid,group=user.pw_gid,env=env,cwd=R);save(R/'ACTIVE_WORKER.json',dict(pid=child.pid,startticks=coord.ticks(child.pid),started=time.time()))
   code=child.wait();assert code==0,('WORKER_FAILED',code,str(R/'SOLVE.log'))
 except BaseException:error=traceback.format_exc()
 finally:
  assert child is None or child.poll() is not None,'LIVE_CHILD_PRESERVED'
  term=dict(status='PASS' if not error else 'FAIL',exit_code=code,error=error,child_dead=True,observed=time.time(),whole_wall_seconds=time.time()-begin,progress=load(R/'PROGRESS.json') if (R/'PROGRESS.json').exists() else None,qualification='production trial only; no speedup claim, no full row qualification; cold only, no warm replay')
  save(R/'TERMINAL.json',term);s.update(status='RELEASED',state='RELEASED',released=True);coord.cas(R/'SHARDS.json',coord.raw(dict(c,K=3,cells=cells,plan_sha256=sha(PLAN),execution_owner='macmini.local',logical_owner='bs09',operation='bounded-missing-main_batch-trial')),coord.raw(s));c.update(status='RELEASED',state='RELEASED',released=True,terminal_path=str(R/'TERMINAL.json'),terminal_sha256=sha(R/'TERMINAL.json'));coord.cas(C,current,coord.raw(c));save(R/'RELEASE_READBACK.json',load(C));print(json.dumps(term),flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker()
 else:supervisor()
