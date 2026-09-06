#!/usr/bin/env python3
"""Resume the released s4 K1 producer after a diagnosed empty fit.

No scientific policy changes: exact original configs and solver remain pinned.
Empty-fit pairs are retained as pending, never solved by a fallback. All previous
outputs and allocation bytes are preserved. One cell per admission transaction.
"""
import hashlib,json,os,pwd,shutil,subprocess,sys,time,traceback
from pathlib import Path

BASIS='98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b'
PIN='dcfd22711a8357e43ad6914680cb11073ecd14a0'
TASK='t_68e5892b'
P=Path('/home/dnola/missions/DS4_Q1_PRODUCTION_t_0640a5bf_s4')
PREV=P/'run8022-valid-only-r2'
CAN=Path('/home/dnola/missions/DS4_Q1_EFFICIENCY_t_0640a5bf/canonical')
D=Path(__file__).resolve().parent
C=Path('/home/dnola/HOST_CLAIM.json')

def missing_cells(configs,accepted,pending):
    keys=[(c['layer'],c['expert'],c['projection']) for c in configs]
    seen=[tuple(a['cell']) for a in accepted]
    assert len(set(keys))==len(keys) and len(set(seen))==len(seen),'DUPLICATE'
    assert set(seen)<=set(keys),'ACCEPTED_OUTSIDE_CONFIGS'
    assert all(36<=l<43 and 0<=e<256 and p in ('fused13','down') for l,e,p in keys),'OUT_OF_SCOPE'
    omitted=set(seen)|set(map(tuple,pending))
    return [list(k) for k in keys if k not in omitted]

def worker_credentials(uid):
    # The inherited s4 solver and cache were root-owned, unlike s8's FUSE lane.
    assert uid==0,'S4_ROOT_CACHE_IDENTITY_REQUIRED'
    return dict(user=0,group=0)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def helpers():
    # Reuse the already deployed canonical successor's fsynced save/exact CAS.
    import importlib.util
    p=D/'clean_successor.py'
    s=importlib.util.spec_from_file_location('coordination',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m

def gate(coord):
    c=json.loads(C.read_text());s=json.loads((D/'SHARDS.json').read_text())
    assert c['task_id']==s['task_id']==TASK and c['mission']==str(D)
    assert c['status']==s['status']=='CLAIMED' and c['expires_unix']>time.time()
    assert [c['controller_pid'],c['controller_startticks']]==[s['controller_pid'],s['controller_startticks']]
    assert coord.ticks(c['controller_pid'])==c['controller_startticks']
    assert c['intended_basis']==s['intended_basis']==BASIS==sha('/home/dnola/models/hf/DeepSeek-V4-Flash-0731/model.safetensors.index.json')
    return s

def worker(config):
    coord=helpers();s=gate(coord);cfg=json.loads(config.read_text())
    assert [cfg['layer'],cfg['expert'],cfg['projection']] in s['cells']
    assert cfg['geometry']==dict(K=1,L=16,V=2) and cfg['tier']=='qtip@1.00'
    sys.path.insert(0,str(CAN/'banana-smasher/src'))
    from banana_smasher.solver_qtip_profile import main_many
    try:
        main_many(config.parent,D,cfg['layer'],batch_size=1,config_paths=[config],kernel_cache_root=CAN.parent/'kernel-cache')
    finally:
        mods={n:dict(path=str(Path(m.__file__).resolve()),sha256=sha(m.__file__)) for n,m in tuple(sys.modules.items()) if getattr(m,'__file__',None) and Path(m.__file__).is_file() and (n.startswith('banana_smasher') or n.startswith('qtip_validation'))}
        coord.save(D/f"L{cfg['layer']:03d}_{config.stem}.IMPORTS.json",mods)

def supervisor():
    import fcntl
    coord=helpers();save=coord.save
    with (D/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (D/'SHARDS.json').exists() and not (D/'TERMINAL.json').exists(),'RESUME_NOT_RELAUNCH'
        old=C.read_bytes();claim=json.loads(old);sr=(P/'SHARDS.json').read_bytes();source=json.loads(sr)
        assert claim['task_id']==source['task_id']=='t_0640a5bf'
        assert claim['status']==source['status']=='RELEASED' and claim['released'] and source['released']
        assert [claim['controller_pid'],claim['controller_startticks']]==[2041695,56094019]
        assert [source['controller_pid'],source['controller_startticks']]==[2041695,56094019]
        assert coord.ticks(2041695) is None
        child=json.loads((PREV/'ACTIVE_WORKER.json').read_text());assert coord.ticks(child['pid']) is None
        assert claim['intended_basis']==source['intended_basis']==BASIS==sha('/home/dnola/models/hf/DeepSeek-V4-Flash-0731/model.safetensors.index.json')
        assert sha(claim['terminal_path'])==claim['terminal_sha256']
        term=json.loads(Path(claim['terminal_path']).read_text())
        assert term['status']=='FAIL' and term['prior_receipts_preserved'] and 'L041_E230_fused13.log' in term['error']
        assert 'empty routed fit population rows=0 mass=0.0' in (PREV/'L041_E230_fused13.log').read_text()
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'FOREIGN_GPU'
        census=subprocess.run(['pgrep','-af','s4_valid_only|s4_missing_resume|solver_qtip_profile|CHAMPION_SHARD_RUNNER'],capture_output=True,text=True).stdout
        assert not [x for x in census.splitlines() if int(x.split()[0]) not in (os.getpid(),os.getppid())],census
        auth=json.loads((D/'AUTHORIZATION.json').read_text())
        assert sha(__file__)==auth['controller_sha256'] and sha(D/'clean_successor.py')==auth['coordination_sha256']
        closure=json.loads((D/'IMPORT_CLOSURE.json').read_text());assert closure['pin']==PIN
        for name,digest in closure['canonical'].items():assert sha(CAN/name)==digest,name
        baseline=json.loads((PREV/'PROGRESS.json').read_text());accepted=list(baseline['accepted'])
        assert len(accepted)==baseline['completed_cells']==term['current_count']==3018
        pending=[[37,130,p] for p in ('fused13','down')]+[[41,230,p] for p in ('fused13','down')]
        configs=sorted((P/'configs').glob('L*/E*.json'))
        rows=[json.loads(p.read_text()) for p in configs]
        paths={(c['layer'],c['expert'],c['projection']):p for p,c in zip(configs,rows)}
        cells=missing_cells(rows,accepted,pending)
        # Both projections route by the same frozen fit population; do not infer a new fit.
        pair=[json.loads(paths[(41,230,p)].read_text()) for p in ('fused13','down')]
        for key in ('fit_capture_root','fit_windows','hessian_layer_manifest_sha256'):
            assert pair[0][key]==pair[1][key]
        assert len(rows)==3584 and all(not (P/f'solve/L{l:03d}/E{e:03d}_{p}/QTIP_SOLVE_RECEIPT.json').exists() for l,e,p in cells),'UNACCOUNTED_OUTPUT'
        mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        assert mem-4*(1<<30)>20*(1<<30) and shutil.disk_usage(D).free>8*(1<<30)+len(cells)*8*(1<<20),'CAPACITY'
        save(D/'PLAN.json',dict(cells=cells,pending_cells=pending,prior_progress=str(PREV/'PROGRESS.json'),prior_progress_sha256=sha(PREV/'PROGRESS.json'),prior_count=len(accepted),production_pin=PIN,basis=BASIS,not_clean_uniform_acceptance=True))
        save(D/'ADOPTION_PREIMAGE.json',dict(claim=claim,shards=source,claim_sha256=hashlib.sha256(old).hexdigest(),shards_sha256=hashlib.sha256(sr).hexdigest(),terminal_sha256=sha(PREV/'TERMINAL.json'),census=census))
        new=dict(task_id=TASK,owner_task_id=TASK,status='CLAIMED',state='CLAIMED',released=False,controller_pid=os.getpid(),controller_startticks=coord.ticks(os.getpid()),expires_unix=time.time()+7200,intended_basis=BASIS,canonical_git_pin=PIN,mission=str(D),receipt_path=str(D/'CLAIM_READBACK.json'))
        current=coord.raw(new);coord.cas(C,old,current)
        shard=dict(new,cells=cells,layers=sorted({c[0] for c in cells}),receipt_path=str(D/'SHARDS_READBACK.json'))
        with (D/'SHARDS.json').open('xb') as f:f.write(coord.raw(shard));f.flush();os.fsync(f.fileno())
        save(D/'CLAIM_READBACK.json',json.loads(C.read_text()));save(D/'SHARDS_READBACK.json',shard)
        save(D/'PROGRESS.json',dict(accepted=accepted,completed_cells=len(accepted),pending_cells=pending,prior_count=len(accepted)))
        print(json.dumps(new),flush=True)
        user=pwd.getpwnam('dnola');os.chown(D,user.pw_uid,user.pw_gid)
        child=None;error=None
        try:
            for layer,expert,projection in cells:
                gate(coord);assert (P/'SHARDS.json').read_bytes()==sr
                new['expires_unix']=time.time()+7200;updated=coord.raw(new);coord.cas(C,current,updated);current=updated
                mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
                assert mem-4*(1<<30)>20*(1<<30) and shutil.disk_usage(D).free>8*(1<<30),'CAPACITY'
                config=paths[(layer,expert,projection)];name=f'L{layer:03d}_{config.stem}'
                env=dict(os.environ,OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',HOME=user.pw_dir,USER='dnola',LOGNAME='dnola')
                argv=['/home/dnola/humming_env/bin/python','-u',str(Path(__file__).resolve()),'worker',str(config)]
                with (D/(name+'.log')).open('x') as log:
                    child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,**worker_credentials(os.getuid()),cwd=D)
                    save(D/'ACTIVE_WORKER.json',dict(pid=child.pid,startticks=coord.ticks(child.pid),argv=argv,cell=[layer,expert,projection]))
                    rc=child.wait()
                assert rc==0,('WORKER_EXIT',rc,str(D/(name+'.log')))
                rp=D/f'solve/L{layer:03d}/E{expert:03d}_{projection}/QTIP_SOLVE_RECEIPT.json';r=json.loads(rp.read_text())
                assert r['status']=='PASS' and sha(rp.parent/r['artifact'])==r['artifact_sha256'] and sha(config)==r['config_sha256']
                assert r['build']['packed_decode']['fp16_bit_exact'] and r['build']['packed_decode']['runtime_check_performed'] and r['build']['canonical_pack']['canonical_pack_roundtrip_exact']
                imports=D/(name+'.IMPORTS.json');coord.verify_imports(json.loads(imports.read_text()),closure)
                accepted.append(dict(cell=[layer,expert,projection],receipt=str(rp),sha256=sha(rp),config=str(config),imports=str(imports)))
                save(D/'PROGRESS.json',dict(accepted=accepted,completed_cells=len(accepted),pending_cells=pending,prior_count=3018,observed_unix=time.time()))
        except BaseException:error=traceback.format_exc()
        finally:
            if child is not None and child.poll() is None:
                save(D/'LIVE_CHILD_ERROR.json',dict(error=error,pid=child.pid));raise RuntimeError('LIVE_CHILD_PRESERVED')
            result=dict(status='FAIL' if error else 'PASS',error=error,accepted=accepted,current_count=len(accepted),pending_cells=pending,prior_receipts_preserved=True,uniform_complete=False)
            save(D/'TERMINAL.json',result)
            assert (P/'SHARDS.json').read_bytes()==sr
            shard.update(status='RELEASED',released=True);save(D/'SHARDS.json',shard)
            new.update(status='RELEASED',state='RELEASED',released=True,terminal_path=str(D/'TERMINAL.json'),terminal_sha256=sha(D/'TERMINAL.json'))
            coord.cas(C,current,coord.raw(new));save(D/'RELEASE_READBACK.json',json.loads(C.read_text()))
            print(json.dumps(result),flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker':worker(Path(sys.argv[2]))
    else:supervisor()
