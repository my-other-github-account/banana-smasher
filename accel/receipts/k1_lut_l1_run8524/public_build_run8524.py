import os,sys,json,time,subprocess,hashlib
from pathlib import Path
R=Path('/dev/shm/t_182fbc9d');O=Path(os.environ['OUT']);PIN='53f05876cb02a95e2accdce8a0bc23efc7edd5f6'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
if '--child' in sys.argv:
 arm=sys.argv[2];sys.path.insert(0,str(R/'public_run8524/banana-smasher/src'))
 import torch
 from banana_smasher import solver_qtip_profile as sp
 from banana_smasher.glm_qtip_source_adapter import capture_source_closure
 torch.set_num_threads(8);D=O/arm;D.mkdir();phases=[]
 for expert,projection in [(242,'down'),(243,'down'),(242,'fused13')]:
  cell=f'E{expert}_{projection}';C=D/cell;C.mkdir()
  source=R/f'pair_pointers/warm1/configs/E{expert}_down.json';cfg=json.loads(source.read_text());cfg['viterbi_lut_l1_retention']=True;cfg['projection']=projection;cfg['rht_seed']=sp._canonical_rht_seed(cfg['rht_seed_material'],4,expert,projection);runner=Path(sp.__file__).with_name('qtip_runner.py');cfg['qtip_runner']=str(runner)
  manifest=json.loads(Path(cfg['materialization']['run_manifest']).read_text());manifest['cells']=[f'L004/{cell}'];manifest['canonical_commit']=PIN;manifest['tiers'][0]['bindings']['qtip_runner']=dict(path=str(runner),sha256=sha(runner));mp=C/'MANIFEST.json';put(mp,manifest);cfg['materialization']['run_manifest']=str(mp);cfg['materialization']['run_manifest_sha256']=sha(mp)
  qv=sp._load_public_qtip_runner(runner,sha(runner));qv.QTIP=Path(cfg['qtip_root']);bits,ldlq,math,kd=qv.load_official_qtip();closure=capture_source_closure(qv,dict(bitshift=bits,ldlq=ldlq,math_utils=math,kernel_decompress=kd));put(C/'IMPORT_CLOSURE.json',closure);cfg['glm_source_closure_sha256']=closure['sha256'];cp=C/'CONFIG.json';put(cp,cfg)
  dest=C/'build';dest.mkdir();put(dest/'SHARDS.json',dict(intended_basis=cfg['input_identity']['model_index']['sha256'],rows=[dict(owner='t_182fbc9d',layer=4,experts=[expert],projections=[projection],operation='candidate_only_L1_retention')]))
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();result=sp.main(cp,dest,4,profile_mode=False);torch.cuda.synchronize();phases.append(dict(cell=cell,public_seconds=time.perf_counter()-start,result=result,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()));put(D/'RESULT.json',dict(phases=phases,pin=PIN))
 raise SystemExit()
assert json.loads(Path('/home/dnola/HOST_CLAIM.json').read_text())['pid']==os.getppid()
rows=[]
for arm in ['public']:
 start=time.perf_counter()
 with (O/f'{arm}.log').open('w') as f:
  p=subprocess.Popen([sys.executable,'-u',__file__,'--child',arm],stdout=f,stderr=subprocess.STDOUT);put(O/'ACTIVE_CHILD.json',dict(pid=p.pid,startticks=int(Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19]),argv=[sys.executable,'-u',__file__,'--child',arm]));rc=p.wait()
 rows.append(dict(arm=arm,returncode=rc,process_seconds=time.perf_counter()-start));put(O/'RESULT.json',dict(pin=PIN,scope='candidate-only actual down242/down243/distinct fused242, retained controls, historical noninterleaved comparison; shared JIT/source caches',rows=rows));assert rc==0
