"""Independent NumPy/fSum adjudication of authorized one-ordinal DS4 output."""
import os,sys,json,hashlib,fcntl,time,traceback,math,subprocess
from pathlib import Path
TASK='t_ebcba52e';CLAIM=Path('/home/dnola/HOST_CLAIM.json');ROOT=Path('/dev/shm/t_ebcba52e_ds4_k3_sixcell_verify_run8474');RUN=Path('/dev/shm/t_ebcba52e_ds4_k3_sixcell_output_run8474');PREFIX=Path('/dev/shm/t_ebcba52e_ds4_prefix_run8461_a2')
BASIS='98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def put(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
 with tmp.open('w') as f:json.dump(d,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
assert (RUN/'RESULT.json').exists();result=json.loads((RUN/'RESULT.json').read_text());assert result['state']=='SEALED_BOUNDED_DS4_PAIRED_ORDINAL0_OUTPUT'
ROOT.mkdir(exist_ok=False);identity=dict(task_id=TASK,pid=os.getpid(),pgid=os.getpgid(0),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),argv=sys.argv,receipt=str(ROOT/'IDENTITY.json'),script_sha256=sha(__file__))
put(ROOT/'IDENTITY.json',identity)
with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);raw=CLAIM.read_bytes();c=json.loads(raw)
 assert hashlib.sha256(raw).hexdigest()==sys.argv[1] and c['task_id']==TASK and c['stage']=='t_ebcba52e_ds4_k3_sixcell_output_run8474_terminal_retained'
 assert not Path('/proc/'+str(c['pid'])).exists()
 assert sha('/home/dnola/models/hf/DeepSeek-V4-Flash-0731/model.safetensors.index.json')==BASIS
 mem=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'));assert mem>6*(1<<30)
 put(ROOT/'CLAIM_PREIMAGE.json',c);put(ROOT/'SHARDS.json',dict(intended_basis=BASIS,rows=[dict(owner=TASK,ordinal=0,positions=1024,operation='independent_output_verification',receipt=str(ROOT/'IDENTITY.json'))]))
 claim=dict(identity,owner=TASK,state='CLAIMED',status='CLAIMED',host='spark-6',stage=ROOT.name,mission=str(ROOT),expiry_unix=time.time()+7200,source_model_index_sha256=BASIS)
 assert CLAIM.read_bytes()==raw;put(CLAIM,claim)
try:
 import numpy as np
 import torch
 torch.set_num_threads(1)
 science=json.loads((RUN/'SCIENCE.json').read_text());assert science['ordinal']==0 and science['window']==28 and science['positions']==1024 and science['suffix_layers']==[9,42] and science['intended_basis']==BASIS
 assert sha(RUN/'SCIENCE.json')==result['science_sha256'] and sha(PREFIX/'RESULT.json')==result['prefix_result_sha256']
 bindings=json.loads((RUN/'REPLACEMENTS.json').read_text())
 owner=json.loads(Path('/dev/shm/t_ebcba52e_ds4_ordinal0_inputs_run8456/source/selection.json').read_text());owner={(r['layer'],r['expert'],r['projection']):r for r in owner}
 counts={};stage_shas={}
 for L in range(9,43):
  p=RUN/f'L{L:03d}_INJECTIONS.json';rows=json.loads(p.read_text());keys=[(r['layer'],r['expert'],r['projection']) for r in rows]
  assert len(keys)==len(set(keys))==512 and set(keys)=={(L,e,p) for e in range(256) for p in ['down','fused13']}
  for r in rows:
   src=owner[(L,r['expert'],r['projection'])];assert r['artifact_sha256']==src['artifact_sha256'] and r['injected_bf16_sha256']==src['injected_bf16_sha256']
  for arm in science['arms']:
   rec=json.loads((RUN/'frontier'/arm/f'L{L:03d}.json').read_text());assert rec['ordinal']==0 and rec['window']==28 and rec['positions']==1024 and rec['science_sha256']==sha(RUN/'SCIENCE.json') and rec['injections_sha256']==sha(p)
   assert sha(rec['path'])==rec['sha256'] and rec['consumed_sha256']==sha(RUN/(arm+f'_L{L:03d}_CONSUMED.json'))
   stage_shas[arm+'/'+str(L)]=rec['sha256']
  counts[str(L)]=len(rows)
 for arm in science['arms']:
  expected={(r['layer'],r['expert'],r['projection']):r for r in bindings['rows'][arm]};assert len(expected)==6
  seen=set()
  for L in range(9,43):
   consumed=json.loads((RUN/(arm+f'_L{L:03d}_CONSUMED.json')).read_text())
   assert len(consumed)==(3 if L in [9,26] else 0)
   for r in consumed:
    key=(L,r['expert'],r['projection']);e=expected[key];assert r['artifact_sha256']==e['sha256'] and sha(e['path'])==e['sha256'];seen.add(key)
  assert seen==set(expected)
 assert len(stage_shas)==102
 pb=json.loads((RUN/'PREFIX_BINDING.json').read_text());assert pb['frontier_sha256']==sha(PREFIX/'frontier/L008.pt')=='fa9f1cdf6f7075da8003efb66fc2d27e5faf2cccc682affe5412de1e3b52b91a'
 assert (RUN/'FROZEN_LIMITS.json').stat().st_mtime_ns<(RUN/'candidate1_READOUT.pt').stat().st_mtime_ns
 teacher_path=RUN/'frozen/t8192_win28.pt';assert sha(teacher_path)=='561753481a1e08aee88e28f5fa0c6e727f4af679494c39679e87ed5189e2653d'
 teacher=torch.load(teacher_path,map_location='cpu',weights_only=True);idx=teacher['idx'][:1024].numpy();tp=teacher['logprob'][:1024].numpy()
 assert hashlib.sha256(idx.tobytes()).hexdigest()=='1a9bf47028f0196f6457c2581a390c1f2995a7d4487dbb760b14ec565e5ac7f2'
 assert hashlib.sha256(tp.tobytes()).hexdigest()=='d3a6f36608300c18355394bca6cbf0a8b52a724e571d6e5a3cca3cc4dcde1a2f'
 def norm(x):
  x=x.astype(np.float64);m=np.max(x,axis=1,keepdims=True);return x-m-np.log(np.sum(np.exp(x-m),axis=1,keepdims=True))
 metrics={}
 for arm in science['arms']:
  rec=result['metrics'][arm];assert sha(rec['path'])==rec['sha256'];d=torch.load(rec['path'],map_location='cpu',weights_only=True);meta=d['metadata']
  assert meta['teacher_sha256']==sha(teacher_path) and meta['science_sha256']==sha(RUN/'SCIENCE.json') and meta['top1_semantics']=='teacher_support_firstmax_token_vs_candidate_fullvocab_argmax'
  assert np.array_equal(d['idx'].numpy(),idx) and d['q_lp_at_ref'].dtype==torch.float16
  q=d['q_lp_at_ref'].numpy();winner=d['q_argmax'].numpy();kl=[];matches=0
  for start in range(0,1024,32):
   p=norm(tp[start:start+32]);qq=norm(q[start:start+32]);v=np.sum(np.exp(p)*(p-qq),axis=1);assert np.isfinite(v).all() and (v>=0).all();kl.extend(v.tolist());t=idx[start:start+32][np.arange(len(p)),p.argmax(1)];matches+=int(np.sum(t==winner[start:start+32]))
  value=math.fsum(kl)/1024;assert abs(value-rec['kl'])<=1e-12 and matches==rec['top1_matches']
  assert np.max(np.abs(np.asarray(kl)-d['kl_per_position'].numpy()))<=1e-11
  metrics[arm]=dict(kl=value,top1_matches=matches,positions=1024,support=8192,scorer_delta=value-rec['kl'],readout_sha256=rec['sha256'])
 limits=json.loads((RUN/'FROZEN_LIMITS.json').read_text());repeat=abs(metrics['baseline1']['kl']-metrics['baseline2']['kl']);tolerance=max(1e-8,5*repeat)
 assert tolerance<=1e-5 and abs(tolerance-limits['tolerance'])<=1e-12 and limits['candidate_readout_not_executed']
 delta=metrics['candidate1']['kl']-(metrics['baseline1']['kl']+metrics['baseline2']['kl'])/2
 passed=delta<=tolerance and metrics['candidate1']['top1_matches']>=min(metrics[a]['top1_matches'] for a in ['baseline1','baseline2']);assert passed==result['quality_pass']
 put(ROOT/'RESULT.json',dict(state='INDEPENDENT_PAIRED_ORDINAL0_PASS' if passed else 'INDEPENDENT_PAIRED_ORDINAL0_RED',identity=identity,producer_result_sha256=sha(RUN/'RESULT.json'),metrics=metrics,delta=delta,tolerance=tolerance,injection_counts=counts,frontiers_verified=len(stage_shas),frontier_sha256=stage_shas,quality_pass=passed,full_model_equivalence=False,production_promotion=False,scope=result['scope']))
except BaseException as e:put(ROOT/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
finally:
 with Path('/home/dnola/HOST_CLAIM.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);c=json.loads(CLAIM.read_text());assert c['pid']==os.getpid();c.update(stage=ROOT.name+'_terminal_retained',terminal_at=time.time());put(CLAIM,c)
