import os,json,time,hashlib
from pathlib import Path
import numpy as np,torch
R=Path('/dev/shm/t_182fbc9d');P=R/'heldout_readout_run8524';I=R/'heldout_input_run8520';O=Path(os.environ['OUT']);torch.set_num_threads(8);start=time.perf_counter()
assert json.loads(Path('/home/dnola/HOST_CLAIM.json').read_text())['pid']==os.getppid()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
assert json.loads((P/'TERMINAL.json').read_text())['returncode']==0
binding=json.loads((P/'BINDING.json').read_text());assert binding['delta_limit']==1e-8 and binding['teacher_sha256']==sha(I/'teacher_capture.json') and binding['ledger_sha256']==sha(I/'token_ledger.json')
for x in json.loads((I/'ACK.json').read_text())['files']:assert sha(I/x['file'])==x['sha256']
assert binding['prefix_sha256']==sha(I/'L003_000.payload')
for x in binding['replacements']:assert sha(x['path'])==x['sha256']
expected={f'L{li:03d}_{arm}.json' for li in range(4,45) for arm in ['baseline1','baseline2','candidate']};actual={p.name for p in P.glob('L*.json')};assert actual==expected
same=[];count=0
for li in range(4,45):
 v={}
 for arm in ['baseline1','baseline2','candidate']:
  p=P/f'L{li:03d}_{arm}.pt';m=json.loads(p.with_suffix('.json').read_text());assert sha(p)==m['sha256'] and p.stat().st_size==m['bytes'] and m['stage']==li and m['arm']==arm;v[arm]=torch.load(p,map_location='cpu',weights_only=True);count+=1
 for arm in ['baseline2','candidate']:
  assert torch.equal(v['baseline1']['activation'],v[arm]['activation']);a=v['baseline1']['topk'];b=v[arm]['topk'];assert (a is None and b is None) or torch.equal(a,b)
 same.append(li)
 del v
with np.load(I/'teacher_row_ordinal0.npz',allow_pickle=False) as z:teacher={k:z[k] for k in z.files}
assert np.array_equal(teacher['position_map'],np.arange(1024));assert teacher['support_token_ids'].shape==(1024,8192)
def logsoftmax(x):
 x=x.astype(np.float64);x=x-x.max(axis=-1,keepdims=True);return x-np.log(np.exp(x).sum(axis=-1,keepdims=True))
a=logsoftmax(teacher['support_logits']);rows=[];record=json.loads((P/'RESULT.json').read_text());reported={x['arm']:x for x in record['rows']}
for arm in ['baseline1','baseline2','candidate']:
 p=P/f'{arm}.npz';assert sha(p)==reported[arm]['row_sha256']
 with np.load(p,allow_pickle=False) as z:out={k:z[k] for k in z.files}
 assert np.array_equal(out['support_token_ids'],teacher['support_token_ids']);b=logsoftmax(out['support_logits']);kl=(np.exp(a)*(a-b)).sum(-1);assert np.isfinite(kl).all();mean=float(kl.mean());assert abs(mean-reported[arm]['kld'])<=1e-10
 rows.append(dict(arm=arm,kld=mean,top1_matches=int((out['top1_token_ids']==teacher['top1_token_ids']).sum()),row_sha256=sha(p)))
d={x['arm']:x['kld'] for x in rows};assert abs(d['baseline1']-d['baseline2'])<=1e-8 and d['candidate']-d['baseline1']<=1e-8
result=dict(status='PASS',frontiers_verified=count,all_paired_frontiers_equal=len(same)==41,replacement_bindings_verified=len(binding['replacements']),rows=rows,delta=d['candidate']-d['baseline1'],tolerance=1e-8,verification_seconds=time.perf_counter()-start,scope=binding['scope'],teacher_sha256=binding['teacher_sha256'],input_prefix_sha256=binding['prefix_sha256'],producer_result_sha256=sha(P/'RESULT.json'),no_forward_replay=True)
with (O/'VERIFIED.json').open('w') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(result),flush=True)
