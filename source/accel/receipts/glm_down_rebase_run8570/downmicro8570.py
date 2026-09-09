import os,json,time,hashlib,sys,importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch
from banana_smasher import qtip_viterbi as candidate
from scope8568 import admit
root=Path('/dev/shm/t_012f1c90');claim=json.loads(Path('/home/dnola/HOST_CLAIM.json').read_text());out=Path(claim['mission']);assert claim['owner']=='t_012f1c90' and claim['pid']==os.getppid()
basis=claim['source_model_index_sha256'];admit(json.loads((root/'SHARDS.json').read_text()),claim['owner'],basis,[(4,242,'down'),(4,243,'down'),(4,242,'fused13')]);assert hashlib.sha256(Path('/dev/shm/t_ebcba52e_ldlq_run8503/selected_pair_run8507/model.safetensors.index.json').read_bytes()).hexdigest()==basis
base=root/'code8564/banana-smasher/src/banana_smasher/qtip_viterbi.py'
assert hashlib.sha256(base.read_bytes()).hexdigest()==sys.argv[1]
spec=importlib.util.spec_from_file_location('banana_smasher._baseline8568',base);baseline=importlib.util.module_from_spec(spec);sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
assert torch.cuda.mem_get_info()[0]>(12+4)<<30
torch.set_num_threads(8);torch.manual_seed(1729);batch=256;steps=128
tlut=torch.randn(512,2,device='cuda');s=torch.arange(65536,device='cuda',dtype=torch.int64);h=s*(s+1);lut=tlut[(h>>6)&511].clone();lut[:,0]*=1-2*((h>>15)&1)
def cb():return SimpleNamespace(L=16,K=1,V=2,decode_mode='quantlut_sym',tlut_bits=9,lut=lut.T.contiguous().clone(),_banana_smasher_structured_gather=True,_banana_smasher_branch_unroll=True,_banana_smasher_distance_alphabet=True,_banana_smasher_viterbi_num_warps=16,_banana_smasher_backpointer_dtype='uint16',_banana_smasher_conditioned_distance_sum=True,_banana_smasher_projection='down')
models={'B':(baseline,cb()),'C':(candidate,cb())};x=torch.randn(steps*2,batch,device='cuda',dtype=torch.float16);overlap=torch.randint(0,16384,(batch,),device='cuda',dtype=torch.int32);rows=[]
for mode,ov in [('seed',None),('conditioned',overlap)]:
 for arm,(mod,book) in models.items():mod.exact_prefix_viterbi(book,x,ov)
 torch.cuda.synchronize()
 for arm in ['B','C','C','B']:
  mod,book=models[arm];torch.cuda.synchronize();start=time.perf_counter()
  for _ in range(3):result=mod.exact_prefix_viterbi(book,x,ov)
  torch.cuda.synchronize();rows.append(dict(arm=arm,mode=mode,seconds=time.perf_counter()-start))
total={a:sum(r['seconds'] for r in rows if r['arm']==a) for a in ['B','C']};ratio=total['B']/total['C']
receipt=dict(rows=rows,ratio=ratio,batch=batch,steps=steps,candidate_source_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),baseline_source_sha256=sys.argv[1],scope='Synthetic focus filter only; not authentic quality or representative build speed',passed=ratio>1.01)
p=out/'MICRO.json'
with p.open('x') as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(receipt));raise SystemExit(0 if ratio>1.01 else 3)
