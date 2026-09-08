import sys,os,json,time,hashlib,types,importlib.util,statistics
from pathlib import Path
import torch
R=Path('/dev/shm/t_ebcba52e_ldlq_run8503');c=json.loads(Path('/home/dnola/HOST_CLAIM.json').read_text());assert c['task_id']=='t_ebcba52e' and c['pid']==os.getppid()
torch.set_num_threads(8)
from banana_smasher import qtip_runner as qv,qtip_viterbi as new
m=json.loads((R/'closure/MANIFEST.json').read_text())
def locate(suffix):
 rows=[r for r in m['files'] if r['archive_path'].endswith(suffix)];assert len(rows)==1,(suffix,len(rows));return R/'closure'/rows[0]['archive_path']
qv.QTIP=locate('lib/algo/ldlq.py').parents[2]
bits,ldlq,math,kd=qv.load_official_qtip()
from banana_smasher.solver_qtip_profile import _load_tlut
lut=_load_tlut(locate('P641_QTIP_TLUT_SOURCE.pt'))
cb=bits.bitshift_codebook(L=16,K=1,V=2,tlut_bits=9,decode_mode='quantlut_sym',tlut=lut.cuda()).cuda()
ms=importlib.util.spec_from_file_location('banana_smasher.qtip_memory',locate('/qtip_memory_run8248.py'));mm=importlib.util.module_from_spec(ms);ms.loader.exec_module(mm);sys.modules[ms.name]=mm
spec=importlib.util.spec_from_file_location('banana_smasher.owner_viterbi',locate('/qtip_viterbi_run8248.py'));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
torch.manual_seed(5413)
rows=[]
for batch in (256,1024):
 x=torch.randn((256,batch),device='cuda');overlap=torch.randint(0,16384,(batch,),device='cuda',dtype=torch.int32)
 ref=None
 for name,dtype,warps,gather,unroll in [('owner','int32',None,None,None),('main','int32',None,None,None),('uint8','uint8',None,None,None),('uint8_unroll','uint8',16,None,True),('uint8_gather','uint8',16,True,True)]:
  cb._banana_smasher_backpointer_dtype=dtype;cb._banana_smasher_viterbi_num_warps=warps;cb._banana_smasher_structured_gather=gather;cb._banana_smasher_branch_unroll=unroll
  fn=old.exact_prefix_viterbi if name=='owner' else new.exact_prefix_viterbi
  outputs=[];times=[];cold=time.perf_counter()
  for ov in (None,overlap):outputs.append(fn(cb,x,ov))
  torch.cuda.synchronize();cold=time.perf_counter()-cold
  for rep in range(3):
   torch.cuda.synchronize();t=time.perf_counter()
   for ov in (None,overlap):fn(cb,x,ov)
   torch.cuda.synchronize();times.append(time.perf_counter()-t)
  if ref is None:ref=outputs
  row=dict(batch=batch,arm=name,cold_seconds=cold,seconds=times,median=statistics.median(times),exact=[torch.equal(a,b) for a,b in zip(ref,outputs)],scope='synthetic_shape_microbenchmark_authentic_TLUT_not_output_quality',peak=torch.cuda.max_memory_allocated())
  rows.append(row);print(json.dumps(row),flush=True);(Path(os.environ['OUT'])/'MICRO.json').write_text(json.dumps(rows,indent=2))
  del outputs;torch.cuda.empty_cache()
