#!/usr/bin/env python3
from __future__ import annotations
import hashlib,importlib.util,json,shlex,struct,subprocess,time
from pathlib import Path
from typing import Any,Mapping
import numpy as np
RECOVERY_TERMINAL=Path('/home/dnola/missions/W328_SEALED_RECON_t_03c6894c_s5w/receipts/L034_E161_E170_RECOVERY3_TERMINAL.json')
EXPECTED_RECOVERY_TERMINAL_SHA='3c62229f5a17eaf9ee4b0923f1048de65b591bea8aaeeb28342bcaa7492baedb'
BASIS='98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b'; TASK='t_1a6fd24f'; EXPECTED_COMMIT='bb807b9f6ffcae211f9dc779b5b576198c3ac6da'
REMOTE_CODE=r'''import hashlib,json,os,struct,sys
rows=json.loads(sys.stdin.buffer.readline()); out=sys.stdout.buffer
def sha_bytes(value): return hashlib.sha256(value).hexdigest()
def sha_file(path):
 h=hashlib.sha256()
 with open(path,"rb",buffering=0) as f:
  for block in iter(lambda:f.read(8<<20),b""): h.update(block)
 return h.hexdigest()
RECOVERY_ROOT="/dev/shm/QTIP2_V7_L034_RECOVERY_STAGE21/output/physical"
RECOVERY_PATHS={f"/home/dnola/missions/Q2_L34_RANGE_t_befdd9ca_s6/artifacts/physical/E{e}/E{e}_{w}.physical.bf16.bin":f"{RECOVERY_ROOT}/E{e}_{w}.bf16.bin" for e in range(161,171) for w in ("w1","w2","w3")}
for row in rows:
 requested_path=row["path"]; p=RECOVERY_PATHS.get(requested_path, requested_path); mode="physical"; receipt_sha=None; artifact_sha=None
 if not os.path.isfile(p):
  receipt_bytes=open(row["member_receipt_path"],"rb").read(); receipt_sha=sha_bytes(receipt_bytes)
  if receipt_sha!=row["member_receipt_sha256"]: raise RuntimeError("member receipt SHA mismatch")
  receipt=json.loads(receipt_bytes); p=receipt.get("artifact_path") or receipt["artifact"]["path"]
  artifact_sha=receipt.get("artifact_sha256") or receipt["artifact"]["sha256"]
  if not os.path.isfile(p) or sha_file(p)!=artifact_sha: raise RuntimeError("compact recovery artifact mismatch")
  mode="compact"
 st=os.stat(p); h=json.dumps({"path":requested_path,"bytes":st.st_size,"mode":mode,"receipt_sha256":receipt_sha,"artifact_sha256":artifact_sha},separators=(",",":")).encode(); out.write(struct.pack("!I",len(h))); out.write(h)
 with open(p,"rb",buffering=0) as f:
  while True:
   b=f.read(8<<20)
   if not b: break
   out.write(b)
out.flush()
'''
def sha256_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def load_module(path,expected,name):
 if sha256_file(path)!=expected: raise ValueError(f'{name} SHA mismatch')
 spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def read_exact(stream,n):
 parts=[]; left=n
 while left:
  b=stream.read(min(left,8<<20))
  if not b: raise EOFError(f'stream ended with {left} bytes remaining')
  parts.append(b); left-=len(b)
 return b''.join(parts)
def open_framed_stream(command,request_bytes,*,attempts=3,popen=subprocess.Popen,retry_delay=5):
 errors=[]
 for attempt in range(attempts):
  proc=popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  assert proc.stdin is not None and proc.stdout is not None
  proc.stdin.write(request_bytes); proc.stdin.close()
  try:
   return proc,read_exact(proc.stdout,4)
  except EOFError as exc:
   rc=proc.wait(); err=(proc.stderr.read().decode(errors='replace') if proc.stderr else '')
   errors.append(f'attempt={attempt+1} rc={rc} error={exc} stderr={err[-1000:]}')
   if attempt+1<attempts: time.sleep(retry_delay)
 raise RuntimeError('initial framed stream failed: '+' | '.join(errors))
def load_l034(*,torch:Any,device:Any,source_state:Mapping[str,Any],config:Mapping[str,Any],basis_sha256:str)->dict[str,Any]:
 if basis_sha256!=BASIS or config.get('basis_sha256')!=BASIS: raise ValueError('complete provider basis mismatch')
 roster_path=Path(config['candidate_roster']['path']); roster_sha=config['candidate_roster']['sha256']
 if sha256_file(roster_path)!=roster_sha: raise ValueError('complete roster SHA mismatch')
 roster=json.loads(roster_path.read_text())
 if roster.get('status')!='PASS' or roster.get('task_id')!=TASK or roster.get('basis_sha256')!=BASIS or roster.get('candidate_commit')!=EXPECTED_COMMIT or roster.get('member_count')!=768 or roster.get('candidate_expert_count')!=256 or roster.get('expert_gaps')!=[] or roster.get('duplicates')!=0 or roster.get('pass_through_bytes')!=0: raise ValueError('complete roster contract mismatch')
 exact=config['exact_provider']; module=load_module(Path(exact['path']),exact['sha256'],'exact_k2_provider_for_complete_overlay'); exact_config=dict(config['exact_config'])
 q2=config['q2_module']; active_lut=config['active_lut']
 started=time.monotonic()
 # Full 256-expert coverage has no authentic complement: allocate target geometry and fill every cell from admitted physical members.
 gate_up=torch.empty((256,4096,4096),dtype=torch.bfloat16,device=device); down=torch.empty((256,4096,2048),dtype=torch.bfloat16,device=device); exact_ev={}
 recovery_bytes=RECOVERY_TERMINAL.read_bytes()
 if hashlib.sha256(recovery_bytes).hexdigest()!=EXPECTED_RECOVERY_TERMINAL_SHA: raise RuntimeError('sealed recovery terminal SHA mismatch')
 recovery=json.loads(recovery_bytes)
 if recovery.get('status')!='PASS' or recovery.get('basis')!=BASIS or recovery.get('complete_members')!=30 or recovery.get('gaps')!=0 or recovery.get('duplicates')!=0 or not recovery.get('physical_parity_exact') or recovery.get('fallback_calls')!=0: raise RuntimeError('sealed recovery terminal contract mismatch')
 recovery_rows={(int(x['member'].split('/')[1][1:]),x['member'].split('/')[2]):x for x in recovery['members']}
 if set(recovery_rows)!={(e,p) for e in range(161,171) for p in ('w1','w2','w3')}: raise RuntimeError('sealed recovery terminal coverage mismatch')
 members=[]; local_recovered_members=0
 for row0 in roster['members']:
  row=dict(row0); key=(int(row['expert']),str(row['member']))
  if key in recovery_rows:
   rr=recovery_rows[key]
   if rr['physical_sha256']!=row['sha256']: raise RuntimeError(f'sealed recovery physical identity mismatch {key}')
  members.append(row)
 by_host={}
 for row in members: by_host.setdefault(row['host'],[]).append(row)
 streamed_bytes=0; transport_bytes=0; compact_recovery_members=0; physical_digest=hashlib.sha256(); member_receipts=[]
 with torch.no_grad():
  for host in sorted(by_host):
   host_rows=by_host[host]
   if host=='__local__': command=['/usr/bin/sudo','-u','dnola','python3','-c',REMOTE_CODE]
   else: command=['/usr/bin/sudo','-u','dnola','/usr/bin/ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'sudo -n python3 -c '+shlex.quote(REMOTE_CODE)]
   def physical_path(row):
    value=row['path']; return value['path'] if isinstance(value,Mapping) else value
   requests=[{'path':physical_path(r),'member_receipt_path':r.get('member_receipt_path'),'member_receipt_sha256':r.get('member_receipt_sha256')} for r in host_rows]
   proc,first_prefix=open_framed_stream(command,(json.dumps(requests,separators=(',',':'))+'\n').encode())
   for row_index,row in enumerate(host_rows):
    prefix=first_prefix if row_index==0 else read_exact(proc.stdout,4)
    expected_path=physical_path(row); n=struct.unpack('!I',prefix)[0]; head=json.loads(read_exact(proc.stdout,n)); mode=head.get('mode')
    data=read_exact(proc.stdout,int(head['bytes'])); transport_bytes+=len(data); e=int(row['expert']); m=row['member']
    if mode=='physical':
     if head['path']!=expected_path or int(head['bytes'])!=int(row['bytes']): raise RuntimeError(f'remote physical header mismatch {expected_path}')
     observed=hashlib.sha256(data).hexdigest()
     if observed!=row['sha256']: raise RuntimeError(f'physical SHA mismatch {expected_path}')
     value=torch.frombuffer(bytearray(data),dtype=torch.bfloat16).reshape(tuple(row['shape'])).to(device)
    elif mode=='compact':
     if head.get('receipt_sha256')!=row['member_receipt_sha256']: raise RuntimeError(f'compact receipt mismatch E{e} {m}')
     artifact_observed=hashlib.sha256(data).hexdigest()
     if artifact_observed!=head.get('artifact_sha256'): raise RuntimeError(f'compact artifact mismatch E{e} {m}')
     if m=='w2': packed_shape,su_count,sv_count=(128,256,32),2048,4096
     elif m in ('w1','w3'): packed_shape,su_count,sv_count=(256,128,32),4096,2048
     else: raise RuntimeError(f'compact projection refused E{e} {m}')
     packed_end=2_097_152; suh_end=packed_end+2*su_count; svh_end=suh_end+2*sv_count; su_end=svh_end+4*su_count; sv_end=su_end+4*sv_count; total=sv_end+4
     if len(data)!=total: raise RuntimeError(f'compact layout bytes mismatch E{e} {m}: {len(data)} != {total}')
     packed=torch.from_numpy(np.frombuffer(data[:packed_end],dtype='<i2').copy().reshape(packed_shape)).to(device)
     su=torch.from_numpy(np.frombuffer(data[svh_end:su_end],dtype='<f4').copy()).to(device)
     sv=torch.from_numpy(np.frombuffer(data[su_end:sv_end],dtype='<f4').copy()).to(device)
     decoded=q2.decode_k2_matrix(packed,active_lut); value=q2.inverse_transform(decoded,su,sv).T.contiguous().to(torch.bfloat16)
     value_bytes=value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes(); observed=hashlib.sha256(value_bytes).hexdigest()
     if observed!=row['sha256']: raise RuntimeError(f'compact recovery physical SHA mismatch E{e} {m}: {observed} != {row["sha256"]}')
     compact_recovery_members+=1; del packed,su,sv,decoded,value_bytes
    else: raise RuntimeError(f'unknown transport mode E{e} {m}: {mode!r}')
    physical_digest.update(f"{e}:{m}:".encode()); physical_digest.update(bytes.fromhex(observed))
    if m=='w1' and tuple(value.shape)==(2048,4096): gate_up[e,:2048].copy_(value)
    elif m=='w3' and tuple(value.shape)==(2048,4096): gate_up[e,2048:].copy_(value)
    elif m=='w2' and tuple(value.shape)==(4096,2048): down[e].copy_(value)
    else: raise RuntimeError(f'candidate geometry/member mismatch E{e} {m} {tuple(value.shape)}')
    streamed_bytes+=int(row['bytes']); member_receipts.append({'expert':e,'member':m,'producer':row['producer'],'physical_sha256':observed,'receipt_sha256':row['member_receipt_sha256'],'transport_mode':mode}); del value,data
   rc=proc.wait(); err=(proc.stderr.read().decode(errors='replace') if proc.stderr else '')
   if rc: raise RuntimeError(f'stream host {host} rc={rc} stderr={err[-1000:]}')
 candidate=set(roster['candidate_experts']); authentic=sorted(set(range(256))-candidate)
 if candidate!=set(range(256)) or authentic: raise RuntimeError('complete candidate coverage mismatch')
 wire=dict(roster['wire_accounting']); weights=int(wire['weights'])
 evidence={'provider':'banana-smasher-q2-complete-assignment-physical-v2','mode':'complete_candidate_assignment_physical_bf16_with_exact_compact_transport_recovery','candidate_roster_path':str(roster_path),'candidate_roster_sha256':roster_sha,'candidate_expert_count':256,'candidate_experts':list(range(256)),'authentic_k2_expert_count':0,'authentic_k2_experts':[],'candidate_member_count':768,'authentic_k2_member_count':0,'physical_stream_bytes':streamed_bytes,'transport_stream_bytes':transport_bytes,'compact_recovery_members':compact_recovery_members,'local_recovered_physical_members':local_recovered_members,'physical_stream_sha256':physical_digest.hexdigest(),'physical_members':member_receipts,'candidate_commit':EXPECTED_COMMIT,'basis_sha256':BASIS,'fallback_calls':0,'nontarget_tensors_changed':0,'changed_target_tensors':512,'roster_count':768,'assignment_physical_scientific_input_bytes':streamed_bytes,'deployment_wire':wire,'elapsed_seconds':time.monotonic()-started}
 if streamed_bytes!=12884901888 or int(wire['complete_wire_bytes'])!=1638929408 or int(wire['selected_payload_bytes_excluding_shared_lut'])!=1638927360: raise RuntimeError('complete physical/wire accounting mismatch')
 return {'gate_up':gate_up,'down':down,'evidence':evidence}
