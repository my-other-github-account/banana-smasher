"""Finite CPU-only forkserver bootstrap; each child retains production max2."""
import os,sys,json,multiprocessing as mp,runpy,time
from pathlib import Path

def job(part):
 import torch
 assert not torch.cuda.is_initialized(), 'forkserver inherited CUDA context'
 os.environ['RESIDENT_PHASES']=part
 sys.argv=['/run/t_5ade4a57/fork_arm8873.py','C1','candidate']
 runpy.run_path(sys.argv[0],run_name='__main__')

if __name__=='__main__':
 mp.set_forkserver_preload(['torch'])
 ctx=mp.get_context('forkserver')
 for part in ['setup,warm','e096,e097']:
  p=ctx.Process(target=job,args=(part,));p.start()
  receipt=Path('/dev/t_5ade4a57/fork8873')/(part+'_FORK_CHILD.json')
  d=dict(pid=p.pid,pgid=os.getpgid(p.pid),start_ticks=int(Path('/proc/'+str(p.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]),part=part)
  with receipt.open('x') as f:json.dump(d,f);f.flush();os.fsync(f.fileno())
  p.join();assert p.exitcode==0,(part,p.exitcode)
