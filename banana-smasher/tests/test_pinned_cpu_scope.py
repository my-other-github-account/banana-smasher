import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace
p=Path(__file__).parents[1]/'src/banana_smasher/pinned_cpu_scope.py'
s=importlib.util.spec_from_file_location('scope',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_reach_sync_fallback_and_restore(self):
  trace=[]
  class T:
   is_cuda=True;requires_grad=False;device='cuda:0';dtype='float32'
   def numel(self):return 262144
   def element_size(self):return 4
   def is_contiguous(self):return True
   def cpu(self,*a,**k):trace.append('fallback');return self
  class Out:
   def copy_(self,t,non_blocking):self.asserted=non_blocking;trace.append('copy')
   def is_pinned(self):return True
  def alloc(t,**kw):self.assertEqual(kw,dict(device='cpu',pin_memory=True));return Out()
  torch=SimpleNamespace(Tensor=T,empty_like=alloc,cuda=SimpleNamespace(current_stream=lambda d:SimpleNamespace(synchronize=lambda:trace.append('sync'))))
  original=T.cpu
  with self.assertRaisesRegex(RuntimeError,'sentinel'):
   with m.pinned_cpu_scope(torch,True) as events:
    out=T().cpu();self.assertIsInstance(out,Out);self.assertEqual(trace,['copy','sync']);self.assertEqual(len(events),1)
    t=T();t.requires_grad=True;self.assertIs(t.cpu(),t)
    t.requires_grad=False;self.assertIs(t.cpu(memory_format='x'),t)
    t.is_cuda=False;self.assertIs(t.cpu(),t)
    raise RuntimeError('sentinel')
  self.assertIs(T.cpu,original)
  with m.pinned_cpu_scope(torch):self.assertIs(T.cpu,original)
  with self.assertRaises(ValueError):
   with m.pinned_cpu_scope(torch,1):pass
if __name__=='__main__':unittest.main()
