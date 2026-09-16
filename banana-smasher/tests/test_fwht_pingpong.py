import importlib.util, math, unittest
from pathlib import Path
import torch
class Test(unittest.TestCase):
 def test_exact_and_immutable(self):
  p=Path(__file__).parents[1]/'src/banana_smasher/fwht_pingpong.py'
  self.assertTrue(p.exists(),'missing bounded FWHT implementation')
  spec=importlib.util.spec_from_file_location('fw',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  for n in [1,2,4,16,256]:
   x=torch.randn(3,n*2)[:,::2];before=x.clone();y=x.contiguous();w=1
   while w<n:
    z=y.reshape(3,n//(2*w),2,w);a,b=z[...,0,:],z[...,1,:];y=torch.cat((a+b,a-b),dim=-1).reshape(3,n);w*=2
   expected=y/math.sqrt(n)
   torch.testing.assert_close(m.fwht_pingpong(x),expected,rtol=0,atol=0)
   torch.testing.assert_close(x,before,rtol=0,atol=0)
if __name__=='__main__':unittest.main()
