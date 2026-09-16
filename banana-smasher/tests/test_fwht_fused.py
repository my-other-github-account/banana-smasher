import unittest,importlib.util
from pathlib import Path
import torch
class Test(unittest.TestCase):
 def test_admission(self):
  p=Path(__file__).parents[1]/'src/banana_smasher/fwht_fused.py'
  self.assertTrue(p.exists(),'missing fused FWHT')
  spec=importlib.util.spec_from_file_location('fused',p);assert spec and spec.loader
  m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  for x in [torch.ones(2,16),torch.ones(2,3),torch.ones(2,16,dtype=torch.float64)]:
   with self.assertRaises(ValueError):m.fwht_fused(x)
if __name__=='__main__':unittest.main()
