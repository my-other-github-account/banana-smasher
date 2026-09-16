"""Admission tests for the exact physically measured finite launcher."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

p=Path(__file__).parents[1]/'experiments/max2_forkserver/fork_pair8873.py'
spec=importlib.util.spec_from_file_location('finite_fork',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Admission(unittest.TestCase):
 def test_reject_initialized_cuda_before_running_payload(self):
  fake=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:True))
  with patch.dict(sys.modules,torch=fake),patch.object(m.runpy,'run_path') as run:
   with self.assertRaisesRegex(AssertionError,'CUDA context'):m.job('e096,e097')
   run.assert_not_called()
 def test_serial_pair_plumbing(self):
  fake=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False))
  for part in ['setup,warm','e096,e097']:
   with patch.dict(sys.modules,torch=fake),patch.dict(os.environ,{},clear=False),patch.object(sys,'argv',[]),patch.object(m.runpy,'run_path') as run:
    m.job(part)
    self.assertEqual(os.environ['RESIDENT_PHASES'],part)
    self.assertEqual(sys.argv[1:],['C1','candidate'])
    run.assert_called_once_with('/run/t_5ade4a57/fork_arm8873.py',run_name='__main__')
if __name__=='__main__':unittest.main()
