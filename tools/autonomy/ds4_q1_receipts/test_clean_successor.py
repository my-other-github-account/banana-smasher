#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import unittest

class BoundaryTests(unittest.TestCase):
 def test_live_predecessor_refused_then_released_accepted(self):
  p=Path(__file__).with_name('clean_successor.py')
  self.assertTrue(p.exists(), 'missing clean successor boundary implementation')
  spec=importlib.util.spec_from_file_location('successor',p)
  m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  c={'task_id':'t_0640a5bf','status':'RELEASED','released':True,'intended_basis':m.BASIS,'controller_pid':1}
  s=dict(c)
  prev={'task_id':'t_46b61159','status':'RELEASED','intended_basis':m.BASIS,'controller_pid':2,'controller_startticks':22}
  terminal={'status':'PASS','pid':2,'startticks':22}
  with self.assertRaisesRegex(AssertionError,'PREDECESSOR_LIVE'):
   m.boundary_gate(c,s,prev,terminal,lambda pid:22 if pid==2 else None)
  m.boundary_gate(c,s,prev,terminal,lambda pid:None)
  for field,value in [('status','CLAIMED'),('intended_basis','wrong')]:
   bad=dict(prev);bad[field]=value
   with self.assertRaises(AssertionError):m.boundary_gate(c,s,bad,terminal,lambda pid:None)
  with self.assertRaises(AssertionError):m.boundary_gate(c,s,prev,dict(terminal,status='FAIL'),lambda pid:None)

 def test_plan_is_exact_missing_only_and_nonoverlapping(self):
  p=Path(__file__).with_name('clean_successor.py')
  spec=importlib.util.spec_from_file_location('successor',p)
  m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  self.assertTrue(hasattr(m,'plan_cells'), 'missing frozen-plan admission')
  plan={'production_pin':m.PIN,'clean_population_sha256':m.POPULATION,'cells':[[30,0,'fused13'],[30,0,'down']], 'empty_cells':[], 'retained_only_qualified_cells':[]}
  self.assertEqual(m.plan_cells(plan),plan['cells'])
  for cells in ([[29,0,'down']],[[30,0,'down'],[30,0,'down']],[[37,130,'down']]):
   with self.assertRaises(AssertionError):m.plan_cells(dict(plan,cells=cells))
  with self.assertRaises(AssertionError):m.plan_cells(dict(plan,retained_only_qualified_cells=[[30,0,'down']]))

 def test_supervisor_refuses_live_before_claim_mutation(self):
  import tempfile,json
  from unittest.mock import patch
  p=Path(__file__).with_name('clean_successor.py')
  spec=importlib.util.spec_from_file_location('successor',p)
  m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  self.assertTrue(hasattr(m,'supervisor'), 'missing successor executor')
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);m.D=d/'run';m.D.mkdir();m.C=d/'HOST_CLAIM.json';m.SOURCE=d/'source';m.SOURCE.mkdir();m.PREV=d/'prev';m.PREV.mkdir()
   c={'task_id':'t_0640a5bf','status':'RELEASED','released':True,'intended_basis':m.BASIS,'controller_pid':1}
   prev={'task_id':'t_46b61159','status':'RELEASED','intended_basis':m.BASIS,'controller_pid':2,'controller_startticks':22}
   m.C.write_text(json.dumps(c));(m.SOURCE/'SHARDS.json').write_text(json.dumps(c));(m.PREV/'SHARDS.json').write_text(json.dumps(prev));(m.PREV/'TERMINAL.json').write_text(json.dumps({'status':'PASS','pid':2,'startticks':22}))
   before=m.C.read_bytes()
   with patch.object(m.os,'getuid',return_value=0),patch.object(m,'ticks',side_effect=lambda pid:22 if pid==2 else None):
    with self.assertRaisesRegex(AssertionError,'PREDECESSOR_LIVE'):m.supervisor()
   self.assertEqual(before,m.C.read_bytes());self.assertFalse((m.D/'SHARDS.json').exists())

if __name__=='__main__':unittest.main()
