import importlib.util
from pathlib import Path
import unittest

class Tests(unittest.TestCase):
 def load(self):
  p=Path(__file__).parents[1]/'src/banana_smasher/host_arena_environment.py'
  self.assertTrue(p.exists(),'missing startup-only arena environment helper')
  spec=importlib.util.spec_from_file_location('arenas',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
 def test_default_off_and_opt_in_fresh_mapping(self):
  m=self.load();env={'X':'Y','MALLOC_ARENA_MAX':'8'}
  self.assertEqual(m.host_arena_environment(env),env)
  self.assertEqual(m.host_arena_environment(env,enabled=True),{'X':'Y','MALLOC_ARENA_MAX':'2'})
  self.assertEqual(env['MALLOC_ARENA_MAX'],'8')
  self.assertIsNot(m.host_arena_environment(env),env)
 def test_strict_boolean_and_real_child_environment(self):
  import subprocess,sys
  m=self.load()
  for flag in [0,1,'true',None]:
   with self.assertRaises(ValueError):m.host_arena_environment({},enabled=flag)
  out=subprocess.check_output([sys.executable,'-c','import os;print(os.environ["MALLOC_ARENA_MAX"])'],env=m.host_arena_environment({},enabled=True),text=True)
  self.assertEqual(out.strip(),'2')
if __name__=='__main__':unittest.main()
