import importlib.util,pathlib,sys,unittest
R=pathlib.Path(__file__).parents[1]
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
original=load('fleet_cap_run8805',R/'tests/fixtures/fleet_cap_run8805.py')
class Tests(unittest.TestCase):
 def test_actual_original_two_cell_verify(self):
  p=R/'src/banana_smasher/resident_two_cap.py'
  cap=load('resident_two_cap',p) if p.exists() else original
  s=dict(K=4,cells=['L014/E180_down','L014/E181_down'],estimated_peak_bytes=56<<30,planned_write_bytes=12<<20)
  self.assertEqual(cap.verify(s,80<<30,8<<30),(56<<30,12<<20))
 def test_original_caps_and_aggregate_boundaries(self):
  import copy,types
  cap=load('resident_two_cap',R/'src/banana_smasher/resident_two_cap.py')
  s=dict(K=4,cells=['L014/E180_down','L014/E181_down'],estimated_peak_bytes=56<<30,planned_write_bytes=12<<20);before=copy.deepcopy(s);events=[]
  resource=types.SimpleNamespace(RLIMIT_AS=9,setrlimit=lambda *args:events.append(args))
  cuda=types.SimpleNamespace(get_device_properties=lambda n:types.SimpleNamespace(total_memory=128<<30),set_per_process_memory_fraction=lambda *args:events.append(args))
  result=cap.apply_cap(s,resource,cuda)
  self.assertEqual(events,[(9,(48<<30,48<<30)),(4/128,0)]);self.assertEqual(result['cuda_allocator_limit'],4<<30);self.assertEqual(s,before)
  for available,free in [((64<<30)+(12<<20),8<<30),(80<<30,(4<<30)+(12<<20))]:
   with self.assertRaises(AssertionError):cap.verify(s,available,free)
  for override in [dict(planned_write_bytes=6<<20),dict(estimated_peak_bytes=112<<30),dict(cells=s['cells'][:1]),dict(cells=[s['cells'][0]]*2),dict(K=1)]:
   with self.assertRaises(ValueError):cap.verify(dict(s,**override),80<<30,8<<30)
 def test_fused_output_is_aggregate_not_singleton(self):
  cap=load('resident_two_cap',R/'src/banana_smasher/resident_two_cap.py')
  s=dict(K=4,cells=['L014/E180_fused13','L014/E181_fused13'],estimated_peak_bytes=56<<30,planned_write_bytes=20<<20)
  self.assertEqual(cap.verify(s,80<<30,8<<30),(56<<30,20<<20))
 def test_explicit_four_cell_integration_preserves_singleton_limits(self):
  cap=load('resident_two_cap',R/'src/banana_smasher/resident_two_cap.py')
  cells=['L040/E047_down','L020/E000_fused13','L026/E096_fused13','L026/E097_fused13']
  s=dict(K=4,cells=cells,resident_cell_limit=4,estimated_peak_bytes=56<<30,planned_write_bytes=36<<20)
  self.assertEqual(cap.verify(s,80<<30,8<<30),(56<<30,36<<20))
  for change in [dict(resident_cell_limit=2),dict(resident_cell_limit=3),dict(cells=cells[:3]),dict(cells=cells[:3]+cells[:1]),dict(planned_write_bytes=35<<20)]:
   with self.assertRaises(ValueError):cap.verify(dict(s,**change),80<<30,8<<30)
  with self.assertRaises(AssertionError):cap.verify(s,(64<<30)+(36<<20),8<<30)
  with self.assertRaises(AssertionError):cap.verify(s,80<<30,(4<<30)+(36<<20))
if __name__=='__main__':unittest.main()
