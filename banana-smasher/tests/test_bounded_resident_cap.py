import unittest,tempfile,json,hashlib,importlib.util,pathlib,types
p=pathlib.Path(__file__).parents[1]/'src/banana_smasher/bounded_resident_cap.py';s=importlib.util.spec_from_file_location('cap',p);cap=importlib.util.module_from_spec(s);s.loader.exec_module(cap)
class Test(unittest.TestCase):
 def test_admission_and_strict_boundaries(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'w.json';p.write_text(json.dumps(dict(proc_status='VmPeak:\t29999660 kB\n',cuda_peak_reserved=1721761792,ru_maxrss_bytes=2142400512)));h=hashlib.sha256(p.read_bytes()).hexdigest()
   self.assertEqual(cap.admit(50<<30,8<<30,16<<20,p,h)['estimated_peak_bytes'],40<<30)
   for a,f in [((48<<30)+(16<<20),8<<30),(50<<30,(4<<30)+(16<<20))]:
    with self.assertRaises(RuntimeError):cap.admit(a,f,16<<20,p,h)
   with self.assertRaises(ValueError):cap.admit(50<<30,8<<30,16<<20,p,'bad')
 def test_enforced_not_just_declared(self):
  events=[];r=types.SimpleNamespace(RLIMIT_AS=9,setrlimit=lambda *a:events.append(a));c=types.SimpleNamespace(get_device_properties=lambda _:types.SimpleNamespace(total_memory=128<<30),set_per_process_memory_fraction=lambda *a:events.append(a))
  cap.enforce(r,c);self.assertEqual(events,[(9,(32<<30,32<<30)),(4/128,0)])
if __name__=='__main__':unittest.main()
