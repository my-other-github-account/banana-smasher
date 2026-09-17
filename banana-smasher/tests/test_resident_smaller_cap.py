import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from test_resident_two_cap import load, R

G = 1 << 30

class SmallerCapTests(unittest.TestCase):
    def test_witnessed_smaller_cap_enforced_with_unchanged_reserves(self):
        cap = load('resident_two_cap', R/'src/banana_smasher/resident_two_cap.py')
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'witness.json'
            p.write_text(json.dumps(dict(proc_status='VmPeak:\t29989864 kB\n', ru_maxrss_bytes=2178117632, cuda_peak_reserved=878706688)))
            s = dict(K=4, cells=['L043/E%03d_down'%i for i in range(8)], resident_cell_limit=8, resource_envelope='sequential30', sequential_address_limit_bytes=29*G, estimated_peak_bytes=37*G, planned_write_bytes=48<<20, working_set_witness=str(p), working_set_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertEqual(cap.verify(s, 46*G, 8*G), (37*G, 48<<20))
            events=[]
            resource=types.SimpleNamespace(RLIMIT_AS=9,setrlimit=lambda *a:events.append(a))
            cuda=types.SimpleNamespace(get_device_properties=lambda n:types.SimpleNamespace(total_memory=128*G),set_per_process_memory_fraction=lambda *a:events.append(a))
            result=cap.apply_cap(s,resource,cuda)
            self.assertEqual(events,[(9,(29*G,29*G)),(4/128,0)])
            self.assertEqual(result['cpu_address_limit'],29*G)
            self.assertEqual(result['overhead_bytes'],4*G)
            for available,free in [(45*G+(48<<20),8*G),(46*G,4*G+(48<<20))]:
                with self.assertRaises(AssertionError):cap.verify(s,available,free)
            for n in [28*G,31*G,0,-1,True,29.0*G]:
                with self.assertRaises(ValueError):cap.verify(dict(s,sequential_address_limit_bytes=n),80*G,8*G)
            with self.assertRaises(ValueError):cap.verify(dict(s,working_set_sha256='wrong'),80*G,8*G)

if __name__ == '__main__': unittest.main()
