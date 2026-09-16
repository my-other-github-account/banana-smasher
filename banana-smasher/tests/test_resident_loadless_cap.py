"""A smaller enforced address envelope is not a smaller safety reserve."""
import copy
import hashlib
import json
import pathlib
import tempfile
import types
import unittest
from test_resident_two_cap import load, R

class Tests(unittest.TestCase):
    def setUp(self):
        self.cap = load('resident_two_cap', R/'src/banana_smasher/resident_two_cap.py')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = pathlib.Path(self.temp.name)/'WORKING_SET.json'
        self.path.write_text(json.dumps(dict(proc_status='VmPeak:\t31109792 kB\n', ru_maxrss_bytes=2015010816, cuda_peak_reserved=2644508672)))
        self.spec = dict(K=4, cells=['L040/E047_down','L020/E000_fused13','L026/E096_fused13','L026/E097_fused13'], resident_cell_limit=4, estimated_peak_bytes=38<<30, planned_write_bytes=36<<20, resource_envelope='sequential30', working_set_witness=str(self.path), working_set_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())
    def test_explicit_smaller_envelope_preserves_reserves(self):
        before = copy.deepcopy(self.spec)
        self.assertEqual(self.cap.verify(self.spec, 47<<30, 8<<30), (38<<30, 36<<20))
        self.assertEqual(self.spec, before)
        for available, free in [((46<<30)+(36<<20), 8<<30), (47<<30, (4<<30)+(36<<20))]:
            with self.assertRaises(AssertionError): self.cap.verify(self.spec, available, free)
    def test_enforcement_is_actual_not_estimate_only(self):
        events=[]
        resource=types.SimpleNamespace(RLIMIT_AS=9,setrlimit=lambda *a:events.append(a))
        cuda=types.SimpleNamespace(get_device_properties=lambda n:types.SimpleNamespace(total_memory=128<<30),set_per_process_memory_fraction=lambda *a:events.append(a))
        result=self.cap.apply_cap(self.spec,resource,cuda)
        self.assertEqual(events, [(9,(30<<30,30<<30)),(4/128,0)])
        self.assertEqual(result['estimated_peak_bytes'],38<<30)
    def test_fail_closed_for_bad_witness_and_undeclared_policy(self):
        for change in [dict(resource_envelope='guess'),dict(working_set_sha256='0'*64),dict(estimated_peak_bytes=37<<30),dict(planned_write_bytes=1)]:
            with self.assertRaises(ValueError): self.cap.verify(dict(self.spec,**change),47<<30,8<<30)
        for field,value in [('proc_status','VmPeak: 31457280 kB\n'),('cuda_peak_reserved',4<<30),('ru_maxrss_bytes',30<<30)]:
            witness=json.loads(self.path.read_text());witness[field]=value;self.path.write_text(json.dumps(witness))
            s=dict(self.spec,working_set_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())
            with self.assertRaises(ValueError):self.cap.verify(s,47<<30,8<<30)

if __name__=='__main__': unittest.main()
