import json,tempfile,shutil,unittest
from pathlib import Path
from reduce_run8503 import reduce,E
class EvidenceTests(unittest.TestCase):
 def test_verified_receipts(self):
  r=reduce(E);self.assertGreater(r['whole_process']['speedup'],1);self.assertTrue(r['quality']['all_decoded_equal_original'])
 def corrupt(self,relative,change):
  with tempfile.TemporaryDirectory(dir=E.parent) as d:
   root=Path(d)
   for sub in ('whole','original_quality'):
    (root/sub).mkdir()
    for p in (E/sub).iterdir():
     if p.is_file():(root/sub/p.name).symlink_to(p.resolve())
   p=root/relative;data=json.loads(p.read_text());p.unlink();change(data);p.write_text(json.dumps(data))
   with self.assertRaises(AssertionError):reduce(root)
 def test_rejects_artifact_digest_mismatch(self):
  self.corrupt('whole/new1_RESULT.json',lambda x:x['rows'][0].update(artifact_sha256='0'*64))
 def test_rejects_duplicate_phase(self):
  self.corrupt('whole/new1_RESULT.json',lambda x:x['rows'].__setitem__(1,x['rows'][0]))
 def test_rejects_quality_regression(self):
  self.corrupt('original_quality/ORIGINAL_QUALITY.json',lambda x:x['rows'][2]['metrics']['qtip_hyb'].update(sse_ratio_vs_true_vq=2.0))
if __name__=='__main__':unittest.main()
