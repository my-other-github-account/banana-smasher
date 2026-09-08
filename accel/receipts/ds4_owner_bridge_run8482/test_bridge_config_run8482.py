import importlib.util
import json
from pathlib import Path
import unittest

class ConfigBindingTest(unittest.TestCase):
    def test_preserve_science_and_rebind_runtime(self):
        path=Path('bridge_config_run8482.py')
        self.assertTrue(path.exists(), 'missing runtime-only config rebinder')
        spec=importlib.util.spec_from_file_location('bridge_config',path)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        src=json.loads(Path('publication-run8461/accel/receipts/ds4_current_k3_sixcell_run8478/candidate_L009_E090.json').read_text())
        snapshot=json.dumps(src,sort_keys=True)
        for grouped in (False,True):
            out=mod.bind_config(src,'pin','/api/runner.py','/out/manifest.json','a'*64,grouped)
            self.assertEqual(json.dumps(src,sort_keys=True),snapshot)
            changes={k for k in src.keys()|out.keys() if src.get(k)!=out.get(k)}
            self.assertLessEqual(changes,{'qtip_runner','exact_solver','materialization','block_ldl_unitwise'})
            self.assertEqual(out['block_ldl_unitwise'],grouped)
            self.assertEqual(out['materialization']['run_manifest_sha256'],'a'*64)
            self.assertEqual(out['rht_seed'],src['rht_seed'])
if __name__=='__main__':unittest.main()
