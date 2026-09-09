"""Offline regression checks for the independently authenticated owner handoff."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'receipts/glm_reference_owner_adoption_run8551'
def load(name):
    return json.loads((ROOT / name).read_text())

class ReferenceOwnerAdoption(unittest.TestCase):
    def test_manifest_is_complete_and_hash_bound(self):
        manifest = load('MANIFEST.json')
        expected = {r['path'] for r in manifest['files']}
        actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p.name != 'MANIFEST.json'}
        self.assertEqual(expected, actual)
        for row in manifest['files']:
            b = (ROOT / row['path']).read_bytes()
            self.assertEqual(len(b), row['bytes'])
            self.assertEqual(hashlib.sha256(b).hexdigest(), row['sha256'])

    def test_actual_new_pair_and_physical_remote_bindings(self):
        proof = load('PRODUCTION_ADOPTION_READBACK_run8551.json')
        remote = load('OWNER_REMOTE_READBACK_run8551.json')
        self.assertEqual(proof['new_outputs'], 2)
        self.assertEqual(proof['files_verified'], remote['files'])
        self.assertEqual(proof['files_count'], len(remote['files']))
        self.assertEqual([r['cell'] for r in proof['rows']], ['L019/E000_down', 'L019/E001_down'])
        self.assertTrue(all(not p['exists'] for p in remote['processes'].values()))
        self.assertEqual(load('MANIFEST.json')['physical_units'], {k:v for k,v in remote['files'].items() if k.endswith('.pt')})

    def test_config_recipe_and_shared_process(self):
        remote = load('OWNER_REMOTE_READBACK_run8551.json')
        processes = []
        for expert in (0, 1):
            cell = f'L019/E{expert:03d}_down'
            config = load(f'actual_product/L019_E{expert:03d}_down_K1.json')
            receipt = load(f'actual_product/solve/{cell}/QTIP_SOLVE_RECEIPT.json')
            prepared = remote['prepared_configs'][cell]['config']
            self.assertTrue(config['block_ldl_unitwise'] and config['block_ldl_reference'])
            for key in ('geometry', 'fit_windows', 'rht_seed', 'input_identity', 'training_ledger_sha256'):
                self.assertEqual(config[key], prepared[key])
            self.assertEqual(receipt['fit_windows'], 16)
            self.assertEqual(receipt['rht_seed'], config['rht_seed'])
            self.assertTrue(receipt['build']['packed_decode']['fp16_bit_exact'])
            self.assertTrue(receipt['build']['canonical_pack']['canonical_pack_roundtrip_exact'])
            processes.append(receipt['cross_unit_batch']['process'])
        self.assertEqual(processes[0], processes[1])
        self.assertEqual(processes[0], {'pid': 1362343, 'startticks': 8481633})

    def test_scope_and_no_replay_are_explicit(self):
        proof = load('PRODUCTION_ADOPTION_READBACK_run8551.json')
        self.assertTrue(proof['no_replay'] and proof['no_foreign_mutation'])
        self.assertFalse(proof['speedup_claim'])
        self.assertIn('not independently rerun', proof['quality_scope'])
        self.assertEqual(proof['runtime_pin'], 'aa455253fa0cd8944a4148112d44c537938d40c4')

if __name__ == '__main__':
    unittest.main()
