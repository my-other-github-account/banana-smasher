import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

MODULE = Path(__file__).with_name('controller.py')

class ControllerTest(unittest.TestCase):
    def test_completed_gate_scores_and_reads_back(self):
        self.assertTrue(MODULE.exists(), 'controller missing: old sensor trusts file existence')
        spec = importlib.util.spec_from_file_location('controller', MODULE)
        c = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(c)
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)
            identity = dict(model_index_sha256='a'*64, artifact_sha256='b'*64,
                            teacher_sha256='c'*64, scorer_sha256='d'*64,
                            window_ids=list(range(100,164)), support_sha256='e'*64,
                            position_policy='frozen-1024', reference='sealed-reference')
            gate = dict(id='q4', owner='owner-a', host='local', marker=str(p/'done'),
                        result=str(p/'score.json'), expected=identity, status='pending',
                        score_argv=[sys.executable, str(p/'score.py')])
            (p/'done').touch()
            (p/'score.py').write_text('import json\nfrom pathlib import Path\n'
                + 'r=' + repr(dict(identity=identity, metric='forward_kl', value=0.1,
                                   window_values=[0.1]*64)) + '\n'
                + 'Path('+repr(gate['result'])+').write_text(json.dumps(r))\n')
            result = c.evaluate_gate(gate, c.LocalReader(), execute=True, owner='owner-a')
            self.assertEqual(result['status'], 'verified')
            self.assertAlmostEqual(result['value'], 0.1)
            self.assertTrue(Path(gate['result']).exists())
            self.assertIn('result_sha256', result)

    def test_failures_identity_and_queue_ownership(self):
        spec = importlib.util.spec_from_file_location('controller', MODULE)
        c = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(c)
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)
            gate = dict(id='failed', owner='owner-a', host='local', marker=str(p/'done'),
                        result=str(p/'score.json'), expected={}, score_argv=[sys.executable, '-c', 'raise SystemExit(7)'])
            (p/'done').touch()
            gate['value'] = .042085
            gate['result_sha256'] = 'obsolete'
            failed = c.evaluate_gate(gate, c.LocalReader(), execute=True, owner='owner-a')
            self.assertNotIn('value', failed)
            self.assertNotIn('result_sha256', failed)
            self.assertEqual(failed['status'], 'pending')
            self.assertIn('exit 7', failed['reason'])
            self.assertEqual(failed['id'], 'failed')
            gate['score_argv'] = [sys.executable, '-c', 'print(0.042085)']
            self.assertEqual(c.evaluate_gate(gate, c.LocalReader(), True, 'owner-a')['status'], 'pending')
            (p/'score.json').touch()
            self.assertEqual(c.evaluate_gate(gate, c.LocalReader())['status'], 'pending')
            expected = {k: 'frozen' for k in c.IDENTITY}
            expected['window_ids'] = list(range(100,164))
            for field in ('teacher_sha256', 'window_ids', 'model_index_sha256'):
                wrong = dict(expected)
                wrong[field] = 'native-or-wrong-windows'
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    c.verify_metric(json.dumps(dict(identity=wrong, metric='forward_kl', value=.042085,
                                                   window_values=[.042085]*64)).encode(), expected)
            queue = [dict(id='blocked', owner='a', status='blocked', priority=0, resource='q1', start=0, end=10),
                     dict(id='overlap', owner='b', status='ready', priority=1, resource='q1', start=0, end=10),
                     dict(id='actionable', owner='c', status='ready', priority=2, resource='q1', start=10, end=20)]
            claims = [dict(host='busy', resource='q1', start=0, end=10, owner='a', active=True)]
            self.assertTrue(hasattr(c, 'select_work'), 'old sensor chooses blocked queue head and overlaps owned range')
            self.assertEqual(c.select_work(queue, claims, 'idle')['id'], 'actionable')
            self.assertIsNone(c.select_work(queue, claims, 'busy'))
            c.retain_gates(p/'GATES.tsv', [failed])
            self.assertEqual(c.read_rows(p/'GATES.tsv')[0]['id'], 'failed')
            self.assertEqual(c.read_rows(p/'GATES.tsv')[0]['status'], 'pending')

    def test_sensor_is_read_only_and_deduplicates_frontiers(self):
        spec = importlib.util.spec_from_file_location('controller', MODULE)
        c = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(c)
        self.assertTrue(hasattr(c, 'tick'), 'bounded read-only control tick missing')
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)
            g = dict(id='pending', owner='a', host='local', marker=str(p/'done'),
                     result=str(p/'metric'), expected={}, score_argv=[sys.executable, '-c', 'raise RuntimeError()'])
            (p/'done').touch()
            c.retain_gates(p/'GATES.tsv', [g])
            c.retain_gates(p/'QUEUE.tsv', [])
            f = dict(id='f1', owner='a', host='local', root=str(p), pattern='done')
            c.retain_gates(p/'FRONTIERS.tsv', [f, dict(f, id='f2')])
            before = (p/'GATES.tsv').read_bytes()
            observation = c.tick(p)
            self.assertEqual(len(observation['frontiers']), 1)
            self.assertEqual(observation['gates'][0]['status'], 'pending')
            self.assertIn('QUEUE_EMPTY', observation['warnings'])
            self.assertFalse((p/'metric').exists())
            self.assertEqual(before, (p/'GATES.tsv').read_bytes())

if __name__ == '__main__':
    unittest.main()
