"""Keep generic traceback code size independent of the sequence length."""
import ast
from pathlib import Path
import unittest


class TracebackCodeSizeTest(unittest.TestCase):
    def test_generic_traceback_is_rolled_and_visits_every_step(self):
        path = Path(__file__).parents[1] / 'src/banana_smasher/qtip_viterbi.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_persistent_prefix_viterbi_generic')
        loop = next(n for n in ast.walk(function) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'back_step')
        self.assertEqual(ast.unparse(loop.iter), 'range(STEPS - 1, -1, -1)')
        for steps in (1, 2, 7, 128):
            self.assertEqual(list(eval(compile(ast.Expression(loop.iter), '<loop>', 'eval'), {'STEPS': steps})), list(reversed(list(range(steps)))))


if __name__ == '__main__':
    unittest.main()
