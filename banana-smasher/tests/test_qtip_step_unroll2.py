"""Bounded recurrence-unrolling experiment schedule contract."""
import ast
from pathlib import Path
import unittest

class StepUnrollTest(unittest.TestCase):
    def test_generic_recurrence_visits_steps_once_with_factor_two(self):
        tree=ast.parse((Path(__file__).parents[1]/'src/banana_smasher/qtip_viterbi.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_persistent_prefix_viterbi_generic')
        loops=[n for n in ast.walk(fn) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step']
        self.assertEqual(len(loops),1)
        self.assertEqual(ast.unparse(loops[0].iter),'tl.range(1, STEPS, loop_unroll_factor=2)')
        self.assertFalse(any(isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='step' for n in ast.walk(fn)))

if __name__=='__main__': unittest.main()
