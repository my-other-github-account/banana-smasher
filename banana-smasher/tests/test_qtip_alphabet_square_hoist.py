"""Research contract: square each alphabet difference before expansion."""
import ast
from pathlib import Path
import unittest

class SquareHoistTests(unittest.TestCase):
    def test_squares_are_computed_before_branch_expansion(self):
        path = Path(__file__).parents[1] / 'src/banana_smasher/qtip_viterbi.py'
        tree = ast.parse(path.read_text())
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.While)
                    and any(isinstance(x, ast.If) and isinstance(x.test, ast.Name)
                            and x.test.id == 'DISTANCE_ALPHABET' for x in n.body))
        branch = next(n for n in loop.body if isinstance(n, ast.For))
        pre = loop.body[:loop.body.index(branch)]
        names = {n.targets[0].id for node in pre for n in ast.walk(node)
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and isinstance(n.value, ast.BinOp) and isinstance(n.value.op, ast.Mult)}
        self.assertTrue({'square_a', 'square_b'} <= names, names)

if __name__ == '__main__':
    unittest.main()
