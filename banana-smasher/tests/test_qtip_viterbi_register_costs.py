"""Guard the K1 LDLQ causal change; GPU parity remains a separate mandatory gate."""
import ast
from pathlib import Path


def test_generic_recurrence_has_no_global_cost_traffic():
    source = (Path(__file__).parents[1] / 'src/banana_smasher/qtip_viterbi.py').read_text()
    tree = ast.parse(source)
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_persistent_prefix_viterbi_generic')
    class RegisterBranch(ast.NodeTransformer):
        def visit_If(self, node):
            text = ast.unparse(node.test)
            if text == 'REGISTER_COSTS':
                return [self.visit(n) for n in node.body]
            if text == 'not REGISTER_COSTS':
                return [self.visit(n) for n in node.orelse]
            return self.generic_visit(node)
    kernel = RegisterBranch().visit(kernel)
    assert 'REGISTER_COSTS=K == 1' in source
    calls = [n for n in ast.walk(kernel) if isinstance(n, ast.Call)]
    gathers = [n for n in calls if ast.unparse(n.func) == 'tl.gather']
    assert gathers, 'generic LDLQ recurrence still reloads predecessor costs from global scratch'
    for call in calls:
        if ast.unparse(call.func) in ('tl.load', 'tl.store'):
            assert 'scratch_ptr' not in ast.unparse(call.args[0])
    # Branch ordering and strict comparison remain unchanged; gather is movement only.
    assert 'candidate < best' in ast.unparse(kernel)
    assert 'for q in tl.range(0, BRANCHES, loop_unroll_factor=BRANCH_UNROLL)' in ast.unparse(kernel)
