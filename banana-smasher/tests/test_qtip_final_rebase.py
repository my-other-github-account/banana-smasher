"""The unconditioned seed retains original ordered arithmetic and offsets."""
import ast
from pathlib import Path
SOURCE=Path(__file__).parents[1]/'src/banana_smasher/qtip_viterbi.py'
def test_seed_path_has_original_delta_arithmetic():
    tree=ast.parse(SOURCE.read_text());kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_persistent_prefix_viterbi_generic')
    guards=[n for n in ast.walk(kernel) if isinstance(n,ast.If) and any(isinstance(a,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='minimum_cost' for t in a.targets) for a in n.body)]
    assert len(guards)==1
    guard=guards[0]
    code=compile(ast.Expression(guard.test),str(SOURCE),'eval')
    assert not eval(code,dict(DISTANCE_ALPHABET=True,HAS_OVERLAP=False,CONDITIONED_DISTANCE_SUM=True)), 'seed cost rebase must be disabled'
    assert eval(code,dict(DISTANCE_ALPHABET=True,HAS_OVERLAP=True,CONDITIONED_DISTANCE_SUM=True))
    for n in ast.walk(kernel):
        if isinstance(n,ast.If) and any(isinstance(a,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='candidate' for t in a.targets) and 'distance_sum' in ast.unparse(a.value) for a in n.body):
            assert ast.unparse(n.test)=='HAS_OVERLAP and CONDITIONED_DISTANCE_SUM'
            assert 'da * da + db * db' in ast.unparse(n.orelse)
            break
    else:raise AssertionError('missing conditioned-only summed candidate')
