"""CPU-only public batch admission and executed call wiring."""
import ast
from pathlib import Path
import pytest

SOURCE = Path(__file__).parents[1] / 'src/banana_smasher/qtip_batch_controller.py'

def resolver():
    tree = ast.parse(SOURCE.read_text())
    names = {'_common', '_capture_hash_workers'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names, 'batch capture worker admission missing'
    ns = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])), str(SOURCE), 'exec'), ns)
    return ns['_capture_hash_workers']

def test_default_and_opt_in():
    f = resolver()
    assert f([{}, {}]) == 1
    for value in (1, 2, 4):
        assert f([{'capture_hash_workers': value}] * 2) == value

@pytest.mark.parametrize('bad', [None, True, False, 0, 3, 8, 1.0, 4.0, '4'])
@pytest.mark.parametrize('reverse', [False, True])
def test_every_member_strict_before_homogeneity(bad, reverse):
    configs = [{'capture_hash_workers': 1}, {'capture_hash_workers': bad}]
    if reverse:
        configs.reverse()
    with pytest.raises(ValueError, match='capture hash workers'):
        resolver()(configs)

@pytest.mark.parametrize('values', [(1, 2), (4, 1), (2, 4)])
def test_mixed_workers_refused(values):
    with pytest.raises(ValueError):
        resolver()([{'capture_hash_workers': v} for v in values])

def test_public_batch_routes_validated_workers_and_receipt():
    tree = ast.parse(SOURCE.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main_batch')
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
    routed = next(n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == '_load_captures')
    kw = next((k for k in routed.keywords if k.arg == 'hash_workers'), None)
    assert kw is not None and isinstance(kw.value, ast.Name) and kw.value.id == 'capture_hash_workers'
    assignment = next(n for n in ast.walk(main) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'capture_hash_workers' for t in n.targets))
    assert isinstance(assignment.value, ast.Call) and isinstance(assignment.value.func, ast.Name) and assignment.value.func.id == '_capture_hash_workers'
    assert ast.dump(assignment.value.args[0]) == ast.dump(ast.Name(id='configs', ctx=ast.Load()))
    assert assignment.lineno < routed.lineno
    receipts = [n for n in ast.walk(main) if isinstance(n, ast.Dict) and any(isinstance(k, ast.Constant) and k.value == 'capture_hash_workers' for k in n.keys)]
    assert receipts, 'effective capture worker count missing from public receipt'
