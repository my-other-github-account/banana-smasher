"""GPU-free validation of opt-in launch scheduling, not output quality."""
import ast
from pathlib import Path
import pytest

SOURCE = Path(__file__).parents[1] / 'src/banana_smasher/qtip_viterbi.py'

def resolver():
    tree = ast.parse(SOURCE.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'resolve_viterbi_num_warps']
    assert len(nodes) == 1, 'missing explicit fail-closed launch schedule resolver'
    env = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), env)
    return env['resolve_viterbi_num_warps']

@pytest.mark.parametrize('requested,expected', [(None,16),(4,4),(8,8),(16,16)])
def test_k3_launch_schedule_preserves_default_and_accepts_explicit_warps(requested, expected):
    assert resolver()((16,3,2),requested) == expected

@pytest.mark.parametrize('requested', [True,0,1,2,32,'4',4.0])
def test_launch_schedule_rejects_invalid_counts(requested):
    with pytest.raises(ValueError):
        resolver()((16,3,2),requested)

@pytest.mark.parametrize('geometry', [(16,1,2),(16,2,2),(16,4,2),(16,3,1)])
def test_explicit_launch_schedule_refuses_other_geometries(geometry):
    with pytest.raises(ValueError):
        resolver()(geometry,4)


def test_configured_installer_binds_opt_in_and_reports_nondefault(monkeypatch):
    import types, sys
    source = SOURCE.with_name('solver_qtip_profile.py')
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_install_configured_viterbi')
    package = types.ModuleType('banana_smasher'); package.__path__ = []
    module = types.ModuleType('banana_smasher.qtip_viterbi')
    module.resolve_viterbi_num_warps = resolver()
    monkeypatch.setitem(sys.modules, 'banana_smasher', package)
    monkeypatch.setitem(sys.modules, 'banana_smasher.qtip_viterbi', module)
    env = dict(Any=object, _ExactTimers=object, __package__='banana_smasher',
        known_qtip_geometries=lambda: {(16,3,2)}, backend_for_geometry=lambda g:'persistent',
        PERSISTENT_BACKENDS={'persistent'},
        _install_profiled_exact_viterbi=lambda *a, **kw: {'production_default':True})
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),env)
    cb = types.SimpleNamespace(L=16,K=3,V=2)
    result=env['_install_configured_viterbi'](cb,None,None,
        {'geometry':{'L':16,'K':3,'V':2},'viterbi_num_warps':4},profile_mode=False)
    assert getattr(cb,'_banana_smasher_viterbi_num_warps',None) == 4
    assert result['viterbi_num_warps'] == 4 and result['production_default'] is False
