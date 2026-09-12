"""Actual source admission and reset tests without accelerator imports."""
import ast
import copy
import subprocess
from test_qtip_stage4_schedule import R


def test_kernel_policy_and_disabled_identity():
    path = 'banana-smasher/src/banana_smasher/qtip_viterbi.py'
    old = ast.parse(subprocess.check_output(['git', '-C', str(R), 'show', '2aa5ef8e46df68156159318adeb9f8f0de1bfa56:' + path], text=True))
    source = ast.parse((R/'qtip_viterbi.py').read_text())
    generic = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and any(a.arg == 'FUSED_SCHEDULE' for a in n.args.args))
    names = [a.arg for a in generic.args.args]
    assert names[-1] == 'TRACEBACK_L2'
    assert ast.literal_eval(generic.args.defaults[-1]) is False
    loads = [n for n in ast.walk(generic) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'load' and any(isinstance(x, ast.Name) and x.id == 'best_state_ptr' for x in ast.walk(n))]
    assert len(loads) == 2
    for n in loads:
        kw = next(k for k in n.keywords if k.arg == 'cache_modifier')
        assert ast.dump(kw.value) == ast.dump(ast.parse("'.cg' if TRACEBACK_L2 else ''", mode='eval').body)
    launches = [n for n in ast.walk(source) if isinstance(n, ast.keyword) and n.arg == 'TRACEBACK_L2']
    assert len(launches) == 1 and isinstance(launches[0].value, ast.Name) and launches[0].value.id == 'traceback_l2'
    class Disable(ast.NodeTransformer):
        def visit_FunctionDef(self, n):
            if n.name == 'resolve_traceback_l2': return None
            if n.args.args and n.args.args[-1].arg == 'TRACEBACK_L2':
                n.args.args.pop(); n.args.defaults.pop()
            return self.generic_visit(n)
        def visit_Assign(self, n):
            if any(isinstance(t, ast.Name) and t.id == 'traceback_l2' or isinstance(t, ast.Attribute) and t.attr == '_banana_smasher_traceback_l2' for t in n.targets): return None
            return self.generic_visit(n)
        def visit_Call(self, n):
            n.keywords = [k for k in n.keywords if k.arg != 'TRACEBACK_L2' and not (k.arg == 'cache_modifier' and isinstance(k.value, ast.IfExp) and isinstance(k.value.test, ast.Name) and k.value.test.id == 'TRACEBACK_L2')]
            return self.generic_visit(n)
    assert ast.dump(Disable().visit(copy.deepcopy(source))) == ast.dump(old)

import types
import pytest
from test_qtip_stage4_schedule import module, load, config, nodes


def batch(monkeypatch):
    module(monkeypatch)
    def common(label, values):
        if any(v != values[0] for v in values):
            raise ValueError(label)
        return values[0]
    return load('qtip_batch_controller.py', '_traceback_l2', dict(__package__='banana_smasher', _common=common))


@pytest.mark.parametrize('projection', ['down', 'fused13'])
def test_batch_admission(monkeypatch, projection):
    f = batch(monkeypatch)
    c = config(projection)
    c['viterbi_traceback_l2'] = True
    assert f([c, c.copy()]) is True
    c['viterbi_conditioned_distance_sum' if projection == 'fused13' else 'viterbi_fused_schedule'] = None
    assert f([c, c.copy()]) is True
    for value in (False, None):
        c['viterbi_traceback_l2'] = value
        assert f([c, c.copy()]) is False
    assert f([{}, {}]) is False


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('change', [
    {'viterbi_traceback_l2': 1}, {'viterbi_traceback_l2': 0},
    {'viterbi_stage4_schedule': 1}, {'viterbi_stage4_schedule': None},
    {'viterbi_distance_alphabet': 1}, {'viterbi_distance_alphabet': None},
    {'viterbi_conditioned_distance_sum': 1}, {'viterbi_conditioned_distance_sum': None},
    {'viterbi_fused_schedule': 0}, {'viterbi_num_warps': 8},
])
def test_batch_validates_each_member(monkeypatch, reverse, change):
    good = config()
    good['viterbi_traceback_l2'] = True
    bad = dict(good, **change)
    pair = [good, bad]
    if reverse: pair.reverse()
    with pytest.raises(ValueError): batch(monkeypatch)(pair)


def test_batch_entry_calls_admission():
    import ast
    main = nodes('qtip_batch_controller.py')['main_batch']
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == '_traceback_l2' for n in ast.walk(main))

@pytest.mark.parametrize('projection', ['down', 'fused13'])
def test_configure_identity_and_reset(monkeypatch, projection):
    module(monkeypatch)
    env = dict(__package__='banana_smasher', known_qtip_geometries=lambda:[(16,1,2)],
               backend_for_geometry=lambda g:'persistent', PERSISTENT_BACKENDS={'persistent'},
               _install_profiled_exact_viterbi=lambda *a,**k:{})
    install = load('solver_qtip_profile.py', '_install_configured_viterbi', env)
    cb = types.SimpleNamespace(L=16,K=1,V=2)
    c = config(projection)
    c['viterbi_traceback_l2'] = True
    identity = install(cb,None,None,c,profile_mode=False)
    assert cb._banana_smasher_traceback_l2 is True
    assert identity['viterbi_traceback_l2'] is True
    assert identity['production_default'] is False
    for value in (None, False, 'absent'):
        if value == 'absent': c.pop('viterbi_traceback_l2')
        else: c['viterbi_traceback_l2'] = value
        cb._banana_smasher_traceback_l2 = True
        identity = install(cb,None,None,c,profile_mode=False)
        assert cb._banana_smasher_traceback_l2 is False
        assert identity.get('viterbi_traceback_l2', False) is False
    public = load('qtip_viterbi.py', 'install_exact_prefix_viterbi', dict(_require_triton=lambda:None,types=types,geometry=lambda cb:{}))
    cb._banana_smasher_traceback_l2 = True
    public(cb)
    assert cb._banana_smasher_traceback_l2 is False
