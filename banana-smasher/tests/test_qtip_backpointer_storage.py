"""CPU admission tests; physical paired canary remains mandatory."""
import ast
from pathlib import Path
import pytest

SOURCE = Path(__file__).parents[1] / 'src/banana_smasher/qtip_viterbi.py'

def resolver():
    tree = ast.parse(SOURCE.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'resolve_backpointer_dtype']
    assert len(nodes) == 1, 'missing backpointer storage admission'
    env = {}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),env)
    return env['resolve_backpointer_dtype']

@pytest.mark.parametrize('requested,expected', [(None,'int32'),('int32','int32'),('uint16','uint16'),('uint8','uint8')])
def test_storage_default_and_opt_in(requested,expected):
    assert resolver()((16,1,2),requested) == expected

@pytest.mark.parametrize('requested', [True,16,'int8','float16',''])
def test_storage_invalid_requested(requested):
    with pytest.raises(ValueError): resolver()((16,1,2),requested)

@pytest.mark.parametrize('geometry', [(17,1,2),(16,2,2),(16,3,2),(16,1,1)])
@pytest.mark.parametrize('dtype', ['uint16', 'uint8'])
def test_opt_in_restricted_to_measured_k1_ring(geometry, dtype):
    with pytest.raises(ValueError): resolver()(geometry,dtype)


@pytest.mark.parametrize("dtype", ["uint16", "uint8"])
def test_configured_installer_binds_storage_and_reports_workspace(monkeypatch, dtype):
    import types, sys
    source = SOURCE.with_name('solver_qtip_profile.py')
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name == '_install_configured_viterbi')
    package=types.ModuleType('banana_smasher'); package.__path__=[]
    module=types.ModuleType('banana_smasher.qtip_viterbi')
    module.resolve_viterbi_num_warps=lambda *a:16
    module.resolve_backpointer_dtype=resolver()
    module.resolve_branch_unroll = lambda *a: 1
    module.resolve_structured_gather = lambda *a: False
    monkeypatch.setitem(sys.modules,'banana_smasher',package)
    monkeypatch.setitem(sys.modules,'banana_smasher.qtip_viterbi',module)
    env=dict(Any=object,_ExactTimers=object,__package__='banana_smasher',known_qtip_geometries=lambda:{(16,1,2)},
        backend_for_geometry=lambda g:'persistent',PERSISTENT_BACKENDS={'persistent'},
        _install_profiled_exact_viterbi=lambda *a,**kw:{'production_default':True,'best_state_dtype':'int32'})
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),env)
    cb=types.SimpleNamespace(L=16,K=1,V=2)
    result=env['_install_configured_viterbi'](cb,None,None,
        {'geometry':{'L':16,'K':1,'V':2},'viterbi_backpointer_dtype':dtype},profile_mode=False)
    assert getattr(cb,'_banana_smasher_backpointer_dtype',None)==dtype
    assert result['best_state_dtype']==dtype
    assert result['best_state_encoding'] == ('branch-index' if dtype == 'uint8' else 'full-state')
    assert result['production_default'] is False


def test_kernel_allocates_requested_storage_but_returns_int32():
    # Execute the allocation AST with a tiny fake allocator; no GPU needed.
    import types
    tree=ast.parse(SOURCE.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='exact_prefix_viterbi')
    selected=[]
    for node in fn.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('backpointer_dtype','best_state','states') for t in node.targets):
            selected.append(node)
    calls=[]
    torch=types.SimpleNamespace(int32='int32',uint16='uint16',empty=lambda shape,**kw:calls.append((shape,kw)) or kw)
    env=dict(torch=torch,steps=128,batch=2,prefixes=16384,x=types.SimpleNamespace(device='cuda'),L=16,K=1,V=2,
        cb=types.SimpleNamespace(_banana_smasher_backpointer_dtype='uint16'),resolve_backpointer_dtype=resolver())
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(SOURCE),'exec'),env)
    assert env['best_state']['dtype']=='uint16'
    assert env['states']['dtype']=='int32'


@pytest.mark.parametrize("compressed", [False, True])
def test_actual_kernel_backpointer_roundtrip_all_k1_states(compressed):
    # Execute the actual store expressions and traceback restoration AST.
    import numpy as np
    import types
    fn = next(n for n in ast.parse(SOURCE.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == '_persistent_prefix_viterbi_generic')
    stores = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute) and n.func.attr == 'store'
              and 'best_state_ptr' in ast.unparse(n.args[0])]
    assert len(stores) == 2
    j = np.tile(np.arange(16384, dtype=np.int32), 4)
    chosen = np.arange(65536, dtype=np.int32)
    for store in stores:
        env = dict(chosen=chosen, PREFIXES=16384, BRANCH_POINTERS=compressed)
        saved = eval(compile(ast.Expression(store.args[1]), str(SOURCE), 'eval'), env)
        saved = saved.astype(np.uint8 if compressed else np.int32).astype(np.int32)
        env = dict(state=saved, prefix=j, PREFIXES=16384, BRANCH_POINTERS=compressed)
        restores = [n for n in ast.walk(fn) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == 'BRANCH_POINTERS']
        for restore in restores:
            exec(compile(ast.Module(body=[restore], type_ignores=[]), str(SOURCE), 'exec'), env)
        assert np.array_equal(env['state'], chosen)


@pytest.mark.parametrize("dtype", ["int32", "uint16", "uint8"])
def test_launch_binds_branch_storage_semantics(dtype):
    fn = next(n for n in ast.parse(SOURCE.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == 'exact_prefix_viterbi')
    call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Subscript)
                and ast.unparse(n.func.value) == '_persistent_prefix_viterbi_generic')
    keywords = {k.arg: k.value for k in call.keywords}
    assert 'BRANCH_POINTERS' in keywords
    value = eval(compile(ast.Expression(keywords['BRANCH_POINTERS']), str(SOURCE), 'eval'),
                 dict(backpointer_dtype=dtype))
    assert value is (dtype == 'uint8')
