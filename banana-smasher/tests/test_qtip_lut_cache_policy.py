"""Only immutable LUT loads retain L1 cache eviction priority."""
import ast
from pathlib import Path

def test_lut_load_cache_policy():
    t=ast.parse((Path(__file__).parents[1]/'src/banana_smasher/qtip_viterbi.py').read_text())
    f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='_persistent_prefix_viterbi_generic')
    loads=[n for n in ast.walk(f) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='load']
    lut=[n for n in loads if 'lut_ptr' in ast.unparse(n.args[0])]
    assert len(lut)==5
    for n in lut:assert any(k.arg=='eviction_policy' and ast.literal_eval(k.value)=='evict_last' for k in n.keywords)
    for n in loads:
        if n not in lut:assert not any(k.arg=='cache_modifier' for k in n.keywords)
