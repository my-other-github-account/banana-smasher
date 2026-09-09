"""CPU structural coverage; CUDA/PRE evidence is measured separately."""
import ast
from pathlib import Path


def test_k1_streaming_store_policy():
    path = Path(__file__).resolve().parents[1] / "src/banana_smasher/qtip_viterbi.py"
    source = ast.parse(path.read_text())
    fn = next(n for n in source.body if isinstance(n, ast.FunctionDef)
              and n.name == "_persistent_prefix_viterbi_generic")
    stores = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute) and n.func.attr == "store"
              and any(isinstance(z, ast.Name) and z.id == "best_state_ptr"
                      for z in ast.walk(n.args[0]))]
    assert len(stores) == 2
    for call in stores:
        kw = {k.arg: k.value for k in call.keywords}
        assert "cache_modifier" in kw
        expr = compile(ast.Expression(kw["cache_modifier"]), "<store-policy>", "eval")
        for prefixes in [1, 16, 256, 1024, 4096, 16384]:
            assert eval(expr, {}, dict(PREFIXES=prefixes)) == (
                ".cs" if prefixes == 16384 else "")


if __name__ == "__main__":
    test_k1_streaming_store_policy()
    print("PASS both actual stores: K1 streaming, other geometries unchanged")
