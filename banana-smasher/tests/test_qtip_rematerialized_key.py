import ast
from pathlib import Path

SOURCE = Path(__file__).parents[1] / 'src/banana_smasher/qtip_viterbi.py'

def test_key_is_rematerialized_inside_recurrence():
    tree = ast.parse(SOURCE.read_text())
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_persistent_prefix_viterbi_generic')
    loop = next(n for n in kernel.body if isinstance(n, ast.While))
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == '_rematerialized_alphabet_key' for n in ast.walk(loop))
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_rematerialized_alphabet_key')
    call = next(n for n in ast.walk(helper) if isinstance(n, ast.Call))
    assert next(k.value.value for k in call.keywords if k.arg == 'is_pure') is False

def test_physical_rematerialized_key():
    import pytest
    torch = pytest.importorskip('torch')
    triton = pytest.importorskip('triton')
    import triton.language as tl
    from banana_smasher.qtip_viterbi import _rematerialized_alphabet_key
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    @triton.jit
    def keys(out):
        i = tl.program_id(0) * 256 + tl.arange(0, 256)
        tl.store(out+i, _rematerialized_alphabet_key(i))
    out = torch.empty(65536, device='cuda', dtype=torch.int32)
    keys[(256,)](out)
    states = torch.arange(65536, device='cuda', dtype=torch.int64)
    expected = ((states * (states + 1)) >> 6) & 1023
    assert torch.equal(out.long(), expected)

def test_alphabet_key_rematerialization_includes_fused():
    tree = ast.parse(SOURCE.read_text())
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_persistent_prefix_viterbi_generic')
    assignments = [n for n in ast.walk(kernel) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'alphabet_key' for t in n.targets)]
    assert len(assignments) == 1
    assert isinstance(assignments[0].value, ast.Call)
    assert assignments[0].value.func.id == '_rematerialized_alphabet_key'
