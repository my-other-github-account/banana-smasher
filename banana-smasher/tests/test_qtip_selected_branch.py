"""The alphabet recurrence retains selected q until wire encoding."""
import ast
from pathlib import Path
import pytest
SOURCE = Path(__file__).parents[1] / 'src/banana_smasher/qtip_viterbi.py'

def test_selected_branch_defers_full_state_encoding():
    tree = ast.parse(SOURCE.read_text())
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_persistent_prefix_viterbi_generic')
    loop = next(n for n in ast.walk(kernel) if isinstance(n, ast.While))
    text = ast.unparse(loop)
    assert 'chosen = tl.where(take, q if DISTANCE_ALPHABET else state, chosen)' in text
    assert 'encoded_chosen = chosen if BRANCH_POINTERS else chosen * PREFIXES + j' in text
    assert 'candidate = predecessor_cost + da * da + db * db' in text
    assert 'encoded_chosen = tl.where(best < float(\'inf\'), encoded_chosen, 0)' in text

def test_cuda_branch_encoding_preserves_all_k1_states():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    import triton
    import triton.language as tl

    @triton.jit
    def encode(out, selected, BLOCK: tl.constexpr):
        j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        for q in tl.static_range(4):
            encoded = q * 16384 + j
            tl.store(out + q * 16384 + j, encoded)
            tl.store(selected + q * 16384 + j, q)
    out = torch.empty(65536, dtype=torch.int32, device='cuda')
    selected = torch.empty_like(out)
    encode[(16,)](out, selected, 1024)
    expected = torch.arange(65536, dtype=torch.int32, device='cuda')
    assert torch.equal(out, expected)
    assert torch.equal(selected, expected // 16384)
