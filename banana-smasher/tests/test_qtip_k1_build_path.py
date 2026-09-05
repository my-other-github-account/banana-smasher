"""K=1..4 admission on the single QTIP build path (regression for the off-main K1 patch set)."""
import types

import pytest

torch = pytest.importorskip("torch")

from banana_smasher import qtip_batch, qtip_batch_controller, solver_qtip_profile as solver  # noqa: E402


def _cb(K, V=2, L=16):
    return types.SimpleNamespace(L=L, K=K, V=V, idx_dtype=torch.int32)


def test_cross_unit_batch_gate_is_ring_wide_not_k_specific():
    src = open(qtip_batch.__file__).read()
    assert "K in 1..4" in src
    assert "sealed for current K2" not in src and "sealed for task K1" not in src
    # geometry/decoder args come from the codebook, not literals
    assert '"K": int(codebook.K)' in src and "int(codebook.V) - 1," in src


def test_controller_geometry_gate_is_ring_wide():
    src = open(qtip_batch_controller.__file__).read()
    assert "sealed_geometry != (16, 2, 2)" not in src
    assert "sealed_geometry != (16, 1, 2)" not in src
    assert 'L=int(geometry["L"])' in src and 'K=int(geometry["K"])' in src


def test_builder_memory_contract_k1_doubles_k2_unchanged():
    src = torch.zeros(56, 28)
    c2 = solver._bind_builder_memory_contract(_cb(2), src)
    c1 = solver._bind_builder_memory_contract(_cb(1), src)
    assert c2["state_elements"] == src.numel() // 2
    assert c1["state_elements"] == 2 * (src.numel() // 2)


def test_profiled_viterbi_routes_wide_calls_to_canonical_chunker():
    src = open(solver.__file__).read()
    assert "base_quantize_seq = getattr(cb, \"quantize_seq\", None)" in src
    assert "int(x.shape[1]) > 8192" in src
