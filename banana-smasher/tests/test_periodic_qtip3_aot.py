from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


def _package_root() -> Path:
    spec = importlib.util.find_spec("banana_smasher")
    assert spec is not None and spec.submodule_search_locations
    return Path(next(iter(spec.submodule_search_locations)))


def test_periodic_qtip3_aot_contract_is_exact_paired_b256() -> None:
    from banana_smasher.periodic_qtip3_aot import geometry

    assert geometry() == {
        "implementation": "periodic-qtip3-pr31-paired-step-graph-b256-v1",
        "L": 16,
        "K": 3,
        "V": 1,
        "steps": 256,
        "full_states": 65536,
        "retained_prefix_costs": 8192,
        "branches_per_prefix": 8,
        "branch_sampling": "full",
        "strict_tie_order": "ascending-q-strict-less-than",
        "backpointer_dtype": "packed-u3-pair-per-byte",
        "batch_contract": "exact-contiguous-B256",
        "launch_submission": "one-cuda-graph-replay-per-pass",
        "fallback": False,
    }


def test_periodic_qtip3_aot_source_packs_every_exact_pair_in_one_graph() -> None:
    root = _package_root() / "periodic_qtip3_aot"
    source = (root / "csrc" / "periodic_qtip3_exact.cu").read_text()
    binding = (root / "csrc" / "binding.cpp").read_text()
    setup = (root / "setup.py").read_text()

    assert "constexpr int STEPS = 256;" in source
    assert "constexpr int PAIRS = 128;" in source
    assert "constexpr int PREFIXES = 8192;" in source
    assert "constexpr int BRANCHES = 8;" in source
    assert "(odd_q << 3) | (even_q & 7)" in source
    assert "cudaGraphAddKernelNode" in source
    assert "cudaGraphLaunch" in source
    assert 'm.def("viterbi"' in binding
    assert 'name="periodic_qtip3_cuda_exact"' in setup


def test_periodic_qtip3_aot_reuses_each_previous_prefix_across_low_bits() -> None:
    source = (
        _package_root()
        / "periodic_qtip3_aot"
        / "csrc"
        / "periodic_qtip3_exact.cu"
    ).read_text()

    assert "constexpr int PREVIOUS_VALUES =" in source
    assert "float* previous_cost_cache" in source
    assert "previous_cost_cache[previous_index]" in source


def test_periodic_qtip3_aot_open_terminal_ties_use_lowest_full_state() -> None:
    source = (
        _package_root()
        / "periodic_qtip3_aot"
        / "csrc"
        / "periodic_qtip3_exact.cu"
    ).read_text()

    assert "best_full_state" in source
    assert "candidate_full_state < best_full_state" in source
    assert "static_cast<int64_t>(PAIRS - 1) * BATCH" in source


def test_periodic_qtip3_cell_solver_refuses_partial_b256_before_cuda_load() -> None:
    import numpy as np
    import torch

    from banana_smasher.periodic_qtip3_aot import solve_periodic_qtip3_cells_exact

    target = np.zeros((1, 64, 4), dtype=np.float32)
    scalar_lut = torch.zeros(65536, dtype=torch.float32)

    with pytest.raises(ValueError, match="multiple of 256 blocks"):
        solve_periodic_qtip3_cells_exact([target], scalar_lut)


@pytest.mark.skipif(
    not os.environ.get("BANANA_SMASHER_PERIODIC_QTIP3_EXTENSION"),
    reason="set the sealed Periodic QTIP3 AOT extension for physical parity",
)
def test_periodic_qtip3_aot_matches_scalar_pr31_eager_with_and_without_overlap() -> None:
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        pytest.fail("Periodic QTIP3 AOT parity is a required CUDA gate, not a skip")

    from banana_smasher.banana_v1 import expand_banana_v1_codebook
    from banana_smasher.periodic_qtip3_aot import solve_periodic_qtip3_exact
    from banana_smasher.qtip25_native_v4 import (
        NativeQtip25Geometry,
        _native_v5_cuda_pass,
    )

    geometry = NativeQtip25Geometry(B=12)
    generator = torch.Generator(device="cuda").manual_seed(20260808)
    phases = torch.randn(
        (256, 256), generator=generator, device="cuda", dtype=torch.float32
    )
    scalar_lut = torch.from_numpy(
        np.ascontiguousarray(expand_banana_v1_codebook(), dtype=np.float32)
    ).cuda()
    values = phases.transpose(0, 1).reshape(256, 64, 4).contiguous()

    expected_open = _native_v5_cuda_pass(
        values, scalar_lut, None, geometry=geometry
    ).reshape(256, 256)
    actual_open = solve_periodic_qtip3_exact(phases, scalar_lut).transpose(0, 1)
    assert torch.equal(actual_open, expected_open)

    overlap = (expected_open[:, 128] >> 3).to(torch.int32).contiguous()
    expected_closed = _native_v5_cuda_pass(
        values, scalar_lut, overlap, geometry=geometry
    ).reshape(256, 256)
    actual_closed = solve_periodic_qtip3_exact(
        phases, scalar_lut, overlap=overlap
    ).transpose(0, 1)
    assert torch.equal(actual_closed, expected_closed)
