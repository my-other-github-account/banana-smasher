from __future__ import annotations

import os

import numpy as np
import pytest

from banana_smasher.qtip_periodic import states_from_symbols
from banana_smasher.qtip_periodic_cuda import (
    periodic_cuda_geometry,
    solve_periodic_cuda,
)


def test_periodic_cuda_geometry_is_full_branch_exact_k2_k3() -> None:
    assert periodic_cuda_geometry() == {
        "implementation": "periodic-k2-k3-cuda-exact-v1",
        "L": 16,
        "V": 2,
        "steps": 128,
        "transition_bits": [4, 6],
        "retained_prefix_costs": [4096, 1024],
        "branches_per_prefix": [16, 64],
        "branch_sampling": "full",
        "passes": ["open", "cyclic-closure"],
        "backpointer_dtype": "uint8",
        "fallback": None,
    }


@pytest.mark.skipif(
    not os.environ.get("BANANA_SMASHER_PERIODIC_CUDA_EXTENSION"),
    reason="exact PERIODIC CUDA extension is not installed",
)
def test_periodic_cuda_recovers_known_cyclic_zero_distortion_path() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(731)
    for _ in range(100):
        symbols = np.empty(128, dtype=np.uint8)
        symbols[0::2] = rng.integers(0, 16, size=64, dtype=np.uint8)
        symbols[1::2] = rng.integers(0, 64, size=64, dtype=np.uint8)
        states = states_from_symbols(symbols)
        if np.unique(states).size == states.size:
            break
    else:
        raise AssertionError("failed to construct a unique-state periodic wire")
    target = np.stack(
        (np.arange(128, dtype=np.float32), -np.arange(128, dtype=np.float32)),
        axis=1,
    )
    lut = np.full((1 << 16, 2), 10_000.0, dtype=np.float32)
    lut[states] = target
    observed = solve_periodic_cuda(
        torch.from_numpy(np.broadcast_to(target, (256, 128, 2)).copy()).cuda(),
        torch.from_numpy(lut).cuda(),
    )
    expected = torch.from_numpy(
        np.broadcast_to(states.astype(np.int32), (256, 128)).copy()
    )
    assert observed.device.type == "cuda"
    assert observed.dtype == torch.int32
    assert torch.equal(observed.cpu(), expected)
