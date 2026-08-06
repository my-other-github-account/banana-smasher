from __future__ import annotations

from banana_smasher.qtip_periodic_cuda import periodic_cuda_geometry


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
