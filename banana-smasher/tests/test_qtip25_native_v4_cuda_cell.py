from __future__ import annotations

import numpy as np
import pytest
import torch

import banana_smasher.qtip25_native_v4_cuda_cell as cuda_cell
from banana_smasher.qtip1 import gaussian_tlut
from banana_smasher.qtip25_native_v4 import NATIVE_QTIP25_GEOMETRY
from banana_smasher.qtip25_native_v4_cuda_cell import validate_input


def test_native_v4_cuda_cell_preflight_binds_exact_basis_and_geometry(tmp_path) -> None:
    target_path = tmp_path / "target.npy"
    tlut_path = tmp_path / "tlut.npy"
    np.save(target_path, np.zeros((3, 64, 4), dtype=np.float32), allow_pickle=False)
    np.save(tlut_path, gaussian_tlut(bits=9, columns=2), allow_pickle=False)

    target, tlut, identity = validate_input(
        target_path,
        tlut_path,
        intended_basis_sha256="9" * 64,
        observed_basis_sha256="9" * 64,
    )
    assert target.shape == (3, 64, 4)
    assert tlut.shape == (512, 2)
    assert identity["basis_sha256"] == "9" * 64
    with pytest.raises(ValueError, match="basis mismatch"):
        validate_input(
            target_path,
            tlut_path,
            intended_basis_sha256="9" * 64,
            observed_basis_sha256="8" * 64,
        )


def test_ldlq_batches_scale_candidates_on_solver_axis(monkeypatch) -> None:
    calls: list[tuple[int, ...]] = []

    def fake_solve(target, *, state_lut, geometry):
        calls.append(tuple(target.shape))
        return torch.zeros(
            (target.shape[0], target.shape[1]),
            dtype=torch.int32,
            device=target.device,
        )

    monkeypatch.setattr(cuda_cell, "solve_native_v4_cuda", fake_solve)
    monkeypatch.setattr(
        cuda_cell,
        "_pack_cuda_states_v4",
        lambda states, *, geometry: torch.zeros(
            (states.shape[0], 8 * geometry.B), dtype=torch.uint8, device=states.device
        ),
    )
    target = np.linspace(-1.0, 1.0, 512, dtype=np.float32).reshape(2, 64, 4)
    basis = np.eye(32, dtype=np.float32) + np.tril(
        np.full((32, 32), 0.01, dtype=np.float32), -1
    )
    hessian = np.ascontiguousarray(basis @ basis.T)
    state_lut = torch.ones((1 << 16, 4), dtype=torch.float32)

    packed, selected_scale, optimization = cuda_cell._ldlq_cuda_matrix(
        target,
        hessian,
        matrix_shape=(16, 32),
        state_lut=state_lut,
        geometry=NATIVE_QTIP25_GEOMETRY,
        solve_batch=2048,
        scale_factors=(0.9, 1.0, 1.1),
    )

    assert calls == [(3, 64, 4), (3, 64, 4)]
    assert packed.shape == (2, 80)
    assert selected_scale > 0
    assert optimization["scale_batch_size"] == 3
