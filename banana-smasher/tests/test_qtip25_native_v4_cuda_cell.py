from __future__ import annotations

import numpy as np
import pytest
import torch

import banana_smasher.qtip25_native_v4 as native_v4
import banana_smasher.qtip25_native_v4_cuda_cell as cuda_cell
from banana_smasher.qtip1 import gaussian_tlut
from banana_smasher.banana_v1 import expand_banana_v1_codebook
from banana_smasher.qtip25_native_v4 import (
    NATIVE_QTIP25_GEOMETRY,
    NativeQtip25Geometry,
    ldlq_native_v4_matrix,
    native_v4_lower_from_hessian,
)
from banana_smasher.qtip25_native_v4_cuda_cell import validate_input


def test_scalar_pr31_b256_dispatches_to_paired_aot(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCudaTensor:
        is_cuda = True
        device = "cuda:0"
        dtype = torch.float32

        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape
            self.ndim = len(shape)

    geometry = NativeQtip25Geometry(B=12)
    target = FakeCudaTensor((256, 64, 4))
    scalar_lut = FakeCudaTensor((geometry.states,))
    expected = object()

    monkeypatch.setattr(
        native_v4,
        "_solve_native_v5_cuda_aot",
        lambda got_target, got_lut, *, geometry: (
            expected
            if got_target is target
            and got_lut is scalar_lut
            and geometry == NativeQtip25Geometry(B=12)
            else None
        ),
    )
    monkeypatch.setattr(
        native_v4,
        "_solve_native_v5_cuda",
        lambda *_args, **_kwargs: pytest.fail("B256 used the scalar eager solver"),
    )

    assert (
        native_v4.solve_native_v4_cuda(
            target,
            state_lut=scalar_lut,
            geometry=geometry,
        )
        is expected
    )


def test_scalar_b10_b256_stays_on_variable_width_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCudaTensor:
        is_cuda = True
        device = "cuda:0"
        dtype = torch.float32

        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape
            self.ndim = len(shape)

    target = FakeCudaTensor((256, 64, 4))
    scalar_lut = FakeCudaTensor((NATIVE_QTIP25_GEOMETRY.states,))
    expected = object()

    monkeypatch.setattr(
        native_v4,
        "_solve_native_v5_cuda_aot",
        lambda *_args, **_kwargs: pytest.fail("B10 entered the B12-only AOT solver"),
    )
    monkeypatch.setattr(
        native_v4,
        "_solve_native_v5_cuda",
        lambda got_target, got_lut, *, geometry: (
            expected
            if got_target is target
            and got_lut is scalar_lut
            and geometry is NATIVE_QTIP25_GEOMETRY
            else None
        ),
    )

    assert (
        native_v4.solve_native_v4_cuda(
            target,
            state_lut=scalar_lut,
            geometry=NATIVE_QTIP25_GEOMETRY,
        )
        is expected
    )


def test_native_v5_phase_accumulation_reuses_error_storage() -> None:
    errors = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    best = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    storage = errors.data_ptr()

    observed = native_v4._native_v5_accumulate_phase_cost_(
        errors,
        best,
        branches=3,
    )

    assert observed.data_ptr() == storage
    assert torch.equal(
        observed,
        torch.tensor(
            [
                [10.0, 11.0, 12.0, 23.0, 24.0, 25.0],
                [36.0, 37.0, 38.0, 49.0, 50.0, 51.0],
            ]
        ),
    )


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
    solver_targets: list[torch.Tensor] = []

    def fake_solve(target, *, state_lut, geometry):
        calls.append(tuple(target.shape))
        solver_targets.append(target.detach().cpu().clone())
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

    packed, selected_scales, optimizations = cuda_cell.ldlq_native_v4_cuda_batch(
        [target, target],
        [hessian, hessian],
        matrix_shape=(16, 32),
        state_lut=state_lut,
        geometry=NATIVE_QTIP25_GEOMETRY,
        solve_batch=2048,
        scale_factors=(1.0,),
        cell_scales=(0.9, 1.1),
    )

    assert calls == [(2, 64, 4), (2, 64, 4)]
    assert len(packed) == 2
    assert all(value.shape == (2, 80) for value in packed)
    assert selected_scales == [0.9, 1.1]
    serial_source = (
        torch.from_numpy(target.copy())
        .reshape(1, 2, 16, 16)
        .permute(0, 2, 1, 3)
        .reshape(16, 32)
        .contiguous()
    )
    expected_last_column = torch.stack(
        [
            serial_source[:, 16:32].reshape(1, 64, 4) / 0.9,
            serial_source[:, 16:32].reshape(1, 64, 4) / 1.1,
        ]
    ).reshape(2, 64, 4)
    assert torch.equal(solver_targets[0], expected_last_column)
    assert [value["selected_factor"] for value in optimizations] == [1.0, 1.0]
    assert all(value["fixed_absolute_scale"] for value in optimizations)
    assert all(value["scale_batch_size"] == 1 for value in optimizations)
    assert all(value["cell_batch_size"] == 2 for value in optimizations)


def test_native_v4_cuda_cells_use_one_homogeneous_solver_batch(monkeypatch) -> None:
    calls: list[tuple[int, ...]] = []

    def fake_solve(target, *, state_lut, geometry):
        calls.append(tuple(target.shape))
        return torch.zeros(
            (target.shape[0], target.shape[1]),
            dtype=torch.int32,
            device=target.device,
        )

    def fake_pack(states, *, geometry):
        return torch.arange(
            states.shape[0] * 80,
            dtype=torch.int64,
            device=states.device,
        ).reshape(states.shape[0], 80).to(torch.uint8)

    monkeypatch.setattr(cuda_cell, "solve_native_v4_cuda", fake_solve)
    monkeypatch.setattr(cuda_cell, "_pack_cuda_states_v4", fake_pack)
    targets = [
        np.zeros((2, 64, 4), dtype=np.float32),
        np.ones((3, 64, 4), dtype=np.float32),
    ]

    packed, counters = cuda_cell.solve_native_v4_cuda_cells(
        targets,
        state_lut=torch.ones((1 << 16, 4), dtype=torch.float32),
        geometry=NATIVE_QTIP25_GEOMETRY,
    )

    assert calls == [(5, 64, 4)]
    assert [value.shape for value in packed] == [(2, 80), (3, 80)]
    assert np.array_equal(packed[0], np.arange(160, dtype=np.uint8).reshape(2, 80))
    assert counters == {
        "cell_batch_calls": 1,
        "cells": 2,
        "blocks": 5,
        "serial_cell_solver_calls": 0,
    }


def test_native_v4_cuda_cells_chunks_solver_rows_at_256(monkeypatch) -> None:
    calls: list[tuple[int, ...]] = []

    def fake_solve(target, *, state_lut, geometry):
        calls.append(tuple(target.shape))
        return torch.zeros(
            (target.shape[0], target.shape[1]),
            dtype=torch.int32,
            device=target.device,
        )

    def fake_pack(states, *, geometry):
        return torch.zeros(
            (states.shape[0], 8 * geometry.B),
            dtype=torch.uint8,
            device=states.device,
        )

    monkeypatch.setattr(cuda_cell, "solve_native_v4_cuda", fake_solve)
    monkeypatch.setattr(cuda_cell, "_pack_cuda_states_v4", fake_pack)
    targets = [
        np.zeros((200, 64, 4), dtype=np.float32),
        np.ones((100, 64, 4), dtype=np.float32),
    ]

    packed, counters = cuda_cell.solve_native_v4_cuda_cells(
        targets,
        state_lut=torch.ones((1 << 16, 4), dtype=torch.float32),
        geometry=NATIVE_QTIP25_GEOMETRY,
    )

    assert calls == [(256, 64, 4), (44, 64, 4)]
    assert [value.shape for value in packed] == [(200, 80), (100, 80)]
    assert counters["cell_batch_calls"] == 2
    assert counters["serial_cell_solver_calls"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
def test_pr31_scalar_ldlq_cuda_matches_cpu_codes_exactly() -> None:
    rng = np.random.default_rng(31)
    matrix = rng.normal(size=(16, 16)).astype(np.float32)
    basis = np.eye(16, dtype=np.float32) + np.tril(
        rng.normal(scale=0.01, size=(16, 16)).astype(np.float32), -1
    )
    hessian = np.ascontiguousarray(basis @ basis.T)
    compact_lut = np.linspace(-3.0, 3.0, 1024, dtype=np.float16)
    scalar_lut = torch.from_numpy(expand_banana_v1_codebook(compact_lut)).to("cuda")
    lower = native_v4_lower_from_hessian(hessian)

    reference = ldlq_native_v4_matrix(
        matrix,
        lower,
        tlut=compact_lut,
        scale_factors=(1.0,),
        scale_semantics="absolute_unit",
    )
    packed, selected_scales, optimization = cuda_cell.ldlq_native_v4_cuda_batch(
        [matrix.reshape(1, 64, 4)],
        [hessian],
        matrix_shape=(16, 16),
        state_lut=scalar_lut,
        geometry=NATIVE_QTIP25_GEOMETRY,
        solve_batch=64,
        scale_factors=(1.0,),
        cell_scales=(1.0,),
    )

    assert np.array_equal(packed[0], reference.packed)
    assert selected_scales == [1.0]
    assert optimization[0]["method"] == "qtip_batch_block_ldl_reverse_16_cell_scale_batched"
