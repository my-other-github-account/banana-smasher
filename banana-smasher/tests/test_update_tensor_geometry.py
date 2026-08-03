from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from banana_smasher.token_sizing import MemoryBudget
from banana_smasher.update import prepare_tensor_segments, run_tensor_update

GiB = 1024**3
MiB = 1024**2


def _identity() -> dict[str, str]:
    return {
        "content_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "assignment_sha256": "3" * 64,
        "aot_sha256": "4" * 64,
        "runtime_sha256": "5" * 64,
        "code_sha256": "6" * 64,
    }


def _budget() -> MemoryBudget:
    return MemoryBudget(
        available_bytes=16 * GiB,
        resident_frozen_bytes=1 * GiB,
        trainable_bytes=1 * MiB,
        optimizer_bytes=2 * MiB,
        staging_bytes=1 * MiB,
        calibrated_activation_bytes_per_token=1 * MiB,
    )


@pytest.mark.parametrize("argument", ["requested_tokens", "segments", "batch_size"])
@pytest.mark.parametrize("invalid", [True, 1.0, 1.5])
def test_tensor_token_geometry_requires_actual_non_bool_integers(
    argument: str, invalid: object
) -> None:
    values: dict[str, Any] = {
        "requested_tokens": 2,
        "segments": 2,
        "batch_size": 1,
    }
    values[argument] = invalid
    tensors = torch.zeros((1, 4))

    with pytest.raises(TypeError, match=rf"{argument} must be an integer"):
        prepare_tensor_segments(
            input_ids=tensors,
            teacher_targets=tensors,
            teacher_mask=torch.ones_like(tensors, dtype=torch.bool),
            positions=torch.arange(4).reshape(1, -1),
            memory_budget=_budget(),
            **values,
        )


def test_tensor_update_uses_actual_1024_token_shape_and_teacher_geometry(
    tmp_path: Path,
) -> None:
    physical_tokens = 1024
    segments = 2
    logical_tokens = physical_tokens * segments
    input_ids = torch.arange(logical_tokens, dtype=torch.float64).reshape(1, -1)
    teacher_targets = torch.zeros((1, logical_tokens, 4), dtype=torch.float64)
    teacher_mask = torch.ones((1, logical_tokens), dtype=torch.bool)
    positions = torch.arange(logical_tokens, dtype=torch.int64).reshape(1, -1)
    parameter = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=1e-9)
    observed: list[
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    ] = []

    def loss_sum(segment: dict) -> torch.Tensor:
        observed.append(
            (
                tuple(segment["input_ids"].shape),
                tuple(segment["teacher_targets"].shape),
                tuple(segment["teacher_mask"].shape),
                tuple(segment["positions"].shape),
            )
        )
        prediction = (
            (parameter * segment["input_ids"])
            .unsqueeze(-1)
            .expand_as(segment["teacher_targets"])
        )
        error = prediction - segment["teacher_targets"]
        return error.masked_select(segment["teacher_mask"].unsqueeze(-1)).square().sum()

    receipt = run_tensor_update(
        parameters=[parameter],
        optimizer=optimizer,
        input_ids=input_ids,
        teacher_targets=teacher_targets,
        teacher_mask=teacher_mask,
        positions=positions,
        requested_tokens=physical_tokens,
        segments=segments,
        batch_size=1,
        memory_budget=_budget(),
        loss_sum=loss_sum,
        output=tmp_path / "update.pt",
        identity=_identity(),
        peak_memory_bytes=lambda: 1234,
    )

    assert observed == [((1, 1024), (1, 1024, 4), (1, 1024), (1, 1024))] * 2
    assert receipt["physical_tokens"] == 1024
    assert receipt["logical_tokens"] == 2048
    assert receipt["segments"] == 2
    assert receipt["observed_input_shape"] == [1, 1024]
    assert receipt["teacher_geometry"] == {
        "target_shape": [1, 1024, 4],
        "mask_shape": [1, 1024],
        "position_shape": [1, 1024],
    }
    assert receipt["forward_count"] == receipt["backward_count"] == 2
    assert receipt["optimizer_steps"] == 1
    assert receipt["finite_required_trainable_gradients"] is True
    assert receipt["peak_memory_bytes"] == 1234
    assert receipt["immutable_identity"] == _identity()
    assert receipt["memory_sizing"]["physical_tokens"] == 1024
    assert receipt["semantic_parity"] == {
        "claim": "causal-segmented-no-equivalence-claim",
        "tested": False,
    }
    assert receipt["timing"]["started_unix"] <= receipt["timing"]["completed_unix"]
    assert len(receipt["timing"]["segments"]) == 2


def test_logical_tokens_are_distinct_from_masked_logical_items(tmp_path: Path) -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.5))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    values = torch.ones((1, 4))
    mask = torch.tensor([[True, False, True, False]])

    receipt = run_tensor_update(
        parameters=[parameter],
        optimizer=optimizer,
        input_ids=values,
        teacher_targets=torch.zeros_like(values),
        teacher_mask=mask,
        positions=torch.arange(4).reshape(1, -1),
        requested_tokens=2,
        segments=2,
        batch_size=1,
        memory_budget=_budget(),
        loss_sum=lambda segment: (
            (parameter * segment["input_ids"] - segment["teacher_targets"])
            .masked_select(segment["teacher_mask"])
            .square()
            .sum()
        ),
        output=tmp_path / "masked-items.pt",
        identity=_identity(),
        peak_memory_bytes=0,
    )

    assert receipt["physical_tokens"] == 2
    assert receipt["logical_tokens"] == 4
    assert receipt["logical_items"] == 2
    assert receipt["segment_items"] == [1, 1]


def test_impossible_memory_geometry_refuses_before_forward(tmp_path: Path) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    calls = 0

    def forbidden(_segment: dict) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return parameter.square()

    impossible = MemoryBudget(
        available_bytes=8 * GiB,
        resident_frozen_bytes=4 * GiB,
        trainable_bytes=1,
        optimizer_bytes=0,
        staging_bytes=0,
        calibrated_activation_bytes_per_token=1,
    )
    logical = 8192
    values = torch.zeros((1, logical))
    with pytest.raises(RuntimeError, match="no physical token geometry fits"):
        run_tensor_update(
            parameters=[parameter],
            optimizer=optimizer,
            input_ids=values,
            teacher_targets=values,
            teacher_mask=torch.ones_like(values, dtype=torch.bool),
            positions=torch.arange(logical).reshape(1, -1),
            requested_tokens=8192,
            segments=1,
            batch_size=1,
            memory_budget=impossible,
            loss_sum=forbidden,
            output=tmp_path / "never.pt",
            identity=_identity(),
            peak_memory_bytes=0,
        )
    assert calls == 0


def test_simple_causal_slicing_cannot_claim_exact_without_parity(
    tmp_path: Path,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    values = torch.ones((1, 4))
    calls = 0

    def forbidden(_segment: dict) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return parameter.square()

    with pytest.raises(ValueError, match="requires a passing semantic parity test"):
        run_tensor_update(
            parameters=[parameter],
            optimizer=optimizer,
            input_ids=values,
            teacher_targets=torch.zeros_like(values),
            teacher_mask=torch.ones_like(values, dtype=torch.bool),
            positions=torch.arange(4).reshape(1, -1),
            requested_tokens=4,
            segments=1,
            batch_size=1,
            memory_budget=_budget(),
            loss_sum=forbidden,
            output=tmp_path / "semantic.pt",
            identity=_identity(),
            peak_memory_bytes=0,
            semantic_claim="exact",
        )
    assert calls == 0


def test_semantic_parity_flag_requires_actual_bool(tmp_path: Path) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    values = torch.ones((1, 2))
    calls = 0

    def forbidden(_segment: dict) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return parameter.square()

    with pytest.raises(TypeError, match="semantic_parity_tested must be a bool"):
        run_tensor_update(
            parameters=[parameter],
            optimizer=optimizer,
            input_ids=values,
            teacher_targets=torch.zeros_like(values),
            teacher_mask=torch.ones_like(values, dtype=torch.bool),
            positions=torch.arange(2).reshape(1, -1),
            requested_tokens=2,
            segments=1,
            batch_size=1,
            memory_budget=_budget(),
            loss_sum=forbidden,
            output=tmp_path / "truthy-parity.pt",
            identity=_identity(),
            peak_memory_bytes=0,
            semantic_claim="exact",
            semantic_parity_tested=1,
        )
    assert calls == 0


def test_noncontiguous_teacher_positions_refuse_before_forward(tmp_path: Path) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    values = torch.ones((1, 4))
    calls = 0

    def forbidden(_segment: dict) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return parameter.square()

    with pytest.raises(ValueError, match="positions are not contiguous"):
        run_tensor_update(
            parameters=[parameter],
            optimizer=optimizer,
            input_ids=values,
            teacher_targets=torch.zeros_like(values),
            teacher_mask=torch.ones_like(values, dtype=torch.bool),
            positions=torch.tensor([[0, 1, 3, 4]]),
            requested_tokens=4,
            segments=1,
            batch_size=1,
            memory_budget=_budget(),
            loss_sum=forbidden,
            output=tmp_path / "positions.pt",
            identity=_identity(),
            peak_memory_bytes=0,
        )
    assert calls == 0


def test_position_gap_at_segment_boundary_refuses_before_forward(
    tmp_path: Path,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    values = torch.ones((1, 4))
    calls = 0

    def forbidden(_segment: dict) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return parameter.square()

    with pytest.raises(ValueError, match="positions are not contiguous"):
        run_tensor_update(
            parameters=[parameter],
            optimizer=optimizer,
            input_ids=values,
            teacher_targets=torch.zeros_like(values),
            teacher_mask=torch.ones_like(values, dtype=torch.bool),
            positions=torch.tensor([[0, 1, 3, 4]]),
            requested_tokens=2,
            segments=2,
            batch_size=1,
            memory_budget=_budget(),
            loss_sum=forbidden,
            output=tmp_path / "boundary-positions.pt",
            identity=_identity(),
            peak_memory_bytes=0,
        )
    assert calls == 0


def test_missing_required_trainable_gradient_fails_loud(tmp_path: Path) -> None:
    used = torch.nn.Parameter(torch.tensor(1.0))
    unused = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([used, unused], lr=0.1)
    values = torch.ones((1, 2))

    with pytest.raises(RuntimeError, match=r"missing=\[1\]"):
        run_tensor_update(
            parameters=[used, unused],
            optimizer=optimizer,
            input_ids=values,
            teacher_targets=torch.zeros_like(values),
            teacher_mask=torch.ones_like(values, dtype=torch.bool),
            positions=torch.arange(2).reshape(1, -1),
            requested_tokens=2,
            segments=1,
            batch_size=1,
            memory_budget=_budget(),
            loss_sum=lambda segment: (
                (used * segment["input_ids"] - segment["teacher_targets"])
                .square()
                .sum()
            ),
            output=tmp_path / "missing-gradient.pt",
            identity=_identity(),
            peak_memory_bytes=0,
        )
    assert not (tmp_path / "missing-gradient.pt").exists()
