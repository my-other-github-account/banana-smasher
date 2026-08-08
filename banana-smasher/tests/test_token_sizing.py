from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from banana_smasher.token_sizing import (
    DEFAULT_OS_FLOOR_BYTES,
    MemoryBudget,
    choose_physical_tokens,
)

GiB = 1024**3
MiB = 1024**2


def test_auto_sizing_shrinks_8192_before_compute_and_preserves_os_floor() -> None:
    budget = MemoryBudget(
        available_bytes=12 * GiB,
        resident_frozen_bytes=2 * GiB,
        trainable_bytes=1 * GiB,
        optimizer_bytes=512 * MiB,
        staging_bytes=512 * MiB,
        calibrated_activation_bytes_per_token=1 * MiB,
    )

    plan = choose_physical_tokens(requested_tokens=8192, batch_size=1, budget=budget)

    assert plan["requested_physical_tokens"] == 8192
    assert plan["physical_tokens"] == 4096
    assert plan["shrunk_before_compute"] is True
    assert (
        plan["estimated_peak_bytes"] + plan["os_floor_bytes"] <= budget.available_bytes
    )


def test_auto_sizing_has_no_arbitrary_small_model_token_cap() -> None:
    budget = MemoryBudget(
        available_bytes=64 * GiB,
        resident_frozen_bytes=256 * MiB,
        trainable_bytes=256 * MiB,
        optimizer_bytes=512 * MiB,
        staging_bytes=0,
        calibrated_activation_bytes_per_token=1 * MiB,
    )

    assert (
        choose_physical_tokens(requested_tokens=32768, batch_size=1, budget=budget)[
            "physical_tokens"
        ]
        == 32768
    )


def test_auto_sizing_refuses_impossible_geometry_before_compute() -> None:
    budget = MemoryBudget(
        available_bytes=8 * GiB,
        resident_frozen_bytes=3 * GiB,
        trainable_bytes=1 * GiB,
        optimizer_bytes=1,
        staging_bytes=0,
        calibrated_activation_bytes_per_token=1,
    )

    with pytest.raises(RuntimeError, match="no physical token geometry fits"):
        choose_physical_tokens(requested_tokens=8192, batch_size=1, budget=budget)


def test_memory_budget_refuses_os_floor_below_four_gibibytes() -> None:
    budget = MemoryBudget(
        available_bytes=16 * GiB,
        resident_frozen_bytes=1 * GiB,
        trainable_bytes=1 * MiB,
        optimizer_bytes=2 * MiB,
        staging_bytes=1 * MiB,
        calibrated_activation_bytes_per_token=1 * MiB,
        os_floor_bytes=DEFAULT_OS_FLOOR_BYTES - 1,
    )

    with pytest.raises(ValueError, match="os_floor_bytes must be at least"):
        budget.validated()


@pytest.mark.parametrize(
    "field",
    [
        "available_bytes",
        "resident_frozen_bytes",
        "trainable_bytes",
        "optimizer_bytes",
        "staging_bytes",
        "calibrated_activation_bytes_per_token",
        "os_floor_bytes",
    ],
)
@pytest.mark.parametrize("invalid", [True, 1.0, 1.5])
def test_memory_budget_requires_actual_non_bool_integers(
    field: str, invalid: object
) -> None:
    budget = MemoryBudget(
        available_bytes=16 * GiB,
        resident_frozen_bytes=1 * GiB,
        trainable_bytes=1 * MiB,
        optimizer_bytes=2 * MiB,
        staging_bytes=1 * MiB,
        calibrated_activation_bytes_per_token=1 * MiB,
    )

    with pytest.raises(TypeError, match=rf"{field} must be an integer"):
        replace(budget, **{field: invalid}).validated()


@pytest.mark.parametrize(
    "argument", ["requested_tokens", "batch_size", "minimum_tokens"]
)
@pytest.mark.parametrize("invalid", [False, 1.0, 1.5])
def test_token_sizing_arguments_require_actual_non_bool_integers(
    argument: str, invalid: object
) -> None:
    values: dict[str, Any] = {
        "requested_tokens": 1024,
        "batch_size": 1,
        "minimum_tokens": 1,
    }
    values[argument] = invalid
    budget = MemoryBudget(
        available_bytes=16 * GiB,
        resident_frozen_bytes=1 * GiB,
        trainable_bytes=1 * MiB,
        optimizer_bytes=2 * MiB,
        staging_bytes=1 * MiB,
        calibrated_activation_bytes_per_token=1 * MiB,
    )

    with pytest.raises(TypeError, match=rf"{argument} must be an integer"):
        choose_physical_tokens(budget=budget, **values)
