from __future__ import annotations

from dataclasses import asdict, dataclass
import operator
from typing import Any

DEFAULT_OS_FLOOR_BYTES = 4 * 1024**3


def require_integer(name: str, value: Any) -> int:
    """Return an integer input without silently truncating floats or accepting bool."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer, got {value!r}") from exc
    return int(result)


@dataclass(frozen=True)
class MemoryBudget:
    """Inputs to the pre-compute physical-token memory model."""

    available_bytes: int
    resident_frozen_bytes: int
    trainable_bytes: int
    optimizer_bytes: int
    staging_bytes: int
    calibrated_activation_bytes_per_token: int
    os_floor_bytes: int = DEFAULT_OS_FLOOR_BYTES

    def validated(self) -> MemoryBudget:
        values = asdict(self)
        for name, value in values.items():
            value = require_integer(name, value)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if self.available_bytes <= 0:
            raise ValueError("available_bytes must be positive")
        if self.calibrated_activation_bytes_per_token <= 0:
            raise ValueError("calibrated_activation_bytes_per_token must be positive")
        if self.os_floor_bytes < DEFAULT_OS_FLOOR_BYTES:
            raise ValueError(
                f"os_floor_bytes must be at least {DEFAULT_OS_FLOOR_BYTES} (4 GiB)"
            )
        return self

    @property
    def fixed_bytes(self) -> int:
        return sum(
            require_integer(name, value)
            for name, value in (
                ("resident_frozen_bytes", self.resident_frozen_bytes),
                ("trainable_bytes", self.trainable_bytes),
                ("optimizer_bytes", self.optimizer_bytes),
                ("staging_bytes", self.staging_bytes),
            )
        )


def choose_physical_tokens(
    *,
    requested_tokens: int,
    batch_size: int,
    budget: MemoryBudget,
    minimum_tokens: int = 1,
) -> dict[str, Any]:
    """Select the largest safe physical token dimension before model compute.

    There is no model-independent upper cap. The requested shape is retained when
    it fits; otherwise it is reduced to the exact memory-derived maximum.
    """
    budget = budget.validated()
    requested_tokens = require_integer("requested_tokens", requested_tokens)
    batch_size = require_integer("batch_size", batch_size)
    minimum_tokens = require_integer("minimum_tokens", minimum_tokens)
    if requested_tokens <= 0:
        raise ValueError("requested_tokens must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if minimum_tokens <= 0 or minimum_tokens > requested_tokens:
        raise ValueError("minimum_tokens must be in [1, requested_tokens]")

    compute_capacity = budget.available_bytes - budget.os_floor_bytes
    activation_per_physical_token = (
        budget.calibrated_activation_bytes_per_token * batch_size
    )
    token_capacity_bytes = compute_capacity - budget.fixed_bytes
    maximum_tokens = token_capacity_bytes // activation_per_physical_token
    selected = min(requested_tokens, maximum_tokens)
    if selected < minimum_tokens:
        raise RuntimeError(
            "no physical token geometry fits before compute: "
            f"available={budget.available_bytes}, os_floor={budget.os_floor_bytes}, "
            f"fixed={budget.fixed_bytes}, activation_per_token="
            f"{activation_per_physical_token}, minimum_tokens={minimum_tokens}"
        )

    activation_bytes = selected * activation_per_physical_token
    estimated_peak = budget.fixed_bytes + activation_bytes
    return {
        "schema": "banana-smasher-token-sizing-v1",
        "batch_size": batch_size,
        "requested_physical_tokens": requested_tokens,
        "physical_tokens": selected,
        "shrunk_before_compute": selected != requested_tokens,
        "available_bytes": budget.available_bytes,
        "os_floor_bytes": budget.os_floor_bytes,
        "resident_frozen_bytes": budget.resident_frozen_bytes,
        "trainable_bytes": budget.trainable_bytes,
        "optimizer_bytes": budget.optimizer_bytes,
        "staging_bytes": budget.staging_bytes,
        "calibrated_activation_bytes_per_token": budget.calibrated_activation_bytes_per_token,
        "activation_bytes": activation_bytes,
        "estimated_peak_bytes": estimated_peak,
        "headroom_bytes": compute_capacity - estimated_peak,
    }
