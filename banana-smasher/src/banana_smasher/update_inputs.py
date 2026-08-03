from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PhysicalBatch:
    """Explicit token, masking, and position semantics for an update batch."""

    input_ids: np.ndarray[Any, Any]
    attention_mask: np.ndarray[Any, Any]
    position_ids: np.ndarray[Any, Any]
    teacher_mask: np.ndarray[Any, Any]
    physical_token_count: int
    teacher_token_count: int


def _shape_2d(value: object, *, label: str) -> np.ndarray[Any, Any]:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{label} must be rank 2, got shape {array.shape}")
    return array


def _boolean_mask(
    value: object | None,
    *,
    label: str,
    shape: tuple[int, ...],
    default: bool,
) -> np.ndarray[Any, Any]:
    if value is None:
        return np.full(shape, default, dtype=np.bool_)
    array = _shape_2d(value, label=label)
    if array.shape != shape:
        raise ValueError(f"{label} shape mismatch: {array.shape} != {shape}")
    if not np.issubdtype(array.dtype, np.bool_) and not (
        np.issubdtype(array.dtype, np.integer) and np.isin(array, (0, 1)).all()
    ):
        raise ValueError(f"{label} must contain only boolean or 0/1 values")
    return array.astype(np.bool_, copy=True)


def prepare_physical_batch(
    input_ids: object,
    *,
    attention_mask: object | None = None,
    position_ids: object | None = None,
    teacher_mask: object | None = None,
) -> PhysicalBatch:
    """Validate and freeze exact physical update-token semantics.

    Padding is represented only by ``attention_mask``; token values are never
    rewritten or inferred. ``teacher_mask`` may narrow supervised positions but
    cannot select a position excluded from physical attention.
    """

    tokens = _shape_2d(input_ids, label="input_ids")
    if not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError("input_ids must use an integer dtype")
    if tokens.size == 0 or tokens.shape[1] == 0:
        raise ValueError("input_ids must contain at least one physical token")
    if np.any(tokens < 0):
        raise ValueError("input_ids must be non-negative")
    tokens = tokens.astype(np.int64, copy=True)
    attended = _boolean_mask(
        attention_mask,
        label="attention_mask",
        shape=tokens.shape,
        default=True,
    )
    if not np.any(attended):
        raise ValueError("attention_mask excludes every physical token")

    if position_ids is None:
        positions = np.broadcast_to(
            np.arange(tokens.shape[1], dtype=np.int64), tokens.shape
        ).copy()
    else:
        positions = _shape_2d(position_ids, label="position_ids")
        if positions.shape != tokens.shape:
            raise ValueError(
                f"position_ids shape mismatch: {positions.shape} != {tokens.shape}"
            )
        if not np.issubdtype(positions.dtype, np.integer) or np.any(positions < 0):
            raise ValueError("position_ids must contain non-negative integers")
        positions = positions.astype(np.int64, copy=True)

    supervised = _boolean_mask(
        teacher_mask,
        label="teacher_mask",
        shape=tokens.shape,
        default=True,
    )
    if teacher_mask is None:
        supervised &= attended
    elif np.any(supervised & ~attended):
        raise ValueError("teacher mask selects a non-attended token")
    if not np.any(supervised):
        raise ValueError("teacher_mask selects no tokens")

    for array in (tokens, attended, positions, supervised):
        array.setflags(write=False)
    return PhysicalBatch(
        input_ids=tokens,
        attention_mask=attended,
        position_ids=positions,
        teacher_mask=supervised,
        physical_token_count=int(np.count_nonzero(attended)),
        teacher_token_count=int(np.count_nonzero(supervised)),
    )
