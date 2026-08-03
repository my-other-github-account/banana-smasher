from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Generic, TypeVar

KeyT = TypeVar("KeyT")
PayloadT = TypeVar("PayloadT")
StableT = TypeVar("StableT")
OutputT = TypeVar("OutputT")
ProgressCallback = Callable[[dict[str, int | float | str]], None]


def run_shape_stable_batch(
    items: Sequence[KeyT] | Iterable[KeyT],
    *,
    batch_stable: Callable[[tuple[KeyT, ...]], Sequence[StableT] | Iterable[StableT]],
    shape_sensitive: Callable[[KeyT, StableT], OutputT],
) -> tuple[OutputT, ...]:
    """Batch the safe path, then preserve one-item calls for shape-sensitive work."""

    ordered_items = tuple(items)
    if not ordered_items:
        return ()
    stable_values = tuple(batch_stable(ordered_items))
    if len(stable_values) != len(ordered_items):
        raise ValueError("batch_stable must return exactly one value per item")
    return tuple(
        shape_sensitive(item, stable_value)
        for item, stable_value in zip(ordered_items, stable_values, strict=True)
    )


@dataclass(frozen=True)
class ActivationCacheBuildResult(Generic[KeyT]):
    completed_keys: tuple[KeyT, ...]
    bytes_written: int
    elapsed_seconds: float
    batches: int


def build_activation_cache(
    keys: Sequence[KeyT] | Iterable[KeyT],
    *,
    batch_size: int,
    build_batch: Callable[[tuple[KeyT, ...]], Iterable[tuple[KeyT, PayloadT]]],
    write_one: Callable[[KeyT, PayloadT], int],
    progress: ProgressCallback | None = None,
) -> ActivationCacheBuildResult[KeyT]:
    """Batch cache computation while preserving serial, ordered persistence."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    ordered_keys = tuple(keys)
    if len(set(ordered_keys)) != len(ordered_keys):
        raise ValueError("cache keys must be unique")
    started = perf_counter()
    bytes_written = 0
    batches = 0
    completed = 0

    def emit(phase: str, **fields: int | float | str) -> None:
        if progress is not None:
            progress({"phase": phase, **fields})

    for offset in range(0, len(ordered_keys), batch_size):
        batch_keys = ordered_keys[offset : offset + batch_size]
        payload_rows = tuple(build_batch(batch_keys))
        payload_keys = tuple(key for key, _payload in payload_rows)
        if payload_keys != batch_keys:
            raise ValueError(
                f"builder keys drift: expected {batch_keys!r}, got {payload_keys!r}"
            )
        for key, payload in payload_rows:
            written = write_one(key, payload)
            if isinstance(written, bool) or not isinstance(written, int) or written < 0:
                raise ValueError("write_one must return a non-negative integer byte count")
            bytes_written += written
        completed += len(batch_keys)
        batches += 1
        emit(
            "persisted",
            completed=completed,
            total=len(ordered_keys),
            bytes_written=bytes_written,
            batches=batches,
        )

    elapsed = perf_counter() - started
    emit(
        "complete",
        completed=completed,
        total=len(ordered_keys),
        bytes_written=bytes_written,
        batches=batches,
        elapsed_seconds=elapsed,
    )
    return ActivationCacheBuildResult(
        completed_keys=ordered_keys,
        bytes_written=bytes_written,
        elapsed_seconds=elapsed,
        batches=batches,
    )
