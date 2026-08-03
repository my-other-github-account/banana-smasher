from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Generic, TypeVar

KeyT = TypeVar("KeyT")
PayloadT = TypeVar("PayloadT")
ProgressCallback = Callable[[dict[str, int | float | str]], None]


@dataclass(frozen=True)
class ActivationCacheBuildResult(Generic[KeyT]):
    completed_keys: tuple[KeyT, ...]
    bytes_written: int
    elapsed_seconds: float
    batches: int


def _batches(keys: tuple[KeyT, ...], size: int) -> Iterable[tuple[KeyT, ...]]:
    for offset in range(0, len(keys), size):
        yield keys[offset : offset + size]


def build_activation_cache(
    keys: Sequence[KeyT] | Iterable[KeyT],
    *,
    batch_size: int,
    build_batch: Callable[[tuple[KeyT, ...]], Iterable[tuple[KeyT, PayloadT]]],
    write_one: Callable[[KeyT, PayloadT], int],
    io_workers: int = 4,
    max_pending_batches: int = 2,
    progress: ProgressCallback | None = None,
) -> ActivationCacheBuildResult[KeyT]:
    """Build cache payloads while prior batches persist in bounded I/O workers.

    ``build_batch`` must return exactly one payload for every requested key and
    in the same order.  The next batch is built before the pending-write bound
    is applied, so compute overlaps the prior batch's writes while resident
    payload memory remains bounded by ``max_pending_batches + 1`` batches.
    Any builder drift or write failure aborts instead of publishing success.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if io_workers < 1:
        raise ValueError("io_workers must be at least 1")
    if max_pending_batches < 1:
        raise ValueError("max_pending_batches must be at least 1")

    ordered_keys = tuple(keys)
    if len(set(ordered_keys)) != len(ordered_keys):
        raise ValueError("cache keys must be unique")

    started = perf_counter()
    completed = 0
    bytes_written = 0
    batch_count = 0
    pending: deque[tuple[tuple[KeyT, ...], tuple[Future[int], ...]]] = deque()

    def emit(phase: str, **fields: int | float | str) -> None:
        if progress is not None:
            progress({"phase": phase, **fields})

    def drain_one() -> None:
        nonlocal completed, bytes_written
        batch_keys, futures = pending.popleft()
        batch_bytes = 0
        for future in futures:
            written = future.result()
            if not isinstance(written, int) or written < 0:
                raise ValueError("write_one must return a non-negative byte count")
            batch_bytes += written
        completed += len(batch_keys)
        bytes_written += batch_bytes
        emit(
            "persisted",
            completed=completed,
            total=len(ordered_keys),
            batch_bytes=batch_bytes,
            bytes_written=bytes_written,
        )

    with ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix="activation-cache-io") as pool:
        for batch_keys in _batches(ordered_keys, batch_size):
            payload_rows = tuple(build_batch(batch_keys))
            payload_keys = tuple(key for key, _payload in payload_rows)
            if payload_keys != batch_keys:
                raise ValueError(
                    f"builder keys drift: expected {batch_keys!r}, got {payload_keys!r}"
                )

            emit(
                "built",
                batch=batch_count,
                batch_keys=len(batch_keys),
                completed=completed,
                total=len(ordered_keys),
                pending_batches=len(pending),
            )
            while len(pending) >= max_pending_batches:
                drain_one()

            futures = tuple(
                pool.submit(write_one, key, payload) for key, payload in payload_rows
            )
            pending.append((batch_keys, futures))
            batch_count += 1

        while pending:
            drain_one()

    elapsed = perf_counter() - started
    emit(
        "complete",
        completed=completed,
        total=len(ordered_keys),
        bytes_written=bytes_written,
        batches=batch_count,
        elapsed_seconds=elapsed,
    )
    return ActivationCacheBuildResult(
        completed_keys=ordered_keys,
        bytes_written=bytes_written,
        elapsed_seconds=elapsed,
        batches=batch_count,
    )
