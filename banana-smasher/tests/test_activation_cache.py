from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from banana_smasher.activation_cache import build_activation_cache, run_shape_stable_batch


def test_serial_cache_batches_build_calls_without_parallel_writes() -> None:
    build_calls: list[tuple[int, ...]] = []
    write_threads: list[int] = []
    persisted: dict[int, bytes] = {}
    caller_thread = threading.get_ident()

    def build_batch(keys: tuple[int, ...]) -> list[tuple[int, bytes]]:
        build_calls.append(keys)
        return [(key, f"payload-{key}".encode()) for key in keys]

    def write_one(key: int, payload: bytes) -> int:
        write_threads.append(threading.get_ident())
        persisted[key] = payload
        return len(payload)

    result = build_activation_cache(
        range(7),
        batch_size=3,
        build_batch=build_batch,
        write_one=write_one,
    )

    assert build_calls == [(0, 1, 2), (3, 4, 5), (6,)]
    assert write_threads == [caller_thread] * 7
    assert persisted == {key: f"payload-{key}".encode() for key in range(7)}
    assert result.completed_keys == tuple(range(7))
    assert result.batches == 3
    assert result.bytes_written == sum(map(len, persisted.values()))


def test_shape_stable_batch_batches_only_the_safe_path() -> None:
    stable_calls: list[tuple[int, ...]] = []
    sensitive_calls: list[tuple[int, int]] = []

    def batch_stable(items: tuple[int, ...]) -> tuple[int, ...]:
        stable_calls.append(items)
        return tuple(item * 10 for item in items)

    def shape_sensitive(item: int, stable_value: int) -> int:
        sensitive_calls.append((item, stable_value))
        return item + stable_value

    result = run_shape_stable_batch(
        [1, 2, 3, 4],
        batch_stable=batch_stable,
        shape_sensitive=shape_sensitive,
    )

    assert result == (11, 22, 33, 44)
    assert stable_calls == [(1, 2, 3, 4)]
    assert sensitive_calls == [(1, 10), (2, 20), (3, 30), (4, 40)]


def test_shape_stable_batch_rejects_cardinality_drift_before_sensitive_work() -> None:
    sensitive_calls: list[tuple[int, int]] = []

    with pytest.raises(ValueError, match="one value per item"):
        run_shape_stable_batch(
            [1, 2],
            batch_stable=lambda _items: [10],
            shape_sensitive=lambda item, value: sensitive_calls.append((item, value)) or 0,
        )

    assert sensitive_calls == []


def test_serial_cache_rejects_builder_key_drift_before_writing() -> None:
    writes: list[int] = []

    with pytest.raises(ValueError, match="builder keys drift"):
        build_activation_cache(
            [0, 1],
            batch_size=2,
            build_batch=lambda _keys: [(0, b"a"), (0, b"b")],
            write_one=lambda key, _payload: writes.append(key) or 1,
        )

    assert writes == []


def test_serial_cache_rejects_boolean_byte_counts() -> None:
    with pytest.raises(ValueError, match="non-negative integer byte count"):
        build_activation_cache(
            [0],
            batch_size=1,
            build_batch=lambda keys: [(keys[0], b"payload")],
            write_one=lambda _key, _payload: True,
        )


@pytest.mark.parametrize("batch_size", [True, 1.5, 0])
def test_serial_cache_rejects_non_positive_integer_batch_size(batch_size: object) -> None:
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        build_activation_cache(
            [],
            batch_size=batch_size,  # type: ignore[arg-type]
            build_batch=lambda _keys: [],
            write_one=lambda _key, _payload: 0,
        )


def test_serial_cache_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="cache keys must be unique"):
        build_activation_cache(
            [1, 1],
            batch_size=1,
            build_batch=lambda keys: [(keys[0], b"payload")],
            write_one=lambda _key, payload: len(payload),
        )


def test_generic_serial_benchmark_emits_exact_equal_receipt(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "tools" / "benchmark_activation_cache.py"
    output = tmp_path / "benchmark.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output",
            str(output),
            "--keys",
            "12",
            "--payload-bytes",
            "65536",
            "--batch-size",
            "4",
            "--build-rounds",
            "1000",
            "--repeats",
            "2",
        ],
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(output.read_text())
    assert receipt["schema"] == "banana-smasher-activation-cache-benchmark-v1"
    assert receipt["parallelism"] is False
    assert receipt["exact_equal"] is True
    assert receipt["baseline"]["batch_size"] == 1
    assert receipt["candidate"]["batch_size"] == 4
    assert receipt["baseline"]["build_calls"] == 12
    assert receipt["candidate"]["build_calls"] == 3
    assert receipt["baseline"]["bytes_written"] == 12 * 65536
    assert receipt["candidate"]["bytes_written"] == 12 * 65536


def test_serial_cache_reports_monotonic_persisted_progress() -> None:
    events: list[dict[str, int | float | str]] = []

    result = build_activation_cache(
        range(5),
        batch_size=2,
        build_batch=lambda keys: [(key, b"payload") for key in keys],
        write_one=lambda _key, payload: len(payload),
        progress=events.append,
    )

    persisted = [event for event in events if event["phase"] == "persisted"]
    assert [event["completed"] for event in persisted] == [2, 4, 5]
    assert [event["bytes_written"] for event in persisted] == [14, 28, 35]
    assert events[-1]["phase"] == "complete"
    assert events[-1]["completed"] == len(result.completed_keys)
