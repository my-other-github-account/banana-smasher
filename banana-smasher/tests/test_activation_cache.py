from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from banana_smasher.activation_cache import build_activation_cache


def _payload(key: int) -> bytes:
    block = hashlib.sha256(f"window-{key}".encode()).digest()
    return block * 4096


def test_overlaps_next_build_with_prior_batch_writes(tmp_path: Path) -> None:
    second_build_started = threading.Event()
    release_first_write = threading.Event()

    def build_batch(keys: tuple[int, ...]) -> list[tuple[int, bytes]]:
        if keys == (2, 3):
            second_build_started.set()
        return [(key, _payload(key)) for key in keys]

    def write_one(key: int, payload: bytes) -> int:
        if key == 0:
            assert second_build_started.wait(1.0), "next build did not overlap prior write"
            release_first_write.set()
        elif key == 1:
            assert release_first_write.wait(1.0)
        path = tmp_path / f"win{key}.bin"
        path.write_bytes(payload)
        return len(payload)

    result = build_activation_cache(
        range(4),
        batch_size=2,
        build_batch=build_batch,
        write_one=write_one,
        io_workers=2,
        max_pending_batches=2,
    )

    assert result.completed_keys == (0, 1, 2, 3)
    assert result.bytes_written == sum(len(_payload(key)) for key in range(4))


def test_persists_exact_payload_bytes_and_reports_monotonic_progress(tmp_path: Path) -> None:
    events: list[dict[str, int | str]] = []

    def build_batch(keys: tuple[int, ...]) -> list[tuple[int, bytes]]:
        return [(key, _payload(key)) for key in keys]

    def write_one(key: int, payload: bytes) -> int:
        path = tmp_path / f"win{key}.bin"
        path.write_bytes(payload)
        return len(payload)

    result = build_activation_cache(
        [7, 8, 9, 10, 11],
        batch_size=2,
        build_batch=build_batch,
        write_one=write_one,
        io_workers=2,
        progress=events.append,
    )

    assert [hashlib.sha256((tmp_path / f"win{key}.bin").read_bytes()).hexdigest() for key in range(7, 12)] == [
        hashlib.sha256(_payload(key)).hexdigest() for key in range(7, 12)
    ]
    assert result.completed_keys == (7, 8, 9, 10, 11)
    persisted = [event["completed"] for event in events if event["phase"] == "persisted"]
    assert persisted == sorted(persisted)
    assert persisted[-1] == 5


def test_rejects_builder_key_drift_before_persisting(tmp_path: Path) -> None:
    writes: list[int] = []

    with pytest.raises(ValueError, match="builder keys drift"):
        build_activation_cache(
            [0, 1],
            batch_size=2,
            build_batch=lambda _keys: [(0, b"a"), (0, b"b")],
            write_one=lambda key, _payload: writes.append(key) or 1,
        )

    assert writes == []


def test_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        build_activation_cache([], batch_size=0, build_batch=lambda _keys: [], write_one=lambda _key, _payload: 0)
    with pytest.raises(ValueError, match="io_workers"):
        build_activation_cache([], batch_size=1, build_batch=lambda _keys: [], write_one=lambda _key, _payload: 0, io_workers=0)
    with pytest.raises(ValueError, match="max_pending_batches"):
        build_activation_cache([], batch_size=1, build_batch=lambda _keys: [], write_one=lambda _key, _payload: 0, max_pending_batches=0)


def test_benchmark_refuses_to_delete_an_existing_output_root(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("preserve")
    script = Path(__file__).parents[1] / "tools" / "benchmark_activation_cache.py"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-root",
            str(existing),
            "--task-id",
            "test-task",
            "--claim-path",
            str(tmp_path / "missing-claim.json"),
            "--expected-claim-sha256",
            "0" * 64,
            "--index-path",
            str(tmp_path / "missing-index.json"),
            "--expected-basis-sha256",
            "1" * 64,
            "--registry-sha256",
            "2" * 64,
        ],
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "output root already exists" in result.stderr
    assert sentinel.read_text() == "preserve"
