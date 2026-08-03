#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from banana_smasher.activation_cache import build_activation_cache


def atomic_json(path: Path, value: object) -> str:
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_for(key: int, size: int, rounds: int) -> bytes:
    seed = f"banana-smasher-activation-window-{key}".encode()
    digest = seed
    for _ in range(rounds):
        digest = hashlib.sha256(digest + seed).digest()
    return hashlib.shake_256(digest).digest(size)


def write_one(directory: Path, key: int, payload: bytes) -> int:
    target = directory / f"window-{key:04d}.bin"
    temporary = target.with_suffix(f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    return len(payload)


def serial_build(
    keys: tuple[int, ...],
    *,
    directory: Path,
    batch_size: int,
    io_workers: int,
    payload_bytes: int,
    rounds: int,
    progress,
) -> float:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        for offset in range(0, len(keys), batch_size):
            batch = keys[offset : offset + batch_size]
            rows = [(key, payload_for(key, payload_bytes, rounds)) for key in batch]
            futures = [pool.submit(write_one, directory, key, payload) for key, payload in rows]
            for future in futures:
                future.result()
            progress({
                "phase": "persisted",
                "completed": min(offset + len(batch), len(keys)),
                "total": len(keys),
            })
    return time.perf_counter() - started


def manifest(directory: Path) -> dict[str, str]:
    return {path.name: sha256(path) for path in sorted(directory.glob("window-*.bin"))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--claim-path", type=Path, required=True)
    parser.add_argument("--expected-claim-sha256", required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--expected-basis-sha256", required=True)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--expected-hostname")
    parser.add_argument("--keys", type=int, default=12)
    parser.add_argument("--payload-mib", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--io-workers", type=int, default=2)
    parser.add_argument("--pending-batches", type=int, default=2)
    parser.add_argument("--build-rounds", type=int, default=50000)
    args = parser.parse_args()

    if args.expected_hostname and socket.gethostname() != args.expected_hostname:
        raise RuntimeError(
            f"wrong host: expected {args.expected_hostname!r}, got {socket.gethostname()!r}"
        )
    output = args.output_root
    if output.exists():
        raise FileExistsError(f"output root already exists: {output}")
    if sha256(args.claim_path) != args.expected_claim_sha256:
        raise RuntimeError("host claim SHA drift")
    claim = json.loads(args.claim_path.read_text())
    if claim.get("task_id") != args.task_id or claim.get("status") != "CLAIMED":
        raise RuntimeError("host claim owner/status drift")
    if sha256(args.index_path) != args.expected_basis_sha256:
        raise RuntimeError("BASIS GATE index drift")
    if len(args.registry_sha256) != 64:
        raise RuntimeError("registry SHA malformed")

    baseline = output / "baseline"
    candidate = output / "candidate"
    output.mkdir(parents=True, exist_ok=False)
    baseline.mkdir(parents=True)
    candidate.mkdir(parents=True)
    progress_path = output / "PROGRESS.json"
    keys = tuple(range(args.keys))
    payload_bytes = args.payload_mib << 20

    atomic_json(progress_path, {
        "schema": "banana-smasher-activation-cache-benchmark-progress-v1",
        "task_id": args.task_id,
        "phase": "BASELINE",
        "completed": 0,
        "total": len(keys),
        "pid": os.getpid(),
        "updated_unix": time.time(),
    })
    def baseline_progress(event: dict[str, int | float | str]) -> None:
        atomic_json(progress_path, {
            "schema": "banana-smasher-activation-cache-benchmark-progress-v1",
            "task_id": args.task_id,
            "phase": "BASELINE",
            "pid": os.getpid(),
            "updated_unix": time.time(),
            **event,
        })

    baseline_seconds = serial_build(
        keys,
        directory=baseline,
        batch_size=args.batch_size,
        io_workers=args.io_workers,
        payload_bytes=payload_bytes,
        rounds=args.build_rounds,
        progress=baseline_progress,
    )
    baseline_manifest = manifest(baseline)

    def build_batch(batch: tuple[int, ...]):
        return [(key, payload_for(key, payload_bytes, args.build_rounds)) for key in batch]

    def persist(key: int, payload: bytes) -> int:
        return write_one(candidate, key, payload)

    events: list[dict[str, int | float | str]] = []

    def progress(event: dict[str, int | float | str]) -> None:
        events.append(event)
        if event.get("phase") != "persisted":
            return
        atomic_json(progress_path, {
            "schema": "banana-smasher-activation-cache-benchmark-progress-v1",
            "task_id": args.task_id,
            "phase": "CANDIDATE",
            "pid": os.getpid(),
            "updated_unix": time.time(),
            **event,
        })

    candidate_result = build_activation_cache(
        keys,
        batch_size=args.batch_size,
        build_batch=build_batch,
        write_one=persist,
        io_workers=args.io_workers,
        max_pending_batches=args.pending_batches,
        progress=progress,
    )
    candidate_manifest = manifest(candidate)
    exact_equal = candidate_manifest == baseline_manifest and len(candidate_manifest) == len(keys)
    speedup = baseline_seconds / candidate_result.elapsed_seconds
    result = {
        "schema": "banana-smasher-activation-cache-overlap-benchmark-v1",
        "task_id": args.task_id,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "claim_sha256": args.expected_claim_sha256,
        "basis_sha256": args.expected_basis_sha256,
        "registry_sha256": args.registry_sha256,
        "keys": len(keys),
        "payload_bytes_per_key": payload_bytes,
        "total_payload_bytes": payload_bytes * len(keys),
        "batch_size": args.batch_size,
        "io_workers": args.io_workers,
        "pending_batches": args.pending_batches,
        "build_rounds": args.build_rounds,
        "baseline_seconds": baseline_seconds,
        "candidate_seconds": candidate_result.elapsed_seconds,
        "speedup": speedup,
        "baseline_manifest": baseline_manifest,
        "candidate_manifest": candidate_manifest,
        "exact_equal": exact_equal,
        "candidate_bytes_written": candidate_result.bytes_written,
        "candidate_completed_keys": list(candidate_result.completed_keys),
        "progress_events": events,
        "status": "PASS" if exact_equal else "FAIL",
        "completed_unix": time.time(),
    }
    receipt_sha = atomic_json(output / "BENCHMARK.json", result)
    atomic_json(progress_path, {
        "schema": "banana-smasher-activation-cache-benchmark-progress-v1",
        "task_id": args.task_id,
        "phase": result["status"],
        "completed": len(keys),
        "total": len(keys),
        "exact_equal": exact_equal,
        "speedup": speedup,
        "receipt_sha256": receipt_sha,
        "pid": os.getpid(),
        "updated_unix": time.time(),
    })
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if exact_equal else 2


if __name__ == "__main__":
    raise SystemExit(main())
