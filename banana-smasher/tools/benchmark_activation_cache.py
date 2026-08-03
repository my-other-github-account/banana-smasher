from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import tempfile
from pathlib import Path

from banana_smasher.activation_cache import build_activation_cache


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _payload(key: int, size: int) -> bytes:
    block = hashlib.sha256(f"activation-window-{key}".encode()).digest()
    return (block * ((size + len(block) - 1) // len(block)))[:size]


def _run_case(
    *,
    root: Path,
    keys: int,
    payload_bytes: int,
    batch_size: int,
    build_rounds: int,
) -> dict[str, object]:
    build_calls = 0

    def build_batch(batch_keys: tuple[int, ...]) -> list[tuple[int, bytes]]:
        nonlocal build_calls
        build_calls += 1
        digest = hashlib.sha256(b"activation-cache-build").digest()
        for _ in range(build_rounds):
            digest = hashlib.sha256(digest).digest()
        if not digest:
            raise AssertionError("unreachable empty digest")
        return [(key, _payload(key, payload_bytes)) for key in batch_keys]

    def write_one(key: int, payload: bytes) -> int:
        path = root / f"window-{key:06d}.bin"
        temporary = root / f".{path.name}.tmp"
        temporary.write_bytes(payload)
        os.replace(temporary, path)
        return len(payload)

    result = build_activation_cache(
        range(keys),
        batch_size=batch_size,
        build_batch=build_batch,
        write_one=write_one,
    )
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.glob("window-*.bin"))
    }
    return {
        "batch_size": batch_size,
        "build_calls": build_calls,
        "bytes_written": result.bytes_written,
        "elapsed_seconds": result.elapsed_seconds,
        "payload_sha256": hashes,
    }


def _measure(
    *,
    parent: Path,
    label: str,
    repeats: int,
    keys: int,
    payload_bytes: int,
    batch_size: int,
    build_rounds: int,
) -> dict[str, object]:
    runs: list[dict[str, object]] = []
    for repeat in range(repeats):
        with tempfile.TemporaryDirectory(prefix=f"{label}-{repeat}-", dir=parent) as directory:
            runs.append(
                _run_case(
                    root=Path(directory),
                    keys=keys,
                    payload_bytes=payload_bytes,
                    batch_size=batch_size,
                    build_rounds=build_rounds,
                )
            )
    elapsed = [float(run["elapsed_seconds"]) for run in runs]
    first = runs[0]
    if any(run["payload_sha256"] != first["payload_sha256"] for run in runs[1:]):
        raise RuntimeError(f"{label} payload hashes drifted across repeats")
    return {
        "batch_size": batch_size,
        "build_calls": first["build_calls"],
        "bytes_written": first["bytes_written"],
        "elapsed_seconds": elapsed,
        "median_seconds": statistics.median(elapsed),
        "payload_sha256": first["payload_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark serial activation-cache batching.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keys", type=_positive_int, default=24)
    parser.add_argument("--payload-bytes", type=_positive_int, default=262144)
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--build-rounds", type=_positive_int, default=50000)
    parser.add_argument("--repeats", type=_positive_int, default=5)
    args = parser.parse_args(argv)

    output = args.output.resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    baseline = _measure(
        parent=output.parent,
        label="baseline",
        repeats=args.repeats,
        keys=args.keys,
        payload_bytes=args.payload_bytes,
        batch_size=1,
        build_rounds=args.build_rounds,
    )
    candidate = _measure(
        parent=output.parent,
        label="candidate",
        repeats=args.repeats,
        keys=args.keys,
        payload_bytes=args.payload_bytes,
        batch_size=args.batch_size,
        build_rounds=args.build_rounds,
    )
    exact_equal = baseline["payload_sha256"] == candidate["payload_sha256"]
    if not exact_equal:
        raise RuntimeError("baseline and candidate payload hashes differ")

    baseline_seconds = float(baseline["median_seconds"])
    candidate_seconds = float(candidate["median_seconds"])
    receipt = {
        "schema": "banana-smasher-activation-cache-benchmark-v1",
        "parallelism": False,
        "exact_equal": exact_equal,
        "keys": args.keys,
        "payload_bytes_per_key": args.payload_bytes,
        "build_rounds_per_batch": args.build_rounds,
        "repeats": args.repeats,
        "baseline": baseline,
        "candidate": candidate,
        "speedup": baseline_seconds / candidate_seconds,
    }
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
