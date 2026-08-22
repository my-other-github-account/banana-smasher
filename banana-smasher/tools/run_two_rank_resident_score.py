#!/usr/bin/env python3
"""Run a provenance-gated two-rank resident score canary or full64."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from banana_smasher.resident_continuation import (
    MODEL_INDEX_SHA256,
    ModernGreenResidentEngine,
    _require_sealed_batch1_parity,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def proc_io() -> dict[str, int]:
    rows: dict[str, int] = {}
    for line in Path("/proc/self/io").read_text().splitlines():
        key, value = line.split(":", 1)
        rows[key] = int(value.strip())
    return rows


def require_claim(task_id: str, rank: int) -> dict[str, Any]:
    path = Path("/home/dnola/HOST_CLAIM.json")
    raw = path.read_bytes()
    claim = json.loads(raw)
    owner = claim.get("task_id", claim.get("owner_task_id"))
    if (
        claim.get("state", claim.get("status")) != "CLAIMED"
        or owner != task_id
        or claim.get("intended_basis") != MODEL_INDEX_SHA256
    ):
        raise RuntimeError(f"host claim drift for rank {rank}: {claim}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "task_id": owner,
        "rank": rank,
        "intended_basis": claim.get("intended_basis"),
    }


def require_shards(path: Path, task_id: str, rank: int) -> dict[str, Any]:
    raw = path.read_bytes()
    shards = json.loads(raw)
    expected = [0, 20] if rank == 0 else [21, 42]
    if (
        shards.get("task_id") != task_id
        or shards.get("intended_basis") != MODEL_INDEX_SHA256
        or shards.get("ranks", {}).get(str(rank), {}).get("layers") != expected
    ):
        raise RuntimeError(f"SHARDS drift for rank {rank}: {shards}")
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def load_checkpoint(path: Path, expected_sha256: str) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(f"checkpoint SHA mismatch: {observed} != {expected_sha256}")
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or set((payload.get("state") or {})) != {
        "luts", "norms", "outputs"
    }:
        raise RuntimeError("checkpoint lacks exact luts/norms/outputs state")
    return payload


def parse_windows(value: str) -> list[int]:
    windows = [int(item) for item in value.split(",") if item.strip()]
    if len(windows) not in (4, 64) or len(set(windows)) != len(windows):
        raise argparse.ArgumentTypeError("windows must contain 4 or 64 unique IDs")
    return windows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--windows", type=parse_windows, required=True)
    parser.add_argument("--mode", choices=("canary", "full64"), required=True)
    parser.add_argument("--canary-reference", type=Path)
    args = parser.parse_args()

    args.run_root.mkdir(parents=True, exist_ok=True)
    claim = require_claim(args.task_id, args.rank)
    shards = require_shards(args.run_root / "SHARDS.json", args.task_id, args.rank)
    config = json.loads(args.config.read_text())
    config.update(rank=args.rank, score_only=True, score_windows=args.windows)
    model_index = Path(config["model_root"]) / "model.safetensors.index.json"
    observed_basis = sha256_file(model_index)
    if observed_basis != MODEL_INDEX_SHA256:
        raise RuntimeError(f"source model basis mismatch: {observed_basis}")
    payload = load_checkpoint(args.checkpoint, args.checkpoint_sha256)

    started_unix = time.time()
    started = time.perf_counter()
    io_before = proc_io()
    engine = ModernGreenResidentEngine(
        payload=payload,
        config=config,
        rank=args.rank,
        layer_ranges={0: (0, 20), 1: (21, 42)},
    )
    construction_seconds = time.perf_counter() - started
    score_started = time.perf_counter()
    if args.mode == "canary":
        if len(args.windows) != 4:
            raise RuntimeError("canary mode requires exactly four windows")
        result = engine._score_live_windows(args.windows)
        gate = None
        projected = float(result["timed_wall_seconds"]) * 16.0
    else:
        if len(args.windows) != 64 or args.canary_reference is None:
            raise RuntimeError("full64 mode requires 64 windows and --canary-reference")
        reference = json.loads(args.canary_reference.read_text())
        canary = reference["observed"]
        gate = _require_sealed_batch1_parity(canary, reference["sealed_reference"])
        projected = float(canary["timed_wall_seconds"]) * 16.0
        best_single = float(reference["sealed_reference"]["best_single_host_wall_seconds"])
        if not math.isfinite(projected) or projected > 300.0 or projected >= best_single:
            raise RuntimeError(
                f"canary projection refused full64: {projected:.3f}s vs {best_single:.3f}s"
            )
        result = engine._score_live_windows(args.windows)
    score_outer_seconds = time.perf_counter() - score_started
    io_after = proc_io()
    receipt = {
        "schema": "banana-smasher-two-rank-resident-score-v1",
        "status": "PASS",
        "mode": args.mode,
        "task_id": args.task_id,
        "rank": args.rank,
        "host": os.uname().nodename,
        "pid": os.getpid(),
        "started_unix": started_unix,
        "sealed_unix": time.time(),
        "claim": claim,
        "shards": shards,
        "model_index_sha256": observed_basis,
        "checkpoint_sha256": args.checkpoint_sha256,
        "construction_seconds": construction_seconds,
        "score_outer_seconds": score_outer_seconds,
        "projected_full64_seconds": projected,
        "proc_read_bytes_during_score": io_after.get("read_bytes", 0) - io_before.get("read_bytes", 0),
        "observed": result,
        "sealed_batch1_gate": gate,
    }
    path = args.run_root / "receipts" / f"{args.mode.upper()}_RANK{args.rank}.json"
    atomic_json(path, receipt)
    print(json.dumps({"receipt": str(path), **receipt}, sort_keys=True), flush=True)
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
