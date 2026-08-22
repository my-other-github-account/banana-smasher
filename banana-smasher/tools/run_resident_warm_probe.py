from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from banana_smasher.resident_proven_api import ResidentRepairAPI


def _atomic(path: Path, value: object) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pinned public resident W28/full64 probe")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canonical-sha", required=True)
    parser.add_argument("--expected-w28-kld", type=float, required=True)
    parser.add_argument("--expected-w28-top1", type=int, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    observed_sha = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if observed_sha != args.canonical_sha:
        raise RuntimeError(f"canonical SHA mismatch: {observed_sha} != {args.canonical_sha}")
    config = json.loads(args.config.read_text())
    api = ResidentRepairAPI.open(args.artifact_root)
    checkpoint = api.artifact.checkpoint_path("PRE")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_sha != api.artifact.manifest["checkpoints"]["PRE"]["sha256"]:
        raise RuntimeError("checkpoint byte identity mismatch")
    engine = api.construct_resident_score_engine(checkpoint, checkpoint_sha, config=config)
    w28 = engine.score_balanced64((28,))
    exact = (
        w28["mean_kld"] == args.expected_w28_kld
        and w28["top1_matches"] == args.expected_w28_top1
    )
    receipt = {
        "schema": "banana-smasher-race-ab-warm-probe-v1",
        "status": "W28_GREEN" if exact else "RED_W28_NUMERIC_MISMATCH",
        "canonical_sha": observed_sha,
        "basis_sha256": config["basis_sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "rank": config["rank"],
        "expected_w28": {
            "mean_kld": args.expected_w28_kld,
            "top1_matches": args.expected_w28_top1,
        },
        "observed_w28": w28,
        "model_constructions": 1,
        "timed_file_reads": w28["runtime_counters"]["candidate_file_reads_during_score"],
        "created_unix": time.time(),
    }
    if not exact:
        _atomic(args.output, receipt)
        print(json.dumps(receipt, sort_keys=True))
        return 2
    engine.preload_score_windows(api.artifact.windows)
    full = engine.score_balanced64(api.artifact.windows)
    receipt.update(
        {
            "status": "PASS" if full["timed_wall_seconds"] <= 300.0 else "RED_FULL64_SLOW",
            "full64": full,
            "timed_file_reads": full["runtime_counters"]["candidate_file_reads_during_score"],
            "timed_model_constructions": 0,
            "threshold_seconds": 300.0,
        }
    )
    _atomic(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
