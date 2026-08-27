"""Run the canonical Modern Green anchors through the resident repair API."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

from . import RepairArtifact
from .production_score_guard import reject_standalone_score_runner

U3_TARGET = 0.22103965283948
MAX_ANCHOR_SECONDS = 1200.0


def _ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: Any) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    return _sha_bytes(payload)


def _claim(claim_path: Path, task_id: str, basis: str, root: Path) -> dict[str, Any]:
    lock_path = claim_path.with_name(f"{claim_path.name}.lock")
    pid = os.getpid()
    startticks = _ticks(pid)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        raw = claim_path.read_bytes()
        previous = json.loads(raw)
        released = (
            previous.get("state") == "RELEASED"
            and previous.get("status") == "RELEASED"
            and previous.get("task_id") is None
            and previous.get("workload_pid") is None
            and previous.get("controller_pid") is None
        )
        same_task_idle = (
            previous.get("state") == "CLAIMED"
            and previous.get("status") == "CLAIMED"
            and previous.get("task_id") == task_id
            and previous.get("workload_pid") is None
            and previous.get("controller_pid") is None
        )
        if not (released or same_task_idle):
            raise RuntimeError("refusing to claim non-released or concurrently-owned host")
        now = time.time()
        claim = dict(previous)
        claim.update(
            {
                "state": "CLAIMED",
                "status": "CLAIMED",
                "task_id": task_id,
                "owner_task_id": task_id,
                "owner": "main-session",
                "owner_profile": "main-session",
                "intended_basis": basis,
                "purpose": "resident in-memory Modern Green Balanced64 campaign",
                "controller_pid": pid,
                "controller_startticks": startticks,
                "workload_pid": pid,
                "workload_startticks": startticks,
                "workload_job": "resident_campaign",
                "workload_argv": [sys.executable, "-m", "repair_api.resident_campaign"],
                "workload_checkpoint_sha256": None,
                "workload_dispatch_sha256": None,
                "claimed_unix": now,
                "lease_until_unix": now + 7200,
                "lease_expires_unix": now + 7200,
                "expiry_unix": now + 7200,
                "claim_preimage_sha256": _sha_bytes(raw),
            }
        )
        post = _atomic_json(claim_path, claim)
    return {
        "status": "CLAIMED",
        "task_id": task_id,
        "pid": pid,
        "startticks": startticks,
        "preimage_sha256": _sha_bytes(raw),
        "post_sha256": post,
    }


def _release(claim_path: Path, task_id: str) -> dict[str, Any]:
    lock_path = claim_path.with_name(f"{claim_path.name}.lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        raw = claim_path.read_bytes()
        claim = json.loads(raw)
        if claim.get("task_id") != task_id or claim.get("workload_pid") != os.getpid():
            return {"status": "NOT_RELEASED", "reason": "claim changed or no longer owned"}
        preimage = _sha_bytes(raw)
        now = time.time()
        claim.update(
            {
                "state": "RELEASED",
                "status": "RELEASED",
                "task_id": None,
                "owner_task_id": None,
                "owner": None,
                "controller_pid": None,
                "controller_startticks": None,
                "workload_pid": None,
                "workload_startticks": None,
                "workload_job": None,
                "released_by": "resident_campaign",
                "released_by_task_id": task_id,
                "release_reason": "RESIDENT_CAMPAIGN_TERMINAL",
                "released_unix": now,
                "claim_release_preimage_sha256": preimage,
                "lease_until_unix": now,
                "lease_expires_unix": now,
                "expiry_unix": now,
            }
        )
        post = _atomic_json(claim_path, claim)
    return {"status": "RELEASED", "preimage_sha256": preimage, "post_sha256": post}


def _run_anchor(artifact: RepairArtifact, key: str, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    reject_standalone_score_runner("repair_api.resident_campaign._run_anchor")
    started = time.perf_counter()
    generation = artifact.generate_candidates(
        key,
        builder_template=args.builder_template,
        ref_dir=args.ref_dir,
        corpus=args.builder_corpus,
        meta_dir=args.meta_dir,
        python_executable=args.python_executable,
        mode=args.mode,
        remote=args.remote,
        local_dir=args.local_dir,
        chunk=args.chunk,
        mb=args.mb,
    )
    generated_at = time.perf_counter()
    result = artifact.score_in_memory(key)
    finished = time.perf_counter()
    score = result.as_dict()
    receipt = {
        "schema": "resident-api-anchor-v1",
        "status": "PASS" if result.positions == 65536 and result.timed_wall_seconds is not None and result.timed_wall_seconds < MAX_ANCHOR_SECONDS else "RED",
        "checkpoint": result.checkpoint,
        "checkpoint_sha256": artifact.manifest["checkpoints"][result.checkpoint].get("sha256"),
        "checkpoint_identity_sha256": artifact.manifest["checkpoints"][result.checkpoint].get("identity_sha256"),
        "generation": generation,
        "score": score,
        "generation_wall_seconds": generated_at - started,
        "anchor_wall_seconds": finished - started,
        "under_20_minute_anchor": (finished - started) < MAX_ANCHOR_SECONDS,
        "score_execution_mode": result.execution_mode,
    }
    _atomic_json(out_dir / f"{result.checkpoint}_RESIDENT.json", receipt)
    if receipt["status"] != "PASS":
        raise RuntimeError(f"resident anchor failed timing/coverage gate: {receipt}")
    return receipt


def main(argv: list[str] | None = None) -> int:
    reject_standalone_score_runner("repair_api.resident_campaign")
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--builder-template", type=Path, required=True)
    parser.add_argument("--ref-dir", type=Path, required=True)
    parser.add_argument("--builder-corpus", type=Path, required=True)
    parser.add_argument("--meta-dir", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--local-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=("w2", "planes"), default="w2")
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--mb", type=int, default=1)
    parser.add_argument("--task-id", default="t_f5d2415c")
    parser.add_argument("--basis", required=True)
    parser.add_argument("--claim-path", type=Path, default=Path("/home/dnola/HOST_CLAIM.json"))
    args = parser.parse_args(argv)

    artifact = RepairArtifact.open(args.artifact_root)
    receipts_dir = args.artifact_root / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    claim = _claim(args.claim_path, args.task_id, args.basis, args.artifact_root)
    _atomic_json(receipts_dir / "CLAIM_RESIDENT_API.json", claim)
    anchors: dict[str, Any] = {}
    status = "FAILED"
    try:
        for key in ("UPDATE_000", "UPDATE_003"):
            anchors[key] = _run_anchor(artifact, key, args, receipts_dir)
        u3_kld = float(anchors["UPDATE_003"]["score"]["kld_mean"])
        if abs(u3_kld - U3_TARGET) > 1e-12:
            raise RuntimeError(f"U3 calibration mismatch: measured {u3_kld!r}, target {U3_TARGET!r}")
        anchors["UPDATE_016"] = _run_anchor(artifact, "UPDATE_016", args, receipts_dir)
        status = "PASS"
    finally:
        release = _release(args.claim_path, args.task_id)
        _atomic_json(receipts_dir / "CLAIM_RELEASE.json", release)
    terminal = {
        "schema": "modern-green-resident-api-terminal-v1",
        "status": status,
        "artifact_root": str(args.artifact_root),
        "accepted_u3_target": U3_TARGET,
        "green_u3_reference": {
            "kld": 0.226162314683653,
            "top1": 56700,
            "source": "/Volumes/U5TDD/t_efa23ac5/U5_BALANCED64_TERMINAL.json",
        },
        "anchors": anchors,
        "side_by_side": {
            "pre_repair": anchors.get("UPDATE_000", {}).get("score", {}).get("kld_mean"),
            "modern_u3": anchors.get("UPDATE_003", {}).get("score", {}).get("kld_mean"),
            "modern_u16": anchors.get("UPDATE_016", {}).get("score", {}).get("kld_mean"),
            "green_u3_reference": 0.226162314683653,
        },
    }
    _atomic_json(receipts_dir / "MODERN_GREEN_RESIDENT_API_TERMINAL.json", terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
