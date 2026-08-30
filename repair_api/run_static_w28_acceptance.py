from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from repair_api import sealed_pre_forward


def startticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def replace_claim(path: Path, claim: dict) -> None:
    stat = path.stat()
    raw = (json.dumps(claim, sort_keys=True, indent=2) + "\n").encode()
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.fchown(fd, stat.st_uid, stat.st_gid)
    os.fchmod(fd, stat.st_mode & 0o777)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--l034-roster", type=Path, required=True)
    parser.add_argument("--canonical-pin", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != args.canonical_pin:
        raise SystemExit(f"CANONICAL_PIN_REFUSED:{head}")
    claim_path = Path("/home/dnola/HOST_CLAIM.json")
    prior = json.loads(claim_path.read_text())
    if prior.get("state", prior.get("status")) != "RELEASED":
        raise SystemExit(f"SEAT_NOT_RELEASED:{prior.get('owner_task_id')}:{prior.get('state')}")
    args.root.mkdir(parents=True, exist_ok=False)
    pid = os.getpid(); ticks = startticks(pid); now = time.time()
    claim = dict(prior)
    claim.update({
        "schema": "banana-smasher-host-claim-v5", "state": "CLAIMED", "status": "CLAIMED",
        "task_id": args.task, "owner_task_id": args.task, "owner": f"{args.task}/macmini4",
        "owner_profile": "macmini4", "board_run_id": args.run_id, "active_board_run_id": args.run_id,
        "host": "spark-1", "host_id": "spark-1", "seat": "spark-1 rank0", "rank": 0,
        "intended_basis": sealed_pre_forward.BASIS_SHA256, "basis_sha256": sealed_pre_forward.BASIS_SHA256,
        "canonical_git_pin": head, "canonical_code_commit": head,
        "workload_pid": pid, "workload_startticks": ticks, "holder_pid": pid, "holder_startticks": ticks,
        "claimed_unix": now, "updated_unix": now, "heartbeat_unix": now,
        "lease_until_unix": now + 8 * 3600, "expires_unix": now + 8 * 3600,
        "phase": "STATIC_W28_PLANES_MB2_PAIRED", "do_not_preempt": True,
        "purpose": "exact sealed static W28 planes-mode acceptance; no full64",
        "run_root": str(args.root), "mission_root": str(args.root),
    })
    replace_claim(claim_path, claim)
    os.environ["HERMES_KANBAN_RUN_ID"] = str(args.run_id)
    launch = {
        "schema": "banana-smasher-static-w28-launch-v1", "status": "RUNNING",
        "task_id": args.task, "board_run_id": args.run_id, "host": "spark-1", "rank": 0,
        "pid": pid, "startticks": ticks, "canonical_git_pin": head,
        "root": str(args.root), "cache_root": str(args.cache_root), "full64_launched": False,
        "created_unix": now,
    }
    launch["receipt_sha256"] = sealed_pre_forward.atomic_json(args.root / "LAUNCH.json", launch)
    try:
        config = {
            "task_id": args.task, "rank": 0, "world_size": 1,
            "l034_roster": str(args.l034_roster),
            "validation_teacher_root": str(args.teacher), "validation_corpus": str(args.corpus),
            "basis_sha256": sealed_pre_forward.BASIS_SHA256,
            "checkpoint_sha256": sealed_pre_forward.CHECKPOINT_SHA256,
            "model_root": str(args.model_root),
            "canonical_git_pin": head, "board_run_id": args.run_id,
            "sealed_pre_cache_root": str(args.cache_root),
            "sealed_pre_use_local_model": True,
            "sealed_builder_window_microbatch": 2, "sealed_builder_chunk": 64,
        }
        receipt = sealed_pre_forward.run_static_w28_acceptance(
            task=args.task, rank=0, root=args.root, config=config,
            checkpoint=args.checkpoint, canonical_pin=head,
        )
        print(json.dumps(receipt, sort_keys=True), flush=True)
    finally:
        current = json.loads(claim_path.read_text())
        if current.get("owner_task_id") == args.task and current.get("workload_pid") == pid:
            released = time.time()
            current.update(state="RELEASED", status="RELEASED", phase="STATIC_W28_RELEASED",
                           workload_pid=None, workload_startticks=None, holder_pid=None, holder_startticks=None,
                           do_not_preempt=False, lease_until_unix=released, expires_unix=released,
                           released_unix=released, updated_unix=released)
            replace_claim(claim_path, current)


if __name__ == "__main__":
    main()
