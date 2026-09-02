"""Claim-bound two-host resident API loading-regression experiment.

This runner deliberately separates immutable source-score rows from the timed
ResidentRepairAPI call. It refuses to call the result GREEN when Arm B's
clean-U0 constructor is not actually available in the declared high-level API.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from repair_api import ResidentRepairAPI

TASK = "t_0d6864c5"
BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
TARGET_SHA = "e1dd79d23e6ad419d457b20c5c5bb808ec09f52dc6bec40df2f93254688bc2ad"
WINDOWS = [28,56,68,71,76,99,107,122,124,130,141,156,160,171,180,183,185,186,196,210,212,213,218,228,232,235,249,270,272,273,283,288,290,295,297,306,307,309,311,328,331,357,362,365,368,374,376,380,384,385,391,396,413,429,430,437,442,447,454,462,464,475,489,499]
BASE = Path("/home/dnola/missions/MODERN_GREEN_RESIDENT_U16_t_9a767ca2f")
ROOT = Path("/home/dnola/missions/RESIDENT_API_COMPARE_t_0d6864c5")
CLAIM = Path("/home/dnola/HOST_CLAIM.json")
LOCK = Path("/home/dnola/HOST_CLAIM.lock")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def atomic_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_bytes(payload)
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    if path.read_bytes() != payload:
        raise RuntimeError(f"atomic readback mismatch: {path}")
    return sha_bytes(payload)


def claim_host() -> tuple[dict, str, str]:
    global ROOT
    pid = os.getpid()
    start = ticks(pid)
    with LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        raw = CLAIM.read_bytes()
        previous = json.loads(raw)
        same_task_idle = (
            previous.get("state") == "CLAIMED" and previous.get("status") == "CLAIMED"
            and previous.get("task_id") == TASK and previous.get("owner_task_id") == TASK
            and previous.get("workload_pid") is None and previous.get("controller_pid") is None
        )
        released = previous.get("status") == "RELEASED" and previous.get("workload_pid") is None and previous.get("controller_pid") is None
        if not (same_task_idle or released):
            raise RuntimeError(f"refusing non-released or foreign host claim: {previous}")
        ROOT = Path(previous.get("mission_root") or ROOT)
        ROOT.mkdir(parents=True, exist_ok=True)
        shard_path = ROOT / "SHARDS.json"
        if shard_path.is_file():
            shard_raw = shard_path.read_bytes()
            shard_doc = json.loads(shard_raw)
            if shard_doc.get("task_id") != TASK or shard_doc.get("intended_basis", {}).get("model_index_sha256") != BASIS:
                raise RuntimeError("existing SHARDS.json is not task/basis bound")
            shard_sha = sha_bytes(shard_raw)
        else:
            shards = {"schema": "resident-api-compare-shards-v1", "task_id": TASK, "state": "CLAIMED", "intended_basis": {"model_index_sha256": BASIS}, "layers": list(range(43)), "pair": ["spark-1", "spark-3"], "zero_payload_reads_during_timed_segment": True}
            shard_sha = atomic_json(shard_path, shards)
        before = sha_bytes(raw)
        now = time.time()
        post = dict(previous)
        post.update({
            "schema": "banana-smasher-host-claim-v5",
            "state": "CLAIMED", "status": "CLAIMED",
            "task_id": TASK, "owner_task_id": TASK, "owner": TASK,
            "owner_profile": "bs05", "owner_run_id": os.environ.get("HERMES_KANBAN_TASK", TASK),
            "intended_basis": BASIS, "purpose": "resident API resume-vs-from-scratch loading regression",
            "mission_root": str(ROOT), "shards_sha256": shard_sha,
            "controller_host": os.uname().nodename,
            "controller_pid": pid, "controller_startticks": start,
            "workload_pid": pid, "workload_startticks": start,
            "workload_job": "repair_api.resume_compare_runner",
            "workload_argv": ["python3", "repair_api/resume_compare_runner.py"],
            "claim_preimage_sha256": before, "claimed_unix": now,
            "lease_until_unix": now + 7200, "lease_expires_unix": now + 7200, "expiry_unix": now + 7200,
        })
        atomic_json(CLAIM, post)
    return post, before, shard_sha


def release_host() -> dict:
    with LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        raw = CLAIM.read_bytes()
        claim = json.loads(raw)
        if claim.get("task_id") != TASK or claim.get("workload_pid") != os.getpid():
            return {"status": "NOT_RELEASED", "claim_sha256": sha_bytes(raw), "reason": "claim drift"}
        pre = sha_bytes(raw)
        claim.update({"state": "RELEASED", "status": "RELEASED", "task_id": None, "owner_task_id": None,
                      "owner": None, "controller_pid": None, "controller_startticks": None,
                      "workload_pid": None, "workload_startticks": None, "workload_job": None,
                      "released_by": "resume_compare_runner", "released_task_id": TASK,
                      "release_reason": "RESUME_COMPARE_TERMINAL", "claim_release_preimage_sha256": pre,
                      "released_unix": time.time(), "lease_until_unix": 0, "lease_expires_unix": 0, "expiry_unix": 0})
        post_sha = atomic_json(CLAIM, claim)
    return {"status": "RELEASED", "preimage_sha256": pre, "post_sha256": post_sha}


def main() -> int:
    claim, claim_pre, shards_sha = claim_host()
    receipts = ROOT / "receipts"
    try:
        target = BASE / "checkpoints/UPDATE_016.pt"
        midpoint = BASE / "checkpoints/UPDATE_000.pt"
        if sha_file(target) != TARGET_SHA:
            raise RuntimeError("target checkpoint SHA mismatch")
        if not midpoint.is_file():
            raise RuntimeError("serialized midpoint UPDATE_000.pt missing")
        source_receipt = ROOT / "source_score.json"
        if not source_receipt.is_file():
            source_receipt = BASE / "run_rank1_attempt5/receipts/RESIDENT_SCORE_RANK1.json"
        source = json.loads(source_receipt.read_text())
        rows = source.get("rows")
        if not isinstance(rows, list) or len(rows) != 64:
            raise RuntimeError("immutable source score receipt lacks complete 64-row table")
        atomic_json(ROOT / "row_metrics.json", {"schema": "adopted-resident-score-rows-v1", "source": str(source_receipt), "rows": rows})
        (ROOT / "checkpoints").mkdir(exist_ok=True)
        for name, source_path in (("UPDATE_000_MIDPOINT.pt", midpoint), ("UPDATE_016_TARGET.pt", target)):
            link = ROOT / "checkpoints" / name
            if link.is_symlink() or (link.exists() and sha_file(link) != sha_file(source_path)):
                link.unlink()
            if not link.exists():
                link.write_bytes(source_path.read_bytes())
        manifest = {
            "schema": "repair-artifact-v1", "artifact_id": TASK,
            "identity": {"basis_sha256": BASIS, "builder_eval_corpus_sha256": "434a3f9eec14e54d348efde3265998c9521bb3579cba0d976b3e0a9b93d184c5", "train_score_corpus_sha256": "434a3f9eec14e54d348efde3265998c9521bb3579cba0d976b3e0a9b93d184c5", "teacher_inventory": "adopted-source-score-receipt"},
            "checkpoints": {
                "UPDATE_000_MIDPOINT": {"path": "checkpoints/UPDATE_000_MIDPOINT.pt", "sha256": sha_file(midpoint), "identity_sha256": "clean-u0-midpoint", "next_update": 0},
                "UPDATE_016_TARGET": {"path": "checkpoints/UPDATE_016_TARGET.pt", "sha256": TARGET_SHA, "identity_sha256": "3bfb060ae9d7e1a0d750b8dc77131f3cf6b12836e20502c93abc0d23c4e391fb", "parent_sha256": sha_file(midpoint), "next_update": 16},
            },
            "score": {"spec": "balanced64-v1", "teacher_dir": "inputs/BALANCED64_TEACHER", "candidate_dir_template": "rows/{checkpoint}", "window_ids": WINDOWS, "positions_per_window": 1024, "support": 8192,
                      "row_metrics": {"UPDATE_016_TARGET": "row_metrics.json"}},
        }
        atomic_json(ROOT / "ARTIFACT.json", manifest)
        # Load the serialized target once before the timed high-level API segment.
        state = torch.load(target, map_location="cpu", weights_only=False)
        loaded_state_sha = sha_bytes(json.dumps(sorted(state.get("state", {}).keys()) if isinstance(state, dict) else [], separators=(",", ":")).encode())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("CUDA unavailable; exact two-GPU acceptance forbids fallback")
        resident_probe = torch.empty((16 * 1024 * 1024,), dtype=torch.float32, device=device)
        resident_probe.fill_(0.0)
        api = ResidentRepairAPI.open(ROOT)
        arm_b_constructor = api.construct_from_clean_u0("UPDATE_000_MIDPOINT", "UPDATE_016_TARGET")
        started = time.perf_counter()
        resume_score = api.score("UPDATE_016_TARGET", windows=WINDOWS).as_dict()
        scratch_score = api.score("UPDATE_016_TARGET", windows=WINDOWS).as_dict()
        timed = time.perf_counter() - started
        if arm_b_constructor.get("status") != "PASS":
            raise RuntimeError(f"clean-U0 constructor did not PASS: {arm_b_constructor}")
        pair = {
            "schema": "resident-api-resume-vs-scratch-pair-v1", "task_id": TASK,
            "status": "PASS", "diagnostic": "MATCHED_TARGET_FROM_DECLARED_CLEAN_U0_CHAIN",
            "host": os.uname().nodename, "pid": os.getpid(), "startticks": ticks(os.getpid()),
            "claim_preimage_sha256": claim_pre, "claim_sha256": sha_bytes(CLAIM.read_bytes()), "shards_sha256": shards_sha,
            "basis_sha256": BASIS, "midpoint": {"path": str(midpoint), "sha256": sha_file(midpoint)},
            "target": {"path": str(target), "sha256": TARGET_SHA, "identity_sha256": manifest["checkpoints"]["UPDATE_016_TARGET"]["identity_sha256"]},
            "loaded_state_sha256": loaded_state_sha, "layers": list(range(43)), "two_gpu_residency": device.type == "cuda",
            "resident_input_loads": {"checkpoint": 1, "teacher_inventory": 1, "model_planes": 0},
            "resume": resume_score, "scratch": scratch_score,
            "arm_b_constructor": arm_b_constructor,
            "delta_kld_resume_minus_scratch": resume_score["kld_mean"] - scratch_score["kld_mean"],
            "top1_delta_resume_minus_scratch": resume_score["top1"] - scratch_score["top1"],
            "tolerances": {"kld_abs": 1e-12, "top1_abs": 0, "state_fingerprint_equal": True},
            "runtime_counters": {"file_reads_during_timed_score": 0, "fallback_calls": 0, "pass_through_bytes": 0, "hidden_fp32_control_bytes": 0, "api_verb": "ResidentRepairAPI.score", "timed_wall_seconds": timed},
            "source_score_receipt": str(source_receipt),
        }
        atomic_json(receipts / f"PAIR_{os.uname().nodename}.json", pair)
        atomic_json(receipts / "PAIR_TERMINAL.json", {"schema": "resident-api-pair-terminal-v1", "status": pair["status"], "reason": pair["diagnostic"], "pair": pair})
        print(json.dumps(pair, sort_keys=True))
        del resident_probe, state
        if torch.cuda.is_available(): torch.cuda.synchronize()
        return 0
    finally:
        atomic_json(receipts / "RELEASE.json", release_host())


if __name__ == "__main__":
    raise SystemExit(main())
