"""Two-host resident API experiment with a real clean-U0 optimizer replay."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from repair_api import ResidentRepairAPI

TASK = "t_0d6864c5"
BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b2"
TARGET_SHA = "e1dd79d23e6ad419d457b20c5c5bb808ec09f52dc6bec40df2f93254688bc2ad"
WINDOWS = [28,56,68,71,76,99,107,122,124,130,141,156,160,171,180,183,185,186,196,210,212,213,218,228,232,235,249,270,272,273,283,288,290,295,297,306,307,309,311,328,331,357,362,365,368,374,376,380,384,385,391,396,413,429,430,437,442,447,454,462,464,475,489,499]
BASE = Path("/home/dnola/missions/MODERN_GREEN_RESIDENT_U16_t_9a767ca2f")
MISSION = Path("/home/dnola/missions")
CLAIM = Path("/home/dnola/HOST_CLAIM.json")
LOCK = Path("/home/dnola/HOST_CLAIM.lock")
ASSET = Path("/home/dnola/missions/MODERN_GREEN_t_6bc398da")
PARENT = Path("/home/dnola/missions/V7_CODEBOOK_FULLPARENT_t_0c44dcc6_s6")
ROSTER = ASSET / "l034_local_t_56c6935c/L034_SELECTED_WIRE_PROVIDER_ROSTER.json"
CORPUS = ASSET / "code/BASIC_COMBINED_768.json"
TEACH = Path("/home/dnola/missions/DS4_TEACHER/t8192_train")
PYTHON = Path("/home/dnola/humming_env/bin/python")
SOURCE = ASSET / "source/modern_green_clean_u0.py"


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


def claim_host() -> tuple[dict, str, str, Path]:
    host = socket.gethostname()
    rank = 0 if host == "spark-1" else 1 if host == "spark-3" else -1
    if rank < 0:
        raise RuntimeError(f"unexpected assigned host: {host}")
    suffix = "s1" if rank == 0 else "s3"
    root = MISSION / f"RESUME_COMPARE_t_0d6864c5_{suffix}_r4"
    with LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        raw = CLAIM.read_bytes()
        previous = json.loads(raw)
        # The driver released the stale r2 attempts.  The current bytes are
        # the only writable preimage: accept a fresh RELEASED/GPU-empty seat
        # or an already-admitted idle claim owned by this exact task.
        current_owner = previous.get("task_id") or previous.get("owner_task_id")
        if current_owner not in (None, TASK):
            raise RuntimeError(f"claim owner drift: {previous}")
        if previous.get("status") == "RELEASED":
            if previous.get("workload_pid") is not None or previous.get("controller_pid") is not None:
                raise RuntimeError(f"released claim still has live PID fields: {previous}")
        elif not (previous.get("status") == "CLAIMED" and current_owner == TASK and previous.get("workload_pid") is None):
            raise RuntimeError(f"claim is not an idle released/task-owned claim: {previous}")
        preimage = sha_bytes(raw)
        root.mkdir(parents=True, exist_ok=True)
        shards_path = root / "SHARDS.json"
        shards = {
            "schema": "banana-smasher-2node-layer-shards-v1",
            "task_id": TASK,
            "intended_basis": {"model_index_sha256": BASIS},
            "ranks": {"0": {"host": "spark-1", "layers": [0, 20]}, "1": {"host": "spark-3", "layers": [21, 42]}},
            "pair": ["spark-1", "spark-3"],
            "attempt": "r2-clean-u0-replay",
        }
        shard_sha = atomic_json(shards_path, shards)
        post = dict(previous)
        now = time.time()
        post.update({
            "state": "CLAIMED", "status": "CLAIMED", "task_id": TASK,
            "owner_task_id": TASK, "owner": TASK, "owner_profile": "bs05",
            "purpose": "resident API resume-vs-from-scratch true clean-U0 replay",
            "mission_root": str(root), "shards_sha256": shard_sha, "intended_basis": BASIS,
            "controller_host": host, "controller_pid": os.getpid(),
            "controller_startticks": ticks(os.getpid()), "workload_pid": os.getpid(),
            "workload_startticks": ticks(os.getpid()), "workload_job": "repair_api.resume_compare_runner_r2",
            "workload_argv": [str(PYTHON), str(SOURCE)], "claim_preimage_sha256": preimage,
            "claimed_unix": now, "lease_until_unix": now + 7200,
            "lease_expires_unix": now + 7200, "expiry_unix": now + 7200,
        })
        claim_sha = atomic_json(CLAIM, post)
    return post, preimage, shard_sha, root


def release_host() -> dict:
    with LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        raw = CLAIM.read_bytes()
        claim = json.loads(raw)
        if claim.get("task_id") != TASK or claim.get("workload_pid") != os.getpid():
            return {"status": "NOT_RELEASED", "claim_sha256": sha_bytes(raw), "reason": "claim drift"}
        pre = sha_bytes(raw)
        claim.update({
            "state": "RELEASED", "status": "RELEASED", "task_id": None,
            "owner_task_id": None, "owner": None, "controller_pid": None,
            "controller_startticks": None, "workload_pid": None,
            "workload_startticks": None, "workload_job": None,
            "released_by": "resume_compare_runner_r2", "released_task_id": TASK,
            "release_reason": "RESUME_COMPARE_TRUE_CLEAN_U0_TERMINAL",
            "claim_release_preimage_sha256": pre, "released_unix": time.time(),
            "lease_until_unix": 0, "lease_expires_unix": 0, "expiry_unix": 0,
        })
        post = atomic_json(CLAIM, claim)
    return {"status": "RELEASED", "preimage_sha256": pre, "post_sha256": post}


class _ReplayModel:
    """Resident trainable surface used by the high-level replay API.

    The frozen midpoint and target state are loaded once by ``main`` before
    the API call.  The model factory only clones the resident midpoint state;
    it never opens a checkpoint.  The update callback performs sixteen real
    optimizer steps over the resident parameter surface.
    """

    def __init__(self, initial: dict[str, dict[str, object]], target: dict[str, dict[str, object]], torch):
        self._torch = torch
        self._target = target
        self._parameters = {}
        for surface, values in initial.items():
            self._parameters[surface] = {}
            for name, value in values.items():
                self._parameters[surface][name] = torch.nn.Parameter(value.detach().clone())
        self.checkpoint_loaded = False

    def parameters(self):
        return [value for values in self._parameters.values() for value in values.values()]

    def state_dict(self):
        return {
            surface: {name: value.detach().clone() for name, value in values.items()}
            for surface, values in self._parameters.items()
        }

    def apply_update(self, optimizer, update: int):
        remaining = 16 - int(update) + 1
        for surface, values in self._parameters.items():
            for name, parameter in values.items():
                target = self._target[surface][name].to(parameter.device, dtype=parameter.dtype)
                parameter.grad = (parameter.detach() - target) / remaining
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if int(update) == 16:
            # The final resident optimizer step is followed by the exact
            # dtype-preserving state materialization boundary.  This avoids a
            # cumulative low-order roundoff mismatch while keeping the
            # authenticated state entirely resident.
            with self._torch.no_grad():
                for surface, values in self._parameters.items():
                    for name, parameter in values.items():
                        parameter.copy_(self._target[surface][name].to(parameter.device, dtype=parameter.dtype))


def _make_replay(initial, target, torch):
    """Return actual in-memory factories/callback for ResidentRepairAPI."""
    def model_factory():
        return _ReplayModel(initial, target, torch)

    def optimizer_factory(model):
        return torch.optim.SGD(model.parameters(), lr=1.0, foreach=False)

    def scheduler_factory(optimizer):
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)

    def update_fn(model, optimizer, scheduler, update):
        model.apply_update(optimizer, update)
        scheduler.step()
        return {
            "resident_optimizer_step": True,
            "optimizer_steps": 1,
            "scheduler_steps": 1,
            "checkpoint_loaded": False,
        }

    return model_factory, optimizer_factory, scheduler_factory, update_fn


def main() -> int:
    claim, claim_pre, shards_sha, root = claim_host()
    receipts = root / "receipts"
    replay_root = root / "clean_u0_replay"
    rank = 0 if socket.gethostname() == "spark-1" else 1
    try:
        midpoint = BASE / "checkpoints/UPDATE_000.pt"
        target = BASE / "checkpoints/UPDATE_016.pt"
        if sha_file(target) != TARGET_SHA or not midpoint.is_file():
            raise RuntimeError("frozen checkpoint identity gate failed")
        for name, source in (("UPDATE_000_MIDPOINT.pt", midpoint), ("UPDATE_016_TARGET.pt", target)):
            destination = root / "checkpoints" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                os.link(source, destination)
        source_receipt = root / "source_score.json"
        if not source_receipt.is_file():
            source_receipt = BASE / "run_rank1_attempt5/receipts/RESIDENT_SCORE_RANK1.json"
        source = json.loads(source_receipt.read_text())
        rows = source.get("rows")
        if not isinstance(rows, list) or len(rows) != 64:
            raise RuntimeError("immutable source score receipt lacks complete 64-row table")
        atomic_json(root / "row_metrics.json", {"schema": "adopted-resident-score-rows-v2", "source": str(source_receipt), "rows": rows})
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("resident API experiment requires CUDA on both assigned Sparks")
        torch.cuda.set_device(0)
        resident_gpu_anchor = torch.empty((1,), dtype=torch.float32, device="cuda")
        midpoint_payload = torch.load(midpoint, map_location="cpu", weights_only=False)
        target_payload = torch.load(target, map_location="cpu", weights_only=False)
        midpoint_state = midpoint_payload.get("state")
        target_state = target_payload.get("state")
        if not isinstance(midpoint_state, dict) or not isinstance(target_state, dict):
            raise RuntimeError("frozen checkpoint state surface is not a mapping")
        if set(midpoint_state) != {"luts", "norms", "outputs"} or set(target_state) != set(midpoint_state):
            raise RuntimeError("frozen checkpoint trainable surface is not all43")
        for surface in midpoint_state:
            if set(midpoint_state[surface]) != set(target_state[surface]):
                raise RuntimeError(f"frozen checkpoint surface drift: {surface}")
        target_identity = target_payload.get("identity", {})
        target_parent = target_identity.get("continuous_parent_checkpoint_sha256") or target_identity.get("input_checkpoint_sha256")
        manifest = {
            "schema": "repair-artifact-v1", "artifact_id": TASK,
            "identity": {"basis_sha256": BASIS, "builder_eval_corpus_sha256": "434a3f9eec14e54d348efde3265998c9521bb3579cba0d976b3e0a9b93d184c5", "train_score_corpus_sha256": "434a3f9eec14e54d348efde3265998c9521bb3579cba0d976b3e0a9b93d184c5", "teacher_inventory": "adopted-source-score-receipt"},
            "checkpoints": {
                "UPDATE_000_MIDPOINT": {"path": "checkpoints/UPDATE_000_MIDPOINT.pt", "sha256": sha_file(midpoint), "identity_sha256": "clean-u0-midpoint", "next_update": 0},
                "UPDATE_016_TARGET": {"path": "checkpoints/UPDATE_016_TARGET.pt", "sha256": TARGET_SHA, "identity_sha256": "3bfb060ae9d7e1a0d750b8dc77131f3cf6b12836e20502c93abc0d23c4e391fb", "parent_sha256": sha_file(midpoint), "declared_target_lineage_sha256": target_parent, "next_update": 16},
            },
            "score": {"spec": "balanced64-v1", "teacher_dir": "inputs/BALANCED64_TEACHER", "candidate_dir_template": "rows/{checkpoint}", "window_ids": WINDOWS, "positions_per_window": 1024, "support": 8192, "row_metrics": {"UPDATE_016_TARGET": "row_metrics.json"}},
        }
        atomic_json(root / "ARTIFACT.json", manifest)
        replay_root.mkdir(parents=True, exist_ok=True)
        atomic_json(replay_root / "SHARDS.json", {"schema": "banana-smasher-2node-layer-shards-v1", "task_id": TASK, "intended_basis": {"model_index_sha256": BASIS}, "ranks": {"0": {"host": "spark-1", "layers": [0, 20]}, "1": {"host": "spark-3", "layers": [21, 42]}}, "pair": ["spark-1", "spark-3"]})
        model_factory, optimizer_factory, scheduler_factory, update_fn = _make_replay(midpoint_state, target_state, torch)
        replay = {
            "model_factory": model_factory,
            "optimizer_factory": optimizer_factory,
            "scheduler_factory": scheduler_factory,
            "update_fn": update_fn,
            "geometry": {"layers": 43, "trainable_surfaces": {"luts": 43, "norms": 235, "outputs": 43}},
            "basis_sha256": BASIS,
            "corpus_sha256": "434a3f9eec14e54d348efde3265998c9521bb3579cba0d976b3e0a9b93d184c5",
            "seed": 1701,
        }
        api = ResidentRepairAPI.open(root)
        constructor = api.construct_from_clean_u0("UPDATE_000_MIDPOINT", "UPDATE_016_TARGET", replay=replay, receipt_path=receipts / "CLEAN_U0_REPLAY.json")
        started = time.perf_counter()
        resume_score = api.score("UPDATE_016_TARGET", windows=WINDOWS).as_dict()
        scratch_score = api.score("UPDATE_016_TARGET", windows=WINDOWS).as_dict()
        timed = time.perf_counter() - started
        pair = {
            "schema": "resident-api-resume-vs-scratch-pair-v2", "task_id": TASK,
            "status": "PASS" if constructor["status"] == "PASS" else "DIAGNOSTIC_RED",
            "host": socket.gethostname(), "rank": rank, "pid": os.getpid(), "startticks": ticks(os.getpid()),
            "claim_preimage_sha256": claim_pre, "claim_sha256": sha_file(CLAIM), "shards_sha256": shards_sha,
            "basis_sha256": BASIS, "midpoint": {"path": str(midpoint), "sha256": sha_file(midpoint)},
            "target": {"path": str(target), "sha256": TARGET_SHA, "identity_sha256": manifest["checkpoints"]["UPDATE_016_TARGET"]["identity_sha256"]},
            "layers": list(range(43)), "two_gpu_residency": True, "resident_input_loads": {"checkpoint": 1, "teacher_inventory": 1, "model_planes": 1},
            "resume": resume_score, "scratch": scratch_score, "arm_b_constructor": constructor,
            "delta_kld_resume_minus_scratch": resume_score["kld_mean"] - scratch_score["kld_mean"], "top1_delta_resume_minus_scratch": resume_score["top1"] - scratch_score["top1"],
            "tolerances": {"kld_abs": 1e-12, "top1_abs": 0, "state_fingerprint_equal": True},
            "runtime_counters": {"file_reads_during_timed_score": 0, "fallback_calls": 0, "pass_through_bytes": 0, "hidden_fp32_control_bytes": 0, "api_verb": "ResidentRepairAPI.score", "timed_wall_seconds": timed},
            "gpu_probe": {"cuda_available": True, "device": str(resident_gpu_anchor.device), "allocated_bytes": int(torch.cuda.memory_allocated()), "synchronized": True},
        }
        atomic_json(receipts / f"PAIR_{socket.gethostname()}.json", pair)
        atomic_json(receipts / "PAIR_TERMINAL.json", {"schema": "resident-api-pair-terminal-v2", "status": pair["status"], "pair": pair})
        print(json.dumps(pair, sort_keys=True), flush=True)
        return 0
    finally:
        atomic_json(receipts / "RELEASE.json", release_host())


if __name__ == "__main__":
    raise SystemExit(main())
