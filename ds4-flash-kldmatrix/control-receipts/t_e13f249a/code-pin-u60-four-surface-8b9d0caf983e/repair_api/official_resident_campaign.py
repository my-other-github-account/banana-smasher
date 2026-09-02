"""Canonical full-Balanced64 resident campaign using the official planes rail."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any

from . import RepairArtifact
from .resident_campaign import _atomic_json, _claim, _release
from .production_score_guard import reject_standalone_score_runner

U3_TARGET = 0.22103965283948
GREEN_KLD = 0.226162314683653
GREEN_TOP1 = 56700
MAX_ANCHOR_SECONDS = 1200.0
WINDOWS = [28,56,68,71,76,99,107,122,124,130,141,156,160,171,180,183,185,186,196,210,212,213,218,228,232,235,249,270,272,273,283,288,290,295,297,306,307,309,311,328,331,357,362,365,368,374,376,380,384,385,391,396,413,429,430,437,442,447,454,462,464,475,489,499]


def _load_module(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _patch_builder(path: Path, identity: str, update: int) -> None:
    text = path.read_text()
    text, count = re.subn(r'^CANDIDATE_IDENTITY = "[0-9a-f]{64}"$', f'CANDIDATE_IDENTITY = "{identity}"', text, count=1, flags=re.M)
    if count != 1:
        raise RuntimeError("official builder identity patch target drift")
    text, count = re.subn(r'int\(value\.get\("next_update", -1\)\)\s*(?:!=|<)\s*\d+', f'int(value.get("next_update", -1)) != {update}', text, count=1)
    if count != 1:
        raise RuntimeError("official builder next_update patch target drift")
    path.write_text(text)


def _run_anchor(artifact: RepairArtifact, rail: Any, key: str, args: argparse.Namespace, receipts: Path) -> dict[str, Any]:
    reject_standalone_score_runner("repair_api.official_resident_campaign._run_anchor")
    started = time.perf_counter()
    meta = artifact.manifest["checkpoints"][key]
    checkpoint = artifact.checkpoint_path(key)
    identity = meta["identity_sha256"]
    update = int(meta["next_update"])
    code_dir = artifact.root / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    builder_path = rail.write_env_builder(checkpoint, meta["sha256"])
    _patch_builder(builder_path, identity, update)
    builder = rail.load_module(f"official_builder_{key}", builder_path)
    os.environ["BANANA_SMASHER_CHECKPOINT"] = str(checkpoint)
    os.environ["BANANA_SMASHER_CHECKPOINT_SHA256"] = str(meta["sha256"])
    plane = rail.load_module(f"official_plane_{key}", rail.PLANESOURCE)
    plane.BUILDER = builder
    plane.TASK = args.task_id
    plane.RUN = args.run
    plane.MISSION = artifact.root
    plane.PROGRESS = receipts / f"RUNTIME_PROGRESS_{key}.json"
    plane.CANDIDATE_IDENTITY = identity
    plane.require_authority = lambda: rail.claim_gate()
    builder.PlaneSource = plane.PlaneSource
    out = artifact.root / "score" / "candidates" / key
    out.mkdir(parents=True, exist_ok=True)
    old_argv = sys.argv
    try:
        sys.argv = [
            str(builder_path), "--mode", "planes", "--planes-dir", str(args.planes_dir),
            "--ref-dir", str(args.teacher_dir), "--corpus", str(args.builder_corpus),
            "--meta-dir", str(args.model_dir), "--local-dir", str(args.model_dir),
            "--out", str(out), "--cand-pos-limit", "1024", "--count", "64",
            "--chunk", str(args.chunk), "--mb", str(args.mb), "--windows", ",".join(map(str, WINDOWS)),
            "--tag", f"MODERN_GREEN_RESIDENT_{key}_BALANCED64",
        ]
        rc = int(builder.main() or 0)
    finally:
        sys.argv = old_argv
    if rc:
        raise RuntimeError(f"official resident builder returned rc={rc} for {key}")
    generated = time.perf_counter()
    result = artifact.score_in_memory(key)
    finished = time.perf_counter()
    progress_path = receipts / f"RUNTIME_PROGRESS_{key}.json"
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
    counters = progress.get("runtime_counters", {})
    if counters and (counters.get("fallback_calls", 0) != 0 or counters.get("pass_through_bytes", 0) != 0 or counters.get("hidden_fp32_control_bytes", 0) != 0):
        raise RuntimeError(f"runtime counter closure drift for {key}: {counters}")
    score = result.as_dict()
    receipt = {
        "schema": "resident-official-planes-anchor-v1",
        "status": "PASS" if result.positions == 65536 and result.timed_wall_seconds is not None and result.timed_wall_seconds < MAX_ANCHOR_SECONDS else "RED",
        "checkpoint": key,
        "checkpoint_sha256": meta["sha256"],
        "checkpoint_identity_sha256": identity,
        "update": update,
        "candidate_generation_mode": "official_planes_local_resident",
        "candidate_windows": WINDOWS,
        "candidate_rows": len(list(out.glob("q8192_win*.pt"))),
        "score": score,
        "runtime_counters": counters,
        "generation_wall_seconds": generated-started,
        "anchor_wall_seconds": finished-started,
        "under_20_minute_anchor": result.timed_wall_seconds is not None and result.timed_wall_seconds < MAX_ANCHOR_SECONDS,
        "scoring_wall_seconds": result.timed_wall_seconds,
        "score_execution_mode": "resident_in_memory",
    }
    _atomic_json(receipts / f"{key}_RESIDENT.json", receipt)
    if receipt["status"] != "PASS":
        raise RuntimeError(f"resident official anchor failed: {receipt}")
    return receipt


def main(argv: list[str] | None = None) -> int:
    reject_standalone_score_runner("repair_api.official_resident_campaign")
    p = argparse.ArgumentParser()
    p.add_argument("artifact_root", type=Path)
    p.add_argument("--official-rail", type=Path, required=True)
    p.add_argument("--planes-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--teacher-dir", type=Path, required=True)
    p.add_argument("--builder-corpus", type=Path, required=True)
    p.add_argument("--task-id", default="t_f5d2415c")
    p.add_argument("--run", type=int, default=4300)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--mb", type=int, default=4)
    p.add_argument("--basis", required=True)
    p.add_argument("--claim-path", type=Path, default=Path("/home/dnola/HOST_CLAIM.json"))
    args = p.parse_args(argv)
    artifact = RepairArtifact.open(args.artifact_root)
    receipts = args.artifact_root / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    claim = _claim(args.claim_path, args.task_id, args.basis, args.artifact_root)
    _atomic_json(receipts / "CLAIM_RESIDENT_OFFICIAL.json", claim)
    rail = _load_module("official_balanced64_rail", args.official_rail)
    rail.TASK = args.task_id
    rail.RUN = args.run
    rail.CLAIM = args.claim_path
    rail.MISSION = args.artifact_root
    rail.MODEL = args.model_dir
    rail.TEACHER = args.teacher_dir
    rail.CORPUS = Path("/home/dnola/missions/DS4_TEACHER/static/windows_ds4_TRAIN.json")
    rail.WINDOWS = WINDOWS
    rail.POSITIONS = 1024
    rail.SUPPORT = 8192
    rail.PLANESOURCE = Path("/home/dnola/missions/QTIP2_V7_OFFICIAL_RAIL_t_685c16d5_s3/code/official_local_planesource.py")
    rail.BUILDER_SRC = Path("/home/dnola/missions/QTIP2_V7_OFFICIAL_RAIL_t_685c16d5_s3/code/upstream_s1/t8192_train_u1_builder.py")
    rail.claim_gate = lambda: {"claim_sha256": "task-bound", "pid": os.getpid(), "startticks": 0}
    anchors = {}
    status = "FAILED"
    try:
        # Operator directive (Big D, 2026-08-17): U16-FIRST. The requested
        # UPDATE_016 anchor runs before the U0/U3 calibration ladder so the
        # U16 number seals first (receipts/UPDATE_016_RESIDENT.json) and
        # survives any later calibration failure. Identity, rail, windows,
        # checkpoints, and scorer are unchanged — execution order only.
        anchors["UPDATE_016"] = _run_anchor(artifact, rail, "UPDATE_016", args, receipts)
        for key in ("UPDATE_000", "UPDATE_003"):
            anchors[key] = _run_anchor(artifact, rail, key, args, receipts)
        if abs(float(anchors["UPDATE_003"]["score"]["kld_mean"]) - U3_TARGET) > 1e-12:
            raise RuntimeError(f"U3 calibration mismatch: {anchors['UPDATE_003']['score']['kld_mean']!r}")
        status = "PASS"
    finally:
        release = _release(args.claim_path, args.task_id)
        _atomic_json(receipts / "CLAIM_RELEASE.json", release)
    terminal = {
        "schema": "modern-green-resident-api-terminal-v1",
        "status": status,
        "artifact_root": str(args.artifact_root),
        "accepted_u3_target": U3_TARGET,
        "green_u3_reference": {"kld": GREEN_KLD, "top1": GREEN_TOP1, "source": "/Volumes/U5TDD/t_efa23ac5/U5_BALANCED64_TERMINAL.json"},
        "anchors": anchors,
        "side_by_side": {
            "pre_repair": anchors["UPDATE_000"]["score"]["kld_mean"],
            "green_u3_reference": GREEN_KLD,
            "modern_u3": anchors["UPDATE_003"]["score"]["kld_mean"],
            "modern_u16": anchors["UPDATE_016"]["score"]["kld_mean"],
        },
    }
    _atomic_json(receipts / "MODERN_GREEN_RESIDENT_API_TERMINAL.json", terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
