#!/usr/bin/env python3
from __future__ import annotations
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import torch

from repair_api import ResidentRepairAPI
from repair_api.modern_green_resident import ModernGreenResidentEngine

TASK = "t_d4dac464"
BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
CHECKPOINT = "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70"
W28_KLD = 0.13712959240533734
W28_TOP1 = 877


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return hashlib.sha256(raw).hexdigest()


def aggregate_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positions = sum(int(row["positions"]) for row in rows)
    kld_sum = math.fsum(float(row["kld_sum_binary64"]) for row in rows)
    top1 = sum(int(row["top1"]) for row in rows)
    return {"positions": positions, "kld_sum": kld_sum, "kld_mean": kld_sum / positions,
            "top1": top1, "top1_rate": top1 / positions}


def main() -> None:
    rank = int(os.environ["RANK"])
    root = Path(os.environ["PHYSICAL_ROOT"])
    config_path = root / f"CONFIG.{TASK}.rank{rank}.json"
    config = json.loads(config_path.read_text())
    pin = str(config["canonical_code_commit"])
    if len(pin) != 40 or config.get("basis_sha256") != BASIS:
        raise RuntimeError("CONFIG_PIN_OR_BASIS_MISMATCH")
    index = Path(config["model_root"]) / "model.safetensors.index.json"
    if sha(index) != BASIS:
        raise RuntimeError("BASIS_GATE_MISMATCH")
    api = ResidentRepairAPI.open(Path(config["artifact_root"]))
    checkpoint_path = api.artifact.checkpoint_path("PRE")
    if sha(checkpoint_path) != CHECKPOINT:
        raise RuntimeError("PUBLISHED_PRE_CHECKPOINT_MISMATCH")
    reference_path = Path(config["reference_terminal"])
    reference = json.loads(reference_path.read_text())
    if reference.get("basis_sha256") != BASIS or reference.get("checkpoint_sha256") != CHECKPOINT:
        raise RuntimeError("REFERENCE_IDENTITY_MISMATCH")
    windows = tuple(int(value) for value in reference["coverage"]["expected_windows"])
    if len(windows) != 64 or len(set(windows)) != 64 or windows[0] != 28:
        raise RuntimeError("REFERENCE_WINDOW_ROSTER_MISMATCH")
    expected_rows = {int(row["window"]): row for row in reference["per_window"]}
    if set(expected_rows) != set(windows):
        raise RuntimeError("REFERENCE_ROW_COVERAGE_MISMATCH")
    if int(config.get("score_window_batch_size", 0)) != 4:
        raise RuntimeError("FULL64_REQUIRES_BATCH4")
    if config.get("resident_validation_expert_implementation") not in {None, "accepted_static_w28"}:
        raise RuntimeError("FULL64_REQUIRES_ACCEPTED_PROVIDER")
    # Eager remains the exact parity-proof admission mode. Production changes
    # only the sanctioned attention mechanics after that receipt is sealed.
    config["resident_validation_attention_implementation"] = "eager"

    os.environ["NCCL_SOCKET_IFNAME"] = str(config["nccl_socket_ifname"])
    os.environ["GLOO_SOCKET_IFNAME"] = str(config["nccl_socket_ifname"])
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(seconds=900))
    if torch.distributed.get_world_size() != 2 or torch.distributed.get_rank() != rank:
        raise RuntimeError("DIST_GEOMETRY_MISMATCH")

    process_started = time.perf_counter()
    load_started = time.perf_counter()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    engine = ModernGreenResidentEngine(payload=payload, config=config, rank=rank,
                                       layer_ranges={0: (0, 20), 1: (21, 42)})
    del payload
    torch.cuda.synchronize()
    resident_load_seconds = time.perf_counter() - load_started

    # The accepted W28 gate was captured in the sealed builder's aligned mb=2
    # fixture. Production then switches only the physical validation geometry
    # to the admitted batch4 resident path; the engine and weights stay loaded.
    config["sealed_builder_window_microbatch"] = 2
    admission_started = time.perf_counter()
    admission = api.validate(engine, (28,), config["validation_teacher_root"])
    torch.cuda.synchronize()
    admission_wall = time.perf_counter() - admission_started
    if admission.get("windows") != [28] or admission.get("kld_mean") != W28_KLD or admission.get("top1") != W28_TOP1:
        raise RuntimeError(f"W28_ADMISSION_RED:{admission.get('kld_mean')}:{admission.get('top1')}")
    admission_path = root / "receipts" / f"W28_ADMISSION.{TASK}.rank{rank}.json"
    admission_row = {"schema": "banana-smasher-resident-w28-admission-v1", "status": "PASS",
                     "task_id": TASK, "rank": rank, "canonical_code_commit": pin,
                     "basis_sha256": BASIS, "checkpoint_sha256": CHECKPOINT,
                     "resident_load_seconds": resident_load_seconds, "admission_wall_seconds": admission_wall,
                     "measurement": admission}
    admission_row["receipt_sha256"] = atomic(admission_path, admission_row)

    config["resident_validation_attention_implementation"] = "sdpa"
    config["sealed_builder_window_microbatch"] = 4
    full_started = time.perf_counter()
    full = api.validate(engine, windows, config["validation_teacher_root"])
    torch.cuda.synchronize()
    post_load_wall = time.perf_counter() - full_started
    rows = list(full.get("per_window", []))
    if len(rows) != 64 or [int(row["window"]) for row in rows] != list(windows):
        raise RuntimeError("FULL64_ROW_COVERAGE_MISMATCH")
    aggregate = aggregate_from_rows(rows)
    diffs = []
    for row in rows:
        window = int(row["window"])
        expected = expected_rows[window]
        observed_mean = float(row["kld_sum_binary64"]) / int(row["positions"])
        expected_mean = float(expected["kld_mean"])
        diffs.append({"window": window, "kld_mean": observed_mean,
                      "expected_kld_mean": expected_mean, "kld_delta": observed_mean - expected_mean,
                      "top1": int(row["top1"]), "expected_top1": int(expected["top1"]),
                      "top1_delta": int(row["top1"]) - int(expected["top1"])})
    directional_shift = math.fsum(item["kld_delta"] for item in diffs) / len(diffs)
    expected_aggregate = reference["aggregate"]
    if post_load_wall >= 300.0:
        rate_low_path = root / "receipts" / f"FULL64_RATE_LOW.{TASK}.rank{rank}.json"
        rate_low = {
            "schema": "banana-smasher-resident-full64-rate-low-v2", "status": "RATE_LOW",
            "task_id": TASK, "rank": rank, "canonical_code_commit": pin,
            "basis_sha256": BASIS, "checkpoint_sha256": CHECKPOINT,
            "post_load_wall_seconds": post_load_wall, "threshold_seconds": 300.0,
            "aggregate": aggregate, "per_window": rows,
            "phase_profiles_by_rank": full.get("phase_profiles_by_rank"),
            "mechanism_counters": full.get("runtime_counters", {}),
        }
        rate_low["receipt_sha256"] = atomic(rate_low_path, rate_low)
        raise RuntimeError(f"RATE_LOW:{post_load_wall}")
    if abs(aggregate["kld_mean"] - float(expected_aggregate["kld_mean"])) > 5e-4:
        raise RuntimeError(f"AGGREGATE_KLD_SHIFT:{aggregate['kld_mean']}")
    if abs(directional_shift) > 5e-4:
        raise RuntimeError(f"DIRECTIONAL_SHIFT:{directional_shift}")
    mechanism = full.get("runtime_counters", {})
    if int(mechanism.get("checkpoint_reloads", -1)) != 0 or int(mechanism.get("reconstruction_calls", -1)) != 0:
        raise RuntimeError("WEIGHT_RELOAD_OR_RECONSTRUCTION_OBSERVED")

    terminal = {"schema": "banana-smasher-resident-full64-terminal-v1", "status": "PASS",
                "task_id": TASK, "rank": rank, "pid": os.getpid(), "canonical_code_commit": pin,
                "basis_sha256": BASIS, "checkpoint_sha256": CHECKPOINT,
                "runtime_provider": "vllm/vllm-openai:v0.24.0", "teacher_root": config["validation_teacher_root"],
                "validation_teacher_sha256_by_window": full.get("validation_teacher_sha256_by_window"),
                "reference_terminal_sha256": sha(reference_path), "resident_load_seconds": resident_load_seconds,
                "admission_receipt": str(admission_path), "admission_receipt_sha256": admission_row["receipt_sha256"],
                "admission_wall_seconds": admission_wall, "post_load_wall_seconds": post_load_wall,
                "process_wall_seconds": time.perf_counter() - process_started,
                "zero_weight_reload_proof": {"resident_engine_instances": 1, "checkpoint_loads": 1,
                                             "reconstruction_calls": 0},
                "aggregate": aggregate, "expected_aggregate": expected_aggregate,
                "directional_kld_shift": directional_shift, "per_window": rows,
                "per_window_diff": diffs, "phase_profiles_by_rank": full.get("phase_profiles_by_rank"),
                "mechanism_counters": mechanism}
    terminal_path = root / "receipts" / f"FULL64_TERMINAL.{TASK}.rank{rank}.json"
    terminal["receipt_sha256"] = atomic(terminal_path, terminal)
    print(json.dumps({"terminal_path": str(terminal_path), **terminal}, sort_keys=True), flush=True)
    engine.close()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
