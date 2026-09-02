#!/usr/bin/env python3
from __future__ import annotations
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from repair_api import ResidentRepairAPI
from repair_api.modern_green_resident import ModernGreenResidentEngine

TASK = "t_75368735"
BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
PRE = "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70"
WINDOWS = (28, 56, 68, 71)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(path: Path, value: object) -> str:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    rank = int(os.environ["RANK"])
    root = Path(os.environ["PHYSICAL_ROOT"])
    config = json.loads((root / f"CONFIG.rank{rank}.json").read_text())
    pin = str(config.get("canonical_code_commit", ""))
    if len(pin) != 40 or config.get("basis_sha256") != BASIS:
        raise RuntimeError("CONFIG_PIN_OR_BASIS_MISMATCH")
    index = Path(config["model_root"]) / "model.safetensors.index.json"
    if sha(index) != BASIS:
        raise RuntimeError("BASIS_GATE_MISMATCH")
    artifact_root = Path(config["artifact_root"])
    api = ResidentRepairAPI.open(artifact_root)
    checkpoint_path = api.artifact.checkpoint_path("UPDATE_000")
    if sha(checkpoint_path) != PRE:
        raise RuntimeError("PUBLISHED_PRE_CHECKPOINT_MISMATCH")
    if int(config.get("sealed_builder_window_microbatch", 0)) != len(WINDOWS):
        raise RuntimeError("PROFILE_REQUIRES_EXACT_BATCH4")
    if config.get("resident_validation_expert_implementation") != "packed_cuda_bf16_boundary":
        raise RuntimeError("PROFILE_REQUIRES_PACKED_RESIDENT_PROVIDER")
    os.environ["NCCL_SOCKET_IFNAME"] = str(config["nccl_socket_ifname"])
    os.environ["GLOO_SOCKET_IFNAME"] = str(config["nccl_socket_ifname"])
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://", timeout=timedelta(seconds=900)
        )
    if torch.distributed.get_world_size() != 2 or torch.distributed.get_rank() != rank:
        raise RuntimeError("DIST_GEOMETRY_MISMATCH")
    ranges = {0: (0, 20), 1: (21, 42)}
    process_started = time.perf_counter()
    load_started = time.perf_counter()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    engine = ModernGreenResidentEngine(
        payload=payload, config=config, rank=rank, layer_ranges=ranges
    )
    del payload
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    post_load_started = time.perf_counter()
    measured = api.validate(engine, WINDOWS, config["teacher_root"])
    torch.cuda.synchronize()
    post_load_wall = time.perf_counter() - post_load_started
    profiles = measured.get("phase_profiles_by_rank")
    if not isinstance(profiles, list) or len(profiles) != 2:
        raise RuntimeError("PROFILE_RANK_CLOSURE_MISMATCH")
    for expected_rank, rows in enumerate(profiles):
        if not isinstance(rows, list) or len(rows) != 1:
            raise RuntimeError("PROFILE_BATCH_CARDINALITY_MISMATCH")
        profile = rows[0]
        if profile.get("rank") != expected_rank or profile.get("batch_windows") != list(WINDOWS):
            raise RuntimeError("PROFILE_BATCH_IDENTITY_MISMATCH")
        if profile.get("weight_reconstruction_ms") != 0.0:
            raise RuntimeError("PROFILE_WEIGHT_RECONSTRUCTION_TIME_NONZERO")
        delta = profile.get("mechanism_counter_delta", {})
        before = profile.get("mechanism_before", {})
        if int(before.get("provider_count", 0)) <= 0:
            raise RuntimeError("PROFILE_RESIDENT_PROVIDER_NOT_OBSERVED")
        if int(delta.get("reconstruction_calls", -1)) != 0:
            raise RuntimeError("PROFILE_WEIGHT_RECONSTRUCTION_CALL_NONZERO")
        if int(delta.get("projection_calls", 0)) <= 0:
            raise RuntimeError("PROFILE_PACKED_PROJECTION_NOT_EXECUTED")
        for field in ("forward_ms", "p2p_ms"):
            if float(profile.get(field, -1.0)) < 0.0:
                raise RuntimeError(f"PROFILE_PHASE_MISSING_{field}")
        if expected_rank == 1 and float(profile.get("readout_ms", -1.0)) <= 0.0:
            raise RuntimeError("PROFILE_PHASE_MISSING_readout_ms")
    row = {
        "schema": "banana-smasher-physical-resident-one-batch-profile-v1",
        "status": "PASS_PROFILE_ONLY",
        "task_id": TASK,
        "rank": rank,
        "pid": os.getpid(),
        "canonical_code_commit": pin,
        "basis_sha256": BASIS,
        "checkpoint_sha256": PRE,
        "windows": list(WINDOWS),
        "configured_batch_size": len(WINDOWS),
        "resident_load_seconds": load_seconds,
        "post_load_wall_seconds": post_load_wall,
        "process_wall_seconds": time.perf_counter() - process_started,
        "measurement": measured,
        "scientific_acceptance": False,
        "successor_gate": "profile must be admitted before the sole full64 production invocation",
    }
    receipt = root / "receipts" / f"ATTEMPT9_ONE_BATCH_PROFILE.rank{rank}.json"
    row["receipt_sha256"] = atomic(receipt, row)
    print(json.dumps(row, sort_keys=True), flush=True)
    engine.close()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
