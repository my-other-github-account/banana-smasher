"""Bounded Spark-7 probe for L028 SU/SV persistence and reload."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

from repair_api.modern_green_resident import (
    EXPERT_PLANE_ROSTER_SHA256,
    EXPERT_PLANE_SURFACE,
    ModernGreenResidentEngine,
    _activate_expert_plane_surface,
    _fp64_state_adam,
    _validated_expert_plane_expansion,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _atomic_json(path: Path, payload: dict[str, Any]) -> str:
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return hashlib.sha256(encoded).hexdigest()


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> str:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return _sha256(path)


def _optimizer(rows: list[tuple[str, Any]], learning_rate: float) -> Any:
    return _fp64_state_adam(
        torch,
        [
            {"params": [], "lr": 0.0, "group_name": "luts"},
            {"params": [], "lr": 0.0, "group_name": "norms"},
            {"params": [], "lr": 0.0, "group_name": "outputs"},
            {
                "params": [parameter for _name, parameter in rows],
                "lr": learning_rate,
                "group_name": EXPERT_PLANE_SURFACE,
            },
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pre-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--canonical-commit", required=True)
    arguments = parser.parse_args()

    started = time.time()
    root = arguments.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "receipts").mkdir(exist_ok=True)
    (root / "checkpoints").mkdir(exist_ok=True)
    config = json.loads(arguments.config.read_text())
    contract = _validated_expert_plane_expansion(config)
    if contract is None or contract["roster_sha256"] != EXPERT_PLANE_ROSTER_SHA256:
        raise RuntimeError("exact L028 SU/SV contract is not admitted")
    if config.get("basis_sha256") != "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b":
        raise RuntimeError("basis mismatch")
    pre_before = _sha256(arguments.pre_checkpoint)
    if pre_before != "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70":
        raise RuntimeError("published PRE checkpoint mismatch")
    repository_head = subprocess.run(
        ["git", "-C", str(arguments.repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if repository_head != arguments.canonical_commit:
        raise RuntimeError("canonical checkout mismatch")

    trainer_path = arguments.repo / "repair_api/assets/static_w28_modern_green_clean_u0.py"
    provider_path = arguments.repo / "runtime/v7/runner/fast_v7_expert_base.py"
    runner_dir = provider_path.parent
    sys.path.insert(0, str(runner_dir))
    try:
        trainer = _load_module("l028_probe_trainer", trainer_path)
        provider = _load_module("l028_probe_provider", provider_path)
    finally:
        sys.path.remove(str(runner_dir))

    admission_path = Path(config["asset_root"]) / "code/JOINT_REPAIR_ADMISSION.json"
    admission = json.loads(admission_path.read_text())
    row = next(
        item for item in admission["trainable_roster"]["luts"] if int(item["layer"]) == 28
    )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    plane_source = trainer.PlaneSource(
        torch=torch,
        np=np,
        row=row,
        parent_root=Path(config["parent_root"]),
        l034_roster=Path(config["l034_roster"]),
        device=device,
    )
    module = provider.FullyResidentGroupedV7Experts(28, plane_source=plane_source)
    student = type("Student", (), {"experts": {28: module}})()
    rows = _activate_expert_plane_surface(student, {}, contract, checkpoint_cursor=0)
    if len(rows) != 1536 or sum(parameter.numel() for _name, parameter in rows) != 4_718_592:
        raise RuntimeError("L028 promoted roster coverage drift")

    optimizer = _optimizer(rows, float(contract["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[lambda _step: 1.0] * 4
    )
    optimizer.zero_grad(set_to_none=True)
    objective = torch.zeros((), device=device)
    for projection in ("w1", "w2", "w3"):
        for component in ("SU", "SV"):
            objective = objective + module.expert_plane_wire_view(
                projection, component
            ).float().mean()
    objective.backward()
    nonzero_gradients = sum(
        int(parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0)
        for _name, parameter in rows
    )
    if nonzero_gradients != 1536:
        raise RuntimeError("expert surface was not fully consumed by the known-value objective")
    optimizer.step()
    scheduler.step()
    state = {
        "luts": {},
        "norms": {},
        "outputs": {},
        EXPERT_PLANE_SURFACE: {
            name: parameter.detach().cpu().clone() for name, parameter in rows
        },
    }
    payload = {
        "schema": "banana-smasher-l028-persistence-probe-checkpoint-v1",
        "next_update": 1,
        "state": state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "identity": {
            "basis_sha256": config["basis_sha256"],
            "published_pre_checkpoint_sha256": pre_before,
            "canonical_git_pin": arguments.canonical_commit,
            "expert_plane_roster_sha256": contract["roster_sha256"],
        },
    }
    checkpoint_path = root / "checkpoints/RESIDENT_UPDATE_001.pt"
    checkpoint_sha = _atomic_torch_save(checkpoint_path, payload)

    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    second_source = trainer.PlaneSource(
        torch=torch,
        np=np,
        row=row,
        parent_root=Path(config["parent_root"]),
        l034_roster=Path(config["l034_roster"]),
        device=device,
    )
    second_module = provider.FullyResidentGroupedV7Experts(28, plane_source=second_source)
    second_student = type("Student", (), {"experts": {28: second_module}})()
    reloaded_rows = _activate_expert_plane_surface(
        second_student, loaded["state"], contract, checkpoint_cursor=1
    )
    target_optimizer = _optimizer(reloaded_rows, float(contract["learning_rate"]))
    engine = ModernGreenResidentEngine.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.luts = []
    engine.norms = []
    engine.outputs = []
    engine.expert_planes = reloaded_rows
    engine.expert_plane_contract = contract
    engine.state = loaded["state"]
    engine.payload = loaded
    engine.optimizer = target_optimizer
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(
        target_optimizer, lr_lambda=[lambda _step: 1.0] * 4
    )
    engine.published_pre_recipe = True
    engine.global_step = 1
    engine.controlled_arm = False
    engine._load_optimizer_scheduler_state()

    source_by_name = dict(rows)
    target_by_name = dict(reloaded_rows)
    if set(source_by_name) != set(target_by_name):
        raise RuntimeError("reloaded expert roster drift")
    if any(
        not torch.equal(source_by_name[name].detach().cpu(), target_by_name[name].detach().cpu())
        for name in source_by_name
    ):
        raise RuntimeError("reloaded expert state bytes drift")
    if len(engine.optimizer.param_groups) != 4 or not engine.optimizer.state:
        raise RuntimeError("reloaded expert optimizer lineage missing")
    pre_after = _sha256(arguments.pre_checkpoint)
    if pre_after != pre_before:
        raise RuntimeError("published PRE bytes changed")
    torch.cuda.synchronize(device)
    wall = time.time() - started
    receipt = {
        "schema": "banana-smasher-l028-persistence-reload-probe-v1",
        "status": "PASS",
        "task_id": "t_a91cc543",
        "host": os.uname().nodename,
        "pid": os.getpid(),
        "canonical_git_pin": arguments.canonical_commit,
        "basis_sha256": config["basis_sha256"],
        "published_pre_checkpoint_sha256_before": pre_before,
        "published_pre_checkpoint_sha256_after": pre_after,
        "expert_surface": EXPERT_PLANE_SURFACE,
        "expert_plane_roster_sha256": contract["roster_sha256"],
        "expert_surface_consumed": True,
        "expert_surface_reloaded": True,
        "expert_parameter_count": len(rows),
        "expert_parameter_elements": sum(parameter.numel() for _name, parameter in rows),
        "nonzero_gradient_parameter_count": nonzero_gradients,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "resident_state_persisted": True,
        "optimizer_group_count": len(engine.optimizer.param_groups),
        "optimizer_state_entries": len(engine.optimizer.state),
        "objective": float(objective.detach().cpu()),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "wall_seconds": wall,
    }
    receipt_path = root / "receipts/L028_PERSISTENCE_RELOAD_PASS.json"
    receipt_sha = _atomic_json(receipt_path, receipt)
    print(json.dumps({**receipt, "receipt_path": str(receipt_path), "receipt_sha256": receipt_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
