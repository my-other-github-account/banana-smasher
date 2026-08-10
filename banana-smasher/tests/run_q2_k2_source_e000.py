from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

from banana_smasher.q2_codec import k2_lut_fp16
from banana_smasher.qtip_k2 import assign_k2_source


MEMBERS = ("w1", "w2", "w3")


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> str:
    payload = canonical(value)
    atomic_bytes(path, payload)
    return hashlib.sha256(payload).hexdigest()


def atomic_numpy(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".npy"
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.ascontiguousarray(value), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "data_bytes": int(value.nbytes),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": sha256_file(path),
        "data_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
    }


def atomic_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    temporary = Path(name)
    try:
        cpu_state = {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in state.items()
        }
        torch.save(cpu_state, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def verify_authority(
    *,
    basis: str,
    task_id: str,
    input_root: Path,
    host_claim: Path,
    shards: Path,
) -> dict[str, Any]:
    manifest_path = input_root / "SOURCE_ONLY_INPUT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != "PASS"
        or manifest.get("basis_sha256") != basis
        or manifest.get("comparator_inputs") != 0
        or manifest.get("artifact_seed_inputs") != 0
        or manifest.get("external_state_maps") != 0
    ):
        raise RuntimeError("SOURCE_ONLY_INPUT_GATE_REFUSED")
    index_path = input_root / "metadata/model.safetensors.index.json"
    if sha256_file(index_path) != basis:
        raise RuntimeError("BASIS_GATE_REFUSED source index identity mismatch")
    for row in manifest["files"]:
        path = input_root / row["relative_path"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"SOURCE_ONLY_INPUT_GATE_REFUSED {path}")

    claim = json.loads(host_claim.read_text())
    if (
        claim.get("owner_task_id") != task_id
        or claim.get("state") != "CLAIMED"
        or claim.get("intended_basis") != basis
    ):
        raise RuntimeError("HOST_CLAIM_GATE_REFUSED")
    shard_state = json.loads(shards.read_text())
    if shard_state.get("intended_basis") != basis:
        raise RuntimeError("SHARD_BASIS_GATE_REFUSED")
    accepted = any(
        row.get("task_id") == task_id
        and row.get("status") == "CLAIMED"
        and row.get("layer") == 34
        and row.get("expert_start") <= 0
        and row.get("expert_stop") >= 1
        for row in shard_state.get("claims", [])
    )
    if not accepted:
        raise RuntimeError("SHARD_RANGE_GATE_REFUSED")
    return {
        "input_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "host_claim": {"path": str(host_claim), "sha256": sha256_file(host_claim)},
        "shards": {"path": str(shards), "sha256": sha256_file(shards)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-claim", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--basis", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.seed != 0:
        raise ValueError("this exact E000 gate requires seed 0")
    conversion_module_index = 34 + 2
    transform_seed = args.seed + conversion_module_index
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("output is nonempty; sealed work is never redone")
    args.output.mkdir(parents=True, exist_ok=True)
    authority = verify_authority(
        basis=args.basis,
        task_id=args.task_id,
        input_root=args.input_root,
        host_claim=args.host_claim,
        shards=args.shards,
    )
    started = time.time()
    weight_path = args.input_root / "weights/L034_E000_original_W.bf16.safetensors"
    with safe_open(weight_path, framework="pt", device="cpu") as handle:
        source = {name: handle.get_tensor(name) for name in MEMBERS}
    hessian_root = args.input_root / "hessian"
    input_count = 512_000
    down_count = 512_000
    if input_count != 512_000 or down_count != 512_000:
        raise RuntimeError("authentic standard250 Hessian count must equal 512000")
    input_sum = torch.from_numpy(
        np.load(
            hessian_root / "L034_E000_shared_w1_w3_H_sum.fp32.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
    )
    down_sum = torch.from_numpy(
        np.load(
            hessian_root / "L034_E000_e000_w2_H_sum.fp32.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
    )
    if input_sum.shape != (4096, 4096) or down_sum.shape != (2048, 2048):
        raise RuntimeError("authentic standard250 Hessian geometry mismatch")
    hessian_sums = {"w1": input_sum, "w2": down_sum, "w3": input_sum}
    hessian_counts = {"w1": input_count, "w2": down_count, "w3": input_count}
    parent_lut = torch.from_numpy(k2_lut_fp16()).to(device="cuda")
    members = []

    for name in MEMBERS:
        member_root = args.output / "members" / name
        checkpoint_path = args.output / "checkpoints" / f"{name}.pt"
        resume_state = None
        if checkpoint_path.exists():
            loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            resume_state = {
                key: value.to("cuda") if isinstance(value, torch.Tensor) else value
                for key, value in loaded.items()
            }

        def progress(event: dict[str, int]) -> None:
            atomic_json(
                args.output / "PROGRESS.json",
                {
                    "schema": "banana-smasher-q2-source-e000-progress-v1",
                    "status": "RUNNING",
                    "task_id": args.task_id,
                    "basis_sha256": args.basis,
                    "active_member": name,
                    "members_complete": len(members),
                    "source_only": True,
                    "comparator_inputs": 0,
                    "event": event,
                    "updated_unix": time.time(),
                },
            )

        result = assign_k2_source(
            source[name].to(device="cuda"),
            None,
            parent_lut,
            raw_hessian_sum=hessian_sums[name],
            raw_hessian_count=hessian_counts[name],
            seed=args.seed,
            conversion_module_index=conversion_module_index,
            progress=progress,
            resume_state=resume_state,
            checkpoint=lambda state: atomic_checkpoint(checkpoint_path, state),
        )
        states = result["states"].cpu().numpy()
        packed = result["packed_codes"].cpu().numpy()
        su = result["su"].view(-1, 1).cpu().numpy()
        sv = result["sv"].view(1, -1).cpu().numpy()
        suh = result["suh"].view(-1, 1).cpu().numpy()
        svh = result["svh"].view(1, -1).cpu().numpy()
        physical_bfloat16 = (
            result["physical_bfloat16"]
            .contiguous()
            .view(torch.uint8)
            .cpu()
            .numpy()
            .tobytes()
        )
        artifacts = {
            "states": atomic_numpy(member_root.with_suffix(".states.npy"), states),
            "codes": atomic_numpy(member_root.with_suffix(".codes.npy"), packed),
            "su": atomic_numpy(member_root.with_suffix(".su.npy"), su),
            "sv": atomic_numpy(member_root.with_suffix(".sv.npy"), sv),
            "suh": atomic_numpy(member_root.with_suffix(".suh.npy"), suh),
            "svh": atomic_numpy(member_root.with_suffix(".svh.npy"), svh),
        }
        physical_path = member_root.with_suffix(".physical.bf16.bin")
        atomic_bytes(physical_path, physical_bfloat16)
        artifacts["physical_bfloat16"] = {
            "path": str(physical_path),
            "bytes": len(physical_bfloat16),
            "dtype": "bfloat16",
            "shape": list(result["physical_bfloat16"].shape),
            "sha256": hashlib.sha256(physical_bfloat16).hexdigest(),
        }
        members.append(
            {
                "member": name,
                "source_sha256": result["boundaries"]["source_sha256"],
                "global_scale": result["global_scale"],
                "inner_sse": result["inner_sse"],
                "objective_proxy_error": result["objective_proxy_error"],
                "physical_sse": result["physical_sse"],
                "physical_bfloat16_sse": result["physical_bfloat16_sse"],
                "calibration_rows": result["calibration_rows"],
                "calibration_mass": result["calibration_mass"],
                "calibration_batches": result["calibration_batches"],
                "solver_counters": result["solver_counters"],
                "boundaries": result["boundaries"],
                "artifacts": artifacts,
            }
        )
        atomic_json(
            args.output / "PROGRESS.json",
            {
                "schema": "banana-smasher-q2-source-e000-progress-v1",
                "status": "RUNNING",
                "task_id": args.task_id,
                "basis_sha256": args.basis,
                "active_member": None,
                "members_complete": len(members),
                "source_only": True,
                "comparator_inputs": 0,
                "updated_unix": time.time(),
            },
        )
        torch.cuda.empty_cache()

    manifest = {
        "schema": "banana-smasher-q2-source-e000-candidate-v1",
        "status": "PASS",
        "task_id": args.task_id,
        "basis_sha256": args.basis,
        "layer": 34,
        "expert": 0,
        "seed": args.seed,
        "conversion_module_index": conversion_module_index,
        "transform_seed": transform_seed,
        "transform_seed_derivation": "base_seed + layer_index + 2",
        "calibration_mode": "standard250_raw_ordered_sum_authentic_cuda_finalized",
        "calibration_rows": 512000,
        "source_only": True,
        "comparator_inputs": 0,
        "artifact_seed_inputs": 0,
        "external_state_map": False,
        "parent_lut": {
            "dtype": "float16",
            "elements": 1024,
            "bytes": 2048,
            "sha256": hashlib.sha256(k2_lut_fp16().tobytes()).hexdigest(),
        },
        "members": members,
        "authority": authority,
        "started_unix": started,
        "ended_unix": time.time(),
    }
    manifest_path = args.output / "CANDIDATE_MANIFEST.json"
    manifest_sha = atomic_json(manifest_path, manifest)
    terminal = {
        "schema": "banana-smasher-q2-source-e000-terminal-v1",
        "status": "PASS",
        "task_id": args.task_id,
        "basis_sha256": args.basis,
        "source_only": True,
        "comparator_inputs": 0,
        "artifact_seed_inputs": 0,
        "external_state_map": False,
        "members": len(members),
        "cuda_calls": sum(row["solver_counters"]["cuda_calls"] for row in members),
        "fallback_calls": 0,
        "candidate_manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "ended_unix": time.time(),
    }
    atomic_json(args.output / "TERMINAL.json", terminal)
    print(canonical(terminal).decode(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
