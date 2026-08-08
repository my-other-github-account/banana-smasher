from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from banana_smasher.hf_deepseek_v4_d4_adapter import DeepseekV4D4Runtime


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _stage(remote_root: str, name: str, destination: Path) -> None:
    free = shutil.disk_usage(destination.parent).free
    if free < 1 << 30:
        raise RuntimeError(f"disk preflight failed before {name}: free={free}")
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.unlink(missing_ok=True)
    subprocess.run(
        [
            "scp",
            "-q",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=15",
            f"{remote_root.rstrip('/')}/{name}",
            str(temporary),
        ],
        check=True,
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _remove(path: Path) -> None:
    path.unlink(missing_ok=True)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _process_startticks() -> int:
    return int(Path("/proc/self/stat").read_text().split()[21])


def _gpu_snapshot() -> str:
    return subprocess.run(
        ["nvidia-smi"], check=True, capture_output=True, text=True
    ).stdout


def _process_census() -> list[str]:
    rows = subprocess.run(
        ["ps", "-eo", "pid=,args="], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    forbidden = ("EngineCore", "vllm.LLM")
    conflicts = [row.strip() for row in rows if any(token in row for token in forbidden)]
    if conflicts:
        raise RuntimeError(f"resident engine process detected: {conflicts[:8]}")
    return conflicts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--index-source", required=True, type=Path)
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--plane-source", required=True)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--teacher-ref", required=True, type=Path)
    parser.add_argument("--expected-index-sha256", required=True)
    parser.add_argument("--plane-0-md5", required=True)
    parser.add_argument("--plane-1-md5", required=True)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--positions", type=int, default=2048)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    model_root = run_root / "model"
    planes_root = run_root / "planes"
    receipts = run_root / "receipts"
    checkpoints = run_root / "checkpoints"
    for path in (model_root, planes_root, receipts, checkpoints):
        path.mkdir(parents=True, exist_ok=True)

    index_sha = _sha256(args.index_source)
    if index_sha != args.expected_index_sha256:
        raise RuntimeError(
            f"basis gate mismatch: expected index {args.expected_index_sha256}, got {index_sha}"
        )
    shutil.copy2(args.index_source, model_root / args.index_source.name)
    _stage(args.model_source, "config.json", model_root / "config.json")
    intended_basis = {
        "schema": "banana-smasher-d4-layerwise-intended-basis-v1",
        "model_index_sha256": index_sha,
        "model_index_source": str(args.index_source),
        "layers": [0, 1],
        "window": args.window,
        "positions": args.positions,
    }
    _atomic_json(run_root / "INTENDED_BASIS.json", intended_basis)

    weight_map = json.loads((model_root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    stage_shards = {
        "initial": weight_map["embed.weight"],
        "layer_0": sorted(
            {value for key, value in weight_map.items() if key.startswith("layers.0.")}
        )[0],
        "layer_1": sorted(
            {value for key, value in weight_map.items() if key.startswith("layers.1.")}
        )[0],
        "terminal": weight_map["head.weight"],
    }
    if len(set(stage_shards.values())) != 4:
        raise RuntimeError(f"unexpected stage shard map: {stage_shards}")

    corpus = json.loads(args.corpus.read_text())
    corpus_row = corpus[args.window]
    tokens = [1] * args.positions
    source_tokens = list(corpus_row["token_ids"])
    tokens[: min(len(source_tokens), args.positions)] = source_tokens[: args.positions]
    reference = torch.load(args.teacher_ref, map_location="cpu", weights_only=True)
    support = reference["idx"][: args.positions, :2].to(torch.int64).tolist()
    if len(support) != args.positions:
        raise RuntimeError("teacher support does not cover requested positions")
    del reference

    argv = [sys.executable, *sys.argv]
    started = time.time()
    process = {
        "pid": os.getpid(),
        "startticks": _process_startticks(),
        "argv": argv,
    }
    progress_path = receipts / "PROGRESS.json"
    if progress_path.is_file():
        progress: dict[str, Any] = json.loads(progress_path.read_text())
        if (
            progress.get("model_index_sha256") != index_sha
            or progress.get("window") != args.window
        ):
            raise RuntimeError("resume progress basis/window mismatch")
        attempts = progress.setdefault("attempts", [progress.get("process")])
        attempts.append(process)
        progress.update({"status": "RUNNING", "process": process})
    else:
        progress = {
            "schema": "banana-smasher-d4-layerwise-hardware-progress-v1",
            "status": "RUNNING",
            "basis_match": True,
            "model_index_sha256": index_sha,
            "configured_layer_count": 2,
            "manifest_layer_count": 43,
            "layer": None,
            "window": args.window,
            "output_rows": 0,
            "bytes_read": 0,
            "resident_peak_bytes": 0,
            "process": process,
            "attempts": [process],
        }
    _atomic_json(progress_path, progress)

    os.environ["BANANA_SMASHER_D4_PLANES_DIR"] = str(planes_root)
    prior_bytes_read = int(progress.get("bytes_read", 0))
    prior_resident_peak = int(progress.get("resident_peak_bytes", 0))
    runtime = DeepseekV4D4Runtime(
        model_root=model_root,
        parameters={"positions": args.positions},
    )

    def update(stage: str, layer: int | None) -> None:
        progress.update(
            {
                "stage": stage,
                "layer": layer,
                "bytes_read": prior_bytes_read + runtime.bytes_read(),
                "resident_peak_bytes": max(
                    prior_resident_peak, runtime.peak_resident_bytes()
                ),
                "elapsed_seconds": time.time() - started,
            }
        )
        progress.setdefault("history", []).append(
            {
                "stage": stage,
                "layer": layer,
                "window": args.window,
                "output_rows": progress["output_rows"],
                "bytes_read": progress["bytes_read"],
                "resident_peak_bytes": progress["resident_peak_bytes"],
                "process": process,
            }
        )
        _atomic_json(progress_path, progress)

    plane_md5 = {0: args.plane_0_md5, 1: args.plane_1_md5}
    for layer in (0, 1):
        _remove(model_root / stage_shards[f"layer_{layer}"])
        _remove(planes_root / f"vq3u_layer_{layer:03d}.pt")

    layer_zero_checkpoint = checkpoints / "layer_0.npy"
    layer_one_checkpoint = checkpoints / "layer_1.npy"
    if layer_one_checkpoint.is_file():
        source_checkpoint = layer_one_checkpoint
        remaining_layers: tuple[int, ...] = ()
        progress["completed_layers"] = [0, 1]
    elif layer_zero_checkpoint.is_file():
        source_checkpoint = layer_zero_checkpoint
        remaining_layers = (1,)
        progress["completed_layers"] = [0]
    else:
        initial_checkpoint = checkpoints / "initial.npy"
        if not initial_checkpoint.is_file():
            initial_shard = model_root / stage_shards["initial"]
            _stage(args.model_source, initial_shard.name, initial_shard)
            with runtime.initial_stage() as embed:
                activation = embed(tokens, window_id=args.window)
                packed = runtime.export_activation(activation)
                _atomic_npy(initial_checkpoint, packed)
                del packed, activation
                runtime.synchronize()
            initial_resident = runtime.resident_bytes()
            _remove(initial_shard)
            update("initial-complete", None)
            if initial_resident != 0:
                raise RuntimeError(
                    f"initial stage retained accelerator storage: {initial_resident} bytes"
                )
        source_checkpoint = initial_checkpoint
        remaining_layers = (0, 1)
        progress["completed_layers"] = []
    layer_checkpoints = dict(progress.get("layer_checkpoints", {}))
    prior_resume = progress.get("resume_checkpoint")
    prior_resume_sha = progress.get("resume_checkpoint_sha256")
    if (
        isinstance(prior_resume, str)
        and prior_resume.endswith("layer_0.npy")
        and isinstance(prior_resume_sha, str)
    ):
        layer_checkpoints.setdefault(
            "0",
            {
                "path": prior_resume,
                "sha256": prior_resume_sha,
                "retained": Path(prior_resume).is_file(),
                "consumed_by_layer": 1,
            },
        )
    for layer, checkpoint in (
        (0, layer_zero_checkpoint),
        (1, layer_one_checkpoint),
    ):
        if checkpoint.is_file():
            layer_checkpoints[str(layer)] = {
                "path": str(checkpoint),
                "sha256": _sha256(checkpoint),
                "retained": True,
            }
    progress["layer_checkpoints"] = layer_checkpoints
    progress["resume_checkpoint"] = str(source_checkpoint)
    progress["resume_checkpoint_sha256"] = _sha256(source_checkpoint)
    update("resumed", None)

    for layer in remaining_layers:
        shard = model_root / stage_shards[f"layer_{layer}"]
        plane = planes_root / f"vq3u_layer_{layer:03d}.pt"
        _stage(args.model_source, shard.name, shard)
        _stage(args.plane_source, plane.name, plane)
        actual_md5 = _md5(plane)
        if actual_md5 != plane_md5[layer]:
            raise RuntimeError(
                f"layer {layer} plane identity mismatch: expected {plane_md5[layer]}, got {actual_md5}"
            )
        target_checkpoint = checkpoints / f"layer_{layer}.npy"
        with runtime.layer_stage(layer) as forward:
            packed = np.load(source_checkpoint, allow_pickle=False)
            activation = runtime.import_activation(packed)
            output = forward(activation, window_id=args.window)
            exported = runtime.export_activation(output)
            _atomic_npy(target_checkpoint, exported)
            del packed, activation, output, exported
            runtime.synchronize()
        layer_resident = runtime.resident_bytes()
        _remove(shard)
        _remove(plane)
        if source_checkpoint.name != "initial.npy":
            _remove(source_checkpoint)
        source_checkpoint = target_checkpoint
        progress["completed_layers"] = list(range(layer + 1))
        progress["layer_checkpoints"][str(layer)] = {
            "path": str(target_checkpoint),
            "sha256": _sha256(target_checkpoint),
        }
        update("layer-complete", layer)
        if layer_resident != 0:
            raise RuntimeError(
                f"layer {layer} retained accelerator storage: {layer_resident} bytes"
            )

    terminal_shard = model_root / stage_shards["terminal"]
    _stage(args.model_source, terminal_shard.name, terminal_shard)
    with runtime.terminal_stage() as score:
        packed = np.load(source_checkpoint, allow_pickle=False)
        activation = runtime.import_activation(packed)
        scored = score(activation, support, window_id=args.window)
        del packed, activation
        runtime.synchronize()
    if runtime.resident_bytes() != 0:
        raise RuntimeError("terminal stage retained accelerator storage")
    _remove(terminal_shard)

    output_row = {
        "window_id": args.window,
        "support_token_ids": support,
        "logits": scored["logits"],
        "top1_token_ids": scored["top1_token_ids"],
    }
    output_path = run_root / "candidate.jsonl"
    payload = (json.dumps(output_row, separators=(",", ":")) + "\n").encode()
    with output_path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    progress["output_rows"] = 1
    progress["status"] = "PASS"
    update("complete", None)

    census = _process_census()
    receipt = {
        **progress,
        "schema": "banana-smasher-d4-layerwise-hardware-receipt-v1",
        "status": "PASS",
        "layers_completed": [0, 1],
        "output_rows": 1,
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "checkpoint": str(source_checkpoint),
        "checkpoint_sha256": _sha256(source_checkpoint),
        "nvidia_smi": _gpu_snapshot(),
        "resident_engine": False,
        "resident_engine_processes": census,
        "plane_md5": plane_md5,
        "stage_shards": stage_shards,
    }
    receipt_path = receipts / "ACCEPTANCE_RECEIPT.json"
    _atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
