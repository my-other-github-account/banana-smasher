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

from .banana_v1 import (
    banana_v1_gaussian_codebook,
    banana_v1_state_levels,
    banana_v1_transform,
    build_banana_v1,
    fit_banana_v1_codebook,
    write_banana_v1_candidate,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
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


def fit_source_matrix_once(source: np.ndarray, *, seed: int) -> tuple[Any, dict[str, Any]]:
    """Run one ordinary assignment/centroid alternation on authentic source values."""
    original = banana_v1_gaussian_codebook()
    first = build_banana_v1(source, seed=seed, codebook=original)
    transformed, _su, _sv = banana_v1_transform(source, seed=seed)
    normalized = transformed.reshape(-1) / np.repeat(first.scales, first.states.shape[1])
    levels = banana_v1_state_levels()[first.states.reshape(-1)]
    fitted, counts = fit_banana_v1_codebook(original, levels, normalized, alpha=1.0)
    final = build_banana_v1(source, seed=seed, codebook=fitted)
    evidence = {
        "assignment_count": int(levels.size),
        "occupied_levels": int(np.count_nonzero(counts)),
        "level_count_sum": int(counts.sum()),
        "initial_distortion": float(first.distortion),
        "fitted_distortion": float(final.distortion),
        "codebook_dtype": str(fitted.dtype),
        "codebook_shape": list(fitted.shape),
    }
    return final, evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded native-Q2 source/TRAIN solver")
    parser.add_argument("--model-index", required=True)
    parser.add_argument("--expected-index-sha256", required=True)
    parser.add_argument("--tensor-key", required=True)
    parser.add_argument("--scale-key", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--canonical-sha", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--columns", type=int, default=16)
    args = parser.parse_args(argv)

    import torch
    from safetensors import safe_open

    index_path = Path(args.model_index).resolve()
    actual_index_sha = _sha256(index_path)
    if actual_index_sha != args.expected_index_sha256:
        raise RuntimeError(f"basis mismatch {actual_index_sha} != {args.expected_index_sha256}")
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    if weight_map.get(args.tensor_key) != weight_map.get(args.scale_key):
        raise RuntimeError("source tensor and scale must share one immutable shard")
    shard = index_path.parent / weight_map[args.tensor_key]
    shard_sha = _sha256(shard)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    progress = output.parent / f".{output.name}.progress.json"
    started = time.time()

    if args.rows != 16 or args.columns != 16:
        raise ValueError("bounded production gate is exactly one authentic 16x16 tile")
    with safe_open(shard, framework="pt", device="cpu") as handle:
        source_i8 = handle.get_slice(args.tensor_key)[: args.rows, : args.columns]
        source_scale = handle.get_slice(args.scale_key)[: args.rows, :1]
    source_cuda = source_i8.to(device="cuda", dtype=torch.float32)
    scale_cuda = source_scale.to(device="cuda", dtype=torch.float32).repeat_interleave(16, dim=1)
    source_cuda.mul_(scale_cuda)
    torch.cuda.synchronize()
    source = source_cuda.detach().cpu().numpy().astype(np.float32, copy=False)
    source_sha = hashlib.sha256(np.ascontiguousarray(source).tobytes()).hexdigest()
    _atomic_json(
        progress,
        {
            "status": "GPU_SOURCE_READY",
            "task_id": args.task_id,
            "pid": os.getpid(),
            "basis_sha256": actual_index_sha,
            "source_sha256": source_sha,
            "tensor_key": args.tensor_key,
            "shape": list(source.shape),
            "unix": time.time(),
        },
    )

    result, fit = fit_source_matrix_once(source, seed=args.seed)
    candidate = write_banana_v1_candidate(output, result)
    terminal = {
        "schema": "banana-smasher-native-q2-train-gate-v1",
        "status": "PASS_BOUNDED_SOURCE_TRAIN_GATE",
        "task_id": args.task_id,
        "canonical_sha": args.canonical_sha,
        "basis_sha256": actual_index_sha,
        "model_index": str(index_path),
        "shard": {"path": str(shard), "sha256": shard_sha},
        "source": {
            "tensor_key": args.tensor_key,
            "scale_key": args.scale_key,
            "shape": list(source.shape),
            "sha256": source_sha,
        },
        "mechanism": "layer-shared-native-q2-banana-l16-b2-v1-fp16x1024",
        "holdout_used": False,
        "repair_gradients_used": False,
        "hessian_used": False,
        "fit": fit,
        "candidate": candidate,
        "elapsed_seconds": time.time() - started,
        "unix": time.time(),
    }
    _atomic_json(output / "TRAIN_RECEIPT.json", terminal)
    _atomic_json(progress, {"status": "PASS", "receipt": str(output / "TRAIN_RECEIPT.json"), "unix": time.time()})
    print(json.dumps(terminal, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
