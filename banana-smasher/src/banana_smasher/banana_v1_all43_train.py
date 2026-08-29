"""Exact all-layer TRAIN-only Banana V1 shared-codebook producer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Sequence

import numpy as np

from .banana_v1 import (
    banana_v1_gaussian_codebook,
    banana_v1_state_levels,
    banana_v1_transform,
    build_banana_v1,
    fit_banana_v1_codebook_from_statistics,
    verify_banana_v1_candidate,
    write_banana_v1_candidate,
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_path(path: Path) -> str:
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


def fit_shared_codebook_from_sources(
    sources: Sequence[np.ndarray], *, seeds: Sequence[int]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit exactly one FP16[1024] LUT from aggregate authentic assignments."""
    if not sources or len(sources) != len(seeds):
        raise ValueError("sources and seeds must be nonempty and matching")
    original = banana_v1_gaussian_codebook()
    state_levels = banana_v1_state_levels()
    counts = np.zeros(1024, dtype=np.int64)
    sums = np.zeros(1024, dtype=np.float64)
    initial_distortions: list[float] = []
    for source, seed in zip(sources, seeds):
        source_counts, source_sums, distortion = _statistics_for_source(
            source, seed=int(seed), original=original, state_levels=state_levels
        )
        counts += source_counts
        sums += source_sums
        initial_distortions.append(distortion)
    fitted = fit_banana_v1_codebook_from_statistics(original, counts, sums, alpha=1.0)
    evidence = {
        "assignment_count": int(counts.sum()),
        "occupied_levels": int(np.count_nonzero(counts)),
        "counts_sha256": _sha_bytes(np.ascontiguousarray(counts).tobytes()),
        "target_sums_sha256": _sha_bytes(np.ascontiguousarray(sums).tobytes()),
        "initial_distortions": initial_distortions,
    }
    return fitted, evidence


def _statistics_for_source(
    source: np.ndarray,
    *,
    seed: int,
    original: np.ndarray,
    state_levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return authentic initial-codebook statistics for one routed matrix."""
    first = build_banana_v1(source, seed=seed, codebook=original)
    transformed, _su, _sv = banana_v1_transform(source, seed=seed)
    rows, columns = transformed.shape
    transformed_tiles = (
        transformed.reshape(rows // 16, 16, columns // 16, 16)
        .transpose(0, 2, 1, 3)
        .reshape(-1, 256)
    )
    if transformed_tiles.shape != first.states.shape:
        raise RuntimeError("TRAIN tile/state geometry mismatch")
    normalized = transformed_tiles / first.scales[:, None]
    levels = state_levels[first.states]
    counts = np.bincount(levels.reshape(-1).astype(np.int64), minlength=1024).astype(np.int64)
    sums = np.bincount(
        levels.reshape(-1).astype(np.int64),
        weights=normalized.reshape(-1),
        minlength=1024,
    )
    return counts, sums, float(first.distortion)


def build_shared_results(
    sources: Sequence[np.ndarray], *, seeds: Sequence[int], codebook: np.ndarray
) -> list[Any]:
    """Reassign every member against the same frozen codebook bytes."""
    frozen = np.ascontiguousarray(codebook, dtype=np.float16)
    if frozen.shape != (1024,):
        raise ValueError("shared codebook must be FP16[1024]")
    return [
        build_banana_v1(source, seed=int(seed), codebook=frozen)
        for source, seed in zip(sources, seeds)
    ]


def _startticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text().split()[21])


def _claim_host(path: Path, *, preimage_sha256: str, task_id: str, run_id: int, root: Path, basis: str, canonical_sha: str) -> tuple[int, int, str]:
    payload = path.read_bytes()
    actual = _sha_bytes(payload)
    if actual != preimage_sha256:
        raise RuntimeError(f"claim CAS mismatch {actual} != {preimage_sha256}")
    claim = json.loads(payload)
    recover_same_task = claim.get("owner_task_id") == task_id and claim.get("state") == "CLAIMED"
    if not (claim.get("state") == "RELEASED" or recover_same_task):
        raise RuntimeError("host claim is neither RELEASED nor same-task recovery")
    pid = os.getpid()
    ticks = _startticks(pid)
    now = time.time()
    claim.update(
        {
            "state": "CLAIMED",
            "status": "RUNNING",
            "phase": "ALL43_SHARED_SOURCE_TRAIN",
            "task_id": task_id,
            "owner_task_id": task_id,
            "owner": f"{task_id}/bs03",
            "owner_profile": "bs03",
            "active_board_run_id": run_id,
            "payload_board_run_id": run_id,
            "claimed_unix": now,
            "updated_unix": now,
            "expires_unix": now + 7200,
            "holder_pid": pid,
            "holder_startticks": ticks,
            "workload_pid": pid,
            "workload_startticks": ticks,
            "mission_root": str(root),
            "run_root": str(root),
            "intended_basis": basis,
            "source_model_index_sha256": basis,
            "canonical_git_pin": canonical_sha,
            "do_not_preempt": True,
            "previous_claim_sha256": actual,
            "release_receipt": None,
            "release_receipt_sha256": None,
        }
    )
    _atomic_json(path, claim)
    return pid, ticks, _sha_path(path)


def _load_source(
    model_index: Path, index: dict[str, Any], layer: int, *, support: str
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from safetensors import safe_open

    tensor_key = f"layers.{layer}.ffn.experts.0.w1.weight"
    scale_key = f"layers.{layer}.ffn.experts.0.w1.scale"
    weight_map = index["weight_map"]
    if weight_map.get(tensor_key) != weight_map.get(scale_key) or tensor_key not in weight_map:
        raise RuntimeError(f"routed source/co-shard mismatch for L{layer:03d}")
    shard = model_index.parent / weight_map[tensor_key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        if support == "corner16":
            quantized = handle.get_slice(tensor_key)[:16, :16]
            scale = handle.get_slice(scale_key)[:16, :1]
        elif support == "full-projection":
            quantized = handle.get_tensor(tensor_key)
            scale = handle.get_tensor(scale_key)
        else:
            raise ValueError(f"unsupported TRAIN support: {support}")
    raw_weight = np.ascontiguousarray(quantized.numpy())
    scale_float = scale.to(dtype=torch.float32)
    raw_scale = np.ascontiguousarray(scale_float.numpy())
    source_cuda = quantized.to(device="cuda", dtype=torch.float32)
    if scale_float.ndim != 2 or scale_float.shape != (quantized.shape[0], 1):
        raise RuntimeError(f"routed scale geometry mismatch L{layer:03d}")
    source_cuda.mul_(scale_float.to(device="cuda").expand(-1, quantized.shape[1]))
    torch.cuda.synchronize()
    source = source_cuda.cpu().numpy().astype(np.float32, copy=False)
    return source, {
        "tensor_key": tensor_key,
        "scale_key": scale_key,
        "shard": str(shard),
        "raw_weight_sha256": _sha_bytes(raw_weight.tobytes()),
        "raw_scale_sha256": _sha_bytes(raw_scale.tobytes()),
        "source_sha256": _sha_bytes(np.ascontiguousarray(source).tobytes()),
        "shape": list(source.shape),
        "train_support": support,
    }


def run_all43(args: argparse.Namespace) -> dict[str, Any]:
    model_index = Path(args.model_index).resolve()
    root = Path(args.output).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    (root / "candidate" / "members").mkdir(parents=True)
    pid, ticks, claim_sha = _claim_host(
        Path(args.host_claim),
        preimage_sha256=args.claim_preimage_sha256,
        task_id=args.task_id,
        run_id=args.run_id,
        root=root,
        basis=args.expected_index_sha256,
        canonical_sha=args.canonical_sha,
    )
    index_bytes = model_index.read_bytes()
    index_sha = _sha_bytes(index_bytes)
    if index_sha != args.expected_index_sha256:
        raise RuntimeError(f"basis mismatch {index_sha} != {args.expected_index_sha256}")
    index = json.loads(index_bytes)
    layers = list(range(43))
    roster = [f"L{layer:03d}/E000/w1/tile-r000-r015-c000-c015" for layer in layers]
    _atomic_json(
        root / "SHARDS.json",
        {
            "schema": "banana-smasher-shards-v1",
            "status": "CLAIMED_ALL43",
            "task_id": args.task_id,
            "board_run_id": args.run_id,
            "basis": index_sha,
            "layers": layers,
            "members": roster,
            "claim_sha256": claim_sha,
            "pid": pid,
            "startticks": ticks,
        },
    )
    materialization_sources: list[np.ndarray] = []
    source_rows: list[dict[str, Any]] = []
    original = banana_v1_gaussian_codebook()
    state_levels = banana_v1_state_levels()
    counts = np.zeros(1024, dtype=np.int64)
    sums = np.zeros(1024, dtype=np.float64)
    initial_distortions: list[float] = []
    started = time.time()
    for position, layer in enumerate(layers, 1):
        source, metadata = _load_source(
            model_index, index, layer, support=args.train_support
        )
        source_counts, source_sums, distortion = _statistics_for_source(
            source, seed=layer, original=original, state_levels=state_levels
        )
        counts += source_counts
        sums += source_sums
        initial_distortions.append(distortion)
        materialization_sources.append(np.ascontiguousarray(source[:16, :16]))
        source_rows.append(metadata)
        _atomic_json(root / "PROGRESS.json", {"status": "TRAIN_STATISTICS", "completed_members": position, "total_members": 43, "active_member": roster[position - 1], "train_support": args.train_support, "train_assignments": int(counts.sum()), "pid": pid, "startticks": ticks, "basis_sha256": index_sha, "unix": time.time()})
    shared = fit_banana_v1_codebook_from_statistics(original, counts, sums, alpha=1.0)
    statistics = {
        "fit_calls": 1,
        "train_support": args.train_support,
        "assignment_count": int(counts.sum()),
        "occupied_levels": int(np.count_nonzero(counts)),
        "counts_sha256": _sha_bytes(np.ascontiguousarray(counts).tobytes()),
        "target_sums_sha256": _sha_bytes(np.ascontiguousarray(sums).tobytes()),
        "initial_distortions": initial_distortions,
    }
    shared_path = root / "candidate" / "shared_codebook.fp16"
    shared_path.write_bytes(shared.tobytes())
    shared_sha = _sha_path(shared_path)
    results = build_shared_results(materialization_sources, seeds=layers, codebook=shared)
    rows: list[dict[str, Any]] = []
    total_bits = 0
    for position, (layer, result) in enumerate(zip(layers, results), 1):
        member_root = root / "candidate" / "members" / f"L{layer:03d}_E000_w1_tile000"
        receipt = write_banana_v1_candidate(member_root, result)
        if not verify_banana_v1_candidate(member_root):
            raise RuntimeError(f"candidate verification failed L{layer:03d}")
        codebook_sha = receipt["artifacts"]["codebook"]["data_sha256"]
        if codebook_sha != shared_sha:
            raise RuntimeError(f"shared codebook drift L{layer:03d}")
        total_bits += int(receipt["code_bits"])
        rows.append(
            {
                "id": roster[position - 1],
                "layer": layer,
                "train_source": source_rows[position - 1],
                "materialization_shape": [16, 16],
                "candidate_locator": str(member_root),
                "candidate_receipt_sha256": _sha_path(member_root / "BANANA_V1_RECEIPT.json"),
                "shared_codebook_sha256": shared_sha,
                "assignment_bits": int(receipt["code_bits"]),
                "position_count": int(receipt["position_count"]),
                "code_bpw": float(receipt["code_bpw"]),
                "distortion": float(receipt["distortion"]),
            }
        )
        _atomic_json(root / "PROGRESS.json", {"status": "MATERIALIZING_FROZEN_SHARED_CODEBOOK", "completed_members": position, "total_members": 43, "active_member": roster[position - 1], "assignment_bits": total_bits, "shared_codebook_sha256": shared_sha, "pid": pid, "startticks": ticks, "basis_sha256": index_sha, "unix": time.time()})
    elapsed = time.time() - started
    for path in sorted((root / "candidate").rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    (root / "candidate").chmod(0o555)
    forbidden = {
        "holdout_used": False,
        "hessian_used": False,
        "repair_gradients_used": False,
        "comparator_tables_used": False,
        "copied_exl_k2_states_used": False,
        "u16_lineage_used": False,
        "dev_windows_used": False,
        "eval_windows_used": False,
        "teacher_logits_used": False,
        "teacher_gradients_used": False,
        "loss_gradients_used": False,
        "second_order_statistics_used": False,
        "activation_statistics_used": False,
        "external_codebook_used": False,
        "per_member_codebook_fit_used": False,
        "post_score_tuning_used": False,
        "copied_quantizer_state_used": False,
        "non_source_train_data_used": False,
    }
    terminal = {
        "schema": "banana-smasher-native-q2-all43-shared-train-terminal-v1",
        "status": "PASS_ALL43_SHARED_SOURCE_TRAIN",
        "task_id": args.task_id,
        "board_run_id": args.run_id,
        "canonical_sha": args.canonical_sha,
        "basis_sha256": index_sha,
        "model_index": str(model_index),
        "model_index_sha256": index_sha,
        "model_index_self_read_sha256": _sha_path(model_index),
        "roster": roster,
        "roster_count": len(roster),
        "layers": layers,
        "no_gap_no_overlap": layers == list(range(43)) and len(set(roster)) == 43,
        "member_rows": rows,
        "single_shared_codebook": {"path": str(shared_path), "sha256": shared_sha, "dtype": "float16", "shape": [1024], "bytes": shared_path.stat().st_size, "referenced_by_members": len(rows)},
        "train_statistics": statistics,
        "assignment_accounting": {"weights": 43 * 256, "code_bits": total_bits, "code_bpw": total_bits / (43 * 256), "exact_2_bpw": total_bits == 43 * 256 * 2},
        "safety": forbidden,
        "throughput": {"elapsed_seconds": elapsed, "train_assignments_per_second": statistics["assignment_count"] / elapsed, "materialization_assignments_per_second": (43 * 256) / elapsed},
        "candidate_locator": str(root / "candidate"),
        "candidate_mode": "0555-directories/0444-files",
        "pid": pid,
        "startticks": ticks,
        "unix": time.time(),
    }
    receipt_path = root / "receipts" / "ALL43_SHARED_TRAIN_TERMINAL.json"
    _atomic_json(receipt_path, terminal)
    terminal_sha = _sha_path(receipt_path)
    _atomic_json(root / "PROGRESS.json", {"status": terminal["status"], "completed_members": 43, "total_members": 43, "shared_codebook_sha256": shared_sha, "terminal_receipt": str(receipt_path), "terminal_sha256": terminal_sha, "pid": pid, "startticks": ticks, "basis_sha256": index_sha, "unix": time.time()})
    return {"status": "PASS", "terminal": str(receipt_path), "terminal_sha256": terminal_sha, "shared_codebook_sha256": shared_sha, "elapsed_seconds": elapsed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exact all43 shared-codebook source/TRAIN producer")
    parser.add_argument("--model-index", required=True)
    parser.add_argument("--expected-index-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--canonical-sha", required=True)
    parser.add_argument("--host-claim", required=True)
    parser.add_argument("--claim-preimage-sha256", required=True)
    parser.add_argument(
        "--train-support",
        choices=("corner16", "full-projection"),
        required=True,
        help="source/TRAIN support used for the one shared-codebook fit",
    )
    args = parser.parse_args(argv)
    print(json.dumps(run_all43(args), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
