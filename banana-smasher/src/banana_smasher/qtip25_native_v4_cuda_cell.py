"""CUDA cell runner for homogeneous QTIP2.5 L16/B10/V4.

Inputs are the same normalized transformed 16x16 QTIP blocks used by the
matched arms: float32 ``[blocks,64,4]`` plus the shared float32 ``[512,2]``
TLUT. The runner seals exact code/aux accounting, reference/CUDA parity,
encode rate, installed-consumer decode rate and no-fallback counters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .qtip25_native_v4 import (
    NATIVE_QTIP25_GEOMETRY,
    decode_native_v4,
    expand_native_v4_tlut,
    native_v4_wire_accounting,
    pack_native_v4_states,
    solve_native_v4_cuda,
)

SCHEMA = "banana-smasher-qtip25-native-v4-cuda-cell-v1"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _basis(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("basis must be lowercase SHA-256")
    return value


def _atomic_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return hashlib.sha256(payload).hexdigest()


def validate_input(
    input_path: str | Path,
    tlut_path: str | Path,
    *,
    intended_basis_sha256: str,
    observed_basis_sha256: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    intended = _basis(intended_basis_sha256)
    observed = _basis(observed_basis_sha256)
    if observed != intended:
        raise ValueError(f"native V4 basis mismatch: {observed} != {intended}")
    source_path = Path(input_path).resolve()
    table_path = Path(tlut_path).resolve()
    target = np.load(source_path, mmap_mode="r", allow_pickle=False)
    tlut = np.load(table_path, allow_pickle=False)
    if target.dtype != np.float32 or target.ndim != 3 or target.shape[1:] != (64, 4):
        raise ValueError("native V4 input must be float32 [blocks,64,4]")
    if tlut.dtype != np.float32 or tuple(tlut.shape) != (512, 2):
        raise ValueError("native V4 TLUT must be float32 [512,2]")
    if not np.isfinite(target).all() or not np.isfinite(tlut).all():
        raise ValueError("native V4 input/TLUT must be finite")
    return target, tlut, {
        "basis_sha256": intended,
        "input": {
            "path": str(source_path),
            "bytes": source_path.stat().st_size,
            "sha256": _sha_file(source_path),
            "shape": list(target.shape),
            "dtype": str(target.dtype),
        },
        "tlut": {
            "path": str(table_path),
            "bytes": table_path.stat().st_size,
            "sha256": _sha_file(table_path),
            "shape": list(tlut.shape),
            "dtype": str(tlut.dtype),
        },
    }


def run_cuda_cell(
    input_path: str | Path,
    tlut_path: str | Path,
    output: str | Path,
    *,
    intended_basis_sha256: str,
    observed_basis_sha256: str,
    solve_batch: int = 256,
    decode_batch: int = 512,
    decode_repeats: int = 5,
    scale_bytes: int = 0,
    transform_bytes: int = 0,
) -> dict[str, Any]:
    target, tlut, identity = validate_input(
        input_path,
        tlut_path,
        intended_basis_sha256=intended_basis_sha256,
        observed_basis_sha256=observed_basis_sha256,
    )
    import torch
    from banana_smasher_plugin.native_qtip25_v4 import (
        dequantize_native_v4_blocks,
        native_v4_decode_counters,
        reset_native_v4_decode_counters,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("native V4 hardware run requires CUDA")
    if min(solve_batch, decode_batch, decode_repeats) < 1:
        raise ValueError("native V4 CUDA batch/repeat values must be positive")
    free, total = torch.cuda.mem_get_info()
    peak_estimate = (256 << 20) + solve_batch * (
        64 * 64 * 4 + 64 * 4 * 4 + 64 * 4
    )
    if peak_estimate + (4 << 30) > free:
        raise RuntimeError(
            f"native V4 CUDA preflight failed: free={free} peak_estimate={peak_estimate} reserve={4 << 30}"
        )
    device = torch.device("cuda")
    table = torch.from_numpy(np.asarray(tlut)).to(device)
    state_lut = torch.from_numpy(expand_native_v4_tlut(tlut)).to(device)
    states_parts: list[np.ndarray] = []
    torch.cuda.synchronize()
    encode_started = time.perf_counter()
    for start in range(0, len(target), solve_batch):
        source = torch.from_numpy(np.asarray(target[start : start + solve_batch])).to(device)
        states_parts.append(solve_native_v4_cuda(source, state_lut=state_lut).cpu().numpy())
    torch.cuda.synchronize()
    encode_seconds = time.perf_counter() - encode_started
    states = np.concatenate(states_parts).astype(np.int32, copy=False)
    packed = pack_native_v4_states(states)

    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    codes_path = output_root / "codes.npy"
    np.save(codes_path, packed, allow_pickle=False)
    reference_blocks = min(4, len(packed))
    reference = decode_native_v4(
        packed[:reference_blocks],
        np.ones(reference_blocks, dtype=np.float32),
        positions=256,
        tlut=tlut,
    ).reshape(reference_blocks, 16, 16)
    observed_parts = []
    reset_native_v4_decode_counters()
    with torch.no_grad():
        for start in range(0, len(packed), decode_batch):
            code = torch.from_numpy(packed[start : start + decode_batch]).to(device)
            observed_parts.append(dequantize_native_v4_blocks(code, table).cpu())
        parity_observed = torch.cat(observed_parts[:1])[:reference_blocks].numpy()
        if not np.array_equal(reference, parity_observed):
            difference = float(np.max(np.abs(reference - parity_observed)))
            raise RuntimeError(f"native V4 reference/CUDA decode mismatch: max_abs={difference}")
        torch.cuda.synchronize()
        decode_started = time.perf_counter()
        for _ in range(decode_repeats):
            for start in range(0, len(packed), decode_batch):
                code = torch.from_numpy(packed[start : start + decode_batch]).to(device)
                dequantize_native_v4_blocks(code, table)
        torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - decode_started
    decoded = torch.cat(observed_parts).reshape(len(target), 64, 4).numpy()
    delta = decoded.astype(np.float64) - np.asarray(target, dtype=np.float64)
    sse = float(np.sum(delta * delta, dtype=np.float64))
    counters = native_v4_decode_counters()
    if counters["fallback_calls"] != 0 or counters["cuda_decode_calls"] < 1:
        raise RuntimeError(f"native V4 installed consumer counters invalid: {counters}")
    positions = int(target.size)
    accounting = native_v4_wire_accounting(
        position_count=positions,
        sequence_count=len(target),
        scale_bytes=scale_bytes,
        transform_bytes=transform_bytes,
        shared_tlut_bytes=int(tlut.nbytes),
    )
    if accounting["code_payload_bytes"] != int(packed.nbytes):
        raise RuntimeError("native V4 packed bytes do not close exact accounting")
    receipt = {
        "schema": SCHEMA,
        "status": "PASS",
        "task_id": "t_57101415",
        **identity,
        "geometry": NATIVE_QTIP25_GEOMETRY.as_mapping(),
        "phase_count": 1,
        "unique_transition_bits": [10],
        "alternation": False,
        "accounting": accounting,
        "direct_error": {"sse": sse, "mse": sse / positions},
        "encode": {
            "wall_seconds": encode_seconds,
            "blocks_per_second": len(target) / encode_seconds,
            "weights_per_second": positions / encode_seconds,
        },
        "installed_cuda_decode": {
            "wall_seconds": decode_seconds,
            "repeats": decode_repeats,
            "weights_per_second": positions * decode_repeats / decode_seconds,
            "reference_parity_blocks": reference_blocks,
            "counters": counters,
        },
        "cuda": {
            "torch": torch.__version__,
            "runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "free_bytes_preflight": int(free),
            "total_bytes": int(total),
            "peak_estimate_bytes": int(peak_estimate),
            "reserve_bytes": 4 << 30,
        },
        "codes": {
            "path": str(codes_path),
            "bytes": codes_path.stat().st_size,
            "data_bytes": int(packed.nbytes),
            "sha256": _sha_file(codes_path),
        },
    }
    receipt_path = output_root / "NATIVE_V4_CELL_RECEIPT.json"
    receipt["receipt_sha256"] = _atomic_json(receipt_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QTIP2.5 native V4 CUDA cell runner")
    parser.add_argument("--input", required=True)
    parser.add_argument("--tlut", required=True)
    parser.add_argument("--output")
    parser.add_argument("--intended-basis", required=True)
    parser.add_argument("--observed-basis", required=True)
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    parser.add_argument("--solve-batch", type=int, default=256)
    parser.add_argument("--decode-batch", type=int, default=512)
    parser.add_argument("--decode-repeats", type=int, default=5)
    args = parser.parse_args(argv)
    if args.mode == "preflight":
        _target, _tlut, identity = validate_input(
            args.input,
            args.tlut,
            intended_basis_sha256=args.intended_basis,
            observed_basis_sha256=args.observed_basis,
        )
        print(json.dumps({"status": "PASS", **identity}, sort_keys=True))
        return 0
    if not args.output:
        parser.error("--output is required in run mode")
    receipt = run_cuda_cell(
        args.input,
        args.tlut,
        args.output,
        intended_basis_sha256=args.intended_basis,
        observed_basis_sha256=args.observed_basis,
        solve_batch=args.solve_batch,
        decode_batch=args.decode_batch,
        decode_repeats=args.decode_repeats,
    )
    print(json.dumps({"status": "PASS", "receipt_sha256": receipt["receipt_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
