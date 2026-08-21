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

from .banana_v1 import expand_banana_v1_codebook
from .qtip25_native_v4 import (
    NATIVE_QTIP25_GEOMETRY,
    NativeQtip25Geometry,
    decode_native_v4,
    decode_native_v4_torch,
    expand_native_v4_tlut,
    native_v4_cyclic_fast_path_counters,
    native_v4_geometry,
    native_v4_wire_accounting,
    solve_native_v4_cuda,
)

SCHEMA = "banana-smasher-qtip25-native-v4-cuda-cell-v1"


def _decode_native_v4_blocks(decoder: Any, packed: Any, tlut: Any) -> Any:
    """Call the installed V4 decoder through its public two-argument ABI."""
    return decoder(packed, tlut)


def _pack_cuda_states_v4(
    states: Any, *, geometry: NativeQtip25Geometry = NATIVE_QTIP25_GEOMETRY
) -> Any:
    """Pack one exact circular transition stream per row without a CPU loop."""
    import torch

    if not states.is_cuda or states.ndim != 2 or states.shape[1] * geometry.B < geometry.L:
        raise ValueError("native V4 CUDA states must be on-device [rows,steps]")
    values = states.to(torch.int32)
    suffix_mask = geometry.prefixes - 1
    if bool(((values[:, :-1] & suffix_mask) != (values[:, 1:] >> geometry.B)).any()):
        raise ValueError("native V4 CUDA state path violates B10 transitions")
    if bool(((values[:, -1] & suffix_mask) != (values[:, 0] >> geometry.B)).any()):
        raise ValueError("native V4 CUDA state path does not close")
    first_shifts = torch.arange(geometry.L - 1, -1, -1, device=values.device)
    branch_shifts = torch.arange(geometry.B - 1, -1, -1, device=values.device)
    first = ((values[:, :1, None] >> first_shifts) & 1).reshape(values.shape[0], -1)
    rest = ((values[:, 1:, None] >> branch_shifts) & 1).reshape(values.shape[0], -1)
    bit_count = values.shape[1] * geometry.B
    bits = torch.cat((first, rest), dim=1)[:, :bit_count]
    padding = (-bit_count) % 8
    if padding:
        bits = torch.nn.functional.pad(bits, (0, padding))
    byte_shifts = torch.arange(7, -1, -1, device=values.device)
    return torch.sum(bits.reshape(values.shape[0], -1, 8) << byte_shifts, dim=2).to(torch.uint8)


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
    old_v4 = tlut.dtype == np.float32 and tuple(tlut.shape) == (512, 2)
    pr31_v5 = tlut.dtype == np.float16 and tuple(tlut.shape) == (1024,)
    if not (old_v4 or pr31_v5):
        raise ValueError(
            "native LUT must be V4 float32 [512,2] or compact PR31 float16 [1024]"
        )
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


def _ldlq_cuda_matrix(
    target: np.ndarray,
    hessian: np.ndarray,
    *,
    matrix_shape: tuple[int, int],
    state_lut: Any,
    table: Any,
    geometry: NativeQtip25Geometry,
    solve_batch: int,
    scale_factors: Sequence[float],
    scale_semantics: str = "absolute_unit",
    regularization_sigma: float = 1e-2,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Run block-LDL/reverse-16 CUDA feedback at unit, RMS, or relative scale."""
    import math

    import torch

    from .qtip_batch import block_ldl_batch

    rows, columns = matrix_shape
    if (
        rows % 16
        or columns % 16
        or len(target) != (rows // 16) * (columns // 16)
        or hessian.shape != (columns, columns)
        or hessian.dtype != np.float32
        or not np.isfinite(hessian).all()
    ):
        raise ValueError("native V4 CUDA LDLQ matrix/Hessian geometry mismatch")
    factors = tuple(float(value) for value in scale_factors)
    if not factors or any(not math.isfinite(value) or value <= 0 for value in factors):
        raise ValueError("native V4 CUDA LDLQ scale factors must be finite and positive")
    if scale_semantics not in {"relative_search", "absolute_unit", "rms_ratio"}:
        raise ValueError(
            "native V4 CUDA LDLQ scale semantics must be relative_search, absolute_unit, or rms_ratio"
        )
    effective_factors = (
        factors if scale_semantics == "relative_search" else (1.0,)
    )
    device = state_lut.device
    row_blocks = rows // 16
    column_blocks = columns // 16
    source = (
        torch.from_numpy(np.asarray(target).copy())
        .to(device)
        .reshape(row_blocks, column_blocks, 16, 16)
        .permute(0, 2, 1, 3)
        .reshape(rows, columns)
        .contiguous()
    )
    hessian_tensor = torch.from_numpy(np.asarray(hessian).copy()).to(device)
    diagonal = hessian_tensor.diagonal()
    diagonal_mean = diagonal.mean()
    if not bool(diagonal_mean > 0):
        raise RuntimeError("native V4 CUDA LDLQ Hessian diagonal mean must be positive")
    diagonal.add_(diagonal_mean * regularization_sigma)
    lower = block_ldl_batch(hessian_tensor.unsqueeze(0), 16)[0]
    lower.diagonal().zero_()
    feedback_nonzero_count = int(torch.count_nonzero(lower).item())
    if feedback_nonzero_count == 0:
        raise RuntimeError("native V4 CUDA LDLQ requires nonzero Hessian feedback")
    source_rms = source.double().square().mean().sqrt()
    lut_rms = state_lut.double().square().mean().sqrt()
    base_scale = float((source_rms / lut_rms).item()) if source_rms.item() else 1.0
    best: tuple[float, float, float, Any] | None = None
    for factor in effective_factors:
        scale = 1.0 if scale_semantics == "absolute_unit" else base_scale * factor
        decoded = torch.zeros_like(source)
        states_grid = torch.empty(
            (row_blocks, column_blocks, 64), device=device, dtype=torch.int32
        )
        for column_block in range(column_blocks - 1, -1, -1):
            start = column_block * 16
            end = start + 16
            corrected = source[:, start:end].clone()
            if end < columns:
                error_right = source[:, end:] - decoded[:, end:]
                corrected.add_((lower[end:, start:end].T @ error_right.T).T)
            tiles = corrected.reshape(row_blocks, 64, geometry.V) / scale
            parts = []
            for batch_start in range(0, row_blocks, solve_batch):
                parts.append(
                    solve_native_v4_cuda(
                        tiles[batch_start : batch_start + solve_batch],
                        state_lut=state_lut,
                        geometry=geometry,
                    )
                )
            states = torch.cat(parts)
            if state_lut.ndim == 1:
                packed_states = _pack_cuda_states_v4(states, geometry=geometry)
                decoded_order = decode_native_v4_torch(
                    packed_states,
                    torch.ones(row_blocks, device=device, dtype=torch.float32),
                    positions=256,
                    tlut=table,
                    geometry=geometry,
                ).reshape(row_blocks, 256)
                decoded[:, start:end] = decoded_order.reshape(rows, 16) * scale
            else:
                decoded[:, start:end] = state_lut[states].reshape(rows, 16) * scale
            states_grid[:, column_block] = states
        distortion = float((decoded.double() - source.double()).square().sum().item())
        if best is None or distortion < best[0]:
            best = (distortion, factor, scale, states_grid.clone())
    assert best is not None
    distortion, selected_factor, selected_scale, states_grid = best
    flat_states = states_grid.reshape(-1, 64)
    packed_parts = []
    for batch_start in range(0, len(flat_states), solve_batch):
        packed_parts.append(
            _pack_cuda_states_v4(
                flat_states[batch_start : batch_start + solve_batch], geometry=geometry
            ).cpu()
        )
    packed = torch.cat(packed_parts).numpy()
    return packed, selected_scale, {
        "method": "qtip_batch_block_ldl_reverse_16",
        "matrix_shape": [rows, columns],
        "base_scale": base_scale,
        "scale_semantics": scale_semantics,
        "selected_factor": selected_factor,
        "selected_scale": selected_scale,
        "scale_factor": selected_scale,
        "scale_factors": list(effective_factors),
        "hessian_regularization_sigma": regularization_sigma,
        "feedback_nonzero_count": feedback_nonzero_count,
        "distortion": distortion,
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
    geometry: NativeQtip25Geometry = NATIVE_QTIP25_GEOMETRY,
    hessian_path: str | Path | None = None,
    matrix_shape: tuple[int, int] | None = None,
    scale_factors: Sequence[float] = (
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
        1.05,
        1.10,
        1.15,
        1.20,
    ),
    ldlq_scale_semantics: str = "absolute_unit",
    feedback_mode: str = "off",
    hessian_regularization_sigma: float = 1e-2,
    cyclic_warmup_cycles: int = 1,
    trellis_objective: str = "sse",
    cyclic_fixed_point_fast_path: bool = False,
    reserve_bytes: int = 4 << 30,
    sequence_scales: np.ndarray | None = None,
    sequence_boundaries: Sequence[int] | None = None,
    defer_full_cuda_decode: bool = False,
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
    if cyclic_warmup_cycles not in {1, 2}:
        raise ValueError("native QTIP CUDA cyclic warmup must use one or two cycles")
    if trellis_objective not in {"sse", "lexicographic_l4"}:
        raise ValueError("native QTIP trellis objective must be sse or lexicographic_l4")
    if isinstance(reserve_bytes, bool) or not isinstance(reserve_bytes, int) or reserve_bytes < 0:
        raise ValueError("native V4 CUDA reserve_bytes must be a non-negative integer")
    if sequence_scales is not None:
        sequence_scales = np.asarray(sequence_scales, dtype=np.float64)
        if sequence_scales.shape != (len(target),) or not np.isfinite(sequence_scales).all() or bool(np.any(sequence_scales <= 0)):
            raise ValueError("native V4 CUDA sequence_scales must be finite positive [blocks]")
    boundaries = tuple(int(value) for value in (sequence_boundaries or (len(target),)))
    if not boundaries or boundaries[-1] != len(target) or any(
        value <= (boundaries[index - 1] if index else 0)
        for index, value in enumerate(boundaries)
    ):
        raise ValueError("native V4 CUDA sequence_boundaries must be increasing and exhaustive")
    free, total = torch.cuda.mem_get_info()
    peak_estimate = (256 << 20) + solve_batch * (
        64 * geometry.prefixes * 4 + 64 * geometry.V * 4 + geometry.prefixes * 8
    )
    if feedback_mode not in {"off", "reverse_16"}:
        raise ValueError("native V4 feedback mode must be off or reverse_16")
    if ldlq_scale_semantics not in {
        "relative_search",
        "absolute_unit",
        "rms_ratio",
    }:
        raise ValueError(
            "native V4 scale semantics must be relative_search, absolute_unit, or rms_ratio"
        )
    if feedback_mode == "reverse_16" and hessian_path is None:
        raise ValueError("native V4 reverse_16 feedback requires a Hessian")
    if trellis_objective != "sse" and feedback_mode != "off":
        raise ValueError("lexicographic L4 requires feedback disabled")
    hessian = None
    if feedback_mode == "reverse_16":
        hessian = np.load(Path(hessian_path).resolve(), allow_pickle=False)
        peak_estimate += int(hessian.nbytes * 2 + target.nbytes * 3)
    margin = int(free) - int(peak_estimate) - int(reserve_bytes)
    if margin < 0:
        raise RuntimeError(
            f"native V4 CUDA preflight failed: free={free} peak_estimate={peak_estimate} "
            f"reserve={reserve_bytes} margin={margin}"
        )
    device = torch.device("cuda")
    table = torch.from_numpy(np.asarray(tlut)).to(device)
    expanded = (
        expand_banana_v1_codebook(tlut)
        if tuple(tlut.shape) == (1024,)
        else expand_native_v4_tlut(tlut, geometry=geometry)
    )
    state_lut = torch.from_numpy(expanded).to(device)

    selected_scale = 1.0
    if ldlq_scale_semantics == "rms_ratio":
        source_rms = float(np.sqrt(np.mean(np.asarray(target, dtype=np.float64) ** 2)))
        lut_rms = float(np.sqrt(np.mean(expanded.astype(np.float64) ** 2)))
        selected_scale = 1.0 if source_rms == 0 else source_rms / lut_rms
    torch.cuda.synchronize()
    encode_started = time.perf_counter()
    cyclic_fast_before = native_v4_cyclic_fast_path_counters()
    optimization: dict[str, Any] = {
        "method": "rms_only_no_feedback",
        "feedback_mode": "off",
        "base_scale": selected_scale,
        "scale_semantics": ldlq_scale_semantics,
        "selected_factor": 1.0,
        "selected_scale": selected_scale,
        "scale_factor": selected_scale,
        "scale_factors": [1.0],
        "hessian_regularization_sigma": 0.0,
        "feedback_nonzero_count": 0,
        "cyclic_warmup_cycles": cyclic_warmup_cycles,
        "trellis_objective": trellis_objective,
    }
    if hessian is not None:
        if matrix_shape is None:
            raise ValueError("native V4 CUDA Hessian path requires matrix_shape")
        packed, selected_scale, optimization = _ldlq_cuda_matrix(
            target,
            hessian,
            matrix_shape=matrix_shape,
            state_lut=state_lut,
            table=table,
            geometry=geometry,
            solve_batch=solve_batch,
            scale_factors=scale_factors,
            scale_semantics=ldlq_scale_semantics,
            regularization_sigma=hessian_regularization_sigma,
        )
        optimization["feedback_mode"] = "reverse_16"
    else:
        packed_parts: list[np.ndarray] = []
        starts = (0,) + boundaries[:-1]
        scaled_target = None
        if sequence_scales is not None:
            scaled_target = torch.cat([
                torch.from_numpy(np.asarray(target[cell_start:boundary]).copy()).to(device)
                / float(sequence_scales[cell_start])
                for cell_start, boundary in zip(starts, boundaries)
            ])
        for start in range(0, len(target), solve_batch):
            end = min(start + solve_batch, len(target))
            source = (
                scaled_target[start:end]
                if scaled_target is not None
                else torch.from_numpy(np.asarray(target[start:end]).copy()).to(device)
            )
            states = solve_native_v4_cuda(
                source if scaled_target is not None else source / selected_scale,
                state_lut=state_lut,
                geometry=geometry,
                cyclic_warmup_cycles=cyclic_warmup_cycles,
                trellis_objective=trellis_objective,
                cyclic_fixed_point_fast_path=cyclic_fixed_point_fast_path,
            )
            packed_parts.append(
                _pack_cuda_states_v4(states, geometry=geometry).cpu().numpy()
            )
        packed = np.concatenate(packed_parts)
    cyclic_fast_after = native_v4_cyclic_fast_path_counters()
    optimization["cyclic_fixed_point_fast_path"] = {
        key: cyclic_fast_after[key] - cyclic_fast_before[key]
        for key in cyclic_fast_after
    }
    torch.cuda.synchronize()
    encode_seconds = time.perf_counter() - encode_started

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
        geometry=geometry,
    ).reshape(reference_blocks, 16, 16)
    observed_parts = []
    decode_extent = reference_blocks if defer_full_cuda_decode else len(packed)
    reset_native_v4_decode_counters()
    with torch.no_grad():
        if defer_full_cuda_decode:
            torch.cuda.synchronize()
            decode_started = time.perf_counter()
        for start in range(0, decode_extent, decode_batch):
            code = torch.from_numpy(packed[start : min(start + decode_batch, decode_extent)]).to(device)
            observed_parts.append(
                _decode_native_v4_blocks(dequantize_native_v4_blocks, code, table).cpu()
            )
        parity_observed = torch.cat(observed_parts[:1])[:reference_blocks].numpy()
        if not np.array_equal(reference, parity_observed):
            difference = float(np.max(np.abs(reference - parity_observed)))
            raise RuntimeError(f"native V4 reference/CUDA decode mismatch: max_abs={difference}")
        if defer_full_cuda_decode:
            torch.cuda.synchronize()
            decode_seconds = time.perf_counter() - decode_started
        else:
            torch.cuda.synchronize()
            decode_started = time.perf_counter()
            for _ in range(decode_repeats):
                for start in range(0, decode_extent, decode_batch):
                    code = torch.from_numpy(packed[start : min(start + decode_batch, decode_extent)]).to(device)
                    _decode_native_v4_blocks(dequantize_native_v4_blocks, code, table)
            torch.cuda.synchronize()
            decode_seconds = time.perf_counter() - decode_started
    sse = None
    if not defer_full_cuda_decode:
        decoded = torch.cat(observed_parts).reshape(len(target), 64, 4).numpy()
        decoded *= (
            sequence_scales.astype(np.float32).reshape(-1, 1, 1)
            if sequence_scales is not None
            else np.float32(selected_scale)
        )
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
        geometry=geometry,
    )
    if accounting["code_payload_bytes"] != int(packed.nbytes):
        raise RuntimeError("native V4 packed bytes do not close exact accounting")
    receipt = {
        "schema": SCHEMA,
        "status": "PASS",
        **identity,
        "geometry": geometry.as_mapping(),
        "phase_count": 1,
        "unique_transition_bits": [geometry.B],
        "alternation": False,
        "optimization": optimization,
        "accounting": accounting,
        "direct_error": (
            {"deferred_to_public_api": True}
            if sse is None
            else {"sse": sse, "mse": sse / positions}
        ),
        "encode": {
            "wall_seconds": encode_seconds,
            "blocks_per_second": len(target) / encode_seconds,
            "weights_per_second": positions / encode_seconds,
        },
        "installed_cuda_decode": {
            "wall_seconds": decode_seconds,
            "repeats": decode_repeats,
            "weights_per_second": decode_extent * 256 * decode_repeats / decode_seconds,
            "reference_parity_blocks": reference_blocks,
            "full_decode_deferred_to_public_api": defer_full_cuda_decode,
            "counters": counters,
        },
        "cuda": {
            "torch": torch.__version__,
            "runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "free_bytes_preflight": int(free),
            "total_bytes": int(total),
            "peak_estimate_bytes": int(peak_estimate),
            "reserve_bytes": int(reserve_bytes),
            "margin_bytes": int(margin),
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
    parser.add_argument("--bpw", default="2.5")
    args = parser.parse_args(argv)
    if args.mode == "preflight":
        _target, _tlut, identity = validate_input(
            args.input,
            args.tlut,
            intended_basis_sha256=args.intended_basis,
            observed_basis_sha256=args.observed_basis,
        )
        geometry = native_v4_geometry(args.bpw)
        print(
            json.dumps(
                {"status": "PASS", "geometry": geometry.as_mapping(), **identity},
                sort_keys=True,
            )
        )
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
        geometry=native_v4_geometry(args.bpw),
    )
    print(json.dumps({"status": "PASS", "receipt_sha256": receipt["receipt_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
