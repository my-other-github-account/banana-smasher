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
    NativeQtip25Geometry,
    decode_native_v4,
    expand_native_v4_tlut,
    native_v4_geometry,
    native_v4_wire_accounting,
    solve_native_v4_cuda,
)

SCHEMA = "banana-smasher-qtip25-native-v4-cuda-cell-v1"


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


def ldlq_native_v4_cuda_batch(
    targets: Sequence[np.ndarray],
    hessians: Sequence[np.ndarray],
    *,
    matrix_shape: tuple[int, int],
    state_lut: Any,
    geometry: NativeQtip25Geometry,
    solve_batch: int,
    scale_factors: Sequence[float],
    cell_scale_factors: Sequence[float] | None = None,
    cell_scales: Sequence[float] | None = None,
    preserve_cell_math: bool = True,
) -> tuple[list[np.ndarray], list[float], list[dict[str, Any]]]:
    """Run reverse-16 native-V4 LDLQ with cells and scale candidates batched."""
    import math

    import torch

    from .qtip_batch import block_ldl_batch

    rows, columns = matrix_shape
    cell_count = len(targets)
    tile_count = (rows // 16) * (columns // 16)
    if (
        not cell_count
        or len(hessians) != cell_count
        or rows % 16
        or columns % 16
        or any(len(target) != tile_count for target in targets)
        or any(
            hessian.shape != (columns, columns)
            or hessian.dtype != np.float32
            or not np.isfinite(hessian).all()
            for hessian in hessians
        )
    ):
        raise ValueError("native V4 CUDA LDLQ matrix/Hessian geometry mismatch")
    factors = tuple(float(value) for value in scale_factors)
    if not factors or any(not math.isfinite(value) or value <= 0 for value in factors):
        raise ValueError("native V4 CUDA LDLQ scale factors must be finite and positive")
    fixed_factors = (
        tuple(float(value) for value in cell_scale_factors)
        if cell_scale_factors is not None
        else None
    )
    fixed_scales = (
        tuple(float(value) for value in cell_scales)
        if cell_scales is not None
        else None
    )
    if fixed_factors is not None and fixed_scales is not None:
        raise ValueError("native V4 cell scale factors and absolute scales are mutually exclusive")
    if fixed_factors is not None and (
        len(fixed_factors) != cell_count
        or factors != (1.0,)
        or any(not math.isfinite(value) or value <= 0 for value in fixed_factors)
    ):
        raise ValueError(
            "native V4 fixed cell scale factors require one positive factor per cell and scale_factors=(1.0,)"
        )
    if fixed_scales is not None and (
        len(fixed_scales) != cell_count
        or factors != (1.0,)
        or any(not math.isfinite(value) or value <= 0 for value in fixed_scales)
    ):
        raise ValueError(
            "native V4 fixed cell scales require one positive absolute scale per cell and scale_factors=(1.0,)"
        )
    device = state_lut.device
    row_blocks = rows // 16
    column_blocks = columns // 16
    source_values = [
        torch.from_numpy(np.asarray(target).copy())
            .reshape(row_blocks, column_blocks, 16, 16)
            .permute(0, 2, 1, 3)
            .reshape(rows, columns)
            .to(device)
        for target in targets
    ]
    source = torch.stack(source_values)
    regularization_sigma = 1e-2
    hessian_values = [
        torch.from_numpy(np.asarray(hessian).copy()).to(device) for hessian in hessians
    ]
    if preserve_cell_math:
        lower_values = []
        for hessian_tensor in hessian_values:
            diagonal_mean = hessian_tensor.diagonal().mean()
            if not bool(diagonal_mean > 0):
                raise RuntimeError(
                    "native V4 CUDA LDLQ Hessian diagonal mean must be positive"
                )
            hessian_tensor.diagonal().add_(
                diagonal_mean * regularization_sigma
            )
            lower_values.append(block_ldl_batch(hessian_tensor[None], 16)[0])
        lower = torch.stack(lower_values)
    else:
        hessian_tensor = torch.stack(hessian_values)
        diagonal = hessian_tensor.diagonal(dim1=-2, dim2=-1)
        diagonal_mean = diagonal.mean(dim=-1)
        if not bool(torch.all(diagonal_mean > 0)):
            raise RuntimeError(
                "native V4 CUDA LDLQ Hessian diagonal mean must be positive"
            )
        diagonal.add_(diagonal_mean[:, None] * regularization_sigma)
        lower = block_ldl_batch(hessian_tensor, 16)
    lower.diagonal(dim1=-2, dim2=-1).zero_()
    feedback_nonzero_counts = torch.count_nonzero(lower, dim=(1, 2))
    if bool(torch.any(feedback_nonzero_counts == 0)):
        raise RuntimeError("native V4 CUDA LDLQ requires nonzero Hessian feedback")
    lut_rms = state_lut.double().square().mean().sqrt()
    base_scale_values = []
    for cell in range(cell_count):
        source_rms = source_values[cell].double().square().mean().sqrt()
        base_scale_values.append(
            float((source_rms / lut_rms).item()) if source_rms.item() else 1.0
        )
    factor_rows = (
        [[fixed_factors[cell]] for cell in range(cell_count)]
        if fixed_factors is not None
        else [[1.0] for _ in range(cell_count)]
        if fixed_scales is not None
        else [list(factors) for _ in range(cell_count)]
    )
    scale_rows = (
        [[fixed_scales[cell]] for cell in range(cell_count)]
        if fixed_scales is not None
        else [
            [base_scale_values[cell] * factor for factor in factor_rows[cell]]
            for cell in range(cell_count)
        ]
    )
    factor_count = len(factors)
    group_count = cell_count * factor_count
    source_group = source[:, None].expand(-1, factor_count, -1, -1).reshape(
        group_count, rows, columns
    )
    lower_group = lower[:, None].expand(-1, factor_count, -1, -1).reshape(
        group_count, columns, columns
    )
    flat_scales = [value for row in scale_rows for value in row]
    decoded = torch.zeros_like(source_group)
    states_grid = torch.empty(
        (group_count, row_blocks, column_blocks, 64),
        device=device,
        dtype=torch.int32,
    )
    max_solver_batch = 0
    for column_block in range(column_blocks - 1, -1, -1):
        start = column_block * 16
        end = start + 16
        corrected = source_group[:, :, start:end].clone()
        if end < columns:
            error_right = source_group[:, :, end:] - decoded[:, :, end:]
            if preserve_cell_math:
                for group in range(group_count):
                    corrected[group].add_(
                        (
                            lower_group[group, end:, start:end].T
                            @ error_right[group].T
                        ).T
                    )
            else:
                corrected.add_(
                    torch.bmm(error_right, lower_group[:, end:, start:end])
                )
        flattened_tiles = torch.stack(
            torch._foreach_div(
                [
                    corrected[group].reshape(row_blocks, 64, geometry.V)
                    for group in range(group_count)
                ],
                flat_scales,
            )
        ).reshape(group_count * row_blocks, 64, geometry.V)
        parts = []
        for batch_start in range(0, len(flattened_tiles), solve_batch):
            part = flattened_tiles[batch_start : batch_start + solve_batch]
            max_solver_batch = max(max_solver_batch, len(part))
            parts.append(
                solve_native_v4_cuda(
                    part,
                    state_lut=state_lut,
                    geometry=geometry,
                )
            )
        states = torch.cat(parts).reshape(group_count, row_blocks, 64)
        decoded[:, :, start:end] = torch.stack(
            torch._foreach_mul(
                [
                    state_lut[states[group]].reshape(rows, 16)
                    for group in range(group_count)
                ],
                flat_scales,
            )
        )
        states_grid[:, :, column_block] = states
    difference = decoded.double() - source_group.double()
    distortions = (
        torch.stack([difference[group].square().sum() for group in range(group_count)])
        if preserve_cell_math
        else difference.square().sum(dim=(1, 2))
    ).reshape(cell_count, factor_count)
    winners = torch.argmin(distortions, dim=1)
    cell_ids = torch.arange(cell_count, device=device)
    selected_states = states_grid.reshape(
        cell_count, factor_count, row_blocks, column_blocks, 64
    )[cell_ids, winners]
    flat_states = selected_states.reshape(cell_count * tile_count, 64)
    packed_parts = []
    for batch_start in range(0, len(flat_states), solve_batch):
        packed_parts.append(
            _pack_cuda_states_v4(
                flat_states[batch_start : batch_start + solve_batch], geometry=geometry
            ).cpu()
        )
    packed_values = torch.cat(packed_parts).reshape(
        cell_count, tile_count, 8 * geometry.B
    ).numpy()
    selected_scales = [
        scale_rows[cell][int(winners[cell].item())] for cell in range(cell_count)
    ]
    results = []
    for cell in range(cell_count):
        winner = int(winners[cell].item())
        selected_scale = selected_scales[cell]
        results.append(
            {
                "method": "qtip_batch_block_ldl_reverse_16_cell_scale_batched",
                "matrix_shape": [rows, columns],
                "base_scale": base_scale_values[cell],
                "selected_factor": factor_rows[cell][winner],
                "selected_scale": selected_scale,
                "scale_factor": selected_scale,
                "scale_factors": list(factor_rows[cell]),
                "cell_batch_size": cell_count,
                "scale_batch_size": factor_count,
                "preserve_cell_math": preserve_cell_math,
                "fixed_absolute_scale": fixed_scales is not None,
                "max_solver_sequence_batch": max_solver_batch,
                "hessian_regularization_sigma": regularization_sigma,
                "feedback_nonzero_count": int(feedback_nonzero_counts[cell].item()),
                "distortion": float(distortions[cell, winner].item()),
            }
        )
    return (
        [np.ascontiguousarray(value) for value in packed_values],
        selected_scales,
        results,
    )


def _ldlq_cuda_matrix(
    target: np.ndarray,
    hessian: np.ndarray,
    *,
    matrix_shape: tuple[int, int],
    state_lut: Any,
    geometry: NativeQtip25Geometry,
    solve_batch: int,
    scale_factors: Sequence[float],
) -> tuple[np.ndarray, float, dict[str, Any]]:
    packed, selected_scales, optimization = ldlq_native_v4_cuda_batch(
        [target],
        [hessian],
        matrix_shape=matrix_shape,
        state_lut=state_lut,
        geometry=geometry,
        solve_batch=solve_batch,
        scale_factors=scale_factors,
    )
    return packed[0], selected_scales[0], optimization[0]


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
        64 * geometry.prefixes * 4 + 64 * geometry.V * 4 + geometry.prefixes * 8
    )
    hessian = None
    if hessian_path is not None:
        hessian = np.load(Path(hessian_path).resolve(), allow_pickle=False)
        peak_estimate += int(hessian.nbytes * 2 + target.nbytes * 3)
    if peak_estimate + (4 << 30) > free:
        raise RuntimeError(
            f"native V4 CUDA preflight failed: free={free} peak_estimate={peak_estimate} reserve={4 << 30}"
        )
    device = torch.device("cuda")
    table = torch.from_numpy(np.asarray(tlut)).to(device)
    state_lut = torch.from_numpy(expand_native_v4_tlut(tlut, geometry=geometry)).to(device)
    torch.cuda.synchronize()
    encode_started = time.perf_counter()
    optimization: dict[str, Any] = {
        "method": "rms_only_no_feedback",
        "base_scale": 1.0,
        "selected_factor": 1.0,
        "selected_scale": 1.0,
        "scale_factors": [1.0],
        "hessian_regularization_sigma": 0.0,
        "feedback_nonzero_count": 0,
    }
    if hessian is not None:
        if matrix_shape is None:
            raise ValueError("native V4 CUDA Hessian path requires matrix_shape")
        packed, selected_scale, optimization = _ldlq_cuda_matrix(
            target,
            hessian,
            matrix_shape=matrix_shape,
            state_lut=state_lut,
            geometry=geometry,
            solve_batch=solve_batch,
            scale_factors=scale_factors,
        )
    else:
        packed_parts: list[np.ndarray] = []
        for start in range(0, len(target), solve_batch):
            source = torch.from_numpy(
                np.asarray(target[start : start + solve_batch]).copy()
            ).to(device)
            states = solve_native_v4_cuda(source, state_lut=state_lut, geometry=geometry)
            packed_parts.append(
                _pack_cuda_states_v4(states, geometry=geometry).cpu().numpy()
            )
        packed = np.concatenate(packed_parts)
        selected_scale = 1.0
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
    reset_native_v4_decode_counters()
    with torch.no_grad():
        for start in range(0, len(packed), decode_batch):
            code = torch.from_numpy(packed[start : start + decode_batch]).to(device)
            observed_parts.append(
                dequantize_native_v4_blocks(
                    code, table, bpw=geometry.rate_num / geometry.rate_den
                ).cpu()
            )
        parity_observed = torch.cat(observed_parts[:1])[:reference_blocks].numpy()
        if not np.array_equal(reference, parity_observed):
            difference = float(np.max(np.abs(reference - parity_observed)))
            raise RuntimeError(f"native V4 reference/CUDA decode mismatch: max_abs={difference}")
        torch.cuda.synchronize()
        decode_started = time.perf_counter()
        for _ in range(decode_repeats):
            for start in range(0, len(packed), decode_batch):
                code = torch.from_numpy(packed[start : start + decode_batch]).to(device)
                dequantize_native_v4_blocks(
                    code, table, bpw=geometry.rate_num / geometry.rate_den
                )
        torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - decode_started
    decoded = (
        torch.cat(observed_parts).reshape(len(target), 64, 4).numpy()
        * np.float32(selected_scale)
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
