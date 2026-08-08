"""Public build and Anchor64 API for homogeneous QTIP2.5 native V4 cells.

The winning codec consumes the normalized 16x16 QTIP blocks defined by an
existing compact QTIP transform.  This module owns the missing public bridge:
physical cell weights -> compact transform -> native V4 producer -> physical
``decoded.npy`` -> the same anchor metric used by Backpack candidates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .backpack import CLASSES, _anchor_metrics
from .banana_v1 import (
    BANANA_V1_MULTIPLIER,
    BANANA_V1_OFFSET,
    expand_banana_v1_codebook,
)
from .qtip25_native_v4 import (
    NATIVE_QTIP25_GEOMETRY,
    decode_native_v4,
    expand_native_v4_tlut,
    ldlq_native_v4_matrix,
    native_v4_lower_from_hessian,
    native_v4_geometry,
    native_v5_phase_widths,
    solve_native_v4,
)

CELL_SCHEMA = "banana-smasher-qtip25-native-v4-cell-v1"
ANCHOR_SCHEMA = "banana-smasher-qtip25-native-v4-anchor-v1"
GENERIC_CELL_SCHEMA = "banana-smasher-qtip-native-v4-cell-v1"
GENERIC_ANCHOR_SCHEMA = "banana-smasher-qtip-native-v4-anchor-v1"
ANCHOR_SET_SCHEMA = "banana-smasher-qtip-native-v4-anchor-set-v1"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _basis(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
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


def _artifact(path: Path, *, data_bytes: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha_file(path),
    }
    if data_bytes is not None:
        result["data_bytes"] = data_bytes
    return result


def _fwht(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float32)
    count = result.shape[-1]
    if count <= 0 or count & (count - 1):
        raise ValueError("native V4 transform axes must be positive powers of two")
    width = 1
    while width < count:
        grouped = result.reshape(*result.shape[:-1], count // (2 * width), 2, width)
        left = grouped[..., 0, :].copy()
        right = grouped[..., 1, :].copy()
        result = np.concatenate((left + right, left - right), axis=-1).reshape(result.shape)
        width *= 2
    return np.ascontiguousarray(result / np.float32(math.sqrt(count)), dtype=np.float32)


def _fwht_blocks(value: np.ndarray, block: Any | None) -> np.ndarray:
    if block is None:
        return _fwht(value)
    width = int(block)
    source = np.asarray(value, dtype=np.float32)
    if width <= 0 or width & (width - 1) or source.shape[-1] % width:
        raise ValueError("native QTIP Hadamard block must divide the transform axis")
    shaped = source.reshape(*source.shape[:-1], source.shape[-1] // width, width)
    return _fwht(shaped).reshape(source.shape)


def _numpy_control(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"SU", "SV", "Wscale", "shape"}
        if not required.issubset(payload.files):
            raise ValueError(f"native V4 NPZ control must contain {sorted(required)}")
        return {name: np.asarray(payload[name]) for name in payload.files}


def _torch_control(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("loading a PT control requires torch; use an NPZ control for CPU fixtures") from exc
    raw = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(raw, Mapping):
        raise ValueError("native V4 PT control must contain a mapping")
    result: dict[str, Any] = {}
    for name in ("SU", "SV", "Wscale", "shape", "qtip_k", "hadamard_block"):
        if name not in raw:
            continue
        value = raw[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        result[name] = np.asarray(value)
    return result


def _load_control(path: str | Path) -> tuple[dict[str, Any], Path]:
    control_path = Path(path).expanduser().resolve()
    if control_path.is_symlink() or not control_path.is_file():
        raise ValueError(f"native V4 control must be a regular file: {control_path}")
    if control_path.suffix == ".npz":
        raw = _numpy_control(control_path)
    elif control_path.suffix in {".pt", ".pth"}:
        raw = _torch_control(control_path)
    else:
        raise ValueError("native V4 control must be NPZ or PT")
    missing = {"SU", "SV", "Wscale", "shape"} - set(raw)
    if missing:
        raise ValueError(f"native V4 control is missing {sorted(missing)}")
    shape_values = np.asarray(raw["shape"]).reshape(-1)
    if shape_values.size != 2:
        raise ValueError("native V4 control shape must contain rows and columns")
    shape = tuple(int(value) for value in shape_values)
    su_storage = np.ascontiguousarray(raw["SU"]).reshape(-1)
    sv_storage = np.ascontiguousarray(raw["SV"]).reshape(-1)
    if su_storage.dtype.kind != "f" or sv_storage.dtype.kind != "f":
        raise ValueError("native V4 SU/SV transforms must use floating storage")
    su = np.ascontiguousarray(su_storage, dtype=np.float32)
    sv = np.ascontiguousarray(sv_storage, dtype=np.float32)
    wscale_values = np.asarray(raw["Wscale"], dtype=np.float32).reshape(-1)
    if (
        shape[0] <= 0
        or shape[1] <= 0
        or shape[0] % 16
        or shape[1] % 16
        or su.shape != (shape[1],)
        or sv.shape != (shape[0],)
        or wscale_values.size != 1
        or not np.isfinite(su).all()
        or not np.isfinite(sv).all()
        or not np.isfinite(wscale_values).all()
        or float(wscale_values[0]) <= 0
    ):
        raise ValueError("native V4 control has incompatible transform geometry")
    return {
        "SU": su,
        "SV": sv,
        "SU_storage": su_storage,
        "SV_storage": sv_storage,
        "Wscale": np.float32(wscale_values[0]),
        "shape": shape,
        **(
            {"qtip_k": int(np.asarray(raw["qtip_k"]).reshape(-1)[0])}
            if "qtip_k" in raw
            else {}
        ),
        **(
            {
                "hadamard_block": int(
                    np.asarray(raw["hadamard_block"]).reshape(-1)[0]
                )
            }
            if "hadamard_block" in raw
            else {}
        ),
    }, control_path


def _to_normalized_blocks(source: np.ndarray, control: Mapping[str, Any]) -> np.ndarray:
    su = np.asarray(control["SU"], dtype=np.float32)
    sv = np.asarray(control["SV"], dtype=np.float32)
    block = control.get("hadamard_block")
    transformed = _fwht_blocks(source / su, block)
    transformed = _fwht_blocks((transformed / sv[:, None]).T, block).T
    transformed = transformed / np.float32(control["Wscale"])
    rows, columns = transformed.shape
    return np.ascontiguousarray(
        transformed.reshape(rows // 16, 16, columns // 16, 16)
        .transpose(0, 2, 1, 3)
        .reshape(-1, 64, 4),
        dtype=np.float32,
    )


def _from_normalized_blocks(blocks: np.ndarray, control: Mapping[str, Any]) -> np.ndarray:
    rows, columns = (int(value) for value in control["shape"])
    transformed = (
        np.asarray(blocks, dtype=np.float32)
        .reshape(rows // 16, columns // 16, 16, 16)
        .transpose(0, 2, 1, 3)
        .reshape(rows, columns)
    )
    transformed = transformed * np.float32(control["Wscale"])
    block = control.get("hadamard_block")
    physical = _fwht_blocks(transformed.T, block).T * np.asarray(
        control["SV"], dtype=np.float32
    )[:, None]
    return np.ascontiguousarray(
        _fwht_blocks(physical, block) * np.asarray(control["SU"], dtype=np.float32)
    )


def _build_qtip_native_v4_cell(
    source: str | Path,
    control: str | Path,
    tlut: str | Path,
    output: str | Path,
    *,
    intended_basis_sha256: str,
    observed_basis_sha256: str,
    bpw: object,
    receipt_schema: str,
    codec_version: Literal["v4", "v5", "v6"] = "v4",
    backend: Literal["cuda", "reference"] = "cuda",
    solve_batch: int = 2048,
    decode_batch: int = 2048,
    decode_repeats: int = 1,
    materialize_decoded: bool = True,
    hessian: str | Path | None = None,
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
    ldlq_scale_semantics: Literal[
        "relative_search", "absolute_unit", "rms_ratio"
    ] = "absolute_unit",
    feedback_mode: Literal["off", "reverse_16"] = "off",
    trellis_objective: Literal["sse", "lexicographic_l4"] = "sse",
) -> dict[str, Any]:
    """Build one physical homogeneous native-V4 candidate cell.

    ``source`` is finite float32 physical cell weights. ``control`` is the
    compact QTIP transform for the same cell (``SU``, ``SV``, ``Wscale``,
    ``shape``). The CUDA backend is the measured fast path; ``reference`` is a
    tiny-fixture smoke backend and is not suitable for model-scale production.
    """

    geometry = native_v4_geometry(bpw)
    intended = _basis(intended_basis_sha256, "intended_basis_sha256")
    observed = _basis(observed_basis_sha256, "observed_basis_sha256")
    if intended != observed:
        raise ValueError(f"native V4 basis mismatch: {observed} != {intended}")
    if backend not in {"cuda", "reference"}:
        raise ValueError("native V4 backend must be cuda or reference")
    if codec_version not in {"v4", "v5", "v6"}:
        raise ValueError("native QTIP codec_version must be v4, v5, or v6")
    if ldlq_scale_semantics not in {
        "relative_search",
        "absolute_unit",
        "rms_ratio",
    }:
        raise ValueError(
            "native V4 LDLQ scale semantics must be relative_search, absolute_unit, or rms_ratio"
        )
    if feedback_mode not in {"off", "reverse_16"}:
        raise ValueError("native V4 feedback mode must be off or reverse_16")
    if trellis_objective not in {"sse", "lexicographic_l4"}:
        raise ValueError("native QTIP trellis objective must be sse or lexicographic_l4")
    if feedback_mode == "reverse_16" and hessian is None:
        raise ValueError("native V4 reverse_16 feedback requires an explicit Hessian artifact")
    if feedback_mode == "off" and ldlq_scale_semantics == "relative_search":
        raise ValueError("native V4 relative scale search requires reverse_16 feedback")
    source_path = Path(source).expanduser().resolve()
    tlut_path = Path(tlut).expanduser().resolve()
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError(f"native V4 source must be a regular NPY file: {source_path}")
    if tlut_path.is_symlink() or not tlut_path.is_file():
        raise ValueError(f"native V4 TLUT must be a regular NPY file: {tlut_path}")
    source_weights = np.load(source_path, allow_pickle=False)
    table = np.load(tlut_path, allow_pickle=False)
    compact, control_path = _load_control(control)
    if (
        source_weights.dtype != np.float32
        or source_weights.shape != compact["shape"]
        or not np.isfinite(source_weights).all()
    ):
        raise ValueError(
            f"native V4 source must be finite float32{compact['shape']}, got "
            f"{source_weights.dtype}{source_weights.shape}"
        )
    old_v4 = table.dtype == np.float32 and table.shape == (512, 2)
    compact_pr31 = table.dtype == np.float16 and table.shape == (1024,)
    if not (old_v4 or compact_pr31) or not np.isfinite(table).all():
        raise ValueError(
            "native LUT must be finite V4 float32 [512,2] or compact PR31 float16 [1024]"
        )
    if codec_version == "v4" and not old_v4:
        raise ValueError("native QTIP codec_version=v4 requires lut_identity=q9-v2-v4")
    if codec_version in {"v5", "v6"} and not compact_pr31:
        raise ValueError(
            f"native QTIP codec_version={codec_version} requires the compact PR31 LUT"
        )
    if codec_version == "v5" and geometry.B != 8:
        raise ValueError("native QTIP codec_version=v5 is fixed to B8")
    if trellis_objective == "lexicographic_l4" and (
        not compact_pr31 or feedback_mode != "off"
    ):
        raise ValueError(
            "lexicographic L4 requires the compact PR31 LUT with feedback disabled"
        )

    output_root = Path(output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    blocks = _to_normalized_blocks(source_weights, compact)
    normalized_sha256 = _sha_array(blocks)
    selected_scale = 1.0
    if ldlq_scale_semantics == "rms_ratio":
        expanded_lut = (
            expand_banana_v1_codebook(table)
            if compact_pr31
            else expand_native_v4_tlut(table, geometry=geometry)
        )
        source_rms = float(np.sqrt(np.mean(blocks.astype(np.float64) ** 2)))
        lut_rms = float(np.sqrt(np.mean(expanded_lut.astype(np.float64) ** 2)))
        selected_scale = 1.0 if source_rms == 0 else source_rms / lut_rms
    cyclic_warmup_cycles = 2 if codec_version == "v6" else 1
    started = time.perf_counter()
    cuda_receipt: dict[str, Any] | None = None
    optimization: dict[str, Any] = {
        "method": "rms_only_no_feedback",
        "feedback_mode": "off",
        "scale_semantics": ldlq_scale_semantics,
        "base_scale": selected_scale,
        "selected_factor": 1.0,
        "selected_scale": selected_scale,
        "scale_factor": selected_scale,
        "scale_factors": [1.0],
        "feedback_nonzero_count": 0,
        "cyclic_warmup_cycles": cyclic_warmup_cycles,
        "trellis_objective": trellis_objective,
    }
    if backend == "cuda":
        from .qtip25_native_v4_cuda_cell import run_cuda_cell

        normalized_path = output_root / ".normalized-input.npy"
        np.save(normalized_path, blocks, allow_pickle=False)
        try:
            cuda_receipt = run_cuda_cell(
                normalized_path,
                tlut_path,
                output_root,
                intended_basis_sha256=intended,
                observed_basis_sha256=observed,
                solve_batch=solve_batch,
                decode_batch=decode_batch,
                decode_repeats=decode_repeats,
                geometry=geometry,
                scale_bytes=4,
                transform_bytes=int(
                    compact["SU_storage"].nbytes + compact["SV_storage"].nbytes
                ),
                hessian_path=hessian,
                matrix_shape=compact["shape"],
                scale_factors=scale_factors,
            )
        finally:
            normalized_path.unlink(missing_ok=True)
        packed = np.load(output_root / "codes.npy", allow_pickle=False)
        encode_seconds = float(cuda_receipt["encode"]["wall_seconds"])
        optimization = dict(cuda_receipt["optimization"])
    else:
        if feedback_mode == "off":
            encoded = solve_native_v4(
                blocks,
                tlut=table,
                scales=np.full(len(blocks), selected_scale, dtype=np.float32),
                geometry=geometry,
                cyclic_warmup_cycles=cyclic_warmup_cycles,
                trellis_objective=trellis_objective,
            )
            packed = encoded.packed
        else:
            assert hessian is not None
            hessian_path = Path(hessian).expanduser().resolve()
            hessian_value = np.load(hessian_path, allow_pickle=False)
            lower = native_v4_lower_from_hessian(hessian_value)
            rows, columns = compact["shape"]
            transformed = (
                blocks.reshape(rows // 16, columns // 16, 16, 16)
                .transpose(0, 2, 1, 3)
                .reshape(rows, columns)
            )
            matrix = ldlq_native_v4_matrix(
                transformed,
                lower,
                tlut=table,
                geometry=geometry,
                scale_factors=scale_factors,
            )
            packed = matrix.packed
            optimization = {
                "method": "qtip_batch_block_ldl_reverse_16",
                "selected_factor": matrix.scale_factor,
                "selected_scale": float(matrix.scales[0]),
                "scale_factor": float(matrix.scales[0]),
                "scale_factors": list(matrix.scale_factors),
                "feedback_nonzero_count": matrix.feedback_nonzero_count,
                "distortion": matrix.distortion,
                "hessian": _artifact(
                    hessian_path, data_bytes=int(hessian_value.nbytes)
                ),
            }
        codes_path = output_root / "codes.npy"
        np.save(codes_path, packed, allow_pickle=False)
        encode_seconds = time.perf_counter() - started

    selected_scale = np.float32(optimization["scale_factor"])
    decoded_blocks = decode_native_v4(
        packed,
        np.full(len(packed), selected_scale, dtype=np.float32),
        positions=256,
        tlut=table,
        geometry=geometry,
    ).reshape(-1, 64, 4)
    decoded = _from_normalized_blocks(decoded_blocks, compact).astype(np.float32)
    decoded_path = output_root / "decoded.npy"
    su_path = output_root / "SU.npy"
    sv_path = output_root / "SV.npy"
    wscale_path = output_root / "Wscale.npy"
    if materialize_decoded:
        np.save(decoded_path, decoded, allow_pickle=False)
    else:
        decoded_path.unlink(missing_ok=True)
    np.save(su_path, compact["SU_storage"], allow_pickle=False)
    np.save(sv_path, compact["SV_storage"], allow_pickle=False)
    np.save(
        wscale_path,
        np.asarray(compact["Wscale"] * selected_scale, dtype=np.float32),
        allow_pickle=False,
    )
    codes_path = output_root / "codes.npy"

    weights = int(source_weights.size)
    code_bits_numerator = weights * geometry.B
    if code_bits_numerator % geometry.V:
        raise RuntimeError("native V4 weights do not close exact B/V accounting")
    code_bits = code_bits_numerator // geometry.V
    if packed.nbytes * 8 != code_bits:
        raise RuntimeError("native V4 codes do not close exact B/V accounting")
    delta = decoded.astype(np.float64) - source_weights.astype(np.float64)
    sse = float(np.sum(delta * delta, dtype=np.float64))
    transform_bytes = int(
        compact["SU_storage"].nbytes + compact["SV_storage"].nbytes
    )
    wscale_bytes = int(np.asarray(compact["Wscale"], dtype=np.float32).nbytes)
    full_wire_bytes = int(packed.nbytes + transform_bytes + wscale_bytes + table.nbytes)
    receipt: dict[str, Any] = {
        "schema": receipt_schema,
        "status": "PASS",
        "backend": backend,
        "codec_version": codec_version,
        "provider": f"qtip-native-{codec_version}@{geometry.rate_num / geometry.rate_den:.2f}",
        "basis_sha256": intended,
        "geometry": geometry.as_mapping(),
        "source": {
            **_artifact(source_path, data_bytes=int(source_weights.nbytes)),
            "shape": list(source_weights.shape),
            "dtype": str(source_weights.dtype),
        },
        "control": {
            **_artifact(control_path),
            "shape": list(compact["shape"]),
            **({"qtip_k": compact["qtip_k"]} if "qtip_k" in compact else {}),
        },
        "tlut": {
            **_artifact(tlut_path, data_bytes=int(table.nbytes)),
            "tensor_sha256": _sha_array(table),
            "shape": list(table.shape),
            "dtype": str(table.dtype),
            "identity": "pr31-affine-gaussian-edge-v1" if compact_pr31 else "q9-v2-v4",
            **(
                {
                    "phase_widths": list(native_v5_phase_widths(geometry=geometry)),
                    "multiplier": BANANA_V1_MULTIPLIER,
                    "offset": BANANA_V1_OFFSET,
                }
                if compact_pr31
                else {}
            ),
        },
        "normalized_tensor_sha256": normalized_sha256,
        "decode_validation": {
            "tensor_sha256": _sha_array(decoded),
            "materialized": materialize_decoded,
        },
        "optimization": optimization,
        "accounting": {
            "weights": weights,
            "exact_code_bits": code_bits,
            "exact_code_bpw": geometry.rate_num / geometry.rate_den,
            "code_data_bytes": int(packed.nbytes),
            "transform_bytes": transform_bytes,
            "Wscale_bytes": wscale_bytes,
            "shared_tlut_bytes": int(table.nbytes),
            "assignment_map_bytes": 0,
            "routing_bytes": 0,
            "full_cell_wire_bytes_including_shared_tlut": full_wire_bytes,
            "full_cell_wire_bpw_including_shared_tlut": full_wire_bytes * 8 / weights,
        },
        "direct_error": {"sse": sse, "mse": sse / weights},
        "encode": {
            "wall_seconds": encode_seconds,
            "weights_per_second": weights / encode_seconds,
        },
        "artifacts": {
            "codes": _artifact(codes_path, data_bytes=int(packed.nbytes)),
            **(
                {"decoded": _artifact(decoded_path, data_bytes=int(decoded.nbytes))}
                if materialize_decoded
                else {}
            ),
            "SU": _artifact(su_path, data_bytes=int(compact["SU_storage"].nbytes)),
            "SV": _artifact(sv_path, data_bytes=int(compact["SV_storage"].nbytes)),
            "Wscale": _artifact(wscale_path, data_bytes=wscale_bytes),
            **(
                {
                    "cuda_receipt": _artifact(
                        output_root / "NATIVE_V4_CELL_RECEIPT.json"
                    )
                }
                if cuda_receipt is not None
                else {}
            ),
        },
        **(
            {
                "installed_cuda_decode": cuda_receipt["installed_cuda_decode"],
                "cuda": cuda_receipt["cuda"],
            }
            if cuda_receipt is not None
            else {}
        ),
    }
    receipt_path = output_root / "CELL_RECEIPT.json"
    receipt["receipt"] = str(receipt_path)
    receipt["receipt_sha256"] = _atomic_json(receipt_path, receipt)
    return receipt


def build_qtip_native_v4_cell(
    source: str | Path,
    control: str | Path,
    tlut: str | Path,
    output: str | Path,
    *,
    bpw: object,
    intended_basis_sha256: str,
    observed_basis_sha256: str,
    backend: Literal["cuda", "reference"] = "cuda",
    solve_batch: int = 2048,
    decode_batch: int = 2048,
    decode_repeats: int = 1,
    codec_version: Literal["v4", "v5", "v6"] = "v4",
    materialize_decoded: bool = True,
    hessian: str | Path | None = None,
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
    ldlq_scale_semantics: Literal[
        "relative_search", "absolute_unit", "rms_ratio"
    ] = "absolute_unit",
    feedback_mode: Literal["off", "reverse_16"] = "off",
    trellis_objective: Literal["sse", "lexicographic_l4"] = "sse",
) -> dict[str, Any]:
    """Build one homogeneous native-V4 cell at an exact quarter-BPW rate."""

    return _build_qtip_native_v4_cell(
        source,
        control,
        tlut,
        output,
        bpw=bpw,
        receipt_schema=GENERIC_CELL_SCHEMA,
        codec_version=codec_version,
        intended_basis_sha256=intended_basis_sha256,
        observed_basis_sha256=observed_basis_sha256,
        backend=backend,
        solve_batch=solve_batch,
        decode_batch=decode_batch,
        decode_repeats=decode_repeats,
        materialize_decoded=materialize_decoded,
        hessian=hessian,
        scale_factors=scale_factors,
        ldlq_scale_semantics=ldlq_scale_semantics,
        feedback_mode=feedback_mode,
        trellis_objective=trellis_objective,
    )


def build_qtip25_native_v4_cell(
    source: str | Path,
    control: str | Path,
    tlut: str | Path,
    output: str | Path,
    *,
    intended_basis_sha256: str,
    observed_basis_sha256: str,
    backend: Literal["cuda", "reference"] = "cuda",
    solve_batch: int = 2048,
    decode_batch: int = 2048,
    decode_repeats: int = 1,
    materialize_decoded: bool = True,
    hessian: str | Path | None = None,
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
    """Backward-compatible fixed-2.50 wrapper around the generic native-V4 API."""

    return _build_qtip_native_v4_cell(
        source,
        control,
        tlut,
        output,
        bpw=NATIVE_QTIP25_GEOMETRY.rate_num / NATIVE_QTIP25_GEOMETRY.rate_den,
        receipt_schema=CELL_SCHEMA,
        intended_basis_sha256=intended_basis_sha256,
        observed_basis_sha256=observed_basis_sha256,
        backend=backend,
        solve_batch=solve_batch,
        decode_batch=decode_batch,
        decode_repeats=decode_repeats,
        materialize_decoded=materialize_decoded,
        hessian=hessian,
        scale_factors=scale_factors,
    )


def _anchor_qtip_native_v4_cell(
    candidate: str | Path,
    *,
    anchor_bank: str | Path,
    teacher: str | Path,
    output: str | Path,
    candidate_schemas: frozenset[str],
    receipt_schema: str,
) -> dict[str, Any]:
    """Measure one built V4 cell with Banana's standard 64-window anchor."""

    candidate_path = Path(candidate).expanduser().resolve()
    receipt_path = candidate_path / "CELL_RECEIPT.json" if candidate_path.is_dir() else candidate_path
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError(f"native V4 candidate receipt is missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") not in candidate_schemas or receipt.get("status") != "PASS":
        raise ValueError("native V4 candidate receipt is incompatible or incomplete")
    decoded_path = Path(receipt["artifacts"]["decoded"]["path"])
    if _sha_file(decoded_path) != receipt["artifacts"]["decoded"]["sha256"]:
        raise ValueError("native V4 decoded artifact hash mismatch")
    decoded = np.load(decoded_path, allow_pickle=False).astype(np.float32).reshape(-1)

    teacher_path = Path(teacher).expanduser().resolve()
    teacher_weights = np.load(teacher_path, allow_pickle=False)
    if (
        teacher_weights.dtype != np.float32
        or teacher_weights.size != decoded.size
        or not np.isfinite(teacher_weights).all()
    ):
        raise ValueError("native V4 anchor teacher must be matching finite float32 NPY")
    bank_path = Path(anchor_bank).expanduser().resolve()
    with np.load(bank_path, allow_pickle=False) as bank:
        if set(bank.files) != {"features", "classes"}:
            raise ValueError("native V4 anchor bank must contain exactly features and classes")
        features = np.asarray(bank["features"], dtype=np.float32)
        classes = np.asarray(bank["classes"])
    if features.shape != (64, decoded.size) or classes.shape != (64,):
        raise ValueError(f"native V4 anchor bank must be features[64,{decoded.size}] and classes[64]")
    class_rows = [str(value) for value in classes.tolist()]
    if set(class_rows) != set(CLASSES):
        raise ValueError("native V4 anchor bank must cover Banana's six anchor classes")

    metrics = _anchor_metrics(
        features,
        np.asarray(class_rows),
        teacher_weights.reshape(-1),
        decoded,
    )
    output_path = Path(output).expanduser().resolve()
    payload: dict[str, Any] = {
        "schema": receipt_schema,
        "status": "PASS",
        "same_instrument": True,
        "windows": 64,
        "candidate_receipt": str(receipt_path),
        "candidate_receipt_sha256": _sha_file(receipt_path),
        "anchor_bank": str(bank_path),
        "anchor_bank_sha256": _sha_file(bank_path),
        "teacher": str(teacher_path),
        "teacher_sha256": _sha_file(teacher_path),
        "geometry": receipt["geometry"],
        "bpw": receipt["accounting"]["exact_code_bpw"],
        "metrics": metrics,
    }
    payload["receipt"] = str(output_path)
    payload["receipt_sha256"] = _atomic_json(output_path, payload)
    return payload


def anchor_qtip_native_v4_cell(
    candidate: str | Path,
    *,
    anchor_bank: str | Path,
    teacher: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Anchor one generic homogeneous native-V4 cell."""

    return _anchor_qtip_native_v4_cell(
        candidate,
        anchor_bank=anchor_bank,
        teacher=teacher,
        output=output,
        candidate_schemas=frozenset((GENERIC_CELL_SCHEMA, CELL_SCHEMA)),
        receipt_schema=GENERIC_ANCHOR_SCHEMA,
    )


def anchor_qtip25_native_v4_cell(
    candidate: str | Path,
    *,
    anchor_bank: str | Path,
    teacher: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Backward-compatible Anchor64 wrapper for fixed native QTIP2.5 cells."""

    return _anchor_qtip_native_v4_cell(
        candidate,
        anchor_bank=anchor_bank,
        teacher=teacher,
        output=output,
        candidate_schemas=frozenset((CELL_SCHEMA,)),
        receipt_schema=ANCHOR_SCHEMA,
    )


def build_qtip_native_v4_anchor_set(
    source: str | Path,
    control: str | Path,
    tlut: str | Path,
    output: str | Path,
    *,
    bpws: Sequence[object],
    anchor_bank: str | Path,
    teacher: str | Path,
    intended_basis_sha256: str,
    observed_basis_sha256: str,
    backend: Literal["cuda", "reference"] = "cuda",
    solve_batch: int = 2048,
    decode_batch: int = 2048,
    decode_repeats: int = 1,
) -> dict[str, Any]:
    """Build and Anchor64 any declared set of homogeneous quarter-rate V4 tiers."""

    geometries = [native_v4_geometry(value) for value in bpws]
    if not geometries:
        raise ValueError("native V4 anchor set requires at least one bpw")
    transition_bits = [geometry.B for geometry in geometries]
    if len(set(transition_bits)) != len(transition_bits):
        raise ValueError("native V4 anchor set contains duplicate quarter-rate tiers")

    output_root = Path(output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for geometry in geometries:
        tier_root = output_root / f"b{geometry.B:02d}"
        candidate = build_qtip_native_v4_cell(
            source,
            control,
            tlut,
            tier_root / "candidate",
            bpw=geometry.rate_num / geometry.rate_den,
            intended_basis_sha256=intended_basis_sha256,
            observed_basis_sha256=observed_basis_sha256,
            backend=backend,
            solve_batch=solve_batch,
            decode_batch=decode_batch,
            decode_repeats=decode_repeats,
        )
        anchor = anchor_qtip_native_v4_cell(
            tier_root / "candidate",
            anchor_bank=anchor_bank,
            teacher=teacher,
            output=tier_root / "ANCHOR.json",
        )
        rows.append(
            {
                "tier": f"qtip-native-v4-b{geometry.B}",
                "bpw": geometry.rate_num / geometry.rate_den,
                "geometry": geometry.as_mapping(),
                "candidate": candidate,
                "anchor": anchor,
            }
        )

    receipt_path = output_root / "ANCHOR_SET.json"
    result: dict[str, Any] = {
        "schema": ANCHOR_SET_SCHEMA,
        "status": "PASS",
        "same_instrument": True,
        "basis_sha256": intended_basis_sha256,
        "tiers": rows,
        "receipt": str(receipt_path),
    }
    result["receipt_sha256"] = _atomic_json(receipt_path, result)
    return result
