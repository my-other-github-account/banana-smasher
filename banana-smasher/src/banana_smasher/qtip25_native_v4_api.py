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


def qtip_transform_seed(
    seed_domain: str,
    seed_material: str,
    layer: int,
    expert: int,
    projection: Literal["down", "fused13"],
) -> int:
    """Derive the public QTIP transform seed for one routed cell."""

    if not seed_domain or not seed_material:
        raise ValueError("QTIP transform seed domain and material must be non-empty")
    if layer < 0 or expert < 0 or projection not in {"down", "fused13"}:
        raise ValueError("invalid routed-cell identity for QTIP transform seed")
    identity = f"L{layer:03d}_E{expert:03d}_{projection}"
    digest = hashlib.sha256(
        f"{seed_domain}|{seed_material}|{identity}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def build_qtip_native_transform_control(
    source: str | Path,
    output: str | Path,
    *,
    transform_seed: int,
    intended_basis_sha256: str,
    observed_basis_sha256: str,
    device: Literal["cpu", "cuda"] = "cuda",
    qtip_k: int = 3,
) -> dict[str, Any]:
    """Materialize a deterministic source-shaped QTIP transform control.

    This is the public recovery seam for a missing historical ``QTIP_UNIT.pt``
    when only its transform role is consumed by a newer native-QTIP build.  It
    recreates the proven seeded RHT sign mechanism and uses unit ``Wscale``;
    the native V6 ``rms_ratio`` path derives and serializes the physical scale.
    """

    intended = _basis(intended_basis_sha256, "intended_basis_sha256")
    observed = _basis(observed_basis_sha256, "observed_basis_sha256")
    if intended != observed:
        raise ValueError(f"native QTIP control basis mismatch: {observed} != {intended}")
    if (
        isinstance(transform_seed, bool)
        or not isinstance(transform_seed, int)
        or not 0 <= transform_seed < (1 << 63)
    ):
        raise ValueError("transform_seed must be an integer in [0, 2^63)")
    if qtip_k < 1:
        raise ValueError("qtip_k must be positive")
    source_path = Path(source).expanduser().resolve()
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError(f"native QTIP control source must be a regular NPY file: {source_path}")
    weights = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if (
        weights.dtype != np.float32
        or weights.ndim != 2
        or any(int(value) <= 0 or int(value) % 16 for value in weights.shape)
    ):
        raise ValueError(
            "native QTIP control source must be a tile-compatible float32 matrix"
        )
    rows, columns = (int(value) for value in weights.shape)
    del weights

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("building a QTIP transform control requires torch") from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for production QTIP transform controls")
    if device not in {"cpu", "cuda"}:
        raise ValueError("QTIP transform-control device must be cpu or cuda")
    fork_devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(transform_seed)
        su = (torch.randn(columns, device=device).sign() + 1e-5).sign().half().cpu()
        sv = (torch.randn(rows, device=device).sign() + 1e-5).sign().half().cpu()
    payload = {
        "schema": "banana-smasher-source-transform-control-v1",
        "basis_sha256": intended,
        "transform_seed": transform_seed,
        "shape": [rows, columns],
        "SU": su,
        "SV": sv,
        "Wscale": torch.tensor(1.0, dtype=torch.float32),
        "qtip_k": qtip_k,
    }
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if output_path.exists():
        existing, _ = _load_control(output_path)
        if (
            existing["shape"] != (rows, columns)
            or not np.array_equal(existing["SU_storage"], su.numpy())
            or not np.array_equal(existing["SV_storage"], sv.numpy())
            or float(existing["Wscale"]) != 1.0
        ):
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"existing QTIP transform control differs: {output_path}")
        temporary.unlink(missing_ok=True)
    else:
        os.replace(temporary, output_path)
        directory_fd = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return {
        "schema": "banana-smasher-source-transform-control-receipt-v1",
        "status": "PASS",
        "basis_sha256": intended,
        "transform_seed": transform_seed,
        "shape": [rows, columns],
        "qtip_k": qtip_k,
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": _sha_file(output_path),
    }


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


def _hessian_regularization_sigma(
    *, codec_version: Literal["v4", "v5", "v6"], hadamard_block: object | None
) -> float:
    """Return the fixed diagonal load for one native-QTIP encoder generation."""

    if codec_version == "v6":
        return 1.0
    return 0.025 if hadamard_block is not None else 0.01


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
    cyclic_fixed_point_fast_path: bool = False,
    reserve_bytes: int = 4 << 30,
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
    pr31_v5 = table.dtype == np.float16 and table.shape == (1024,)
    if not (old_v4 or pr31_v5) or not np.isfinite(table).all():
        raise ValueError(
            "native LUT must be finite V4 float32 [512,2] or compact PR31 float16 [1024]"
        )
    if codec_version == "v4" and not old_v4:
        raise ValueError("native QTIP codec_version=v4 requires lut_identity=q9-v2-v4")
    if codec_version in {"v5", "v6"} and not (pr31_v5 or old_v4):
        raise ValueError(
            f"native QTIP codec_version={codec_version} requires a supported V4/PR31 LUT"
        )
    if codec_version == "v5" and geometry.B != 8:
        raise ValueError("native QTIP codec_version=v5 is fixed to B8")
    if trellis_objective != "sse" and (not pr31_v5 or feedback_mode != "off"):
        raise ValueError(
            "lexicographic L4 requires the compact PR31 LUT with feedback disabled"
        )

    output_root = Path(output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    blocks = _to_normalized_blocks(source_weights, compact)
    normalized_sha256 = _sha_array(blocks)
    effective_hessian = Path(hessian).expanduser().resolve() if hessian is not None else None
    transformed_hessian_path: Path | None = None
    if compact.get("hadamard_block") is not None and feedback_mode == "reverse_16":
        assert effective_hessian is not None
        hessian_value = np.load(effective_hessian, allow_pickle=False).astype(
            np.float32, copy=True
        )
        signs = -np.sign(np.asarray(compact["SU"], dtype=np.float32))
        block = compact.get("hadamard_block")
        hessian_value *= signs[None, :]
        hessian_value = _fwht_blocks(hessian_value, block)
        hessian_value *= signs[:, None]
        hessian_value = _fwht_blocks(hessian_value.T, block).T
        transformed_hessian_path = output_root / ".normalized-hessian.npy"
        np.save(transformed_hessian_path, hessian_value, allow_pickle=False)
        effective_hessian = transformed_hessian_path
    selected_scale = 1.0
    cyclic_warmup_cycles = 2 if codec_version == "v6" else 1
    if ldlq_scale_semantics == "rms_ratio":
        expanded_lut = (
            expand_banana_v1_codebook(table)
            if pr31_v5
            else expand_native_v4_tlut(table, geometry=geometry)
        )
        source_rms = float(np.sqrt(np.mean(blocks.astype(np.float64) ** 2)))
        lut_rms = float(np.sqrt(np.mean(expanded_lut.astype(np.float64) ** 2)))
        selected_scale = 1.0 if source_rms == 0 else source_rms / lut_rms
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
                hessian_path=(
                    effective_hessian if feedback_mode == "reverse_16" else None
                ),
                matrix_shape=compact["shape"],
                scale_factors=scale_factors,
                ldlq_scale_semantics=ldlq_scale_semantics,
                feedback_mode=feedback_mode,
                hessian_regularization_sigma=_hessian_regularization_sigma(
                    codec_version=codec_version,
                    hadamard_block=compact.get("hadamard_block"),
                ),
                cyclic_warmup_cycles=cyclic_warmup_cycles,
                trellis_objective=trellis_objective,
                cyclic_fixed_point_fast_path=cyclic_fixed_point_fast_path,
                reserve_bytes=reserve_bytes,
            )
        finally:
            normalized_path.unlink(missing_ok=True)
            if transformed_hessian_path is not None:
                transformed_hessian_path.unlink(missing_ok=True)
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
            assert effective_hessian is not None
            hessian_path = effective_hessian
            hessian_value = np.load(hessian_path, allow_pickle=False)
            lower = native_v4_lower_from_hessian(
                hessian_value,
                regularization_sigma=_hessian_regularization_sigma(
                    codec_version=codec_version,
                    hadamard_block=compact.get("hadamard_block"),
                ),
            )
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
                scale_semantics=ldlq_scale_semantics,
            )
            packed = matrix.packed
            optimization = {
                "method": "qtip_batch_block_ldl_reverse_16",
                "feedback_mode": "reverse_16",
                "scale_semantics": ldlq_scale_semantics,
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
    np.save(decoded_path, decoded, allow_pickle=False)
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
            "identity": "pr31-affine-gaussian-edge-v1" if pr31_v5 else "q9-v2-v4",
            **(
                {
                    "phase_widths": list(native_v5_phase_widths(geometry=geometry)),
                    "multiplier": BANANA_V1_MULTIPLIER,
                    "offset": BANANA_V1_OFFSET,
                }
                if pr31_v5
                else {}
            ),
        },
        "normalized_tensor_sha256": normalized_sha256,
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
            "decoded": _artifact(decoded_path, data_bytes=int(decoded.nbytes)),
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
    reserve_bytes: int = 4 << 30,
) -> dict[str, Any]:
    """Build one homogeneous native-V4 cell at an exact quarter-BPW rate."""

    return _build_qtip_native_v4_cell(
        source,
        control,
        tlut,
        output,
        bpw=bpw,
        receipt_schema=GENERIC_CELL_SCHEMA,
        codec_version="v4",
        intended_basis_sha256=intended_basis_sha256,
        observed_basis_sha256=observed_basis_sha256,
        backend=backend,
        solve_batch=solve_batch,
        decode_batch=decode_batch,
        decode_repeats=decode_repeats,
        hessian=hessian,
        scale_factors=scale_factors,
        ldlq_scale_semantics=ldlq_scale_semantics,
        feedback_mode=feedback_mode,
        reserve_bytes=reserve_bytes,
    )


def build_qtip_native_cell(
    source: str | Path,
    control: str | Path,
    tlut: str | Path,
    output: str | Path,
    *,
    bpw: object,
    intended_basis_sha256: str,
    observed_basis_sha256: str,
    codec_version: Literal["v4", "v5", "v6"] = "v6",
    backend: Literal["cuda", "reference"] = "cuda",
    solve_batch: int = 2048,
    decode_batch: int = 2048,
    decode_repeats: int = 1,
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
    ]
    | None = None,
    feedback_mode: Literal["off", "reverse_16"] = "off",
    trellis_objective: Literal["sse", "lexicographic_l4"] = "sse",
    cyclic_fixed_point_fast_path: bool = True,
    reserve_bytes: int = 4 << 30,
) -> dict[str, Any]:
    """Build one native QTIP cell, defaulting to the V6 PR31-edge codec."""

    if codec_version not in {"v4", "v5", "v6"}:
        raise ValueError("native QTIP codec_version must be v4, v5, or v6")
    scale_semantics = ldlq_scale_semantics or (
        "rms_ratio" if codec_version == "v6" else "absolute_unit"
    )
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
        hessian=hessian,
        scale_factors=scale_factors,
        ldlq_scale_semantics=scale_semantics,
        feedback_mode=feedback_mode,
        trellis_objective=trellis_objective,
        cyclic_fixed_point_fast_path=cyclic_fixed_point_fast_path,
        reserve_bytes=reserve_bytes,
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
) -> dict[str, Any]:
    """Backward-compatible fixed-2.50 wrapper around the generic native-V4 API."""

    return _build_qtip_native_v4_cell(
        source,
        control,
        tlut,
        output,
        bpw=NATIVE_QTIP25_GEOMETRY.rate_num / NATIVE_QTIP25_GEOMETRY.rate_den,
        receipt_schema=CELL_SCHEMA,
        codec_version="v4",
        intended_basis_sha256=intended_basis_sha256,
        observed_basis_sha256=observed_basis_sha256,
        backend=backend,
        solve_batch=solve_batch,
        decode_batch=decode_batch,
        decode_repeats=decode_repeats,
        hessian=hessian,
        scale_factors=scale_factors,
        ldlq_scale_semantics=ldlq_scale_semantics,
        feedback_mode=feedback_mode,
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


def build_qtip_native_cells(
    cells: Sequence[Mapping[str, str | Path]],
    tlut: str | Path,
    *,
    bpw: object,
    intended_basis_sha256: str,
    observed_basis_sha256: str,
    codec_version: Literal["v6"] = "v6",
    backend: Literal["cuda"] = "cuda",
    solve_batch: int = 65_536,
    decode_batch: int = 65_536,
    decode_repeats: int = 1,
    scale_factors: Sequence[float] = (1.0,),
    ldlq_scale_semantics: Literal["rms_ratio"] = "rms_ratio",
    feedback_mode: Literal["off"] = "off",
    trellis_objective: Literal["sse"] = "sse",
    cyclic_fixed_point_fast_path: bool = True,
    reserve_bytes: int = 4 << 30,
) -> list[dict[str, Any]]:
    """Build independent QTIP3 cells in one GPU-resident recurrence call.

    Each mapping contains ``source``, ``control``, and ``output``.  This seam is
    intentionally fail-closed to the immutable QTIP3 B12/L16/V4 contract;
    :func:`build_qtip_native_cell` remains the scalar compatibility wrapper.
    """

    geometry = native_v4_geometry(bpw)
    intended = _basis(intended_basis_sha256, "intended_basis_sha256")
    observed = _basis(observed_basis_sha256, "observed_basis_sha256")
    if intended != observed:
        raise ValueError(f"native V4 basis mismatch: {observed} != {intended}")
    if (
        not cells
        or codec_version != "v6"
        or backend != "cuda"
        or (geometry.B, geometry.L, geometry.V) != (12, 16, 4)
        or tuple(float(value) for value in scale_factors) != (1.0,)
        or ldlq_scale_semantics != "rms_ratio"
        or feedback_mode != "off"
        or trellis_objective != "sse"
    ):
        raise ValueError("QTIP3 batch API is fixed to CUDA v6 B12/L16/V4 rms_ratio")

    tlut_path = Path(tlut).expanduser().resolve()
    table = np.load(tlut_path, allow_pickle=False)
    if table.dtype != np.float32 or table.shape != (512, 2) or not np.isfinite(table).all():
        raise ValueError("QTIP3 batch API requires finite float32 TLUT [512,2]")
    expanded = expand_native_v4_tlut(table, geometry=geometry)
    lut_rms = float(np.sqrt(np.mean(expanded.astype(np.float64) ** 2)))
    prepared: list[dict[str, Any]] = []
    for index, item in enumerate(cells):
        if set(item) != {"source", "control", "output"}:
            raise ValueError(f"QTIP3 batch cell {index} must contain source/control/output")
        source_path = Path(item["source"]).expanduser().resolve()
        output_root = Path(item["output"]).expanduser().resolve()
        source_weights = np.load(source_path, allow_pickle=False)
        compact, control_path = _load_control(item["control"])
        if source_weights.dtype != np.float32 or source_weights.shape != compact["shape"] or not np.isfinite(source_weights).all():
            raise ValueError(f"QTIP3 batch source {index} has incompatible shape/dtype")
        blocks = _to_normalized_blocks(source_weights, compact)
        source_rms = float(np.sqrt(np.mean(blocks.astype(np.float64) ** 2)))
        scale = 1.0 if source_rms == 0 else source_rms / lut_rms
        prepared.append({
            "source_path": source_path, "source_weights": source_weights,
            "control_path": control_path, "compact": compact, "blocks": blocks,
            "scale": scale, "output": output_root,
        })

    from .qtip25_native_v4_cuda_cell import run_cuda_cell
    import shutil

    batch_tmpdir = os.environ.get("QTIP3_BATCH_TMPDIR")
    if batch_tmpdir is not None:
        Path(batch_tmpdir).mkdir(parents=True, exist_ok=True)
    batch_parent = Path(tempfile.mkdtemp(prefix="banana-smasher-qtip3-batch-", dir=batch_tmpdir))
    normalized_path = batch_parent / "normalized.npy"
    cuda_root = batch_parent / "cuda"
    try:
        normalized = np.concatenate([row["blocks"] for row in prepared])
        sequence_scales = np.concatenate([
            np.full(len(row["blocks"]), row["scale"], dtype=np.float64)
            for row in prepared
        ])
        sequence_boundaries = np.cumsum(
            [len(row["blocks"]) for row in prepared], dtype=np.int64
        ).tolist()
        np.save(normalized_path, normalized, allow_pickle=False)
        cuda_receipt = run_cuda_cell(
            normalized_path, tlut_path, cuda_root,
            intended_basis_sha256=intended, observed_basis_sha256=observed,
            solve_batch=solve_batch, decode_batch=decode_batch,
            decode_repeats=decode_repeats, geometry=geometry,
            ldlq_scale_semantics="absolute_unit", feedback_mode="off",
            cyclic_warmup_cycles=2, trellis_objective="sse",
            cyclic_fixed_point_fast_path=cyclic_fixed_point_fast_path,
            reserve_bytes=reserve_bytes,
            sequence_scales=sequence_scales,
            sequence_boundaries=sequence_boundaries,
            defer_full_cuda_decode=True,
        )
        combined_packed = np.load(cuda_root / "codes.npy", allow_pickle=False)
        combined_decoded_blocks = decode_native_v4(
            combined_packed,
            sequence_scales.astype(np.float32),
            positions=256,
            tlut=table,
            geometry=geometry,
        ).reshape(-1, 64, 4)
        from concurrent.futures import ThreadPoolExecutor
        block_starts = np.concatenate((np.asarray([0]), np.cumsum(
            [len(row["blocks"]) for row in prepared], dtype=np.int64
        )))
        def finalize_decoded(index: int) -> np.ndarray:
            row = prepared[index]
            return _from_normalized_blocks(
                combined_decoded_blocks[block_starts[index] : block_starts[index + 1]],
                row["compact"],
            ).astype(np.float32)
        workers = min(int(os.environ.get("QTIP3_FINALIZE_WORKERS", "4")), len(prepared))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            decoded_cells = list(pool.map(finalize_decoded, range(len(prepared))))
        results: list[dict[str, Any]] = []
        cursor = 0
        total_blocks = len(combined_packed)
        for row_index, row in enumerate(prepared):
            count = len(row["blocks"])
            cell_start = cursor
            packed = np.ascontiguousarray(combined_packed[cell_start : cell_start + count])
            cursor += count
            compact = row["compact"]
            output_root = row["output"]
            output_root.mkdir(parents=True, exist_ok=True)
            codes_path, decoded_path = output_root / "codes.npy", output_root / "decoded.npy"
            su_path, sv_path, wscale_path = output_root / "SU.npy", output_root / "SV.npy", output_root / "Wscale.npy"
            np.save(codes_path, packed, allow_pickle=False)
            decoded = decoded_cells[row_index]
            np.save(decoded_path, decoded, allow_pickle=False)
            np.save(su_path, compact["SU_storage"], allow_pickle=False)
            np.save(sv_path, compact["SV_storage"], allow_pickle=False)
            np.save(wscale_path, np.asarray(compact["Wscale"] * row["scale"], dtype=np.float32), allow_pickle=False)
            source_weights = row["source_weights"]
            delta = decoded.astype(np.float64) - source_weights.astype(np.float64)
            sse = float(np.sum(delta * delta, dtype=np.float64))
            optimization = dict(cuda_receipt["optimization"])
            optimization.update({
                "method": "qtip3_cross_cell_cuda_float4",
                "base_scale": row["scale"], "selected_scale": row["scale"],
                "scale_factor": row["scale"], "scale_semantics": "rms_ratio",
                "cross_cell_batch_cells": len(prepared),
                "cross_cell_batch_sequences": total_blocks,
            })
            weights = int(source_weights.size)
            transform_bytes = int(compact["SU_storage"].nbytes + compact["SV_storage"].nbytes)
            wscale_bytes = int(np.asarray(compact["Wscale"], dtype=np.float32).nbytes)
            encode_seconds = float(cuda_receipt["encode"]["wall_seconds"]) * count / total_blocks
            full_wire_bytes = int(packed.nbytes + transform_bytes + wscale_bytes + table.nbytes)
            receipt: dict[str, Any] = {
                "schema": GENERIC_CELL_SCHEMA, "status": "PASS", "backend": "cuda",
                "codec_version": "v6", "provider": "qtip-native-v6@3.00",
                "basis_sha256": intended, "geometry": geometry.as_mapping(),
                "source": {**_artifact(row["source_path"], data_bytes=int(source_weights.nbytes)), "shape": list(source_weights.shape), "dtype": str(source_weights.dtype)},
                "control": {**_artifact(row["control_path"]), "shape": list(compact["shape"])},
                "tlut": {**_artifact(tlut_path, data_bytes=int(table.nbytes)), "tensor_sha256": _sha_array(table), "shape": list(table.shape), "identity": "q9-v2-v4"},
                "normalized_tensor_sha256": _sha_array(row["blocks"]),
                "optimization": optimization,
                "accounting": {"weights": weights, "exact_code_bits": weights * geometry.B // geometry.V, "exact_code_bpw": 3.0, "code_data_bytes": int(packed.nbytes), "transform_bytes": transform_bytes, "Wscale_bytes": wscale_bytes, "shared_tlut_bytes": int(table.nbytes), "assignment_map_bytes": 0, "routing_bytes": 0, "full_cell_wire_bytes_including_shared_tlut": full_wire_bytes, "full_cell_wire_bpw_including_shared_tlut": full_wire_bytes * 8 / weights},
                "direct_error": {"sse": sse, "mse": sse / weights},
                "encode": {"wall_seconds": encode_seconds, "weights_per_second": weights / encode_seconds},
                "artifacts": {"codes": _artifact(codes_path, data_bytes=int(packed.nbytes)), "decoded": _artifact(decoded_path, data_bytes=int(decoded.nbytes)), "SU": _artifact(su_path, data_bytes=int(compact["SU_storage"].nbytes)), "SV": _artifact(sv_path, data_bytes=int(compact["SV_storage"].nbytes)), "Wscale": _artifact(wscale_path, data_bytes=wscale_bytes)},
                "installed_cuda_decode": cuda_receipt["installed_cuda_decode"],
                "cuda": cuda_receipt["cuda"],
            }
            receipt_path = output_root / "CELL_RECEIPT.json"
            receipt["receipt"] = str(receipt_path)
            receipt["receipt_sha256"] = _atomic_json(receipt_path, receipt)
            results.append(receipt)
        if cursor != total_blocks:
            raise RuntimeError("QTIP3 batch split did not consume every packed sequence")
        return results
    finally:
        shutil.rmtree(batch_parent, ignore_errors=True)
