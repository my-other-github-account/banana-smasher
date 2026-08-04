from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_SCHEMA = "banana-smasher-fixed-d4-materialization-v1"
_PROJECTIONS = ("down", "fused13")
_TIER_SPECS = {
    "d4_k2048": {"k": 2048, "bits": 11},
    "d4_k4096": {"k": 4096, "bits": 12},
}
_HEX = frozenset("0123456789abcdef")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and set(value) <= _HEX
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
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


def _bound_array(root: Path, value: object, *, label: str) -> tuple[np.ndarray, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a bound array object")
    relative = value.get("path")
    expected_bytes = value.get("bytes")
    expected_sha = value.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label}.path must be a relative file")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{label}.path escapes the manifest root")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 1
        or not _is_sha256(expected_sha)
    ):
        raise ValueError(f"{label} requires bytes and lowercase SHA-256")
    path = root / relative_path
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{label} byte count mismatch")
    if _sha256_file(path) != expected_sha:
        raise ValueError(f"{label} SHA-256 mismatch")
    value_array = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(value_array, np.ndarray):
        raise ValueError(f"{label} must be one NPY array")
    return value_array, {
        "path": relative_path.as_posix(),
        "bytes": expected_bytes,
        "sha256": expected_sha,
    }


def _verify_basis_index(root: Path, value: object, *, basis_sha256: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("basis_index must be a bound source-model index")
    declared_sha = value.get("sha256")
    if declared_sha != basis_sha256:
        raise ValueError("basis_index SHA-256 must equal basis_sha256")
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("basis_index.path must be a non-empty string")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root / path
    try:
        actual_sha = _sha256(path.read_bytes())
    except OSError as exc:
        raise ValueError(f"cannot read basis_index {path}: {exc}") from exc
    if actual_sha != basis_sha256:
        raise ValueError(
            f"basis_index SHA-256 mismatch: expected {basis_sha256}, got {actual_sha}"
        )


def _packed_assignments(
    assignments: np.ndarray,
    *,
    label: str,
    k: int,
    bits_per_assignment: int,
) -> bytes:
    _validate_assignments(assignments, label=label, k=k)
    values = assignments.astype(np.int64, copy=False)
    rows = values.astype(np.uint16).reshape(256, -1)
    bits = (
        (rows[..., None] >> np.arange(bits_per_assignment, dtype=np.uint16)) & 1
    ).astype(np.uint8)
    packed = np.packbits(bits.reshape(256, -1), axis=1, bitorder="little")
    return packed.tobytes(order="C")


def _validate_assignments(assignments: np.ndarray, *, label: str, k: int) -> None:
    if assignments.ndim < 2 or assignments.shape[0] != 256:
        raise ValueError(f"{label} must have 256 expert rows")
    if assignments.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{label} must have an integer dtype")
    rows = assignments.reshape(256, -1)
    for expert in range(256):
        row = rows[expert]
        for start in range(0, row.size, 1 << 20):
            chunk = row[start : start + (1 << 20)]
            if chunk.size and (int(chunk.min()) < 0 or int(chunk.max()) >= k):
                raise ValueError(f"{label} contains an assignment outside D4K{k}")


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _atomic_save_npy(path: Path, value: np.ndarray) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
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


def persist_fixed_d4_solve(
    output_root: str | Path,
    *,
    tier: str,
    layer: int,
    basis_index: str | Path,
    basis_sha256: str,
    projections: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    """Persist exact in-memory solve winners before the solver releases them.

    Exact solvers call this once per completed layer while their winner arrays
    are still available.  The durable result is the ordinary public
    materialization manifest consumed by :func:`materialize_fixed_d4`.
    """

    if tier not in _TIER_SPECS:
        raise ValueError("fixed D4 solve persistence requires tier d4_k2048 or d4_k4096")
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise ValueError("fixed D4 solve persistence layer must be non-negative")
    if not _is_sha256(basis_sha256):
        raise ValueError("basis_sha256 must be a lowercase SHA-256")
    basis_index = Path(basis_index).expanduser().resolve()
    basis_payload = basis_index.read_bytes()
    if _sha256(basis_payload) != basis_sha256:
        raise ValueError("fixed D4 solve persistence basis_index SHA-256 mismatch")
    if set(projections) != set(_PROJECTIONS):
        raise ValueError("fixed D4 solve persistence requires down and fused13 projections")

    spec = _TIER_SPECS[tier]
    k = spec["k"]
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    assignment_count = 0
    try:
        basis_name = "model.safetensors.index.json"
        _atomic_write(staging / basis_name, basis_payload)
        persisted: dict[str, dict[str, dict[str, object]]] = {}
        for projection in _PROJECTIONS:
            row = projections[projection]
            if not isinstance(row, Mapping) or set(row) != {
                "assignments",
                "scales",
                "codebook",
            }:
                raise ValueError(
                    f"fixed D4 solve projection {projection} requires assignments, scales, and codebook"
                )
            assignments = np.asarray(row["assignments"])
            scales = np.asarray(row["scales"])
            codebook = np.asarray(row["codebook"])
            _validate_assignments(
                assignments, label=f"projections.{projection}.assignments", k=k
            )
            if scales.dtype != np.uint8 or scales.ndim < 2 or scales.shape[0] != 256:
                raise ValueError(
                    f"projections.{projection}.scales must be uint8 with 256 expert rows"
                )
            if codebook.shape != (k, 4) or codebook.dtype.kind != "f":
                raise ValueError(
                    f"projections.{projection}.codebook must be floating [{k}, 4]"
                )
            persisted[projection] = {}
            for field, value in (
                ("assignments", assignments),
                ("scales", scales),
                ("codebook", codebook),
            ):
                name = f"{projection}.{field}.npy"
                path = staging / name
                _atomic_save_npy(path, value)
                persisted[projection][field] = {
                    "path": name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            assignment_count += int(assignments.size)

        manifest = {
            "schema": _SCHEMA,
            "tier": tier,
            "layer": layer,
            "basis_sha256": basis_sha256,
            "basis_index": {"path": basis_name, "sha256": basis_sha256},
            "projections": persisted,
        }
        _atomic_write(
            staging / "materialize.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
        )
        os.rename(staging, output_root)
        directory = os.open(output_root.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {
        "schema": "banana-smasher-fixed-d4-solve-persistence-v1",
        "status": "PASS",
        "tier": tier,
        "layer": layer,
        "basis_sha256": basis_sha256,
        "assignment_count": assignment_count,
        "manifest": str(output_root / "materialize.json"),
    }


def materialize_fixed_d4(
    manifest_path: str | Path,
    output_root: str | Path,
    *,
    basis_sha256: str,
) -> dict[str, Any]:
    """Materialize bound full fixed-D4 assignments as executable bs-pack wire planes."""

    if not _is_sha256(basis_sha256):
        raise ValueError("basis_sha256 must be a lowercase SHA-256")
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest_payload = manifest_path.read_bytes()
    manifest = json.loads(manifest_payload)
    if not isinstance(manifest, dict) or manifest.get("schema") != _SCHEMA:
        raise ValueError(f"materialization manifest schema must be {_SCHEMA}")
    tier = manifest.get("tier")
    if tier not in _TIER_SPECS:
        raise ValueError("fixed D4 materialization requires tier d4_k2048 or d4_k4096")
    spec = _TIER_SPECS[str(tier)]
    k = spec["k"]
    bits_per_assignment = spec["bits"]
    if manifest.get("basis_sha256") != basis_sha256:
        raise ValueError("fixed D4 materialization basis mismatch")
    _verify_basis_index(
        manifest_path.parent,
        manifest.get("basis_index"),
        basis_sha256=basis_sha256,
    )
    layer = manifest.get("layer")
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise ValueError("fixed D4 materialization layer must be non-negative")
    projections = manifest.get("projections")
    if not isinstance(projections, Mapping) or set(projections) != set(_PROJECTIONS):
        raise ValueError("fixed D4 materialization requires down and fused13 projections")

    output_root = Path(output_root).expanduser().resolve()
    layer_root = output_root / f"layer_{layer:03d}"
    if layer_root.exists():
        raise FileExistsError(layer_root)
    layer_root.mkdir(parents=True)

    files: list[dict[str, object]] = []
    sources: dict[str, Any] = {}
    assignment_count = 0
    for projection in _PROJECTIONS:
        row = projections[projection]
        if not isinstance(row, Mapping):
            raise ValueError(f"projection {projection} must be an object")
        assignments, assignment_source = _bound_array(
            manifest_path.parent,
            row.get("assignments"),
            label=f"projections.{projection}.assignments",
        )
        scales, scale_source = _bound_array(
            manifest_path.parent,
            row.get("scales"),
            label=f"projections.{projection}.scales",
        )
        codebook, codebook_source = _bound_array(
            manifest_path.parent,
            row.get("codebook"),
            label=f"projections.{projection}.codebook",
        )
        if scales.dtype != np.uint8 or scales.ndim < 2 or scales.shape[0] != 256:
            raise ValueError(f"projections.{projection}.scales must be uint8 with 256 expert rows")
        if codebook.shape != (k, 4) or codebook.dtype.kind != "f":
            raise ValueError(
                f"projections.{projection}.codebook must be floating [{k}, 4]"
            )

        payloads = {
            f"{tier}.{projection}.codebook.fp16.bin": np.ascontiguousarray(
                codebook, dtype="<f2"
            ).tobytes(),
            f"{tier}.{projection}.codes.le{bits_per_assignment}.bin": _packed_assignments(
                assignments,
                label=f"projections.{projection}.assignments",
                k=k,
                bits_per_assignment=bits_per_assignment,
            ),
            f"{tier}.{projection}.expert_ids.i16.bin": np.arange(
                256, dtype="<i2"
            ).tobytes(),
            f"{tier}.{projection}.scales.e8m0.bin": np.ascontiguousarray(
                scales, dtype=np.uint8
            ).tobytes(),
        }
        for name, payload in payloads.items():
            path = layer_root / name
            _atomic_write(path, payload)
            files.append(_file_record(path))
        assignment_count += int(assignments.size)
        sources[projection] = {
            "assignments": assignment_source,
            "scales": scale_source,
            "codebook": codebook_source,
            "assignment_count": int(assignments.size),
            "assignment_dtype": str(assignments.dtype),
        }

    receipt = {
        "schema": "banana_smasher-materialized-layer-v1",
        "status": "PASS",
        "layer": layer,
        "tier": tier,
        "basis_sha256": basis_sha256,
        "basis_index_sha256": basis_sha256,
        "materialization_manifest_sha256": _sha256(manifest_payload),
        "assignment_count": assignment_count,
        "assignment_dtype": f"source-bound-integer; wire=le{bits_per_assignment}",
        "sources": sources,
        "files": sorted(files, key=lambda row: str(row["path"])),
    }
    _atomic_write(
        layer_root / "LAYER_RECEIPT.json",
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
    )
    return {
        "schema": "banana-smasher-fixed-d4-materialization-receipt-v1",
        "status": "PASS",
        "layer": layer,
        "tier": tier,
        "basis_sha256": basis_sha256,
        "assignment_count": assignment_count,
        "output": str(layer_root),
        "receipt": str(layer_root / "LAYER_RECEIPT.json"),
    }
