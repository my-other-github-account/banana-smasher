"""Public fixed-assignment QTIP3 artifacts and continuous-state repair."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .update_engine import run_segmented_update

QTIP3_FIXED_SCHEMA = "banana-smasher-qtip3-fixed-member-v1"
QTIP3_UPDATE_SCHEMA = "banana-smasher-qtip3-fixed-update-request-v1"
QTIP3_PROVIDER_ID = "periodic-qtip3@3.00"
QTIP3_LEGACY_PROVIDER_ID = "qtip-native-v6@3.00"
QTIP3_PROVIDER_IDS = frozenset((QTIP3_PROVIDER_ID, QTIP3_LEGACY_PROVIDER_ID))
QTIP3_GEOMETRY = {
    "L": 16,
    "B": 12,
    "V": 4,
    "layout": "homogeneous",
    "phase_widths": [3, 3, 3, 3],
}
QTIP3_LEGACY_GEOMETRY = {
    "L": 16,
    "B": 12,
    "V": 4,
    "phase_widths": [3, 3, 3, 3],
}


def _tensor_sha256(tensor: torch.Tensor) -> str:
    payload = (
        tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a SHA-256 hex digest") from exc
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_qtip3_fixed_manifest(
    *,
    cell_receipts: Sequence[str | Path],
    lut_source: str | Path,
    manifest_output: str | Path,
    terminal_output: str | Path,
    intended_basis_sha256: str,
    expected_identities: Sequence[tuple[int, int, str]],
) -> dict[str, Any]:
    """Bind physical cell NPYs into a streaming fixed-QTIP3 wire manifest."""

    intended_basis = _require_sha256(intended_basis_sha256, "intended basis SHA-256")
    expected = tuple(
        (int(layer), int(expert), str(projection))
        for layer, expert, projection in expected_identities
    )
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("QTIP3 fixed manifest expected identities must be non-empty and unique")
    if any(projection not in {"fused13", "down"} for _, _, projection in expected):
        raise ValueError("QTIP3 fixed manifest projection must be fused13 or down")

    lut_path = Path(lut_source).expanduser().resolve()
    lut = np.load(lut_path, mmap_mode="r", allow_pickle=False)
    lut_tensor_sha = hashlib.sha256(memoryview(lut).cast("B")).hexdigest()
    if lut.dtype != np.float16 or tuple(lut.shape) != (1024,):
        raise ValueError("QTIP3 fixed manifest LUT requires float16[1024]")
    lut_data_bytes = int(lut.nbytes)
    lut_activation = {
        "id": f"periodic-qtip3-pr31-lut-{lut_tensor_sha[:16]}",
        "bytes": lut_data_bytes,
        "sha256": lut_tensor_sha,
    }

    rows_by_identity: dict[tuple[int, int, str], dict[str, Any]] = {}
    totals = {
        "logical_weights": 0,
        "code_data_bytes": 0,
        "transform_data_bytes": 0,
        "wscale_data_bytes": 0,
    }
    for raw_receipt_path in cell_receipts:
        receipt_path = Path(raw_receipt_path).expanduser().resolve()
        receipt = json.loads(receipt_path.read_text())
        identity = (
            int(receipt.get("layer", -1)),
            int(receipt.get("expert", -1)),
            str(receipt.get("projection", "")),
        )
        geometry = receipt.get("geometry")
        accounting = receipt.get("accounting")
        artifacts = receipt.get("artifacts")
        if (
            receipt.get("schema") != "banana-smasher-periodic-qtip3-pr31-cell-v1"
            or receipt.get("status") != "PASS"
            or receipt.get("basis_index_sha256") != intended_basis
            or not isinstance(geometry, Mapping)
            or tuple(geometry.get(name) for name in ("L", "B", "V")) != (16, 12, 4)
            or geometry.get("rate_num") != 3
            or geometry.get("rate_den") != 1
            or geometry.get("phase_count") != 1
            or geometry.get("alternation") is not False
            or geometry.get("member_averaging") is not False
            or not isinstance(accounting, Mapping)
            or accounting.get("exact_code_bpw") != 3.0
            or not isinstance(artifacts, Mapping)
        ):
            raise ValueError(f"QTIP3 physical cell identity or geometry mismatch: {receipt_path}")
        if identity in rows_by_identity:
            raise ValueError(f"duplicate QTIP3 physical cell identity: {identity}")
        lut_binding = receipt.get("tlut")
        if (
            not isinstance(lut_binding, Mapping)
            or lut_binding.get("tensor_sha256") != lut_tensor_sha
        ):
            raise ValueError(f"QTIP3 physical cell LUT binding mismatch: {receipt_path}")

        arrays: dict[str, np.ndarray[Any, Any]] = {}
        artifact_rows: dict[str, dict[str, Any]] = {}
        for name in ("codes", "SU", "SV", "Wscale"):
            binding = artifacts.get(name)
            if not isinstance(binding, Mapping):
                raise ValueError(f"QTIP3 physical cell lacks artifact {name}: {receipt_path}")
            path = Path(str(binding.get("path", ""))).expanduser().resolve()
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != binding.get("bytes")
                or _sha256_file(path) != binding.get("sha256")
            ):
                raise ValueError(f"QTIP3 physical cell artifact drift: {name}: {receipt_path}")
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            if int(value.nbytes) != binding.get("data_bytes"):
                raise ValueError(f"QTIP3 physical cell data byte drift: {name}: {receipt_path}")
            arrays[name] = value
            artifact_rows[name] = {
                "path": str(path),
                "bytes": int(binding["bytes"]),
                "data_bytes": int(binding["data_bytes"]),
                "sha256": str(binding["sha256"]),
                "dtype": value.dtype.str,
                "shape": list(value.shape),
            }
        weights = int(arrays["SU"].size * arrays["SV"].size)
        code_bytes = int(arrays["codes"].nbytes)
        transform_bytes = int(arrays["SU"].nbytes + arrays["SV"].nbytes)
        wscale_bytes = int(arrays["Wscale"].nbytes)
        if (
            arrays["codes"].dtype != np.uint8
            or tuple(arrays["codes"].shape) != (weights // 256, 96)
            or weights % 256
            or code_bytes * 8 != weights * 3
            or arrays["SU"].dtype != np.float16
            or arrays["SU"].ndim != 1
            or arrays["SV"].dtype != np.float16
            or arrays["SV"].ndim != 1
            or arrays["Wscale"].dtype != np.float32
            or arrays["Wscale"].size != 1
            or accounting.get("weights") != weights
            or accounting.get("exact_code_bits") != weights * 3
            or accounting.get("code_data_bytes") != code_bytes
            or accounting.get("transform_bytes") != transform_bytes
            or accounting.get("Wscale_bytes") != wscale_bytes
            or accounting.get("shared_tlut_bytes") != lut_data_bytes
        ):
            raise ValueError(f"QTIP3 physical cell wire geometry mismatch: {receipt_path}")
        source = receipt.get("source_model_shard")
        control = receipt.get("control")
        if not isinstance(source, Mapping) or not isinstance(control, Mapping):
            raise ValueError(f"QTIP3 physical cell source/control binding missing: {receipt_path}")
        payload_bytes = code_bytes + transform_bytes + wscale_bytes
        row = {
            "schema": "banana-smasher-qtip3-fixed-manifest-member-v1",
            "status": "PASS",
            "codec_provider_id": QTIP3_PROVIDER_ID,
            "basis_index_sha256": intended_basis,
            "layer": identity[0],
            "expert": identity[1],
            "projection": identity[2],
            "source_weight_sha256": _require_sha256(
                source.get("tensor_sha256"), "source weight SHA-256"
            ),
            "hessian_sha256": _require_sha256(control.get("sha256"), "Hessian SHA-256"),
            "geometry": dict(QTIP3_GEOMETRY),
            "receipt": {
                "path": str(receipt_path),
                "bytes": receipt_path.stat().st_size,
                "sha256": _sha256_file(receipt_path),
            },
            "artifacts": artifact_rows,
            "cell_payload_bytes": payload_bytes,
            "physical_bytes": payload_bytes,
            "logical_weights": weights,
            "activation_artifacts": [lut_activation],
        }
        rows_by_identity[identity] = row
        totals["logical_weights"] += weights
        totals["code_data_bytes"] += code_bytes
        totals["transform_data_bytes"] += transform_bytes
        totals["wscale_data_bytes"] += wscale_bytes

    if set(rows_by_identity) != set(expected):
        raise ValueError(
            "QTIP3 fixed manifest coverage mismatch: "
            f"missing={sorted(set(expected) - set(rows_by_identity))} "
            f"extras={sorted(set(rows_by_identity) - set(expected))}"
        )
    rows = [rows_by_identity[identity] for identity in expected]
    manifest_payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    manifest_path = Path(manifest_output).expanduser().resolve()
    _atomic_bytes(manifest_path, manifest_payload)
    full_wire_bytes = (
        totals["code_data_bytes"]
        + totals["transform_data_bytes"]
        + totals["wscale_data_bytes"]
        + lut_data_bytes
    )
    terminal = {
        "schema": "banana-smasher-qtip3-fixed-manifest-v1",
        "status": "PASS",
        "codec_provider_id": QTIP3_PROVIDER_ID,
        "basis_index_sha256": intended_basis,
        "member_count": len(rows),
        "coverage": {
            "layers": sorted({identity[0] for identity in expected}),
            "members": len(rows),
        },
        "manifest": {
            "path": str(manifest_path),
            "bytes": len(manifest_payload),
            "sha256": hashlib.sha256(manifest_payload).hexdigest(),
        },
        "shared_lut": {
            "path": str(lut_path),
            "storage_bytes": lut_path.stat().st_size,
            "storage_sha256": _sha256_file(lut_path),
            "data_bytes": lut_data_bytes,
            "tensor_sha256": lut_tensor_sha,
            "deduplicated_instances": 1,
        },
        "wire": {
            **totals,
            "shared_lut_data_bytes": lut_data_bytes,
            "full_routed_wire_bytes": full_wire_bytes,
            "exact_code_bpw": totals["code_data_bytes"] * 8 / totals["logical_weights"],
            "full_routed_wire_bpw": full_wire_bytes * 8 / totals["logical_weights"],
        },
    }
    terminal_path = Path(terminal_output).expanduser().resolve()
    _atomic_bytes(
        terminal_path,
        (json.dumps(terminal, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    return terminal


def build_qtip3_retained_index(
    *,
    source_model_root: str | Path,
    output: str | Path,
    intended_basis_sha256: str,
) -> dict[str, Any]:
    """Hash-index every source tensor not replaced by routed QTIP3 members."""

    from safetensors import safe_open

    source_root = Path(source_model_root).expanduser().resolve()
    source_index_path = source_root / "model.safetensors.index.json"
    intended_basis = _require_sha256(intended_basis_sha256, "intended basis SHA-256")
    if _sha256_file(source_index_path) != intended_basis:
        raise ValueError("retained QTIP3 source model index basis mismatch")
    source_index = json.loads(source_index_path.read_text())
    source_weight_map = source_index.get("weight_map")
    if not isinstance(source_weight_map, Mapping):
        raise ValueError("retained QTIP3 source model index lacks weight_map")

    def is_routed_expert(name: str) -> bool:
        parts = name.split(".")
        if (
            len(parts) != 7
            or parts[0] != "layers"
            or parts[2:4] != ["ffn", "experts"]
            or parts[5] not in {"w1", "w2", "w3"}
            or parts[6] not in {"weight", "scale"}
        ):
            return False
        try:
            layer = int(parts[1])
            expert = int(parts[4])
        except ValueError:
            return False
        return 0 <= layer <= 42 and 0 <= expert <= 255

    dtype_names = {
        torch.bfloat16: "BF16",
        torch.float16: "F16",
        torch.float32: "F32",
        torch.float64: "F64",
        torch.float8_e4m3fn: "F8_E4M3",
        torch.float8_e8m0fnu: "F8_E8M0",
        torch.int8: "I8",
        torch.int16: "I16",
        torch.int32: "I32",
        torch.int64: "I64",
        torch.uint8: "U8",
        torch.bool: "BOOL",
    }
    names_by_shard: dict[str, list[str]] = {}
    excluded = 0
    for name, shard in source_weight_map.items():
        if not isinstance(name, str) or not isinstance(shard, str):
            raise ValueError("retained QTIP3 source weight map is invalid")
        if is_routed_expert(name):
            excluded += 1
        else:
            names_by_shard.setdefault(shard, []).append(name)

    rows: list[dict[str, Any]] = []
    data_bytes = 0
    for shard, names in sorted(names_by_shard.items()):
        with safe_open(source_root / shard, framework="pt", device="cpu") as handle:
            for name in sorted(names):
                tensor = handle.get_tensor(name).detach().cpu().contiguous()
                tensor_bytes = tensor.numel() * tensor.element_size()
                dtype_name = dtype_names.get(tensor.dtype)
                if dtype_name is None:
                    raise ValueError(f"unsupported retained QTIP3 tensor dtype: {tensor.dtype}")
                rows.append(
                    {
                        "name": name,
                        "shape": list(tensor.shape),
                        "dtype": dtype_name,
                        "bytes": tensor_bytes,
                        "sha256": _tensor_sha256(tensor),
                    }
                )
                data_bytes += tensor_bytes
    payload = {
        "schema": "banana-smasher-qtip3-retained-index-v1",
        "basis_model_index_sha256": intended_basis,
        "retained_tensors": rows,
    }
    output_path = Path(output).expanduser().resolve()
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    _atomic_bytes(output_path, raw)
    return {
        "schema": "banana-smasher-qtip3-retained-index-build-v1",
        "status": "PASS",
        "basis_model_index_sha256": intended_basis,
        "retained_tensor_count": len(rows),
        "excluded_routed_tensor_count": excluded,
        "data_bytes": data_bytes,
        "index": {
            "path": str(output_path),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
    }


def materialize_qtip3_retained_weights(
    *,
    source_model_root: str | Path,
    retained_index: str | Path,
    output_root: str | Path,
    intended_basis_sha256: str,
) -> dict[str, Any]:
    """Write only retained non-routed tensors into sparse shipping shards."""

    from safetensors import safe_open
    from safetensors.torch import save_file

    source_root = Path(source_model_root).expanduser().resolve()
    source_index_path = source_root / "model.safetensors.index.json"
    intended_basis = _require_sha256(intended_basis_sha256, "intended basis SHA-256")
    if _sha256_file(source_index_path) != intended_basis:
        raise ValueError("retained QTIP3 source model index basis mismatch")
    source_index = json.loads(source_index_path.read_text())
    source_weight_map = source_index.get("weight_map")
    if not isinstance(source_weight_map, Mapping):
        raise ValueError("retained QTIP3 source model index lacks weight_map")
    retained_path = Path(retained_index).expanduser().resolve()
    retained = json.loads(retained_path.read_text())
    rows = retained.get("retained_tensors")
    if retained.get("basis_model_index_sha256") != intended_basis or not isinstance(rows, list):
        raise ValueError("retained QTIP3 tensor index basis mismatch")
    rows_by_name: dict[str, Mapping[str, Any]] = {}
    names_by_shard: dict[str, list[str]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("name"), str):
            raise ValueError("retained QTIP3 tensor row is invalid")
        name = str(raw["name"])
        if name in rows_by_name:
            raise ValueError(f"duplicate retained QTIP3 tensor: {name}")
        shard = source_weight_map.get(name)
        if not isinstance(shard, str):
            raise ValueError(f"retained QTIP3 tensor is absent from source index: {name}")
        rows_by_name[name] = raw
        names_by_shard.setdefault(shard, []).append(name)
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    dtype_names = {
        torch.bfloat16: "BF16",
        torch.float16: "F16",
        torch.float32: "F32",
        torch.float64: "F64",
        torch.float8_e4m3fn: "F8_E4M3",
        torch.float8_e8m0fnu: "F8_E8M0",
        torch.int8: "I8",
        torch.int16: "I16",
        torch.int32: "I32",
        torch.int64: "I64",
        torch.uint8: "U8",
        torch.bool: "BOOL",
    }
    output_weight_map: dict[str, str] = {}
    shard_receipts: list[dict[str, Any]] = []
    data_bytes = 0
    for shard, names in sorted(names_by_shard.items()):
        source_shard = source_root / shard
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(source_shard, framework="pt", device="cpu") as handle:
            for name in sorted(names):
                tensor = handle.get_tensor(name).detach().cpu().contiguous()
                row = rows_by_name[name]
                tensor_bytes = tensor.numel() * tensor.element_size()
                if (
                    list(tensor.shape) != row.get("shape")
                    or dtype_names.get(tensor.dtype) != row.get("dtype")
                    or tensor_bytes != row.get("bytes")
                    or _tensor_sha256(tensor) != row.get("sha256")
                ):
                    raise ValueError(f"retained QTIP3 tensor identity mismatch: {name}")
                tensors[name] = tensor
                output_weight_map[name] = shard
                data_bytes += tensor_bytes
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination, prefix=f".{Path(shard).name}."
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        output_shard = destination / shard
        try:
            save_file(tensors, temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, output_shard)
        finally:
            temporary.unlink(missing_ok=True)
        shard_receipts.append(
            {
                "path": str(output_shard),
                "bytes": output_shard.stat().st_size,
                "sha256": _sha256_file(output_shard),
                "tensor_count": len(tensors),
            }
        )
    output_index = {
        "metadata": {"total_size": data_bytes},
        "weight_map": output_weight_map,
    }
    output_index_path = destination / "model.safetensors.index.json"
    _atomic_bytes(
        output_index_path,
        (json.dumps(output_index, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    return {
        "schema": "banana-smasher-qtip3-retained-shipping-v1",
        "status": "PASS",
        "basis_model_index_sha256": intended_basis,
        "tensor_count": len(rows_by_name),
        "data_bytes": data_bytes,
        "index": {
            "path": str(output_index_path),
            "bytes": output_index_path.stat().st_size,
            "sha256": _sha256_file(output_index_path),
        },
        "shards": shard_receipts,
    }


def _verify_file_binding(binding: object, label: str) -> Path:
    if not isinstance(binding, Mapping):
        raise ValueError(f"{label} binding is missing")
    path_value = binding.get("path")
    expected_bytes = binding.get("bytes")
    expected_sha = binding.get("sha256")
    if (
        not isinstance(path_value, str)
        or isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
    ):
        raise ValueError(f"{label} binding is invalid")
    path = Path(path_value).expanduser().resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != expected_bytes
        or _sha256_file(path) != _require_sha256(expected_sha, f"{label} SHA-256")
    ):
        raise ValueError(f"{label} byte or SHA-256 identity mismatch")
    return path


def verify_qtip3_shipping_pack(
    *,
    routed_terminal: str | Path,
    retained_terminal: str | Path,
    output: str | Path,
    intended_basis_sha256: str,
    expected_layers: Sequence[int],
    expected_member_count: int,
    base_model_parameters: int,
) -> dict[str, Any]:
    """Verify routed and retained payloads as one complete uniform-QTIP3 pack."""

    basis = _require_sha256(intended_basis_sha256, "intended basis SHA-256")
    layers = tuple(int(layer) for layer in expected_layers)
    if not layers or len(set(layers)) != len(layers) or any(layer < 0 for layer in layers):
        raise ValueError("QTIP3 shipping pack expected layers must be non-empty and unique")
    if (
        isinstance(expected_member_count, bool)
        or expected_member_count <= 0
        or isinstance(base_model_parameters, bool)
        or base_model_parameters <= 0
    ):
        raise ValueError("QTIP3 shipping pack counts must be positive integers")

    routed_path = Path(routed_terminal).expanduser().resolve()
    routed = json.loads(routed_path.read_text())
    coverage = routed.get("coverage")
    manifest_binding = routed.get("manifest")
    lut_binding = routed.get("shared_lut")
    wire = routed.get("wire")
    if (
        routed.get("schema") != "banana-smasher-qtip3-fixed-manifest-v1"
        or routed.get("status") != "PASS"
        or routed.get("codec_provider_id") != QTIP3_PROVIDER_ID
        or routed.get("basis_index_sha256") != basis
        or routed.get("member_count") != expected_member_count
        or not isinstance(coverage, Mapping)
        or coverage.get("layers") != list(layers)
        or coverage.get("members") != expected_member_count
        or not isinstance(wire, Mapping)
        or not isinstance(lut_binding, Mapping)
    ):
        raise ValueError("QTIP3 routed shipping terminal identity or coverage mismatch")
    manifest_path = _verify_file_binding(manifest_binding, "QTIP3 routed manifest")
    lut_path_value = lut_binding.get("path")
    if not isinstance(lut_path_value, str):
        raise ValueError("QTIP3 shared LUT path is missing")
    lut_path = Path(lut_path_value).expanduser().resolve()
    if (
        not lut_path.is_file()
        or lut_path.is_symlink()
        or lut_path.stat().st_size != lut_binding.get("storage_bytes")
        or _sha256_file(lut_path)
        != _require_sha256(lut_binding.get("storage_sha256"), "shared LUT storage SHA-256")
        or lut_binding.get("data_bytes") != 2048
        or lut_binding.get("deduplicated_instances") != 1
    ):
        raise ValueError("QTIP3 shared LUT storage identity mismatch")
    lut = np.load(lut_path, mmap_mode="r", allow_pickle=False)
    lut_tensor_sha = hashlib.sha256(memoryview(lut).cast("B")).hexdigest()
    if (
        lut.dtype != np.float16
        or tuple(lut.shape) != (1024,)
        or lut_tensor_sha != lut_binding.get("tensor_sha256")
    ):
        raise ValueError("QTIP3 shared LUT tensor identity mismatch")

    raw_rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    identities: set[tuple[int, int, str]] = set()
    routed_artifacts: dict[Path, int] = {}
    routed_payload_bytes = 0
    routed_logical_weights = 0
    loaded_members = 0
    for row, member in zip(
        raw_rows,
        iter_qtip3_fixed_manifest(manifest_path, lut_path=lut_path),
        strict=True,
    ):
        identity = (int(row["layer"]), int(row["expert"]), str(row["projection"]))
        if (
            identity in identities
            or identity[0] not in layers
            or identity[2] not in {"fused13", "down"}
            or member.basis_index_sha256 != basis
        ):
            raise ValueError(f"QTIP3 routed member identity mismatch: {identity}")
        identities.add(identity)
        loaded_members += 1
        routed_payload_bytes += int(row["cell_payload_bytes"])
        routed_logical_weights += int(row["logical_weights"])
        for name in ("codes", "SU", "SV", "Wscale"):
            artifact_path = _verify_file_binding(
                row.get("artifacts", {}).get(name), f"QTIP3 routed {identity}/{name}"
            )
            prior = routed_artifacts.get(artifact_path)
            size = artifact_path.stat().st_size
            if prior is not None and prior != size:
                raise ValueError(f"QTIP3 routed artifact size drift: {artifact_path}")
            routed_artifacts[artifact_path] = size
    if loaded_members != expected_member_count or len(identities) != expected_member_count:
        raise ValueError("QTIP3 routed manifest member count mismatch")
    if expected_member_count == len(layers) * 512:
        for layer in layers:
            expected = {
                (layer, expert, projection)
                for expert in range(256)
                for projection in ("fused13", "down")
            }
            if {identity for identity in identities if identity[0] == layer} != expected:
                raise ValueError(f"QTIP3 routed layer {layer} is not complete 256x2 coverage")
    expected_routed_wire = routed_payload_bytes + int(lut_binding["data_bytes"])
    if (
        wire.get("logical_weights") != routed_logical_weights
        or wire.get("full_routed_wire_bytes") != expected_routed_wire
        or wire.get("exact_code_bpw") != 3.0
    ):
        raise ValueError("QTIP3 routed wire ledger mismatch")

    retained_path = Path(retained_terminal).expanduser().resolve()
    retained = json.loads(retained_path.read_text())
    retained_rows = retained.get("shards")
    retained_bytes = retained.get("data_bytes")
    retained_tensors = retained.get("tensor_count")
    if (
        retained.get("schema") != "banana-smasher-qtip3-retained-shipping-v1"
        or retained.get("status") != "PASS"
        or retained.get("basis_model_index_sha256") != basis
        or not isinstance(retained_rows, list)
        or not isinstance(retained_bytes, int)
        or retained_bytes < 0
        or not isinstance(retained_tensors, int)
        or retained_tensors < 0
    ):
        raise ValueError("QTIP3 retained shipping terminal identity mismatch")
    retained_index_path = _verify_file_binding(retained.get("index"), "QTIP3 retained index")
    retained_storage_bytes = retained_index_path.stat().st_size
    retained_shard_tensors = 0
    for index, row in enumerate(retained_rows):
        path = _verify_file_binding(row, f"QTIP3 retained shard {index}")
        retained_storage_bytes += path.stat().st_size
        count = row.get("tensor_count")
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"QTIP3 retained shard {index} tensor count is invalid")
        retained_shard_tensors += count
    if retained_shard_tensors != retained_tensors:
        raise ValueError("QTIP3 retained tensor count mismatch")

    whole_model_bytes = expected_routed_wire + retained_bytes
    terminal = {
        "schema": "banana-smasher-qtip3-shipping-pack-v1",
        "status": "PASS",
        "codec_provider_id": QTIP3_PROVIDER_ID,
        "basis_index_sha256": basis,
        "coverage": {
            "layers": len(layers),
            "members": loaded_members,
            "retained_tensors": retained_tensors,
        },
        "shared_lut": dict(lut_binding),
        "wire": {
            "routed_bytes": expected_routed_wire,
            "retained_bytes": retained_bytes,
            "whole_model_bytes": whole_model_bytes,
            "base_model_parameters": base_model_parameters,
            "whole_model_bpw": whole_model_bytes * 8 / base_model_parameters,
        },
        "physical_tree_bytes": (
            sum(routed_artifacts.values())
            + lut_path.stat().st_size
            + manifest_path.stat().st_size
            + routed_path.stat().st_size
            + retained_storage_bytes
            + retained_path.stat().st_size
        ),
        "routed_terminal": {
            "path": str(routed_path),
            "bytes": routed_path.stat().st_size,
            "sha256": _sha256_file(routed_path),
        },
        "retained_terminal": {
            "path": str(retained_path),
            "bytes": retained_path.stat().st_size,
            "sha256": _sha256_file(retained_path),
        },
    }
    output_path = Path(output).expanduser().resolve()
    _atomic_bytes(
        output_path,
        (json.dumps(terminal, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    return terminal


def materialize_qtip3_fixed_member(
    *,
    cell_receipt: str | Path,
    lut_source: str | Path,
    member_output: str | Path,
    lut_output: str | Path,
    intended_basis_sha256: str,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """Convert one sealed physical PR31 cell into the public fixed-member ABI."""

    receipt_path = Path(cell_receipt).expanduser().resolve()
    receipt = json.loads(receipt_path.read_text())
    intended_basis = _require_sha256(intended_basis_sha256, "intended basis SHA-256")
    geometry = receipt.get("geometry")
    if (
        receipt.get("schema") != "banana-smasher-periodic-qtip3-pr31-cell-v1"
        or receipt.get("status") != "PASS"
        or receipt.get("basis_index_sha256") != intended_basis
        or receipt.get("accounting", {}).get("exact_code_bpw") != 3.0
        or not isinstance(geometry, Mapping)
        or geometry.get("L") != 16
        or geometry.get("B") != 12
        or geometry.get("V") != 4
        or geometry.get("rate_num") != 3
        or geometry.get("rate_den") != 1
        or geometry.get("phase_count") != 1
        or geometry.get("alternation") is not False
        or geometry.get("member_averaging") is not False
    ):
        raise ValueError("QTIP3 physical cell receipt identity or geometry mismatch")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("QTIP3 physical cell receipt lacks artifacts")
    relocated_root = (
        Path(artifact_root).expanduser().resolve() if artifact_root is not None else None
    )
    tensors: dict[str, torch.Tensor] = {}
    for name in ("codes", "SU", "SV", "Wscale"):
        binding = artifacts.get(name)
        if not isinstance(binding, Mapping):
            raise ValueError(f"QTIP3 physical cell receipt lacks artifact {name}")
        declared_path = Path(str(binding.get("path", "")))
        path = (
            (relocated_root / declared_path.name).resolve()
            if relocated_root is not None
            else declared_path.expanduser().resolve()
        )
        if (
            not path.is_file()
            or path.stat().st_size != binding.get("bytes")
            or _sha256_file(path) != binding.get("sha256")
        ):
            raise ValueError(f"QTIP3 physical cell artifact drift: {name}")
        array = np.asarray(np.load(path, allow_pickle=False))
        if array.nbytes != binding.get("data_bytes"):
            raise ValueError(f"QTIP3 physical cell data byte drift: {name}")
        tensors[name] = torch.from_numpy(np.ascontiguousarray(array))

    lut_array = np.asarray(np.load(Path(lut_source).expanduser().resolve(), allow_pickle=False))
    lut_binding = receipt.get("tlut")
    if (
        lut_array.dtype != np.float16
        or tuple(lut_array.shape) != (1024,)
        or not isinstance(lut_binding, Mapping)
        or hashlib.sha256(np.ascontiguousarray(lut_array).tobytes()).hexdigest()
        != lut_binding.get("tensor_sha256")
    ):
        raise ValueError("QTIP3 physical cell LUT binding mismatch")
    source = receipt.get("source_model_shard")
    control = receipt.get("control")
    if not isinstance(source, Mapping) or not isinstance(control, Mapping):
        raise ValueError("QTIP3 physical cell lacks source or Hessian binding")
    payload = {
        "schema": QTIP3_FIXED_SCHEMA,
        "codec_provider_id": QTIP3_PROVIDER_ID,
        "basis_index_sha256": intended_basis,
        "source_weight_sha256": _require_sha256(
            source.get("tensor_sha256"), "source weight SHA-256"
        ),
        "hessian_sha256": _require_sha256(control.get("sha256"), "Hessian SHA-256"),
        "geometry": dict(QTIP3_GEOMETRY),
        "lut": {
            "identity": lut_binding.get("identity", "pr31-affine-gaussian-edge-v1"),
            "tensor_sha256": lut_binding["tensor_sha256"],
            "data_bytes": int(lut_array.nbytes),
        },
        **tensors,
    }
    member_path = Path(member_output).expanduser().resolve()
    lut_path = Path(lut_output).expanduser().resolve()
    _atomic_torch_save(lut_path, torch.from_numpy(np.ascontiguousarray(lut_array)))
    _atomic_torch_save(member_path, payload)
    return {
        "schema": "banana-smasher-qtip3-fixed-member-materialization-v1",
        "status": "PASS",
        "basis_index_sha256": intended_basis,
        "layer": int(receipt["layer"]),
        "expert": int(receipt["expert"]),
        "projection": str(receipt["projection"]),
        "member": {
            "path": str(member_path),
            "bytes": member_path.stat().st_size,
            "sha256": _sha256_file(member_path),
        },
        "lut": {
            "path": str(lut_path),
            "bytes": lut_path.stat().st_size,
            "sha256": _sha256_file(lut_path),
            "tensor_sha256": lut_binding["tensor_sha256"],
        },
    }


@dataclass(frozen=True)
class Qtip3FixedMember:
    artifact_path: Path
    codec_provider_id: str
    basis_index_sha256: str
    source_weight_sha256: str
    hessian_sha256: str
    geometry: dict[str, Any]
    lut_identity: str
    lut_tensor_sha256: str
    lut: torch.Tensor
    codes: torch.Tensor
    SU: torch.Tensor
    SV: torch.Tensor
    Wscale: torch.Tensor


def iter_qtip3_fixed_manifest(
    manifest_path: str | Path, *, lut_path: str | Path
) -> Iterator[Qtip3FixedMember]:
    """Stream verified physical NPY members from a fixed-QTIP3 manifest."""

    manifest = Path(manifest_path).expanduser().resolve()
    lut_source = Path(lut_path).expanduser().resolve()
    lut_array = np.load(lut_source, allow_pickle=False)
    if lut_array.dtype != np.float16 or tuple(lut_array.shape) != (1024,):
        raise ValueError("QTIP3 manifest LUT requires float16[1024]")
    lut_tensor_sha = hashlib.sha256(memoryview(lut_array).cast("B")).hexdigest()
    lut = torch.from_numpy(np.ascontiguousarray(lut_array)).detach().cpu()
    for line_number, line in enumerate(manifest.read_text().splitlines(), 1):
        row = json.loads(line)
        if (
            not isinstance(row, Mapping)
            or row.get("schema") != "banana-smasher-qtip3-fixed-manifest-member-v1"
            or row.get("status") != "PASS"
            or row.get("codec_provider_id") != QTIP3_PROVIDER_ID
            or row.get("geometry") != QTIP3_GEOMETRY
        ):
            raise ValueError(f"invalid QTIP3 manifest row {line_number}")
        activations = row.get("activation_artifacts")
        if (
            not isinstance(activations, list)
            or len(activations) != 1
            or activations[0].get("sha256") != lut_tensor_sha
            or activations[0].get("bytes") != int(lut_array.nbytes)
        ):
            raise ValueError(f"QTIP3 manifest LUT binding mismatch at row {line_number}")
        artifacts = row.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError(f"QTIP3 manifest artifacts missing at row {line_number}")
        arrays: dict[str, np.ndarray[Any, Any]] = {}
        for name in ("codes", "SU", "SV", "Wscale"):
            binding = artifacts.get(name)
            if not isinstance(binding, Mapping):
                raise ValueError(f"QTIP3 manifest artifact {name} missing at row {line_number}")
            path = Path(str(binding.get("path", ""))).expanduser().resolve()
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != binding.get("bytes")
                or _sha256_file(path) != binding.get("sha256")
            ):
                raise ValueError(f"QTIP3 manifest artifact {name} drift at row {line_number}")
            value = np.load(path, mmap_mode="c", allow_pickle=False)
            if (
                int(value.nbytes) != binding.get("data_bytes")
                or value.dtype.str != binding.get("dtype")
                or list(value.shape) != binding.get("shape")
            ):
                raise ValueError(f"QTIP3 manifest artifact {name} geometry drift at row {line_number}")
            arrays[name] = value
        weights = int(arrays["SU"].size * arrays["SV"].size)
        if (
            arrays["codes"].dtype != np.uint8
            or tuple(arrays["codes"].shape) != (weights // 256, 96)
            or arrays["codes"].nbytes * 8 != weights * 3
            or arrays["SU"].dtype != np.float16
            or arrays["SU"].ndim != 1
            or arrays["SV"].dtype != np.float16
            or arrays["SV"].ndim != 1
            or arrays["Wscale"].dtype != np.float32
            or arrays["Wscale"].size != 1
            or row.get("logical_weights") != weights
        ):
            raise ValueError(f"QTIP3 manifest wire geometry mismatch at row {line_number}")
        receipt = row.get("receipt")
        if not isinstance(receipt, Mapping):
            raise ValueError(f"QTIP3 manifest receipt missing at row {line_number}")
        yield Qtip3FixedMember(
            artifact_path=Path(str(receipt.get("path", ""))).expanduser().resolve(),
            codec_provider_id=QTIP3_PROVIDER_ID,
            basis_index_sha256=_require_sha256(
                row.get("basis_index_sha256"), "basis index SHA-256"
            ),
            source_weight_sha256=_require_sha256(
                row.get("source_weight_sha256"), "source weight SHA-256"
            ),
            hessian_sha256=_require_sha256(
                row.get("hessian_sha256"), "Hessian SHA-256"
            ),
            geometry=dict(QTIP3_GEOMETRY),
            lut_identity=str(activations[0]["id"]),
            lut_tensor_sha256=lut_tensor_sha,
            lut=lut,
            codes=torch.from_numpy(arrays["codes"]),
            SU=torch.from_numpy(arrays["SU"]),
            SV=torch.from_numpy(arrays["SV"]),
            Wscale=torch.from_numpy(arrays["Wscale"]),
        )


def load_qtip3_fixed_member(
    artifact_path: str | Path, *, lut_path: str | Path
) -> Qtip3FixedMember:
    """Load one fixed QTIP3 member and verify its shared PR31 LUT binding."""

    path = Path(artifact_path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("schema") != QTIP3_FIXED_SCHEMA:
        raise ValueError(f"QTIP3 member requires schema {QTIP3_FIXED_SCHEMA!r}")
    provider_id = payload.get("codec_provider_id")
    if provider_id not in QTIP3_PROVIDER_IDS:
        raise ValueError(
            "QTIP3 member requires provider "
            f"{QTIP3_PROVIDER_ID!r} or legacy identity {QTIP3_LEGACY_PROVIDER_ID!r}"
        )
    geometry = dict(payload.get("geometry", {}))
    expected_geometry = (
        QTIP3_GEOMETRY
        if provider_id == QTIP3_PROVIDER_ID
        else QTIP3_LEGACY_GEOMETRY
    )
    if geometry != expected_geometry:
        raise ValueError(f"QTIP3 member geometry mismatch: {geometry!r}")
    lut_binding = payload.get("lut")
    if not isinstance(lut_binding, Mapping):
        raise ValueError("QTIP3 member requires an owned LUT binding")
    identity = lut_binding.get("identity")
    if not isinstance(identity, str) or not identity:
        raise ValueError("QTIP3 member LUT identity must be non-empty")
    expected_lut_sha = _require_sha256(
        lut_binding.get("tensor_sha256"), "QTIP3 member LUT tensor SHA-256"
    )
    lut = torch.load(Path(lut_path).expanduser().resolve(), map_location="cpu", weights_only=True)
    if not isinstance(lut, torch.Tensor):
        raise ValueError("QTIP3 member LUT artifact must contain one tensor")
    lut = lut.detach().cpu().contiguous()
    if _tensor_sha256(lut) != expected_lut_sha:
        raise ValueError("LUT tensor SHA-256 mismatch")
    expected_lut_bytes = lut_binding.get("data_bytes")
    if expected_lut_bytes != lut.numel() * lut.element_size():
        raise ValueError("QTIP3 member LUT data byte count mismatch")
    tensors: dict[str, torch.Tensor] = {}
    for name in ("codes", "SU", "SV", "Wscale"):
        value = payload.get(name)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"QTIP3 member requires tensor {name}")
        tensors[name] = value.detach().cpu().contiguous()
    if tensors["codes"].dtype != torch.uint8 or tensors["codes"].ndim != 2:
        raise ValueError("QTIP3 packed codes require uint8 [rows,bytes]")
    if tensors["Wscale"].dtype != torch.float32 or tensors["Wscale"].numel() != 1:
        raise ValueError("QTIP3 Wscale requires one float32 value")
    input_features = tensors["SU"].numel()
    output_features = tensors["SV"].numel()
    weights = input_features * output_features
    if (
        weights % 256
        or tuple(tensors["codes"].shape) != (weights // 256, 96)
        or tensors["codes"].numel() * 8 != weights * 3
    ):
        raise ValueError("QTIP3 packed codes do not match exact 3-BPW fixed geometry")
    return Qtip3FixedMember(
        artifact_path=path,
        codec_provider_id=provider_id,
        basis_index_sha256=_require_sha256(
            payload.get("basis_index_sha256"), "basis index SHA-256"
        ),
        source_weight_sha256=_require_sha256(
            payload.get("source_weight_sha256"), "source weight SHA-256"
        ),
        hessian_sha256=_require_sha256(payload.get("hessian_sha256"), "Hessian SHA-256"),
        geometry=geometry,
        lut_identity=identity,
        lut_tensor_sha256=expected_lut_sha,
        lut=lut,
        codes=tensors["codes"],
        SU=tensors["SU"],
        SV=tensors["SV"],
        Wscale=tensors["Wscale"],
    )


def decode_qtip3_fixed_member(
    member: Qtip3FixedMember, *, device: str | torch.device
) -> torch.Tensor:
    """Decode one fixed Periodic-QTIP3 member through the public runtime path."""

    from .qtip25_native_v4 import decode_native_v4_torch, native_v4_geometry

    target = torch.device(device)
    input_features = int(member.SU.numel())
    output_features = int(member.SV.numel())
    base = decode_native_v4_torch(
        member.codes.to(target),
        member.Wscale.to(target, torch.float32).expand(member.codes.shape[0]),
        positions=256,
        tlut=member.lut.to(target, torch.float32),
        geometry=native_v4_geometry(3.0),
    ).reshape(output_features, input_features)
    return (
        base
        * member.SV.to(target, torch.float32)[:, None]
        * member.SU.to(target, torch.float32)[None, :]
    )


def compare_qtip3_fixed_member_devices(
    member: Qtip3FixedMember,
    *,
    reference_device: str | torch.device,
    candidate_device: str | torch.device,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Execute one fixed member on two devices and fail on parity drift."""

    absolute_tolerance = float(atol)
    relative_tolerance = float(rtol)
    if (
        not math.isfinite(absolute_tolerance)
        or not math.isfinite(relative_tolerance)
        or absolute_tolerance < 0
        or relative_tolerance < 0
    ):
        raise ValueError("QTIP3 parity tolerances must be finite and non-negative")
    reference_target = torch.device(reference_device)
    candidate_target = torch.device(candidate_device)
    reference = decode_qtip3_fixed_member(member, device=reference_target)
    candidate = decode_qtip3_fixed_member(member, device=candidate_target)
    if reference_target.type == "cuda" or candidate_target.type == "cuda":
        torch.cuda.synchronize()
    reference = reference.detach().cpu().contiguous()
    candidate = candidate.detach().cpu().contiguous()
    if reference.shape != candidate.shape:
        raise RuntimeError("QTIP3 runtime parity output shape mismatch")
    difference = torch.abs(reference - candidate)
    max_abs_error = float(difference.max().item())
    reference_max_abs = float(torch.abs(reference).max().item())
    allowed_max_abs = absolute_tolerance + relative_tolerance * reference_max_abs
    if not torch.allclose(
        reference,
        candidate,
        atol=absolute_tolerance,
        rtol=relative_tolerance,
    ):
        raise RuntimeError(
            "QTIP3 runtime parity mismatch: "
            f"max_abs_error={max_abs_error} allowed_max_abs={allowed_max_abs}"
        )
    reference_sha = hashlib.sha256(reference.numpy().tobytes()).hexdigest()
    candidate_sha = hashlib.sha256(candidate.numpy().tobytes()).hexdigest()
    return {
        "schema": "banana-smasher-qtip3-fixed-runtime-parity-v1",
        "status": "PASS",
        "codec_provider_id": QTIP3_PROVIDER_ID,
        "basis_index_sha256": member.basis_index_sha256,
        "reference_device": str(reference_target),
        "candidate_device": str(candidate_target),
        "shape": list(reference.shape),
        "atol": absolute_tolerance,
        "rtol": relative_tolerance,
        "max_abs_error": max_abs_error,
        "allowed_max_abs": allowed_max_abs,
        "reference_sha256": reference_sha,
        "candidate_sha256": candidate_sha,
        "fallback": {"used": False},
    }


def smoke_qtip3_fixed_manifest(
    *,
    manifest_path: str | Path,
    lut_path: str | Path,
    output: str | Path,
    intended_basis_sha256: str,
    expected_identities: Sequence[tuple[int, int, str]],
    device: str | torch.device,
) -> dict[str, Any]:
    """Load every member and execute one runtime path per layer/projection."""

    basis = _require_sha256(intended_basis_sha256, "intended basis SHA-256")
    expected = tuple(
        (int(layer), int(expert), str(projection))
        for layer, expert, projection in expected_identities
    )
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("QTIP3 runtime smoke identities must be non-empty and unique")
    manifest = Path(manifest_path).expanduser().resolve()
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    observed: list[tuple[int, int, str]] = []
    representatives: dict[tuple[int, str], Qtip3FixedMember] = {}
    for row, member in zip(
        rows,
        iter_qtip3_fixed_manifest(manifest, lut_path=lut_path),
        strict=True,
    ):
        identity = (int(row["layer"]), int(row["expert"]), str(row["projection"]))
        if member.basis_index_sha256 != basis:
            raise ValueError(f"QTIP3 runtime smoke basis mismatch: {identity}")
        observed.append(identity)
        representatives.setdefault((identity[0], identity[2]), member)
    if tuple(observed) != expected:
        raise ValueError("QTIP3 runtime smoke manifest identity order mismatch")

    target = torch.device(device)
    runtime_groups = []
    for (layer, projection), member in sorted(representatives.items()):
        started = time.monotonic()
        weight = decode_qtip3_fixed_member(member, device=target)
        activation = torch.linspace(
            -1.0,
            1.0,
            steps=weight.shape[1],
            dtype=torch.float32,
            device=target,
        )
        runtime_output = torch.matmul(activation, weight.transpose(0, 1))
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        runtime_output = runtime_output.detach().cpu().contiguous()
        finite = bool(torch.isfinite(runtime_output).all().item())
        if not finite:
            raise RuntimeError(f"QTIP3 runtime smoke produced non-finite output: {(layer, projection)}")
        runtime_groups.append(
            {
                "layer": layer,
                "projection": projection,
                "representative_receipt": str(member.artifact_path),
                "device": str(target),
                "weight_shape": list(weight.shape),
                "output_shape": list(runtime_output.shape),
                "output_finite": finite,
                "output_sha256": hashlib.sha256(runtime_output.numpy().tobytes()).hexdigest(),
                "elapsed_seconds": time.monotonic() - started,
            }
        )
    receipt = {
        "schema": "banana-smasher-qtip3-fixed-runtime-smoke-v1",
        "status": "PASS",
        "codec_provider_id": QTIP3_PROVIDER_ID,
        "basis_index_sha256": basis,
        "manifest": {
            "path": str(manifest),
            "bytes": manifest.stat().st_size,
            "sha256": _sha256_file(manifest),
        },
        "coverage": {
            "members_loaded": len(observed),
            "runtime_groups_executed": len(runtime_groups),
        },
        "runtime_groups": runtime_groups,
        "fallback": {"used": False},
    }
    _atomic_bytes(
        Path(output).expanduser().resolve(),
        (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    return receipt


class Qtip3FixedRepairRuntime:
    """Train only the shared LUT while keeping assignments and geometry immutable."""

    def __init__(
        self,
        *,
        members: Sequence[Qtip3FixedMember],
        learning_rate: float,
        device: str | torch.device,
    ) -> None:
        if not members:
            raise ValueError("QTIP3 repair requires at least one fixed member")
        first = members[0]
        if any(
            member.lut_identity != first.lut_identity
            or member.lut_tensor_sha256 != first.lut_tensor_sha256
            for member in members[1:]
        ):
            raise ValueError("QTIP3 repair members do not share one sealed LUT")
        self.members = tuple(members)
        self.device = torch.device(device)
        self.shared_lut = torch.nn.Parameter(first.lut.to(self.device, torch.float32))
        self.optimizer = torch.optim.SGD((self.shared_lut,), lr=float(learning_rate))
        self.acceleration_counters = {
            "periodic_qtip3_lut_gather_calls": 0,
            "periodic_qtip3_lut_vjp_calls": 0,
            "fallback_calls": 0,
        }
        self.shared_lut.register_hook(self._record_lut_vjp)

    def _record_lut_vjp(self, gradient: torch.Tensor) -> torch.Tensor:
        self.acceleration_counters["periodic_qtip3_lut_vjp_calls"] += 1
        return gradient

    def _weight(self, member: Qtip3FixedMember) -> torch.Tensor:
        input_features = int(member.SU.numel())
        output_features = int(member.SV.numel())
        from .qtip25_native_v4 import decode_native_v4_torch, native_v4_geometry

        base = decode_native_v4_torch(
            member.codes.to(self.device),
            member.Wscale.to(self.device, torch.float32).expand(member.codes.shape[0]),
            positions=256,
            tlut=self.shared_lut,
            geometry=native_v4_geometry(3.0),
        ).reshape(output_features, input_features)
        self.acceleration_counters["periodic_qtip3_lut_gather_calls"] += 1
        return (
            base
            * member.SV.to(self.device, torch.float32)[:, None]
            * member.SU.to(self.device, torch.float32)[None, :]
        )

    def _loss_sum(
        self,
        activation_inputs: torch.Tensor,
        teacher_targets: torch.Tensor,
        teacher_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        member = self.members[0]
        inputs = activation_inputs.to(self.device, torch.float32)
        targets = teacher_targets.to(self.device, torch.float32)
        mask = teacher_mask.to(self.device)
        if mask.dtype != torch.bool or tuple(inputs.shape[:-1]) != tuple(mask.shape):
            raise ValueError("QTIP3 teacher mask does not match activation token geometry")
        if inputs.shape[-1] != member.SU.numel() or targets.shape[-1] != member.SV.numel():
            raise ValueError("QTIP3 activation/teacher feature geometry mismatch")
        outputs = torch.matmul(inputs, self._weight(member).transpose(0, 1))
        selected = mask.unsqueeze(-1).expand_as(outputs)
        count = int(selected.sum().item())
        if count <= 0:
            raise ValueError("QTIP3 microdose requires at least one teacher target")
        return torch.square(outputs - targets).masked_select(selected).sum(), count

    def microdose(
        self,
        *,
        activation_inputs: torch.Tensor,
        teacher_targets: torch.Tensor,
        teacher_mask: torch.Tensor,
    ) -> dict[str, Any]:
        code_hashes = tuple(_tensor_sha256(member.codes) for member in self.members)
        geometries = tuple(dict(member.geometry) for member in self.members)
        before = self.shared_lut.detach().clone()
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum, count = self._loss_sum(
            activation_inputs, teacher_targets, teacher_mask
        )
        (loss_sum / count).backward()
        gradient = self.shared_lut.grad
        finite_nonzero = bool(
            gradient is not None
            and torch.isfinite(gradient).all().item()
            and torch.count_nonzero(gradient).item() > 0
        )
        if not finite_nonzero:
            raise RuntimeError("QTIP3 microdose produced no finite nonzero LUT gradient")
        self.optimizer.step()
        delta = float(torch.linalg.vector_norm(self.shared_lut.detach() - before).item())
        codes_unchanged = code_hashes == tuple(
            _tensor_sha256(member.codes) for member in self.members
        )
        geometry_unchanged = geometries == tuple(
            dict(member.geometry) for member in self.members
        )
        if not codes_unchanged or not geometry_unchanged:
            raise RuntimeError("QTIP3 microdose changed immutable assignments or geometry")
        return {
            "schema": "banana-smasher-qtip3-fixed-microdose-v1",
            "status": "PASS_UPDATE",
            "finite_nonzero_gradients": finite_nonzero,
            "authorized_parameter_delta": delta,
            "packed_codes_unchanged": codes_unchanged,
            "geometry_unchanged": geometry_unchanged,
            "loss_sum": float(loss_sum.detach().cpu().item()),
            "target_count": count,
            "acceleration_counters": dict(self.acceleration_counters),
            "fallback": {"used": False},
        }


def run_qtip3_fixed_update(
    *,
    request: Path,
    output: Path,
    receipt: Path | None,
    identity: dict[str, Any],
    requested_tokens: int,
    physical_tokens: int,
    segments: int,
    batch_size: int,
    memory_sizing: dict[str, Any],
    resume: bool,
    restart: bool,
) -> dict[str, Any]:
    """Installed ``smash update`` backend for one fixed-QTIP3 microdose."""

    if batch_size != 1:
        raise ValueError("QTIP3 fixed repair requires batch size one")
    request_root = request.parent
    spec = json.loads(request.read_text())
    if spec.get("schema") != QTIP3_UPDATE_SCHEMA:
        raise ValueError(f"QTIP3 update requires schema {QTIP3_UPDATE_SCHEMA!r}")
    members = [
        load_qtip3_fixed_member(
            request_root / row["artifact"], lut_path=request_root / row["lut"]
        )
        for row in spec["members"]
    ]
    tensors = torch.load(
        request_root / spec["teacher_batch"], map_location="cpu", weights_only=True
    )
    logical_tokens = int(physical_tokens) * int(segments)
    activation_inputs = tensors["activation_inputs"][:, :logical_tokens]
    teacher_targets = tensors["teacher_targets"][:, :logical_tokens]
    teacher_mask = tensors["teacher_mask"][:, :logical_tokens]
    if activation_inputs.shape[1] != logical_tokens:
        raise ValueError("QTIP3 update request has insufficient logical tokens")
    runtime = Qtip3FixedRepairRuntime(
        members=members,
        learning_rate=float(spec["learning_rate"]),
        device=spec.get("device", "cuda"),
    )
    member_state = tuple(
        {
            "codes": _tensor_sha256(member.codes),
            "SU": _tensor_sha256(member.SU),
            "SV": _tensor_sha256(member.SV),
            "Wscale": _tensor_sha256(member.Wscale),
            "geometry": dict(member.geometry),
        }
        for member in members
    )
    work = []
    for index in range(int(segments)):
        start = index * int(physical_tokens)
        stop = start + int(physical_tokens)
        work.append(
            {
                "activation_inputs": activation_inputs[:, start:stop],
                "teacher_targets": teacher_targets[:, start:stop],
                "teacher_mask": teacher_mask[:, start:stop],
            }
        )
    output_features = int(members[0].SV.numel())

    def loss_sum(segment: dict[str, torch.Tensor]) -> torch.Tensor:
        value, _count = runtime._loss_sum(
            segment["activation_inputs"],
            segment["teacher_targets"],
            segment["teacher_mask"],
        )
        return value

    def validate_fixed_state() -> dict[str, Any]:
        observed = tuple(
            {
                "codes": _tensor_sha256(member.codes),
                "SU": _tensor_sha256(member.SU),
                "SV": _tensor_sha256(member.SV),
                "Wscale": _tensor_sha256(member.Wscale),
                "geometry": dict(member.geometry),
            }
            for member in members
        )
        codes_unchanged = all(
            before["codes"] == after["codes"]
            for before, after in zip(member_state, observed)
        )
        transforms_unchanged = all(
            all(before[name] == after[name] for name in ("SU", "SV", "Wscale"))
            for before, after in zip(member_state, observed)
        )
        geometry_unchanged = all(
            before["geometry"] == after["geometry"]
            for before, after in zip(member_state, observed)
        )
        counters = dict(runtime.acceleration_counters)
        if not codes_unchanged or not transforms_unchanged or not geometry_unchanged:
            raise RuntimeError("QTIP3 update changed immutable fixed-assignment state")
        if (
            counters["periodic_qtip3_lut_gather_calls"] <= 0
            or counters["periodic_qtip3_lut_vjp_calls"] <= 0
            or counters["fallback_calls"] != 0
        ):
            raise RuntimeError(f"QTIP3 update acceleration counters are invalid: {counters}")
        return {
            "fixed_qtip3": {
                "provider_id": QTIP3_PROVIDER_ID,
                "packed_codes_unchanged": codes_unchanged,
                "transforms_unchanged": transforms_unchanged,
                "geometry_unchanged": geometry_unchanged,
                "member_state": list(observed),
                "acceleration_counters": counters,
            }
        }

    synchronize = None
    peak_memory_bytes: Any = 0
    if runtime.device.type == "cuda":
        synchronize = torch.cuda.synchronize

        def cuda_peak_memory_bytes() -> int:
            return int(torch.cuda.max_memory_allocated(runtime.device))

        peak_memory_bytes = cuda_peak_memory_bytes
    return run_segmented_update(
        parameters=[runtime.shared_lut],
        optimizer=runtime.optimizer,
        segments=work,
        item_count=lambda segment: int(segment["teacher_mask"].sum().item())
        * output_features,
        loss_sum=loss_sum,
        output=output,
        receipt=receipt,
        identity=identity,
        physical_tokens=int(physical_tokens),
        observed_input_shape=[1, int(physical_tokens)],
        teacher_geometry={
            "target_shape": [1, int(physical_tokens), output_features],
            "mask_shape": [1, int(physical_tokens)],
            "position_shape": [1, int(physical_tokens)],
        },
        peak_memory_bytes=peak_memory_bytes,
        backend="accelerated",
        resume=bool(resume),
        restart=bool(restart),
        synchronize=synchronize,
        receipt_fields={
            "requested_physical_tokens": int(requested_tokens),
            "memory_sizing": memory_sizing,
        },
        post_step_validate=validate_fixed_state,
    )
