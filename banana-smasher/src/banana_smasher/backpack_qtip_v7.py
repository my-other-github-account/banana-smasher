"""Native QTIP2-V7 Backpack lifecycle adapter.

Ordinary QTIP2 Backpack declarations enter this module.  The explicit legacy
packaged-unit importer deliberately remains outside this call graph.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

_METHOD = "qtip2-v7-native"
_CALIBRATION_SCHEMA = "banana-smasher-qtip-v7-calibration-v1"
_PROJECTIONS = ("w1", "w2", "w3")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _binding(path: Path, *, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _bound_npy(root: Path, row: Mapping[str, Any], *, label: str) -> np.ndarray:
    relative = row.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label}.path must be non-empty")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file inside the calibration root")
    if row.get("sha256") != _sha256(path):
        raise ValueError(f"{label} SHA-256 mismatch")
    return np.load(path, allow_pickle=False)


def _load_calibration(path: Path, *, model_basis_sha256: str) -> dict[int, dict[str, Any]]:
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or document.get("schema") != _CALIBRATION_SCHEMA:
        raise ValueError(f"QTIP V7 calibration schema must be {_CALIBRATION_SCHEMA}")
    if document.get("model_basis_sha256") != model_basis_sha256:
        raise ValueError("QTIP V7 calibration/model basis mismatch")
    rows = document.get("layers")
    if not isinstance(rows, list) or not rows:
        raise ValueError("QTIP V7 calibration requires layer rows")
    root = path.parent.resolve()
    layers: dict[int, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("QTIP V7 calibration layer must be an object")
        layer = int(raw["layer"])
        if layer in layers:
            raise ValueError(f"duplicate QTIP V7 calibration layer {layer}")
        lut = _bound_npy(root, raw["lut"], label=f"layers[{layer}].lut")
        if lut.dtype != np.float16 or lut.shape != (1024,):
            raise ValueError(f"QTIP V7 layer {layer} LUT must be float16[1024]")
        hessians = raw.get("hessians")
        if not isinstance(hessians, Mapping) or set(hessians) != set(_PROJECTIONS):
            raise ValueError(f"QTIP V7 layer {layer} requires w1/w2/w3 Hessians")
        parsed_hessians: dict[str, dict[str, Any]] = {}
        for projection in _PROJECTIONS:
            declaration = hessians[projection]
            if not isinstance(declaration, Mapping):
                raise ValueError(f"QTIP V7 {projection} Hessian must be an object")
            count = declaration.get("count")
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"QTIP V7 {projection} Hessian count must be positive")
            value = _bound_npy(
                root,
                declaration,
                label=f"layers[{layer}].hessians.{projection}",
            )
            if value.dtype != np.float32 or value.ndim != 2 or value.shape[0] != value.shape[1]:
                raise ValueError(f"QTIP V7 {projection} Hessian must be square float32")
            parsed_hessians[projection] = {
                "value": np.ascontiguousarray(value),
                "count": count,
                "sha256": str(declaration["sha256"]),
                "path": str((root / str(declaration["path"])).resolve()),
            }
        layers[layer] = {
            "lut": np.ascontiguousarray(lut),
            "lut_path": str((root / str(raw["lut"]["path"])).resolve()),
            "hessians": parsed_hessians,
        }
    return layers


def _tensor_numpy(value: Any, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().contiguous().numpy()
    array = np.asarray(value)
    return np.ascontiguousarray(array if dtype is None else array.astype(dtype, copy=False))


def _produce_native_v7_batch(
    units: Sequence[dict[str, Any]], parent_lut: np.ndarray, *, output_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Invoke the shipped V7 producer in roster chunks with no fallback route."""
    import torch

    from .qtip_v7_batch import produce_qtip2_v7_batch

    if not torch.cuda.is_available():
        raise RuntimeError("native QTIP2-V7 Backpack candidate generation requires CUDA")
    device = torch.device("cuda")
    torch_lut = torch.from_numpy(parent_lut).to(device=device)
    results: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    experts = sorted({int(unit["expert"]) for unit in units})
    for start in range(0, len(experts), 10):
        roster = set(experts[start : start + 10])
        chunk = [
            {
                **unit,
                "source": torch.from_numpy(
                    np.asarray(unit["source"], dtype=np.float32)
                ).to(device),
                "raw_h": torch.from_numpy(
                    np.asarray(unit["raw_h"], dtype=np.float32)
                ),
            }
            for unit in units
            if int(unit["expert"]) in roster
        ]
        chunk_results, chunk_receipt = produce_qtip2_v7_batch(chunk, torch_lut)
        for result in chunk_results:
            packed = result["packed_codes"]
            decoded = result.get(
                "decoded", result.get("physical_fp32", result.get("physical_bfloat16"))
            )
            if decoded is None:
                raise RuntimeError("native QTIP2-V7 producer omitted decoded physical weights")
            results.append(
                {
                    key: result[key]
                    for key in ("layer", "expert", "projection", "member")
                    if key in result
                }
                | {
                    "packed_codes": (
                        bytes(packed)
                        if isinstance(packed, (bytes, bytearray, memoryview))
                        else _tensor_numpy(packed)
                    ),
                    "suh": _tensor_numpy(result["suh"], dtype=np.dtype("<f2")),
                    "svh": _tensor_numpy(result["svh"], dtype=np.dtype("<f2")),
                    "global_scale": float(result["global_scale"]),
                    "decoded": _tensor_numpy(decoded, dtype=np.dtype("<f4")),
                }
            )
        receipts.append(chunk_receipt)
        del chunk_results, chunk
    counters = {
        name: sum(int(receipt.get("counters", {}).get(name, 0)) for receipt in receipts)
        for name in ("qfn_calls", "extension_calls", "cuda_tiles")
    }
    if any(value <= 0 for value in counters.values()):
        raise RuntimeError("native QTIP2-V7 producer omitted positive CUDA execution counters")
    return results, {
        "schema": "banana-smasher-backpack-qtip2-v7-producer-v1",
        "status": "PASS",
        "method": _METHOD,
        "batch_calls": len(receipts),
        "batch_receipts": receipts,
        **counters,
        "fallback_calls": 0,
        "generic_fallback_calls": 0,
    }


def _materialize_native_v7_layer(**kwargs: Any) -> dict[str, Any]:
    from .qtip_v7_wire import pack_qtip_v7_layer, verify_qtip_v7_layer

    packed = pack_qtip_v7_layer(**kwargs)
    return verify_qtip_v7_layer(
        wire=Path(str(packed["wire"])),
        receipt=Path(str(packed["receipt"])),
    )


def materialize_qtip_v7_backpack_layer(**kwargs: Any) -> dict[str, Any]:
    """Pack and verify one native V7 layer through the public provider seam."""
    return _materialize_native_v7_layer(**kwargs)


def decode_selected_qtip_v7_backpack_weights(
    *,
    selected_layers: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Decode selected complete V7 wires for the pre-repair anchor instrument."""
    import torch

    from .qtip_k2 import decode_k2_matrix, inverse_transform
    from .qtip_v7_wire import QtipV7LayerMapping

    layer_paths: dict[int, Path] = {}
    for row in selected_layers:
        layer = int(row["layer"])
        if layer in layer_paths:
            raise ValueError(f"duplicate selected QTIP V7 layer {layer}")
        wire = Path(str(row["path"])).expanduser().resolve()
        if wire.is_symlink() or not wire.is_file():
            raise ValueError(f"selected QTIP V7 wire is unavailable: {wire}")
        if row.get("bytes") != wire.stat().st_size or row.get("sha256") != _sha256(wire):
            raise ValueError(f"selected QTIP V7 wire identity drift: {wire}")
        layer_paths[layer] = wire

    decoded_cells: list[np.ndarray] = []
    mappings: dict[int, QtipV7LayerMapping] = {}
    controls: dict[int, bytearray] = {}
    try:
        for cell in cells:
            layer = int(cell["layer"])
            if layer not in layer_paths:
                raise ValueError(f"selected QTIP V7 wire is missing layer {layer}")
            mapping = mappings.get(layer)
            if mapping is None:
                mapping = QtipV7LayerMapping(layer_paths[layer])
                mappings[layer] = mapping
                controls[layer] = mapping.transient_controls()
            shape = cell.get("matrix_shape")
            if not isinstance(shape, (list, tuple)) or len(shape) != 2:
                raise ValueError(f"QTIP V7 cell {cell['cell_id']} lacks matrix_shape")
            output_width, input_width = (int(shape[0]), int(shape[1]))
            if output_width % 16 or input_width % 16:
                raise ValueError("QTIP V7 selected-wire decode requires 16-aligned matrices")
            projection = str(cell["projection"])
            projection_index = _PROJECTIONS.index(projection)
            lut = torch.frombuffer(mapping.lut_view(), dtype=torch.float16, count=1024).clone()
            expert_rows: list[np.ndarray] = []
            for expert in cell["expert_ids"]:
                expert_id = int(expert)
                packed = torch.frombuffer(
                    mapping.packed_view(expert_id, projection),
                    dtype=torch.int16,
                    count=(input_width // 16) * (output_width // 16) * 32,
                ).reshape(input_width // 16, output_width // 16, 32).clone()
                member_index = expert_id * len(_PROJECTIONS) + projection_index
                start = member_index * mapping.geometry.control_bytes
                control = memoryview(controls[layer])[
                    start : start + mapping.geometry.control_bytes
                ]
                su_bytes = input_width * 2
                sv_bytes = output_width * 2
                su = torch.frombuffer(control[:su_bytes], dtype=torch.float16).clone()
                sv = torch.frombuffer(
                    control[su_bytes : su_bytes + sv_bytes], dtype=torch.float16
                ).clone()
                decoded = inverse_transform(decode_k2_matrix(packed, lut), su, sv).T
                expert_rows.append(
                    decoded.to(torch.float32).contiguous().numpy().reshape(-1)
                )
            decoded_cells.append(np.concatenate(expert_rows).astype(np.float32, copy=False))
    finally:
        controls.clear()
        for mapping in mappings.values():
            mapping.close()
    return np.concatenate(decoded_cells).astype(np.float32, copy=False)


def _account_native_v7_model(**kwargs: Any) -> dict[str, Any]:
    from .qtip_v7_wire import account_qtip_v7_model

    return account_qtip_v7_model(**kwargs)


def _load_legacy_packaged_unit(*_args: Any, **_kwargs: Any) -> None:
    """Forbidden-path sentinel used by focused lifecycle proof tests."""
    raise RuntimeError("legacy packaged QTIP units are forbidden in native V7 lifecycle")


def _write_member(path: Path, result: Mapping[str, Any]) -> None:
    packed = result["packed_codes"]
    packed_bytes = (
        bytes(packed)
        if isinstance(packed, (bytes, bytearray, memoryview))
        else _tensor_numpy(packed).view(np.uint8).tobytes()
    )
    suh = _tensor_numpy(result["suh"], dtype=np.dtype("<f2")).tobytes()
    svh = _tensor_numpy(result["svh"], dtype=np.dtype("<f2")).tobytes()
    scale = np.asarray([float(result["global_scale"])], dtype="<f4").tobytes()
    path.write_bytes(packed_bytes + suh + svh + scale)


def generate_qtip_v7_backpack_candidates(
    run_root: str | Path,
    *,
    tier: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    model_basis_sha256: str,
    weight_denominator: int,
    materialize: Callable[..., dict[str, Any]] = materialize_qtip_v7_backpack_layer,
) -> dict[str, Any]:
    """Generate fresh native QTIP2-V7 members and complete per-layer wires."""
    if tier.get("backend", "native_v7") != "native_v7" or float(tier.get("bpw", 0)) != 2.0:
        raise ValueError("native QTIP2-V7 generation requires family=qtip, bpw=2.0")
    calibration_path = Path(str(tier["calibration"])).expanduser().resolve()
    if calibration_path.is_symlink() or not calibration_path.is_file():
        raise ValueError(f"QTIP V7 calibration manifest is unavailable: {calibration_path}")
    calibration = _load_calibration(
        calibration_path, model_basis_sha256=model_basis_sha256
    )
    input_bindings = [_binding(calibration_path, role="calibration_manifest")]
    for layer, row in sorted(calibration.items()):
        input_bindings.append(_binding(Path(row["lut_path"]), role=f"layer_{layer}_lut"))
        input_bindings.extend(
            _binding(Path(row["hessians"][projection]["path"]), role=f"layer_{layer}_{projection}_hessian")
            for projection in _PROJECTIONS
        )
    root = Path(run_root).expanduser().resolve()
    v7_base = root / "qtip-v7"
    if v7_base.is_symlink():
        raise ValueError(f"managed qtip-v7 root cannot be a symlink: {v7_base}")
    v7_root = v7_base / str(tier["id"])
    if v7_root.is_symlink():
        raise ValueError(f"managed QTIP V7 tier root cannot be a symlink: {v7_root}")
    if v7_root.exists():
        shutil.rmtree(v7_root)
    v7_root.mkdir(parents=True)

    by_layer: dict[int, list[Mapping[str, Any]]] = {}
    for cell in cells:
        projection = str(cell["projection"])
        if projection not in _PROJECTIONS:
            raise ValueError("native QTIP2-V7 model cells must use w1/w2/w3 projections")
        by_layer.setdefault(int(cell["layer"]), []).append(cell)
    if set(by_layer) != set(calibration):
        raise ValueError("QTIP V7 model/calibration layer roster mismatch")

    cell_receipts: list[dict[str, Any]] = []
    layer_receipts: list[dict[str, Any]] = []
    producer_calls = 0
    wire_calls = 0
    qfn_calls = 0
    extension_calls = 0
    cuda_tiles = 0
    fallback_calls = 0
    for layer in sorted(by_layer):
        layer_root = v7_root / f"L{layer:03d}"
        member_root = layer_root / "members"
        member_root.mkdir(parents=True)
        declarations = by_layer[layer]
        units: list[dict[str, Any]] = []
        owners: dict[tuple[int, str], Mapping[str, Any]] = {}
        for cell in declarations:
            experts = [int(value) for value in cell["expert_ids"]]
            shape = cell.get("matrix_shape")
            if (
                not isinstance(shape, (list, tuple))
                or len(shape) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
            ):
                raise ValueError(f"QTIP V7 cell {cell['cell_id']} requires matrix_shape [out,in]")
            weights = np.asarray(cell["weights"], dtype=np.float32)
            expected = len(experts) * int(shape[0]) * int(shape[1])
            if weights.size != expected:
                raise ValueError(f"QTIP V7 cell {cell['cell_id']} matrix geometry mismatch")
            rows = weights.reshape(len(experts), int(shape[0]), int(shape[1]))
            projection = str(cell["projection"])
            hessian = calibration[layer]["hessians"][projection]
            if hessian["value"].shape != (int(shape[1]), int(shape[1])):
                raise ValueError(f"QTIP V7 cell {cell['cell_id']} Hessian geometry mismatch")
            for expert, weight in zip(experts, rows, strict=True):
                identity = (expert, projection)
                if identity in owners:
                    raise ValueError(f"duplicate QTIP V7 member {layer}:{expert}:{projection}")
                owners[identity] = cell
                units.append(
                    {
                        "layer": layer,
                        "expert": expert,
                        "projection": projection,
                        "source": np.ascontiguousarray(weight),
                        "weight": np.ascontiguousarray(weight),
                        "raw_h": hessian["value"],
                        "raw_h_count": hessian["count"],
                        "input_identity": {
                            "raw_hessian_data_sha256": hessian["sha256"]
                        },
                    }
                )
        results, producer_receipt = _produce_native_v7_batch(
            units, calibration[layer]["lut"], output_root=layer_root
        )
        producer_calls += 1
        qfn_calls += int(producer_receipt.get("qfn_calls", 0))
        extension_calls += int(producer_receipt.get("extension_calls", 0))
        cuda_tiles += int(producer_receipt.get("cuda_tiles", 0))
        fallback_calls += int(producer_receipt.get("generic_fallback_calls", producer_receipt.get("fallback_calls", 0)))
        if fallback_calls:
            raise RuntimeError("native QTIP2-V7 producer reported a forbidden fallback")
        if min(qfn_calls, extension_calls, cuda_tiles) <= 0:
            raise RuntimeError("native QTIP2-V7 producer omitted positive execution counters")
        result_map: dict[tuple[int, str], Mapping[str, Any]] = {}
        for result in results:
            if "expert" in result and "projection" in result:
                identity = (int(result["expert"]), str(result["projection"]))
            else:
                parts = str(result["member"]).split("/")
                identity = (int(parts[-2][1:]), parts[-1])
            result_map[identity] = result
        if set(result_map) != set(owners):
            raise RuntimeError("native QTIP2-V7 producer result roster drift")
        for (expert, projection), result in sorted(result_map.items()):
            _write_member(member_root / f"E{expert:03d}_{projection}.q2v7wire", result)

        wire_path = layer_root / f"L{layer:03d}.q2v7layer"
        wire_receipt_path = layer_root / "WIRE_RECEIPT.json"
        wire_receipt = materialize(
            source_root=member_root,
            lut=Path(calibration[layer]["lut_path"]),
            layer=layer,
            output=wire_path,
            receipt=wire_receipt_path,
        )
        wire_calls += 1
        if int(wire_receipt.get("generic_fallback_calls", 0)):
            raise RuntimeError("native QTIP2-V7 wire reported a forbidden fallback")
        wire_sha = str(wire_receipt.get("complete_wire_sha256", _sha256(wire_path)))
        activation = {
            "id": f"qtip2-v7-layer-{layer}",
            "bytes": wire_path.stat().st_size,
            "path": str(wire_path),
            "sha256": wire_sha,
        }
        layer_receipts.append(
            {
                "layer": layer,
                "wire": str(wire_path),
                "wire_bytes": wire_path.stat().st_size,
                "wire_sha256": wire_sha,
                "receipt": str(wire_receipt_path),
                "receipt_bytes": wire_receipt_path.stat().st_size,
                "receipt_sha256": _sha256(wire_receipt_path),
                "runtime_family": "qtip2_v7",
                "generic_fallback_calls": 0,
            }
        )
        for cell in declarations:
            identities = [(int(expert), str(cell["projection"])) for expert in cell["expert_ids"]]
            decoded = np.concatenate(
                [
                    _tensor_numpy(
                        result_map[identity].get(
                            "decoded", result_map[identity].get("physical_bfloat16")
                        ),
                        dtype=np.dtype("<f4"),
                    ).reshape(-1)
                    for identity in identities
                ]
            )
            destination = root / "candidates" / str(tier["id"]) / str(cell["cell_id"])
            destination.mkdir(parents=True, exist_ok=True)
            np.save(destination / "decoded.npy", decoded, allow_pickle=False)
            receipt = {
                "schema": "banana-smasher-backpack-candidate-cell-v1",
                "status": "PASS",
                "tier": tier["id"],
                "family": "qtip",
                "cell_id": cell["cell_id"],
                "algorithm": _METHOD,
                "method": _METHOD,
                "backend": "native_v7",
                "runtime_family": "qtip2_v7",
                "weight_count": int(decoded.size),
                "cell_payload_bytes": 0,
                "physical_bytes": 0,
                "activation_artifacts": [activation],
                "decoded": {
                    "path": str(destination / "decoded.npy"),
                    "sha256": _sha256(destination / "decoded.npy"),
                },
                "v7_wire": {
                    "path": str(wire_path),
                    "sha256": wire_sha,
                    "layer": layer,
                },
                "producer_calls": 1,
                "qfn_calls": int(producer_receipt["qfn_calls"]),
                "extension_calls": int(producer_receipt["extension_calls"]),
                "cuda_tiles": int(producer_receipt["cuda_tiles"]),
                "legacy_packaged_loader_calls": 0,
                "generic_fallback_calls": 0,
                "receipt": str(destination / "RECEIPT.json"),
            }
            _atomic_json(destination / "RECEIPT.json", receipt)
            cell_receipts.append(receipt)
    accounting_path = v7_root / "MODEL_ACCOUNTING.json"
    model_accounting = _account_native_v7_model(
        receipts=[row["receipt"] for row in layer_receipts],
        output=accounting_path,
        weight_denominator=weight_denominator,
        weight_denominator_label="Backpack model manifest weight_count",
    )
    if model_accounting.get("status") != "PASS":
        raise RuntimeError("native QTIP2-V7 model accounting did not pass")
    model_accounting_binding = {
        **model_accounting,
        "receipt": str(accounting_path),
        "receipt_bytes": accounting_path.stat().st_size,
        "receipt_sha256": _sha256(accounting_path),
    }
    return {
        "method": _METHOD,
        "runtime_family": "qtip2_v7",
        "producer_calls": producer_calls,
        "wire_calls": wire_calls,
        "qfn_calls": qfn_calls,
        "extension_calls": extension_calls,
        "cuda_tiles": cuda_tiles,
        "legacy_packaged_loader_calls": 0,
        "generic_fallback_calls": fallback_calls,
        "input_bindings": input_bindings,
        "model_accounting": model_accounting_binding,
        "cells": cell_receipts,
        "layers": layer_receipts,
    }


__all__ = [
    "decode_selected_qtip_v7_backpack_weights",
    "generate_qtip_v7_backpack_candidates",
    "materialize_qtip_v7_backpack_layer",
]
