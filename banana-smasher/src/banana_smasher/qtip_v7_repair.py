"""Fixed-byte QTIP V7 layer-LUT repair artifact contract."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

_SCHEMA = "banana-smasher-qtip-v7-artifact-v1"
_MANIFEST = "QTIP_V7_MANIFEST.json"
_LUT_BYTES = 1024 * 2
_V7_PACKED_BYTES = 2_097_152
_V7_PROJECTION_SHAPES = {
    "w1": (2048, 4096),
    "w2": (4096, 2048),
    "w3": (2048, 4096),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _bound_file(root: Path, row: dict[str, Any], *, expected_bytes: int | None = None) -> Path:
    relative = row.get("path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"QTIP V7 artifact has unsafe path {relative!r}")
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file() or path.is_symlink():
        raise ValueError(f"QTIP V7 artifact file is unavailable: {path}")
    declared_bytes = row.get("bytes")
    if not isinstance(declared_bytes, int) or path.stat().st_size != declared_bytes:
        raise ValueError(f"QTIP V7 artifact byte count drift: {path}")
    if expected_bytes is not None and declared_bytes != expected_bytes:
        raise ValueError(f"QTIP V7 artifact requires exactly {expected_bytes} bytes: {path}")
    observed = _sha256(path)
    if row.get("sha256") != observed:
        raise ValueError(f"QTIP V7 artifact SHA-256 mismatch: {path}")
    return path


@dataclass(frozen=True)
class QtipV7Artifact:
    manifest_path: Path
    rate: int
    document: dict[str, Any]
    member_paths: tuple[Path, ...]
    member_wire_sha256: tuple[str, ...]
    external_layers: tuple[int, ...]
    external_member_count: int
    external_wire_sha256: tuple[str, ...]
    layer_luts: dict[int, np.ndarray]
    complete_wire_bytes: int


def load_qtip_v7_artifact(manifest: str | Path) -> QtipV7Artifact:
    """Load one fixed-wire V7 artifact with separate layer-shared FP16 LUT slots."""
    manifest_path = Path(manifest).expanduser().resolve()
    document = json.loads(manifest_path.read_text())
    if not isinstance(document, dict) or document.get("schema") != _SCHEMA:
        raise ValueError(f"QTIP V7 manifest schema must be {_SCHEMA!r}")
    rate = document.get("rate")
    if isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0:
        raise ValueError("QTIP V7 rate must be a positive integer")
    members = document.get("members")
    luts = document.get("layer_luts")
    if not isinstance(members, list) or not members or not isinstance(luts, list) or not luts:
        raise ValueError("QTIP V7 manifest requires members and layer_luts")
    root = manifest_path.parent
    member_root_value = document.get("member_root")
    member_root = (
        root
        if member_root_value is None
        else Path(str(member_root_value)).expanduser().resolve()
    )
    member_paths: list[Path] = []
    member_shas: list[str] = []
    identities: set[tuple[int, int, str]] = set()
    layers: set[int] = set()
    for row in members:
        if not isinstance(row, dict):
            raise ValueError("QTIP V7 member row must be an object")
        identity = (int(row["layer"]), int(row["expert"]), str(row["projection"]))
        if identity in identities:
            raise ValueError(f"duplicate QTIP V7 member {identity}")
        identities.add(identity)
        layers.add(identity[0])
        path = _bound_file(member_root, row)
        packed_bytes = row.get("packed_code_bytes")
        if not isinstance(packed_bytes, int) or not 0 < packed_bytes <= path.stat().st_size:
            raise ValueError(f"QTIP V7 member packed byte accounting drift: {path}")
        member_paths.append(path)
        member_shas.append(str(row["sha256"]))
    external = document.get("external_layers", [])
    if not isinstance(external, list):
        raise ValueError("QTIP V7 external_layers must be a list")
    external_layers: set[int] = set()
    external_member_count = 0
    external_wire_bytes = 0
    external_shas: list[str] = []
    for row in external:
        if not isinstance(row, dict):
            raise ValueError("QTIP V7 external layer row must be an object")
        layer = int(row["layer"])
        if layer in layers or layer in external_layers:
            raise ValueError(f"duplicate QTIP V7 physical layer {layer}")
        member_count = row.get("member_count")
        complete_wire_bytes = row.get("complete_wire_bytes")
        identity_sha256 = row.get("identity_sha256")
        provider = row.get("provider")
        if isinstance(member_count, bool) or not isinstance(member_count, int) or member_count <= 0:
            raise ValueError(f"QTIP V7 external layer {layer} requires positive member_count")
        if (
            isinstance(complete_wire_bytes, bool)
            or not isinstance(complete_wire_bytes, int)
            or complete_wire_bytes <= 0
        ):
            raise ValueError(
                f"QTIP V7 external layer {layer} requires positive complete_wire_bytes"
            )
        if (
            not isinstance(identity_sha256, str)
            or len(identity_sha256) != 64
            or any(character not in "0123456789abcdef" for character in identity_sha256)
        ):
            raise ValueError(f"QTIP V7 external layer {layer} requires lowercase SHA-256 identity")
        if not isinstance(provider, str) or not provider:
            raise ValueError(f"QTIP V7 external layer {layer} requires provider identity")
        external_layers.add(layer)
        external_member_count += member_count
        external_wire_bytes += complete_wire_bytes
        external_shas.append(identity_sha256)
    layer_luts: dict[int, np.ndarray] = {}
    for row in luts:
        if not isinstance(row, dict) or row.get("dtype") != "float16" or row.get("shape") != [1024]:
            raise ValueError("QTIP V7 layer LUT must be float16 [1024]")
        layer = int(row["layer"])
        if layer in layer_luts:
            raise ValueError(f"duplicate QTIP V7 layer LUT {layer}")
        path = _bound_file(root, row, expected_bytes=_LUT_BYTES)
        layer_luts[layer] = np.fromfile(path, dtype="<f2").astype(np.float16, copy=True)
    physical_layers = layers | external_layers
    if set(layer_luts) != physical_layers:
        raise ValueError("QTIP V7 artifact requires exactly one shared LUT per physical layer")
    complete = (
        sum(path.stat().st_size for path in member_paths)
        + external_wire_bytes
        + len(layer_luts) * _LUT_BYTES
    )
    return QtipV7Artifact(
        manifest_path=manifest_path,
        rate=rate,
        document=document,
        member_paths=tuple(member_paths),
        member_wire_sha256=tuple(member_shas),
        external_layers=tuple(sorted(external_layers)),
        external_member_count=external_member_count,
        external_wire_sha256=tuple(external_shas),
        layer_luts=layer_luts,
        complete_wire_bytes=complete,
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _parse_v7_member(path: Path, *, projection: str) -> dict[str, Any]:
    """Parse the canonical raw V7 member without inventing a container wire."""
    import torch

    try:
        m, k = _V7_PROJECTION_SHAPES[projection]
    except KeyError as exc:
        raise ValueError(f"unsupported QTIP V7 projection {projection!r}") from exc
    payload = path.read_bytes()
    expected = _V7_PACKED_BYTES + (k + m) * 2 + 4
    if len(payload) != expected:
        raise ValueError(
            f"QTIP V7 member byte geometry drift for {projection}: {len(payload)} != {expected}"
        )
    packed_stop = _V7_PACKED_BYTES
    su_stop = packed_stop + k * 2
    sv_stop = su_stop + m * 2
    return {
        "shape": [m, k],
        "trellis": torch.from_numpy(
            np.frombuffer(payload[:packed_stop], dtype="<u2").copy()
        ),
        "SU": torch.from_numpy(
            np.frombuffer(payload[packed_stop:su_stop], dtype="<f2").copy()
        ),
        "SV": torch.from_numpy(
            np.frombuffer(payload[su_stop:sv_stop], dtype="<f2").copy()
        ),
        "Wscale": torch.from_numpy(
            np.frombuffer(payload[sv_stop:], dtype="<f4").copy()
        ).reshape(()),
    }


def build_qtip_v7_repair_bundle(
    *,
    manifest: str | Path,
    training: str | Path,
    output: str | Path,
    learning_rate: float,
    experts: list[int] | None = None,
    members: list[str] | None = None,
) -> dict[str, Any]:
    """Build the public physical-update bundle from fixed V7 wire plus TRAIN tensors."""
    import torch

    source = load_qtip_v7_artifact(manifest)
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("QTIP V7 repair learning_rate must be finite and positive")
    selected_experts = None if experts is None else {int(value) for value in experts}
    selected_members = None if members is None else set(members)
    rows = []
    for row, path in zip(source.document["members"], source.member_paths, strict=True):
        expert = int(row["expert"])
        if selected_experts is not None and expert not in selected_experts:
            continue
        layer = int(row["layer"])
        projection = str(row["projection"])
        member = f"{layer}:{expert}:{projection}"
        if selected_members is not None and member not in selected_members:
            continue
        parsed = _parse_v7_member(path, projection=projection)
        rows.append({
            "schema": "banana-smasher-qtip-v7-public-unit-v1",
            "rate": source.rate,
            "layer": layer,
            "expert": expert,
            "projection": projection,
            "layer_lut": layer,
            "tlut": torch.from_numpy(source.layer_luts[layer].copy()).reshape(512, 2),
            "source_member_sha256": str(row["sha256"]),
            "geometry": {
                "L": 16,
                "K": source.rate,
                "V": 2,
                "tlut_bits": 9,
                "decode_mode": "quantlut_sym",
                "td_x": 16,
                "td_y": 16,
            },
            **parsed,
        })
    if not rows:
        raise ValueError("QTIP V7 repair selected no fixed members")
    rows.sort(key=lambda row: (int(row["layer"]), int(row["expert"]), ("w1", "w2", "w3").index(row["projection"])))
    roster: dict[tuple[int, int], set[str]] = {}
    for row in rows:
        roster.setdefault((int(row["layer"]), int(row["expert"])), set()).add(
            str(row["projection"])
        )
    incomplete = {
        key: sorted(value)
        for key, value in roster.items()
        if value != set(_V7_PROJECTION_SHAPES)
    }
    if incomplete:
        raise ValueError(
            f"QTIP V7 repair requires complete w1/w2/w3 causal members: {incomplete}"
        )

    training_path = Path(training).expanduser().resolve()
    train = torch.load(training_path, map_location="cpu", weights_only=False)
    required = ("input_ids", "activation_inputs", "teacher_targets", "teacher_mask", "positions")
    if not isinstance(train, dict) or any(not isinstance(train.get(name), torch.Tensor) for name in required):
        raise ValueError("QTIP V7 repair TRAIN input requires the five public training tensors")
    bundle = {
        "schema": "banana-smasher-physical-repair-bundle-v1",
        **{name: train[name] for name in required},
        "layers": rows,
        "layer_luts": {
            layer: torch.from_numpy(values.copy())
            for layer, values in source.layer_luts.items()
            if any(int(row["layer"]) == layer for row in rows)
        },
        "optimizer": {"name": "sgd", "learning_rate": float(learning_rate)},
        "qtip_v7": {
            "manifest_sha256": _sha256(source.manifest_path),
            "rate": source.rate,
            "train_inputs_only": True,
            "fixed_member_sha256": list(source.member_wire_sha256),
            "selected_members": [
                f"{row['layer']}:{row['expert']}:{row['projection']}" for row in rows
            ],
        },
    }
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    try:
        torch.save(bundle, temporary)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {
        "schema": "banana-smasher-qtip-v7-repair-bundle-receipt-v1",
        "status": "PASS",
        "rate": source.rate,
        "fixed_members": len(rows),
        "trainable_layer_luts": len(bundle["layer_luts"]),
        "trainable_bytes": len(bundle["layer_luts"]) * _LUT_BYTES,
        "packed_member_bytes": sum(path.stat().st_size for path in source.member_paths),
        "bundle_sha256": _sha256(output_path),
    }
    _write_json(output_path.with_name(f"{output_path.name}.receipt.json"), receipt)
    return receipt


def export_qtip_v7_artifact(
    *,
    manifest: str | Path,
    output: str | Path,
    update_artifact: str | Path | list[str | Path] | None = None,
) -> dict[str, Any]:
    """Export update 0 or a complete set of layer-bound repaired updates."""
    source = load_qtip_v7_artifact(manifest)
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.mkdir(parents=True)
    try:
        trained: dict[int, np.ndarray] = dict(source.layer_luts)
        update = 0
        updated_layers: list[int] = []
        artifacts = (
            []
            if update_artifact is None
            else list(update_artifact)
            if isinstance(update_artifact, list)
            else [update_artifact]
        )
        if artifacts:
            import torch

            layers = set(source.layer_luts)
            bound: dict[int, Path] = {}
            for artifact in artifacts:
                value = str(artifact)
                if "=" not in value:
                    if len(layers) != 1 or len(artifacts) != 1:
                        raise ValueError(
                            "QTIP V7 multi-layer export requires explicit LAYER=PATH update bindings"
                        )
                    layer, path_value = next(iter(layers)), value
                else:
                    layer_value, path_value = value.split("=", 1)
                    try:
                        layer = int(layer_value)
                    except ValueError as exc:
                        raise ValueError(
                            f"QTIP V7 update binding has invalid layer {layer_value!r}"
                        ) from exc
                if layer not in layers:
                    raise ValueError(f"QTIP V7 update binding names foreign layer {layer}")
                if layer in bound:
                    raise ValueError(f"duplicate QTIP V7 update binding for layer {layer}")
                bound[layer] = Path(path_value).expanduser().resolve()
            missing = sorted(layers.difference(bound))
            if missing:
                raise ValueError(f"QTIP V7 update bindings are missing layers {missing}")
            for layer in sorted(bound):
                payload = torch.load(bound[layer], map_location="cpu", weights_only=True)
                if not isinstance(payload, dict) or payload.get("schema") != "banana-smasher-update-artifact-v2" or payload.get("optimizer_steps") != 1:
                    raise ValueError(
                        f"QTIP V7 export requires one completed update artifact for layer {layer}"
                    )
                parameters = payload.get("parameters")
                if not isinstance(parameters, list) or len(parameters) != 1:
                    raise ValueError(
                        f"QTIP V7 layer {layer} update artifact must contain one layer-shared LUT"
                    )
                parameter = parameters[0]
                array = parameter.detach().cpu().reshape(-1).numpy()
                if array.shape != (1024,) or not np.isfinite(array).all():
                    raise ValueError("QTIP V7 update parameter must be finite [1024]")
                trained[layer] = np.ascontiguousarray(array, dtype=np.float16)
                updated_layers.append(layer)
            update = 1
        document = json.loads(json.dumps(source.document))
        # Packed members are immutable parent wires. Export is a tiny LUT overlay
        # manifest that binds them by hash instead of duplicating tens of GiB.
        document["member_root"] = str(source.member_paths[0].parent)
        for row, source_path in zip(document["members"], source.member_paths, strict=True):
            row["path"] = source_path.name
        for row in document["layer_luts"]:
            layer = int(row["layer"])
            target = output_path / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            trained[layer].astype("<f2", copy=False).tofile(target)
            row["bytes"] = _LUT_BYTES
            row["sha256"] = _sha256(target)
        document["update"] = update
        _write_json(output_path / _MANIFEST, document)
        readback = load_qtip_v7_artifact(output_path / _MANIFEST)
        receipt = {
            "schema": "banana-smasher-qtip-v7-export-receipt-v1",
            "status": "PASS",
            "update": update,
            "updated_layers": updated_layers,
            "rate": source.rate,
            "layers": sorted(source.layer_luts),
            "members": len(source.member_paths) + source.external_member_count,
            "external_layers": list(source.external_layers),
            "packed_identity": (
                readback.member_wire_sha256 == source.member_wire_sha256
                and readback.external_wire_sha256 == source.external_wire_sha256
            ),
            "complete_wire_bytes": readback.complete_wire_bytes,
            "wire_size_delta": readback.complete_wire_bytes - source.complete_wire_bytes,
            "layer_lut_bytes": len(source.layer_luts) * _LUT_BYTES,
        }
        if not receipt["packed_identity"] or receipt["wire_size_delta"] != 0:
            raise RuntimeError("QTIP V7 fixed-wire export identity failed")
        _write_json(output_path / "QTIP_V7_EXPORT_RECEIPT.json", receipt)
        return receipt
    except Exception:
        shutil.rmtree(output_path, ignore_errors=True)
        raise


__all__ = [
    "QtipV7Artifact",
    "build_qtip_v7_repair_bundle",
    "export_qtip_v7_artifact",
    "load_qtip_v7_artifact",
]
