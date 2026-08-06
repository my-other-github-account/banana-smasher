from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .contract import (
    MANIFEST_NAME,
    PackValidationError,
    _canonical_json_bytes,
    _file_entry,
    _sha256_file,
    _write_bytes_durable,
    export_pack,
    verify_pack,
)
from .qtip25_codecs import resolve_qtip25_codec_provider

_FIXED_SCHEMA = "banana-smasher-fixed-qtip-members-v1"
_PROJECTIONS = {"fused13": "13", "down": "2"}
_FAMILY_CODES = {"qtip2": 0, "qtip3": 1, "d4": 2, "native": 3}
_REQUIRED_TENSORS = ("trellis", "SU", "SV", "Wscale")


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except Exception as exc:
        raise PackValidationError(f"cannot read fixed-assignment JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PackValidationError(f"fixed-assignment JSON must contain an object: {path}")
    return value, raw


def _read_members(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    try:
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except Exception as exc:
        raise PackValidationError(f"cannot read fixed member manifest {path}: {exc}") from exc
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise PackValidationError("fixed member manifest must contain JSON objects")
    return rows, raw


def _require_sha(path: Path, expected: str, label: str) -> None:
    actual = _sha256_file(path)
    if not isinstance(expected, str) or len(expected) != 64 or actual != expected:
        raise PackValidationError(
            f"{label} SHA-256 mismatch: expected={expected!r} actual={actual} path={path}"
        )


def _member_path(row: dict[str, Any], member_root: Path | None) -> Path:
    artifact = row.get("artifact")
    if not isinstance(artifact, dict):
        raise PackValidationError("fixed member row lacks artifact binding")
    raw_path = artifact.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise PackValidationError("fixed member artifact path is missing")
    if member_root is None:
        return Path(raw_path).expanduser().resolve()
    try:
        layer = int(row["layer"])
        expert = int(row["expert"])
        projection = str(row["projection"])
    except Exception as exc:
        raise PackValidationError(f"malformed fixed member identity: {row}") from exc
    return (
        member_root
        / f"L{layer:03d}"
        / f"E{expert:03d}_{projection}"
        / Path(raw_path).name
    ).resolve()


def _tensor_sha256(tensor: Any) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _load_member(row: dict[str, Any], member_root: Path | None) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise PackValidationError("fixed QTIP member export requires torch") from exc

    path = _member_path(row, member_root)
    artifact = row["artifact"]
    if not path.is_file() or path.is_symlink():
        remote = artifact.get("ssh")
        hint = (
            f"; stage {remote}:{artifact.get('path')} under --fixed-member-root"
            if isinstance(remote, str) and remote
            else ""
        )
        raise PackValidationError(f"fixed QTIP member is unavailable: {path}{hint}")
    if path.stat().st_size != artifact.get("bytes"):
        raise PackValidationError(f"fixed QTIP member byte count drift: {path}")
    _require_sha(path, str(artifact.get("sha256", "")), "fixed QTIP member")
    try:
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception as exc:
        raise PackValidationError(f"cannot load fixed QTIP member {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PackValidationError(f"fixed QTIP member payload is not an object: {path}")
    geometry = payload.get("geometry")
    row_geometry = row.get("geometry")
    if not isinstance(geometry, dict) or not isinstance(row_geometry, dict):
        raise PackValidationError(f"fixed QTIP member geometry is missing: {path}")
    observed = tuple(geometry.get(key) for key in ("L", "K", "V"))
    declared = tuple(row_geometry.get(key) for key in ("L", "K", "V"))
    if observed != declared or declared not in {(16, 2, 2), (16, 3, 2)}:
        raise PackValidationError(
            f"fixed QTIP member geometry drift: row={declared} payload={observed} path={path}"
        )
    tensors = {}
    for name in _REQUIRED_TENSORS:
        tensor = payload.get(name)
        if not isinstance(tensor, torch.Tensor):
            raise PackValidationError(f"fixed QTIP member lacks tensor {name}: {path}")
        tensors[name] = tensor.detach().cpu().contiguous()
    packed_sha = _tensor_sha256(tensors["trellis"])
    expected_packed_sha = row.get("canonical_packed_sha256")
    if not isinstance(expected_packed_sha, str) or packed_sha != expected_packed_sha:
        raise PackValidationError(
            f"fixed QTIP canonical payload drift: expected={expected_packed_sha!r} "
            f"actual={packed_sha} path={path}"
        )
    return tensors


def _validate_contract(
    rows: list[dict[str, Any]],
    admission: dict[str, Any],
    *,
    members_sha256: str,
) -> dict[tuple[int, str, int], list[dict[str, Any]]]:
    physical = admission.get("physical_payload")
    pack_gate = admission.get("pack_gate")
    coverage = admission.get("coverage")
    nested_manifest = physical.get("members_manifest") if isinstance(physical, dict) else None
    declared_members_sha = (
        physical.get("members_manifest_sha256")
        if isinstance(physical, dict)
        else None
    ) or (nested_manifest.get("sha256") if isinstance(nested_manifest, dict) else None)
    repair_absent = admission.get("repair_status") == "absent" or (
        isinstance(pack_gate, dict) and pack_gate.get("repair_training_used") is False
    )
    if (
        admission.get("status") != "PASS"
        or admission.get("tier") != "qtip@2.50"
        or not isinstance(physical, dict)
        or declared_members_sha != members_sha256
        or not repair_absent
    ):
        raise PackValidationError("fixed QTIP pack admission is not a sealed repair-absent qtip@2.50 payload")
    expected_members = physical.get("artifact_count", physical.get("expected_cells"))
    if expected_members is None and isinstance(coverage, dict):
        expected_members = coverage.get("members")
    if expected_members != len(rows):
        raise PackValidationError(
            f"fixed member population drift: admission={expected_members} manifest={len(rows)}"
        )

    grouped: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    identities: set[tuple[int, int, str]] = set()
    layers: set[int] = set()
    for row in rows:
        try:
            layer = int(row["layer"])
            expert = int(row["expert"])
            projection = str(row["projection"])
            geometry = row["geometry"]
            tier = str(row.get("tier", admission["tier"]))
            k = int(geometry["K"])
        except Exception as exc:
            raise PackValidationError(f"malformed fixed member row: {row}") from exc
        identity = (layer, expert, projection)
        if (
            not 0 <= expert < 256
            or projection not in _PROJECTIONS
            or tier != "qtip@2.50"
            or tuple(geometry.get(key) for key in ("L", "K", "V")) not in {(16, 2, 2), (16, 3, 2)}
            or identity in identities
        ):
            raise PackValidationError(f"invalid or duplicate fixed member identity: {identity}")
        identities.add(identity)
        layers.add(layer)
        grouped[(layer, projection, k)].append(row)

    for layer in sorted(layers):
        expected = {
            (layer, expert, projection)
            for expert in range(256)
            for projection in _PROJECTIONS
        }
        actual = {identity for identity in identities if identity[0] == layer}
        if actual != expected:
            raise PackValidationError(
                f"fixed member layer {layer} is not a complete 256x2 assignment: "
                f"missing={len(expected - actual)} extras={len(actual - expected)}"
            )
        for projection in _PROJECTIONS:
            counts = Counter(
                int(row["geometry"]["K"])
                for row in rows
                if int(row["layer"]) == layer and row["projection"] == projection
            )
            if counts != {2: 128, 3: 128}:
                raise PackValidationError(
                    f"fixed member layer {layer}/{projection} is not 50/50 K2/K3: {dict(counts)}"
                )
    return grouped


def _write_group(
    source_root: Path,
    *,
    layer: int,
    projection: str,
    k: int,
    rows: list[dict[str, Any]],
    member_root: Path | None,
) -> tuple[dict[str, dict[str, str]], dict[int, int], dict[str, int]]:
    rows = sorted(rows, key=lambda row: int(row["expert"]))
    first = _load_member(rows[0], member_root)
    suffix = _PROJECTIONS[projection]
    tier = f"qtip25k{k}"
    arrays: dict[str, np.memmap] = {}
    specs: dict[str, dict[str, str]] = {}
    dimensions = {"input": int(first["SU"].numel()), "output": int(first["SV"].numel())}
    try:
        for name, tensor in first.items():
            array = tensor.numpy()
            filename = f"layer_{layer:03d}.{tier}.{suffix}.{name}.npy"
            target = source_root / filename
            arrays[name] = np.lib.format.open_memmap(
                target,
                mode="w+",
                dtype=array.dtype,
                shape=(len(rows), *array.shape),
            )
            specs[name] = {"file": filename}
        expert_name = f"layer_{layer:03d}.{tier}.{suffix}.expert_ids.npy"
        expert_ids = np.lib.format.open_memmap(
            source_root / expert_name,
            mode="w+",
            dtype=np.int16,
            shape=(len(rows),),
        )
        specs["expert_ids"] = {"file": expert_name}
        slots: dict[int, int] = {}
        for slot, row in enumerate(rows):
            tensors = first if slot == 0 else _load_member(row, member_root)
            for name, target in arrays.items():
                value = tensors[name].numpy()
                if value.shape != target.shape[1:] or value.dtype != target.dtype:
                    raise PackValidationError(
                        f"fixed QTIP tensor shape/dtype drift in layer {layer}/{projection}/K{k}/{name}"
                    )
                target[slot] = value
            expert = int(row["expert"])
            expert_ids[slot] = expert
            slots[expert] = slot
        for target in [*arrays.values(), expert_ids]:
            target.flush()
    finally:
        arrays.clear()
    return specs, slots, dimensions


def materialize_fixed_qtip_source(
    *,
    members_manifest: str | Path,
    members_manifest_sha256: str,
    pack_admission: str | Path,
    pack_admission_sha256: str,
    output: str | Path,
    member_root: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize a sealed fixed QTIP assignment as selected P1016 planes."""
    members_path = Path(members_manifest).expanduser().resolve()
    admission_path = Path(pack_admission).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    _require_sha(members_path, members_manifest_sha256, "fixed member manifest")
    _require_sha(admission_path, pack_admission_sha256, "fixed pack admission")
    rows, _ = _read_members(members_path)
    admission, _ = _read_json(admission_path)
    grouped = _validate_contract(rows, admission, members_sha256=members_manifest_sha256)
    relocated_root = Path(member_root).expanduser().resolve() if member_root is not None else None

    output.mkdir(parents=True)
    layer_documents: dict[int, dict[str, Any]] = {}
    try:
        for layer in sorted({key[0] for key in grouped}):
            document: dict[str, Any] = {
                "format": "p1016-true-c-native-planes-v1",
                "layer": layer,
                "E": 256,
                "family_codes": dict(_FAMILY_CODES),
                "payloads": {projection: {} for projection in _PROJECTIONS},
            }
            for projection, suffix in (("fused13", "13"), ("down", "2")):
                tiers = [""] * 256
                families = [-1] * 256
                slots = [-1] * 256
                projection_dimensions: dict[str, int] | None = None
                for k in (2, 3):
                    specs, tier_slots, dimensions = _write_group(
                        output,
                        layer=layer,
                        projection=projection,
                        k=k,
                        rows=grouped[(layer, projection, k)],
                        member_root=relocated_root,
                    )
                    if projection_dimensions is None:
                        projection_dimensions = dimensions
                    elif projection_dimensions != dimensions:
                        raise PackValidationError(
                            f"fixed QTIP projection dimensions drift in layer {layer}/{projection}"
                        )
                    tier_name = f"qtip25k{k}"
                    document["payloads"][projection][tier_name] = {
                        "family": f"qtip{k}",
                        "geometry": {"L": 16, "K": k, "V": 2},
                        "tensors": specs,
                    }
                    for expert, slot in tier_slots.items():
                        tiers[expert] = tier_name
                        families[expert] = _FAMILY_CODES[f"qtip{k}"]
                        slots[expert] = slot
                assert projection_dimensions is not None
                document[f"tier{suffix}"] = tiers
                document[f"family{suffix}"] = families
                document[f"slot{suffix}"] = slots
                document[f"K{suffix}"] = projection_dimensions["input"]
                document[f"N{suffix}"] = projection_dimensions["output"]
            _write_bytes_durable(
                output / f"layer_{layer:03d}.meta.json",
                _canonical_json_bytes(document),
            )
            layer_documents[layer] = document
        receipt = {
            "schema": _FIXED_SCHEMA,
            "status": "PASS",
            "tier": "qtip@2.50",
            "repair_status": "absent",
            "layers": sorted(layer_documents),
            "member_count": len(rows),
            "members_manifest_sha256": members_manifest_sha256,
            "pack_admission_sha256": pack_admission_sha256,
        }
        _write_bytes_durable(output / "FIXED_QTIP_SOURCE.json", _canonical_json_bytes(receipt))
        return receipt
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def export_fixed_qtip_pack(
    *,
    members_manifest: str | Path,
    members_manifest_sha256: str,
    pack_admission: str | Path,
    pack_admission_sha256: str,
    output: str | Path,
    model_id: str,
    instance_id: str,
    serving_model_root: str | Path,
    runtime_floor_bytes: int,
    member_root: str | Path | None = None,
    link_mode: Literal["hardlink", "copy", "auto"] = "hardlink",
) -> dict[str, Any]:
    """Export a repair-absent movable bs-pack from a sealed fixed QTIP manifest."""
    output = Path(output).expanduser().resolve()
    members_path = Path(members_manifest).expanduser().resolve()
    admission_path = Path(pack_admission).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(tempfile.mkdtemp(prefix=f".{output.name}.fixed-qtip-", dir=output.parent))
    shutil.rmtree(source)
    try:
        source_receipt = materialize_fixed_qtip_source(
            members_manifest=members_path,
            members_manifest_sha256=members_manifest_sha256,
            pack_admission=admission_path,
            pack_admission_sha256=pack_admission_sha256,
            output=source,
            member_root=member_root,
        )
        manifest = export_pack(
            source_root=source,
            output=output,
            model_id=model_id,
            instance_id=instance_id,
            link_mode=link_mode,
            repair=None,
            serving_model_root=serving_model_root,
            runtime_floor_bytes=runtime_floor_bytes,
        )
        provenance = output / "provenance"
        provenance.mkdir()
        members_relative = Path("provenance/FIXED_QTIP_MEMBERS.jsonl")
        admission_relative = Path("provenance/FIXED_QTIP_PACK_ADMISSION.json")
        _write_bytes_durable(output / members_relative, members_path.read_bytes())
        _write_bytes_durable(output / admission_relative, admission_path.read_bytes())
        manifest["files"].extend(
            [
                _file_entry(output, members_relative, "fixed_qtip_member_manifest"),
                _file_entry(output, admission_relative, "fixed_qtip_pack_admission"),
            ]
        )
        manifest["files"].sort(key=lambda row: row["path"])
        manifest["links"].extend(
            [
                {"path": members_relative.as_posix(), "mode": "copy", "role": "fixed_qtip_member_manifest"},
                {"path": admission_relative.as_posix(), "mode": "copy", "role": "fixed_qtip_pack_admission"},
            ]
        )
        manifest["fixed_assignment"] = {
            **source_receipt,
            "members_manifest": members_relative.as_posix(),
            "pack_admission": admission_relative.as_posix(),
            "codec": resolve_qtip25_codec_provider("qtip@2.50").as_dict(
                requested_id="qtip@2.50"
            ),
        }
        manifest["provenance"]["fixed_member_root"] = (
            str(Path(member_root).expanduser().resolve()) if member_root is not None else None
        )
        manifest["selected_payloads"]["producer_stage"] = (
            "smash export:fixed-qtip-member-manifest-v1"
        )
        _write_bytes_durable(output / MANIFEST_NAME, _canonical_json_bytes(manifest))
        verify_pack(output)
        return manifest
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(source, ignore_errors=True)
