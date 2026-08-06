from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .backpack_contextual import ContextualValuationError, _atomic_json
from .backpack_virtual import (
    ASSIGNMENT_FILE,
    INDEX_FILE,
    VIRTUAL_MANIFEST,
    _atomic_write,
    _canonical,
    _parse_cell,
    verify_virtual_backpack,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ContextualValuationError(f"expected JSON object: {path}")
    return value


def _bound_file(root: Path, descriptor: Mapping[str, Any], *, role: str) -> Path:
    declared = descriptor.get("file", descriptor.get("path"))
    if not isinstance(declared, str) or not declared:
        raise ContextualValuationError(f"{role} descriptor path is invalid")
    path = Path(declared).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if (
        not path.is_file()
        or path.stat().st_size != descriptor.get("bytes")
        or _sha256(path) != descriptor.get("sha256")
    ):
        raise ContextualValuationError(f"{role} descriptor identity mismatch")
    return path


def materialize_contextual_change(
    baseline_virtual_manifest_path: str | Path,
    option_inventory_path: str | Path,
    change_request_path: str | Path,
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    """Create one zero-copy physical substitution candidate from prepared artifacts."""

    baseline_path = Path(baseline_virtual_manifest_path).expanduser().resolve()
    baseline_root = baseline_path.parent
    baseline = _object(baseline_path)
    verify_virtual_backpack(baseline_root)
    if baseline.get("status") != "PASS_LOGICAL_FULL_WIRE":
        raise ContextualValuationError("baseline virtual Backpack is not PASS")
    basis_sha256 = baseline.get("basis_sha256")
    assignment_descriptor = baseline.get("assignment")
    index_descriptor = baseline.get("materialization_index")
    if not isinstance(assignment_descriptor, Mapping) or not isinstance(
        index_descriptor, Mapping
    ):
        raise ContextualValuationError("baseline virtual descriptors are missing")
    assignment_path = _bound_file(
        baseline_root, assignment_descriptor, role="assignment"
    )
    index_path = _bound_file(
        baseline_root, index_descriptor, role="materialization_index"
    )

    inventory_path = Path(option_inventory_path).expanduser().resolve()
    request_path = Path(change_request_path).expanduser().resolve()
    inventory = _object(inventory_path)
    request = _object(request_path)
    if (
        inventory.get("schema")
        != "banana-smasher-contextual-option-inventory-v1"
        or inventory.get("status") != "READY"
        or inventory.get("basis_sha256") != basis_sha256
        or inventory.get("anchor_assignment_sha256")
        != baseline.get("assignment_map_sha256")
        or not isinstance(inventory.get("options"), list)
    ):
        raise ContextualValuationError("option inventory is not baseline-bound READY v1")
    if (
        request.get("schema")
        != "banana-smasher-contextual-change-request-v1"
        or request.get("status") != "READY"
        or request.get("anchor_assignment_sha256")
        != baseline.get("assignment_map_sha256")
    ):
        raise ContextualValuationError("change request is not baseline-bound READY v1")
    requested_change = request.get("change")
    if not isinstance(requested_change, Mapping):
        raise ContextualValuationError("change request has no physical change")
    cell = requested_change.get("cell")
    identity = requested_change.get("physical_identity")
    if not isinstance(cell, str) or not cell:
        raise ContextualValuationError("change request cell is invalid")
    if not isinstance(identity, str) or not identity:
        raise ContextualValuationError("change request physical identity is invalid")
    matches = [
        row
        for row in inventory["options"]
        if isinstance(row, Mapping)
        and row.get("cell") == cell
        and row.get("physical_identity") == identity
    ]
    if len(matches) != 1:
        raise ContextualValuationError("change request must resolve one physical option")
    option = matches[0]
    tier = option.get("option")
    source_key = option.get("physical_source_key")
    payload_bytes = option.get("payload_bytes")
    activations = option.get("activations", [])
    if not isinstance(tier, str) or not tier:
        raise ContextualValuationError("target logical option is invalid")
    if not isinstance(source_key, str) or not source_key:
        raise ContextualValuationError("target physical source is invalid")
    if source_key not in baseline.get("source_bindings", {}):
        raise ContextualValuationError("target physical source is not baseline-bound")
    if not isinstance(payload_bytes, int) or payload_bytes < 0:
        raise ContextualValuationError("target payload bytes are invalid")
    if not isinstance(activations, list):
        raise ContextualValuationError("target activations must be an array")
    target_artifacts: dict[str, dict[str, Any]] = {}
    for activation in activations:
        if not isinstance(activation, Mapping):
            raise ContextualValuationError("target activation is invalid")
        artifact_id = activation.get("id")
        byte_count = activation.get("bytes")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ContextualValuationError("target activation id is invalid")
        if not isinstance(byte_count, int) or byte_count < 0:
            raise ContextualValuationError("target activation bytes are invalid")
        if artifact_id in target_artifacts:
            raise ContextualValuationError("target activation ids must be unique")
        target_artifacts[artifact_id] = {
            "artifact_id": artifact_id,
            "bytes": byte_count,
        }

    assignment = _object(assignment_path)
    layer, expert, projection = _parse_cell(cell)
    try:
        assignment[str(layer)][str(expert)][projection] = tier
    except (KeyError, TypeError) as exc:
        raise ContextualValuationError("change cell is absent from assignment") from exc
    assignment_raw = _canonical(assignment)
    assignment_sha256 = hashlib.sha256(assignment_raw).hexdigest()

    rows: list[dict[str, Any]] = []
    changed_rows = 0
    for line in index_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ContextualValuationError("materialization row is invalid")
        if row.get("cell_id") == cell:
            row.update(
                {
                    "tier": tier,
                    "source_key": source_key,
                    "physical_bytes": payload_bytes,
                    "activation_artifact_ids": sorted(target_artifacts),
                }
            )
            changed_rows += 1
        rows.append(row)
    if changed_rows != 1:
        raise ContextualValuationError("change cell must resolve one materialization row")
    rows.sort(key=lambda row: _parse_cell(str(row["cell_id"])))
    index_raw = b"".join(_canonical(row) for row in rows)

    baseline_artifacts = baseline.get("activated_artifacts", [])
    if not isinstance(baseline_artifacts, list):
        raise ContextualValuationError("baseline activated artifacts are invalid")
    artifact_by_id: dict[str, dict[str, Any]] = {}
    for artifact in baseline_artifacts:
        if not isinstance(artifact, Mapping):
            raise ContextualValuationError("baseline activation artifact is invalid")
        artifact_id = artifact.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ContextualValuationError("baseline activation id is invalid")
        artifact_by_id[artifact_id] = dict(artifact)
    for artifact_id, artifact in target_artifacts.items():
        if (
            artifact_id in artifact_by_id
            and artifact_by_id[artifact_id].get("bytes") != artifact["bytes"]
        ):
            raise ContextualValuationError("activation artifact byte conflict")
        artifact_by_id.setdefault(artifact_id, artifact)

    tier_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    tier_payload: Counter[str] = Counter()
    selected_artifact_ids: set[str] = set()
    for row in rows:
        row_tier = str(row["tier"])
        row_source = str(row["source_key"])
        row_bytes = int(row["physical_bytes"])
        ids = row.get("activation_artifact_ids", [])
        if not isinstance(ids, list):
            raise ContextualValuationError("materialization activation ids are invalid")
        tier_counts[row_tier] += 1
        source_counts[row_source] += 1
        tier_payload[row_tier] += row_bytes
        selected_artifact_ids.update(str(value) for value in ids)
    missing_artifacts = selected_artifact_ids - set(artifact_by_id)
    if missing_artifacts:
        raise ContextualValuationError(
            f"selected activation metadata missing: {sorted(missing_artifacts)}"
        )
    activation_bytes = sum(
        int(artifact_by_id[artifact_id]["bytes"])
        for artifact_id in selected_artifact_ids
    )
    payload_total = sum(tier_payload.values())
    baseline_accounting = baseline.get("byte_accounting")
    if not isinstance(baseline_accounting, Mapping):
        raise ContextualValuationError("baseline byte accounting is missing")
    fixed_bytes = int(baseline_accounting["fixed_nonexpert_bytes"])
    all_tiers = set(baseline.get("tier_counts", {})) | set(tier_counts)
    derived_accounting = {
        "payload_bytes": payload_total,
        "activation_bytes": activation_bytes,
        "assigned_expert_bytes": payload_total + activation_bytes,
        "fixed_nonexpert_bytes": fixed_bytes,
        "assigned_package_bytes": payload_total + activation_bytes + fixed_bytes,
        "tier_payload_bytes": {
            key: tier_payload.get(key, 0) for key in sorted(all_tiers)
        },
    }
    denominator = int(baseline["expert_parameter_denominator"])
    if denominator <= 0:
        raise ContextualValuationError("expert parameter denominator is invalid")

    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty contextual candidate: {output}")
    output.mkdir(parents=True, exist_ok=True)
    assignment_output = output / ASSIGNMENT_FILE
    index_output = output / INDEX_FILE
    manifest_output = output / VIRTUAL_MANIFEST
    _atomic_write(assignment_output, assignment_raw)
    _atomic_write(index_output, index_raw)
    candidate = {
        **baseline,
        "arm_name": "contextual-physical-change",
        "assignment_map_sha256": assignment_sha256,
        "assignment": {
            "file": ASSIGNMENT_FILE,
            "sha256": assignment_sha256,
            "bytes": len(assignment_raw),
            "rows": len(rows),
        },
        "materialization_index": {
            "file": INDEX_FILE,
            "sha256": hashlib.sha256(index_raw).hexdigest(),
            "bytes": len(index_raw),
            "rows": len(rows),
        },
        "tier_counts": {key: tier_counts.get(key, 0) for key in sorted(all_tiers)},
        "source_component_counts": dict(sorted(source_counts.items())),
        "byte_accounting": derived_accounting,
        "activated_artifacts": [
            artifact_by_id[artifact_id] for artifact_id in sorted(selected_artifact_ids)
        ],
        "expert_wire_bpw": derived_accounting["assigned_expert_bytes"]
        * 8
        / denominator,
    }
    _atomic_json(manifest_output, candidate)
    verification = verify_virtual_backpack(output)
    pack_files = [
        {
            "file": path.name,
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in (assignment_output, manifest_output, index_output)
    ]
    pack_files.sort(key=lambda row: row["file"])
    candidate_pack_sha256 = hashlib.sha256(_canonical(pack_files)).hexdigest()
    scope = request.get("scope")
    if not isinstance(scope, str) or not scope:
        raise ContextualValuationError("change request scope is invalid")
    change = {
        "schema": "banana-smasher-contextual-change-v1",
        "status": "READY",
        "anchor_assignment_sha256": baseline.get("assignment_map_sha256"),
        "candidate_assignment_sha256": assignment_sha256,
        "candidate_pack_sha256": candidate_pack_sha256,
        "scope": scope,
        "change": {"cell": cell, "physical_identity": identity},
        "input_bindings": {
            "baseline_virtual_manifest_sha256": _sha256(baseline_path),
            "option_inventory_sha256": _sha256(inventory_path),
            "change_request_sha256": _sha256(request_path),
        },
    }
    change_output = output / "CHANGE.json"
    _atomic_json(change_output, change)
    return {
        "schema": "banana-smasher-contextual-candidate-receipt-v1",
        "status": "PASS",
        "root": str(output),
        "candidate_assignment_sha256": assignment_sha256,
        "candidate_pack_sha256": candidate_pack_sha256,
        "assigned_package_bytes": derived_accounting["assigned_package_bytes"],
        "change": change,
        "virtual_verification": verification,
    }
