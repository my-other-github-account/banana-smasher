from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

VIRTUAL_MANIFEST = "BACKPACK_VIRTUAL_MANIFEST.json"
ASSIGNMENT_FILE = "ASSIGNMENT.json"
INDEX_FILE = "MATERIALIZATION_INDEX.jsonl"
SCHEMA = "banana-smasher-backpack-virtual-assignment-v1"
SOURCES_SCHEMA = "banana-smasher-backpack-virtual-sources-v1"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def _load_object(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must contain an object: {path}")
    return value, raw


def _evidence(path: Path, raw: bytes | None = None) -> dict[str, Any]:
    path = path.resolve()
    raw = path.read_bytes() if raw is None else raw
    return {"path": str(path), "sha256": _sha256(raw), "bytes": len(raw)}


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _parse_cell(cell_id: str) -> tuple[int, int, str]:
    try:
        layer_text, expert_text, projection = cell_id.split(":")
        layer = int(layer_text.removeprefix("L"))
        expert = int(expert_text.removeprefix("E"))
    except Exception as exc:
        raise ValueError(f"invalid Backpack cell id: {cell_id!r}") from exc
    if layer < 0 or expert < 0 or projection not in {"down", "fused13"}:
        raise ValueError(f"invalid Backpack cell id: {cell_id!r}")
    return layer, expert, projection


def _assignment_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    assignment: dict[str, Any] = {}
    seen: set[str] = set()
    for row in rows:
        cell_id = str(row["cell_id"])
        if cell_id in seen:
            raise ValueError(f"duplicate assignment cell: {cell_id}")
        seen.add(cell_id)
        layer, expert, projection = _parse_cell(cell_id)
        assignment.setdefault(str(layer), {}).setdefault(str(expert), {})[
            projection
        ] = str(row["tier"])
    return assignment


def _source_key(row: dict[str, Any], ledger_row: dict[str, Any]) -> str:
    tier = str(row["tier"])
    if tier != "qtip2_5":
        return tier
    components = ledger_row.get("prediction_components")
    qtip_k = components.get("qtip_k") if isinstance(components, dict) else None
    if qtip_k == 2:
        return "qtip2"
    if qtip_k == 3:
        return "qtip3"
    raise ValueError(
        f"qtip2_5 cell {row['cell_id']} lacks exact qtip_k=2/3 component binding"
    )


def _bind_sources(
    source_bindings: Path, basis_sha256: str, required: set[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    declaration, declaration_raw = _load_object(source_bindings)
    if declaration.get("schema") != SOURCES_SCHEMA:
        raise ValueError(f"source binding schema must be {SOURCES_SCHEMA}")
    sources = declaration.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("source binding declaration has no sources object")
    missing = sorted(required - set(sources))
    if missing:
        raise ValueError(f"source binding declaration is missing {missing}")

    bound: dict[str, Any] = {}
    for key in sorted(required):
        source = sources[key]
        if not isinstance(source, dict):
            raise ValueError(f"source binding {key} is not an object")
        root_value = source.get("root")
        identity_value = source.get("identity")
        expected_sha = source.get("identity_sha256")
        if not isinstance(root_value, str) or not root_value:
            raise ValueError(f"source binding {key} has no root")
        if not isinstance(identity_value, str) or not identity_value:
            raise ValueError(f"source binding {key} has no identity path")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError(f"source binding {key} has no exact identity_sha256")
        if source.get("basis_sha256") != basis_sha256:
            raise ValueError(f"source binding {key} basis mismatch")
        root = Path(root_value).expanduser().resolve()
        unresolved = root / identity_value
        identity = unresolved.resolve()
        if root != identity and root not in identity.parents:
            raise ValueError(f"source binding {key} identity escapes its root")
        if not unresolved.is_file() or unresolved.is_symlink():
            raise ValueError(f"source binding {key} identity is missing or unsafe")
        identity_raw = unresolved.read_bytes()
        actual_sha = _sha256(identity_raw)
        if actual_sha != expected_sha:
            raise ValueError(
                f"source binding {key} identity mismatch: expected={expected_sha} actual={actual_sha}"
            )
        bound[key] = {
            "root": str(root),
            "identity": str(Path(identity_value)),
            "identity_sha256": actual_sha,
            "identity_bytes": len(identity_raw),
            "basis_sha256": basis_sha256,
        }
    return bound, _evidence(source_bindings, declaration_raw)


def materialize_virtual_backpack(
    source_receipt: str | Path,
    option_ledger: str | Path,
    source_bindings: str | Path,
    output: str | Path,
    *,
    arm_name: str,
    expected_assignment_sha256: str,
    expected_source_receipt_sha256: str | None = None,
    solve_input: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize one Backpack assignment as a zero-copy source-bound manifest.

    The output owns only assignment/index metadata. Tensor bytes remain in the
    immutable family roots named by ``source_bindings`` and are charged by the
    exact full-wire ledger, once per selected cell plus each shared activation.
    """

    receipt_path = Path(source_receipt).expanduser().resolve()
    receipt, receipt_raw = _load_object(receipt_path)
    receipt_sha = _sha256(receipt_raw)
    if (
        expected_source_receipt_sha256 is not None
        and receipt_sha != expected_source_receipt_sha256
    ):
        raise ValueError(
            "source receipt SHA-256 mismatch: "
            f"expected={expected_source_receipt_sha256} actual={receipt_sha}"
        )
    basis_sha256 = receipt.get("basis_sha256")
    if not isinstance(basis_sha256, str) or len(basis_sha256) != 64:
        raise ValueError("source receipt has no exact basis_sha256")
    arms = receipt.get("arms")
    arm = arms.get(arm_name) if isinstance(arms, dict) else None
    if not isinstance(arm, dict):
        raise ValueError(f"source receipt has no arm {arm_name!r}")
    if arm.get("assignment_map_sha256") != expected_assignment_sha256:
        raise ValueError("selected arm assignment SHA-256 does not match expectation")
    rows = arm.get("assignment_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("selected arm has no assignment rows")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("selected arm assignment rows are malformed")
    assignment = _assignment_from_rows(rows)
    assignment_raw = _canonical(assignment)
    assignment_sha = _sha256(assignment_raw)
    if assignment_sha != expected_assignment_sha256:
        raise ValueError(
            f"assignment rows hash to {assignment_sha}, expected {expected_assignment_sha256}"
        )
    if assignment != arm.get("assignment"):
        raise ValueError("assignment rows do not reproduce the sealed assignment map")

    immutable = receipt.get("immutable_input_set")
    ledger_descriptor = immutable.get("option_ledger") if isinstance(immutable, dict) else None
    if not isinstance(ledger_descriptor, dict):
        raise ValueError("source receipt lacks option-ledger evidence")
    ledger_path = Path(option_ledger).expanduser().resolve()
    ledger_raw = ledger_path.read_bytes()
    if (
        _sha256(ledger_raw) != ledger_descriptor.get("sha256")
        or len(ledger_raw) != ledger_descriptor.get("bytes")
    ):
        raise ValueError("option ledger does not match the source receipt")
    ledger_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for line in ledger_raw.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["cell_id"]), str(row["tier"]))
        if key in ledger_by_key:
            raise ValueError(f"duplicate option-ledger row: {key}")
        ledger_by_key[key] = row

    tier_counts: Counter[str] = Counter()
    tier_payload_bytes: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    selected_activation_ids: set[str] = set()
    index_rows: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row["cell_id"]), str(row["tier"]))
        ledger_row = ledger_by_key.get(key)
        if ledger_row is None:
            raise ValueError(f"selected cell is absent from option ledger: {key}")
        payload = int(ledger_row["physical_bytes"])
        if payload != int(row["physical_bytes"]):
            raise ValueError(f"selected payload byte mismatch: {key}")
        activation_ids = sorted(str(value) for value in ledger_row["activation_artifact_ids"])
        if activation_ids != sorted(str(value) for value in row["activation_artifact_ids"]):
            raise ValueError(f"selected activation binding mismatch: {key}")
        source_key = _source_key(row, ledger_row)
        tier_counts[key[1]] += 1
        tier_payload_bytes[key[1]] += payload
        source_counts[source_key] += 1
        selected_activation_ids.update(activation_ids)
        parsed = _parse_cell(key[0])
        index_rows.append(
            {
                "cell_id": key[0],
                "layer": parsed[0],
                "expert": parsed[1],
                "projection": parsed[2],
                "tier": key[1],
                "source_key": source_key,
                "physical_bytes": payload,
                "activation_artifact_ids": activation_ids,
            }
        )

    expected_counts = {str(k): int(v) for k, v in arm["tier_counts"].items()}
    if dict(tier_counts) != {k: v for k, v in expected_counts.items() if v}:
        actual_with_zero = {key: tier_counts.get(key, 0) for key in expected_counts}
        if actual_with_zero != expected_counts:
            raise ValueError(
                f"tier count mismatch: expected={expected_counts} actual={actual_with_zero}"
            )
    expected_tier_payload = {
        str(k): int(v) for k, v in arm["byte_accounting"]["tier_payload_bytes"].items()
    }
    actual_tier_payload = {
        key: tier_payload_bytes.get(key, 0) for key in expected_tier_payload
    }
    if actual_tier_payload != expected_tier_payload:
        raise ValueError(
            f"tier payload mismatch: expected={expected_tier_payload} actual={actual_tier_payload}"
        )

    activated = arm.get("activated_artifacts")
    if not isinstance(activated, list):
        raise ValueError("selected arm has no activated artifact list")
    activated_by_id = {str(row["artifact_id"]): row for row in activated}
    if set(activated_by_id) != selected_activation_ids:
        raise ValueError("shared activation set does not match selected ledger rows")
    activation_bytes = sum(int(activated_by_id[key]["bytes"]) for key in selected_activation_ids)
    payload_bytes = sum(tier_payload_bytes.values())
    accounting = arm.get("byte_accounting")
    if not isinstance(accounting, dict):
        raise ValueError("selected arm has no byte accounting")
    fixed_nonexpert_bytes = int(accounting["fixed_nonexpert_bytes"])
    derived = {
        "payload_bytes": payload_bytes,
        "activation_bytes": activation_bytes,
        "assigned_expert_bytes": payload_bytes + activation_bytes,
        "fixed_nonexpert_bytes": fixed_nonexpert_bytes,
        "assigned_package_bytes": payload_bytes
        + activation_bytes
        + fixed_nonexpert_bytes,
        "tier_payload_bytes": actual_tier_payload,
    }
    for key, value in derived.items():
        if accounting.get(key) != value:
            raise ValueError(
                f"sealed byte accounting mismatch for {key}: receipt={accounting.get(key)} derived={value}"
            )

    solve_descriptor = immutable.get("solve_input") if isinstance(immutable, dict) else None
    if not isinstance(solve_descriptor, dict):
        raise ValueError("source receipt lacks solve-input evidence")
    solve_path = (
        Path(solve_input).expanduser().resolve()
        if solve_input is not None
        else Path(str(solve_descriptor["path"])).expanduser().resolve()
    )
    solve_value, solve_raw = _load_object(solve_path)
    if (
        _sha256(solve_raw) != solve_descriptor.get("sha256")
        or len(solve_raw) != solve_descriptor.get("bytes")
    ):
        raise ValueError("solve input does not match the source receipt")
    denominator = int(solve_value["byte_accounting"]["expert_parameter_denominator"])
    if denominator <= 0:
        raise ValueError("solve input has invalid expert parameter denominator")
    geometry = solve_value.get("geometry")
    if not isinstance(geometry, dict) or int(geometry.get("cells", -1)) != len(rows):
        raise ValueError("solve-input geometry does not match assignment rows")

    bound_sources, source_declaration_evidence = _bind_sources(
        Path(source_bindings).expanduser().resolve(), basis_sha256, set(source_counts)
    )
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        if (output_path / VIRTUAL_MANIFEST).is_file():
            existing = verify_virtual_backpack(output_path)
            if existing["assignment_map_sha256"] == expected_assignment_sha256:
                return existing
        if any(output_path.iterdir()):
            raise FileExistsError(f"refusing non-empty virtual output: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    index_rows.sort(key=lambda row: _parse_cell(row["cell_id"]))
    index_raw = b"".join(_canonical(row) for row in index_rows)
    _atomic_write(output_path / ASSIGNMENT_FILE, assignment_raw)
    _atomic_write(output_path / INDEX_FILE, index_raw)
    manifest = {
        "schema": SCHEMA,
        "status": "PASS_LOGICAL_FULL_WIRE",
        "storage": {
            "kind": "external-family-roots-v1",
            "tensor_payload_copy_bytes": 0,
            "source_roots_bound_once": True,
        },
        "basis_sha256": basis_sha256,
        "arm_name": arm_name,
        "assignment_map_sha256": assignment_sha,
        "source_receipt": _evidence(receipt_path, receipt_raw),
        "option_ledger": _evidence(ledger_path, ledger_raw),
        "solve_input": _evidence(solve_path, solve_raw),
        "source_declaration": source_declaration_evidence,
        "source_bindings": bound_sources,
        "assignment": {
            "file": ASSIGNMENT_FILE,
            "sha256": _sha256(assignment_raw),
            "bytes": len(assignment_raw),
            "rows": len(rows),
        },
        "materialization_index": {
            "file": INDEX_FILE,
            "sha256": _sha256(index_raw),
            "bytes": len(index_raw),
            "rows": len(index_rows),
        },
        "tier_counts": {key: tier_counts.get(key, 0) for key in expected_counts},
        "source_component_counts": dict(sorted(source_counts.items())),
        "byte_accounting": derived,
        "activated_artifacts": [activated_by_id[key] for key in sorted(activated_by_id)],
        "expert_parameter_denominator": denominator,
        "expert_wire_bpw": derived["assigned_expert_bytes"] * 8 / denominator,
        "geometry": geometry,
    }
    manifest_raw = _canonical(manifest)
    _atomic_write(output_path / VIRTUAL_MANIFEST, manifest_raw)
    return verify_virtual_backpack(output_path)


def verify_virtual_backpack(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    manifest_path = root_path / VIRTUAL_MANIFEST
    manifest, manifest_raw = _load_object(manifest_path)
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "PASS_LOGICAL_FULL_WIRE":
        raise ValueError("virtual Backpack manifest schema/status mismatch")
    assignment_descriptor = manifest.get("assignment")
    index_descriptor = manifest.get("materialization_index")
    if not isinstance(assignment_descriptor, dict) or not isinstance(index_descriptor, dict):
        raise ValueError("virtual Backpack payload descriptors are missing")
    files: list[dict[str, Any]] = []
    for descriptor in (assignment_descriptor, index_descriptor):
        relative = Path(str(descriptor["file"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("virtual Backpack descriptor path is unsafe")
        path = root_path / relative
        raw = path.read_bytes()
        if _sha256(raw) != descriptor.get("sha256") or len(raw) != descriptor.get("bytes"):
            raise ValueError(f"virtual Backpack payload drift: {relative}")
        files.append({"file": str(relative), "sha256": _sha256(raw), "bytes": len(raw)})
    assignment = json.loads((root_path / ASSIGNMENT_FILE).read_bytes())
    if _sha256(_canonical(assignment)) != manifest.get("assignment_map_sha256"):
        raise ValueError("virtual Backpack assignment hash mismatch")
    for key, source in manifest["source_bindings"].items():
        root = Path(source["root"])
        identity = root / source["identity"]
        raw = identity.read_bytes()
        if _sha256(raw) != source["identity_sha256"] or len(raw) != source["identity_bytes"]:
            raise ValueError(f"virtual Backpack source identity drift: {key}")
    manifest_evidence = _evidence(manifest_path, manifest_raw)
    files.append(
        {
            "file": VIRTUAL_MANIFEST,
            "sha256": manifest_evidence["sha256"],
            "bytes": manifest_evidence["bytes"],
        }
    )
    files.sort(key=lambda row: row["file"])
    artifact_sha = _sha256(_canonical(files))
    return {
        "schema": "banana-smasher-backpack-virtual-verification-v1",
        "status": "PASS",
        "root": str(root_path),
        "assignment_map_sha256": manifest["assignment_map_sha256"],
        "basis_sha256": manifest["basis_sha256"],
        "tier_counts": manifest["tier_counts"],
        "source_component_counts": manifest["source_component_counts"],
        "byte_accounting": manifest["byte_accounting"],
        "expert_parameter_denominator": manifest["expert_parameter_denominator"],
        "expert_wire_bpw": manifest["expert_wire_bpw"],
        "logical_materialized_bytes": manifest["byte_accounting"]["assigned_package_bytes"],
        "tensor_payload_copy_bytes": 0,
        "unique_manifest_bytes": sum(row["bytes"] for row in files),
        "files": files,
        "artifact_sha256": artifact_sha,
        "logical_full_wire_verified": True,
        "source_roots_bound_once": True,
        "shared_activation_bytes_charged_once": manifest["byte_accounting"][
            "activation_bytes"
        ],
    }
