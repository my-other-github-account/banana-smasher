from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from decimal import Decimal, localcontext
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


def _materialize_legacy_virtual_backpack(
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
                "selection_group": str(row.get("selection_group", key[0])),
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


def _canonical_stage_results(
    run_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    from .backpack import (
        STAGES,
        BackpackPlan,
        _execution_plan_sha,
        _load_verified_stage,
        _sha_file,
    )

    plan_path = run_root / "PLAN.json"
    plan_value, _plan_raw = _load_object(plan_path)
    plan = BackpackPlan.from_mapping(plan_value, base_dir=run_root)
    plan_sha256 = _execution_plan_sha(plan)
    prior_stage_sha256: dict[str, str] = {}
    results: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    through = STAGES.index("solve_materialize") + 1
    for index, stage in enumerate(STAGES[:through], 1):
        path = run_root / "stages" / f"{index:02d}-{stage.replace('_', '-')}.json"
        result = _load_verified_stage(
            path,
            stage=stage,
            plan_sha256=plan_sha256,
            prior_stage_sha256=prior_stage_sha256,
            plan=plan,
        )
        if result is None:
            raise ValueError(f"canonical Backpack stage binding is invalid: {path}")
        results[stage] = result
        paths[stage] = path
        prior_stage_sha256[stage] = _sha_file(path)
    return results, paths


def _materialize_canonical_backpack_run(
    run_root: str | Path, output: str | Path
) -> dict[str, Any]:
    """Project a completed canonical Backpack solve into zero-copy contextual wire."""

    root = Path(run_root).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    stage_results, stage_paths = _canonical_stage_results(root)
    inspect = stage_results["inspect"]
    candidates = stage_results["candidates"]
    prediction = stage_results["pred"]
    solved = stage_results["solve_materialize"]
    candidates_path = stage_paths["candidates"]
    solved_path = stage_paths["solve_materialize"]
    basis_sha256 = inspect.get("model_manifest", {}).get("sha256")
    if not isinstance(basis_sha256, str) or len(basis_sha256) != 64:
        raise ValueError("canonical Backpack inspect stage has no model basis SHA-256")
    assignments = solved.get("assignment")
    option_rows = prediction.get("rows")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("canonical Backpack solve has no assignment rows")
    if not isinstance(option_rows, list) or not option_rows:
        raise ValueError("canonical Backpack prediction has no option rows")
    selected = {
        str(row["cell_id"]): str(row["tier"])
        for row in assignments
        if isinstance(row, dict)
    }
    if len(selected) != len(assignments):
        raise ValueError("canonical Backpack assignment cells are not unique")
    selection_group_by_cell = {
        str(row["cell_id"]): str(row.get("selection_group", row["cell_id"]))
        for row in assignments
        if isinstance(row, dict)
    }
    if any(not group for group in selection_group_by_cell.values()):
        raise ValueError("canonical Backpack selection groups are invalid")

    candidate_receipts: dict[tuple[str, str], tuple[Path, str]] = {}
    for tier in candidates.get("candidate_tiers", []):
        if not isinstance(tier, dict):
            raise ValueError("canonical candidate tier is malformed")
        tier_id = str(tier.get("tier"))
        for cell in tier.get("cells", []):
            if not isinstance(cell, dict):
                raise ValueError("canonical candidate cell is malformed")
            receipt_path = Path(str(cell.get("receipt"))).expanduser().resolve()
            receipt_raw = receipt_path.read_bytes()
            candidate_receipts[(str(cell.get("cell_id")), tier_id)] = (
                receipt_path,
                _sha256(receipt_raw),
            )

    ledger_rows: list[dict[str, Any]] = []
    available_activation_bytes: dict[str, int] = {}
    for row in option_rows:
        if not isinstance(row, dict):
            raise ValueError("canonical prediction option is malformed")
        cell_id = str(row.get("cell_id"))
        tier = str(row.get("tier"))
        receipt = candidate_receipts.get((cell_id, tier))
        if receipt is None:
            raise ValueError(f"candidate receipt missing for {(cell_id, tier)!r}")
        activations = row.get("activation_artifacts", [])
        if not isinstance(activations, list):
            raise ValueError("canonical activation artifacts must be an array")
        for artifact in activations:
            artifact_id = str(artifact["id"])
            byte_count = int(artifact["bytes"])
            if (
                artifact_id in available_activation_bytes
                and available_activation_bytes[artifact_id] != byte_count
            ):
                raise ValueError("canonical activation artifact bytes conflict")
            available_activation_bytes[artifact_id] = byte_count
        ledger_rows.append(
            {
                "schema": "banana-smasher-backpack-option-row-v1",
                "basis_sha256": basis_sha256,
                "cell_id": cell_id,
                "selection_group": selection_group_by_cell[cell_id],
                "tier": tier,
                "source_key": tier,
                "physical_receipt_sha256": receipt[1],
                "physical_receipt_path": str(receipt[0]),
                "physical_bytes": int(row["physical_bytes"]),
                "activation_artifact_ids": sorted(
                    str(artifact["id"]) for artifact in activations
                ),
                "prediction_by_class": dict(row.get("prediction_by_class", {})),
            }
        )
    ledger_rows.sort(key=lambda row: (str(row["cell_id"]), str(row["tier"])))

    assignment = dict(sorted(selected.items()))
    assignment_raw = _canonical(assignment)
    assignment_sha256 = _sha256(assignment_raw)
    selected_rows = {
        str(row["cell_id"]): row for row in ledger_rows if selected.get(str(row["cell_id"])) == row["tier"]
    }
    if len(selected_rows) != len(selected):
        raise ValueError("canonical selected assignment is absent from option ledger")
    index_rows = [
        {
            "cell_id": cell_id,
            "selection_group": row["selection_group"],
            "tier": row["tier"],
            "source_key": row["source_key"],
            "physical_receipt_sha256": row["physical_receipt_sha256"],
            "physical_bytes": row["physical_bytes"],
            "activation_artifact_ids": row["activation_artifact_ids"],
        }
        for cell_id, row in sorted(selected_rows.items())
    ]
    index_raw = b"".join(_canonical(row) for row in index_rows)
    ledger_raw = b"".join(_canonical(row) for row in ledger_rows)

    accounting = solved.get("byte_accounting")
    if not isinstance(accounting, dict):
        raise ValueError("canonical Backpack solve has no byte accounting")
    activated = solved.get("activated_artifacts", [])
    if not isinstance(activated, list):
        raise ValueError("canonical Backpack solve activation list is malformed")
    selected_activation_rows = [
        {"artifact_id": str(row["id"]), "bytes": int(row["bytes"])}
        for row in activated
    ]
    available_activation_rows = [
        {"artifact_id": artifact_id, "bytes": available_activation_bytes[artifact_id]}
        for artifact_id in sorted(available_activation_bytes)
    ]
    solve_input = {
        "schema": "banana-smasher-backpack-exact-matched-full-wire-input-v2",
        "status": "PASS",
        "basis_sha256": basis_sha256,
        "budget": {
            "fixed_nonexpert_bytes": int(accounting["fixed_bytes"]),
            "total_package_bytes": int(accounting["target_whole_model_bytes"]),
        },
        "activation_artifacts": available_activation_rows,
        "byte_accounting": {
            "expert_parameter_denominator": int(inspect["weight_count"])
        },
        "geometry": {"cells": len(selected)},
    }
    solve_raw = _canonical(solve_input)

    if destination.exists() and any(destination.iterdir()):
        if (destination / VIRTUAL_MANIFEST).is_file():
            existing = verify_virtual_backpack(destination)
            if existing["assignment_map_sha256"] == assignment_sha256:
                return existing
        raise FileExistsError(f"refusing non-empty virtual output: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    assignment_path = destination / ASSIGNMENT_FILE
    index_path = destination / INDEX_FILE
    ledger_path = destination / "OPTION_LEDGER.jsonl"
    solve_input_path = destination / "SOLVE_INPUT.json"
    _atomic_write(assignment_path, assignment_raw)
    _atomic_write(index_path, index_raw)
    _atomic_write(ledger_path, ledger_raw)
    _atomic_write(solve_input_path, solve_raw)

    candidates_raw = candidates_path.read_bytes()
    source_bindings = {
        tier: {
            "root": str(root),
            "identity": str(candidates_path.relative_to(root)),
            "identity_sha256": _sha256(candidates_raw),
            "identity_bytes": len(candidates_raw),
            "basis_sha256": basis_sha256,
        }
        for tier in sorted({str(row["tier"]) for row in ledger_rows})
    }
    tier_counts = Counter(selected.values())
    tier_payload_bytes = Counter()
    for cell_id, row in selected_rows.items():
        tier_payload_bytes[str(row["tier"])] += int(row["physical_bytes"])
    payload_bytes = sum(tier_payload_bytes.values())
    activation_bytes = sum(int(row["bytes"]) for row in selected_activation_rows)
    fixed_bytes = int(accounting["fixed_bytes"])
    all_tiers = sorted({str(row["tier"]) for row in ledger_rows})
    derived_accounting = {
        "payload_bytes": payload_bytes,
        "activation_bytes": activation_bytes,
        "assigned_expert_bytes": payload_bytes + activation_bytes,
        "fixed_nonexpert_bytes": fixed_bytes,
        "assigned_package_bytes": payload_bytes + activation_bytes + fixed_bytes,
        "tier_payload_bytes": {
            tier: tier_payload_bytes.get(tier, 0) for tier in all_tiers
        },
    }
    manifest = {
        "schema": SCHEMA,
        "status": "PASS_LOGICAL_FULL_WIRE",
        "storage": {
            "kind": "canonical-candidate-roots-v1",
            "tensor_payload_copy_bytes": 0,
            "source_roots_bound_once": True,
        },
        "basis_sha256": basis_sha256,
        "arm_name": "canonical-backpack-solve",
        "assignment_map_sha256": assignment_sha256,
        "source_receipt": _evidence(solved_path),
        "option_ledger": _evidence(ledger_path, ledger_raw),
        "solve_input": _evidence(solve_input_path, solve_raw),
        "source_bindings": source_bindings,
        "assignment": {
            "file": ASSIGNMENT_FILE,
            "sha256": assignment_sha256,
            "bytes": len(assignment_raw),
            "rows": len(selected),
        },
        "materialization_index": {
            "file": INDEX_FILE,
            "sha256": _sha256(index_raw),
            "bytes": len(index_raw),
            "rows": len(index_rows),
        },
        "tier_counts": {tier: tier_counts.get(tier, 0) for tier in all_tiers},
        "source_component_counts": dict(sorted(tier_counts.items())),
        "byte_accounting": derived_accounting,
        "activated_artifacts": selected_activation_rows,
        "available_artifacts": available_activation_rows,
        "expert_parameter_denominator": int(inspect["weight_count"]),
        "expert_wire_bpw": (payload_bytes + activation_bytes)
        * 8
        / int(inspect["weight_count"]),
        "geometry": {"cells": len(selected)},
    }
    _atomic_write(destination / VIRTUAL_MANIFEST, _canonical(manifest))
    return verify_virtual_backpack(destination)


def materialize_provenance_virtual_backpack(
    assignment: str | Path,
    solve_receipt: str | Path,
    option_ledger: str | Path,
    source_bindings: str | Path,
    output: str | Path,
    *,
    expected_assignment_sha256: str,
    expected_solve_receipt_sha256: str,
    logical_base_parameters: int,
) -> dict[str, Any]:
    """Project a terminal provenance-weighted solve into exact64 virtual wire."""

    if (
        isinstance(logical_base_parameters, bool)
        or not isinstance(logical_base_parameters, int)
        or logical_base_parameters <= 0
    ):
        raise ValueError("logical_base_parameters must be a positive integer")
    assignment_path = Path(assignment).expanduser().resolve()
    receipt_path = Path(solve_receipt).expanduser().resolve()
    ledger_path = Path(option_ledger).expanduser().resolve()
    assignment_value, assignment_raw = _load_object(assignment_path)
    receipt_value, receipt_raw = _load_object(receipt_path)
    ledger_raw = ledger_path.read_bytes()
    assignment_sha = _sha256(assignment_raw)
    receipt_sha = _sha256(receipt_raw)

    if assignment_sha != expected_assignment_sha256:
        raise ValueError("provenance assignment SHA-256 mismatch")
    if receipt_sha != expected_solve_receipt_sha256:
        raise ValueError("provenance solve receipt SHA-256 mismatch")
    if (
        assignment_value.get("schema")
        != "banana-smasher-provenance-weighted-assignment-v1"
        or assignment_value.get("status") != "PASS_PREDICTION_ONLY"
    ):
        raise ValueError("provenance assignment schema/status mismatch")
    if (
        receipt_value.get("schema")
        != "banana-smasher-provenance-weighted-solve-receipt-v1"
        or receipt_value.get("status") != "PASS"
    ):
        raise ValueError("provenance solve receipt schema/status mismatch")
    for label, descriptor, path, raw in (
        ("assignment", receipt_value.get("assignment"), assignment_path, assignment_raw),
        ("option ledger", receipt_value.get("option_ledger"), ledger_path, ledger_raw),
    ):
        if (
            not isinstance(descriptor, dict)
            or Path(str(descriptor.get("path"))).expanduser().resolve() != path
            or descriptor.get("sha256") != _sha256(raw)
            or descriptor.get("bytes") != len(raw)
        ):
            raise ValueError(f"provenance solve receipt {label} binding mismatch")

    identities = (
        "model_id",
        "model_revision",
        "basis_sha256",
        "bank_sha256",
        "teacher_sha256",
        "scorer_sha256",
    )
    rows = assignment_value.get("assignments")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("provenance assignment rows are missing or malformed")
    selected = {(str(row["cell_id"]), str(row["tier"])): row for row in rows}
    selected_cells = {cell_id for cell_id, _tier in selected}
    if len(selected) != len(rows) or len(selected_cells) != len(rows):
        raise ValueError("provenance assignment contains duplicate cells")

    ledger_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, line in enumerate(ledger_raw.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"provenance option row {line_number} is not an object")
        key = (str(row.get("cell_id")), str(row.get("tier")))
        if key in ledger_by_key:
            raise ValueError(f"duplicate provenance option row: {key}")
        ledger_by_key[key] = row

    tier_counts: Counter[str] = Counter()
    tier_payload_bytes: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    activation_ids: set[str] = set()
    index_rows: list[dict[str, Any]] = []
    for key, selected_row in selected.items():
        ledger_row = ledger_by_key.get(key)
        if ledger_row is None:
            raise ValueError(f"selected provenance option is absent from ledger: {key}")
        for identity in identities:
            if ledger_row.get(identity) != assignment_value.get(identity):
                raise ValueError(f"selected provenance option {identity} mismatch: {key}")
        byte_count = int(ledger_row["physical_bytes"])
        if int(selected_row["bytes"]) != byte_count:
            raise ValueError(f"selected provenance option byte mismatch: {key}")
        producer = ledger_row.get("physical_producer")
        if not isinstance(producer, dict):
            raise ValueError(f"selected provenance option lacks physical producer: {key}")
        producer_path = Path(str(producer.get("path"))).expanduser().resolve()
        expected_producer_sha = str(producer.get("sha256"))
        if len(expected_producer_sha) != 64 or any(
            value not in "0123456789abcdef" for value in expected_producer_sha
        ):
            raise ValueError(f"selected provenance physical producer identity is invalid: {key}")
        # The solve receipt already binds the complete ledger bytes. Producer
        # receipts were verified when that ledger was built and need not remain
        # mounted on the machine that projects the terminal assignment.
        actual_producer_sha = expected_producer_sha
        source_key = key[1]
        layer, expert, projection = _parse_cell(key[0])
        ids = sorted(str(value) for value in ledger_row.get("activation_ids", []))
        tier_counts[key[1]] += 1
        tier_payload_bytes[key[1]] += byte_count
        source_counts[source_key] += 1
        activation_ids.update(ids)
        index_rows.append(
            {
                "cell_id": key[0],
                "selection_group": key[0],
                "layer": layer,
                "expert": expert,
                "projection": projection,
                "tier": key[1],
                "source_key": source_key,
                "physical_bytes": byte_count,
                "physical_receipt_path": str(producer_path),
                "physical_receipt_sha256": actual_producer_sha,
                "physical_artifact_sha256": producer.get("artifact_sha256"),
                "activation_artifact_ids": ids,
            }
        )

    raw_activated = assignment_value.get("activation_artifacts", [])
    if not isinstance(raw_activated, list) or not all(
        isinstance(row, dict) for row in raw_activated
    ):
        raise ValueError("provenance activation artifacts are malformed")
    activated_artifacts: list[dict[str, Any]] = []
    activated_ids: set[str] = set()
    activation_bytes = 0
    for row in raw_activated:
        artifact_id = str(row.get("id", row.get("artifact_id", "")))
        byte_count = row.get("bytes")
        if (
            not artifact_id
            or artifact_id in activated_ids
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise ValueError("provenance activation artifact identity/bytes are invalid")
        activated_ids.add(artifact_id)
        activation_bytes += byte_count
        activated_artifacts.append(
            {**row, "artifact_id": artifact_id, "bytes": byte_count}
        )
    if activation_ids != activated_ids:
        raise ValueError("selected provenance activation-artifact set mismatch")

    source_accounting = assignment_value.get("whole_model_accounting")
    if (
        not isinstance(source_accounting, dict)
        or receipt_value.get("whole_model_accounting") != source_accounting
    ):
        raise ValueError("provenance whole-model accounting binding mismatch")
    payload_bytes = sum(tier_payload_bytes.values())
    fixed_bytes = int(source_accounting["fixed_nonexpert_bytes"])
    raw_padding_bytes = source_accounting.get("padding_bytes", 0)
    if (
        isinstance(raw_padding_bytes, bool)
        or not isinstance(raw_padding_bytes, int)
        or raw_padding_bytes < 0
    ):
        raise ValueError("provenance padding bytes must be a non-negative integer")
    padding_bytes = raw_padding_bytes
    whole_bytes = int(source_accounting["whole_shipping_bytes"])
    if (
        int(source_accounting["selected_expert_bytes"])
        != payload_bytes + activation_bytes
        or whole_bytes
        != payload_bytes + activation_bytes + fixed_bytes + padding_bytes
        or fixed_bytes
        != int(source_accounting["dense_nonrouted_bytes"])
        + int(source_accounting["repair_bytes"])
        + int(source_accounting["metadata_bytes"])
        or int(source_accounting["shipping_slack_bytes"])
        != int(source_accounting["shipping_bytes_cap"]) - whole_bytes
    ):
        raise ValueError("provenance whole-model accounting equation mismatch")
    numerator = whole_bytes * 8
    with localcontext() as context:
        context.prec = 80
        decimal_bpw = format(
            Decimal(numerator) / Decimal(logical_base_parameters), "f"
        )
    whole_model_accounting = {
        "expert_physical_wire_bytes": payload_bytes + activation_bytes,
        "dense_nonrouted_bytes": int(source_accounting["dense_nonrouted_bytes"]),
        "repair_bytes": int(source_accounting["repair_bytes"]),
        "metadata_bytes": int(source_accounting["metadata_bytes"]),
        "fixed_nonexpert_bytes": fixed_bytes,
        "padding_bytes": padding_bytes,
        "padding_policy": source_accounting.get("padding_policy"),
        "whole_shipping_bytes": whole_bytes,
        "shipping_bytes_cap": int(source_accounting["shipping_bytes_cap"]),
        "shipping_slack_bytes": int(source_accounting["shipping_slack_bytes"]),
        "logical_base_parameters": logical_base_parameters,
        "whole_model_bpw_numerator_bits": numerator,
        "whole_model_bpw_exact_ratio": f"{numerator}/{logical_base_parameters}",
        "whole_model_bpw_decimal": decimal_bpw,
    }
    derived = {
        "payload_bytes": payload_bytes,
        "activation_bytes": activation_bytes,
        "assigned_expert_bytes": payload_bytes + activation_bytes,
        "fixed_nonexpert_bytes": fixed_bytes,
        "padding_bytes": padding_bytes,
        "assigned_package_bytes": whole_bytes,
        "tier_payload_bytes": dict(sorted(tier_payload_bytes.items())),
    }
    required_sources = set(source_counts)
    bound_sources, source_declaration = _bind_sources(
        Path(source_bindings).expanduser().resolve(),
        str(assignment_value["basis_sha256"]),
        required_sources,
    )
    assignment_map = _assignment_from_rows(rows)
    assignment_map_raw = _canonical(assignment_map)
    index_rows.sort(key=lambda row: _parse_cell(row["cell_id"]))
    index_raw = b"".join(_canonical(row) for row in index_rows)
    output_path = Path(output).expanduser().resolve()
    if output_path.exists() and any(output_path.iterdir()):
        if (output_path / VIRTUAL_MANIFEST).is_file():
            existing_manifest, _ = _load_object(output_path / VIRTUAL_MANIFEST)
            if existing_manifest.get("source_assignment", {}).get("sha256") == assignment_sha:
                return verify_virtual_backpack(output_path)
        raise FileExistsError(f"refusing non-empty virtual output: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    _atomic_write(output_path / ASSIGNMENT_FILE, assignment_map_raw)
    _atomic_write(output_path / INDEX_FILE, index_raw)
    manifest = {
        "schema": SCHEMA,
        "status": "PASS_LOGICAL_FULL_WIRE",
        "storage": {
            "kind": "external-family-roots-v1",
            "tensor_payload_copy_bytes": 0,
            "source_roots_bound_once": True,
        },
        "model_id": assignment_value["model_id"],
        "model_revision": assignment_value["model_revision"],
        "basis_sha256": assignment_value["basis_sha256"],
        "bank_sha256": assignment_value["bank_sha256"],
        "teacher_sha256": assignment_value["teacher_sha256"],
        "scorer_sha256": assignment_value["scorer_sha256"],
        "arm_name": "provenance-weighted-clean102",
        "assignment_map_sha256": _sha256(assignment_map_raw),
        "source_assignment": _evidence(assignment_path, assignment_raw),
        "source_receipt": _evidence(receipt_path, receipt_raw),
        "option_ledger": _evidence(ledger_path, ledger_raw),
        "source_declaration": source_declaration,
        "source_bindings": bound_sources,
        "assignment": {
            "file": ASSIGNMENT_FILE,
            "sha256": _sha256(assignment_map_raw),
            "bytes": len(assignment_map_raw),
            "rows": len(rows),
        },
        "materialization_index": {
            "file": INDEX_FILE,
            "sha256": _sha256(index_raw),
            "bytes": len(index_raw),
            "rows": len(index_rows),
        },
        "tier_counts": dict(sorted(tier_counts.items())),
        "source_component_counts": dict(sorted(source_counts.items())),
        "byte_accounting": derived,
        "whole_model_accounting": whole_model_accounting,
        "activated_artifacts": sorted(
            activated_artifacts, key=lambda row: row["artifact_id"]
        ),
        "expert_parameter_denominator": logical_base_parameters,
        "expert_wire_bpw": payload_bytes * 8 / logical_base_parameters,
        "geometry": {"cells": len(rows)},
    }
    _atomic_write(output_path / VIRTUAL_MANIFEST, _canonical(manifest))
    return verify_virtual_backpack(output_path)


def materialize_mixed_v7_virtual_backpack(
    solve_root: str | Path,
    materialized_members: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Project solve-mixed output plus sealed V7 members into canonical runtime wire."""

    from .repack import bind_mixed_v7_member_contract

    solve = Path(solve_root).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        if (destination / VIRTUAL_MANIFEST).is_file():
            manifest, _ = _load_object(destination / VIRTUAL_MANIFEST)
            if manifest.get("storage", {}).get("kind") == "mixed-v7-member-contract-v1":
                solve_receipt_path = solve / "RECEIPT.json"
                if solve_receipt_path.is_file():
                    solve_receipt, _ = _load_object(solve_receipt_path)
                    solve_accounting = solve_receipt.get("byte_accounting")
                    accounting = manifest.get("byte_accounting")
                    if isinstance(solve_accounting, dict) and isinstance(accounting, dict):
                        logical_payload = int(solve_accounting.get("candidate_payload_bytes", 0))
                        fixed = int(solve_accounting.get("fixed_nonexpert_bytes", 0))
                        if logical_payload > 0:
                            accounting.setdefault("physical_payload_bytes", accounting["payload_bytes"])
                            accounting["payload_bytes"] = logical_payload
                            accounting["assigned_expert_bytes"] = logical_payload
                            accounting["fixed_nonexpert_bytes"] = fixed
                            accounting["assigned_package_bytes"] = logical_payload + fixed
                            _atomic_write(destination / VIRTUAL_MANIFEST, _canonical(manifest))
            return verify_virtual_backpack(destination)
        raise FileExistsError(f"refusing non-empty virtual output: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    contract_path = destination / "MIXED_V7_MEMBER_CONTRACT.json"
    bind_receipt = bind_mixed_v7_member_contract(
        solve, materialized_members, output=contract_path
    )
    identity, identity_raw = _load_object(solve / "identity.json")
    contract, contract_raw = _load_object(contract_path)
    by_expert: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for member in contract["members"]:
        layer_text, expert_text, _ = member["cell_id"].split(".")
        key = (
            int(layer_text.removeprefix("L")),
            int(expert_text.removeprefix("E")),
        )
        by_expert.setdefault(key, []).append(member)
    assignment_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    tier_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    payload_bytes = 0
    for (layer, expert), rows in sorted(by_expert.items()):
        tier = rows[0]["tier"]
        source_key = f"{tier}_v7"
        by_projection = {row["cell_id"].split(".")[2]: row for row in rows}
        physical_by_logical = (
            {"down": ["w2"], "fused13": ["w1", "w3"]}
            if tier == "qtip2"
            else {"down": ["down"], "fused13": ["fused13"]}
        )
        for projection, physical in physical_by_logical.items():
            cell_id = f"L{layer}:E{expert}:{projection}"
            byte_count = sum(
                int(by_projection[name]["payload"]["bytes"]) for name in physical
            )
            assignment_rows.append(
                {"cell_id": cell_id, "tier": tier}
            )
            index_rows.append(
                {
                    "cell_id": cell_id,
                    "selection_group": f"L{layer}.E{expert}",
                    "layer": layer,
                    "expert": expert,
                    "projection": projection,
                    "tier": tier,
                    "source_key": source_key,
                    "physical_bytes": byte_count,
                    "activation_artifact_ids": [],
                    "mixed_v7_members": [by_projection[name]["cell_id"] for name in physical],
                }
            )
            tier_counts[tier] += 1
            source_counts[source_key] += 1
            payload_bytes += byte_count
    assignment = _assignment_from_rows(assignment_rows)
    assignment_raw = _canonical(assignment)
    index_raw = b"".join(_canonical(row) for row in index_rows)
    _atomic_write(destination / ASSIGNMENT_FILE, assignment_raw)
    _atomic_write(destination / INDEX_FILE, index_raw)
    fixed_bytes = 0
    logical_payload_bytes = payload_bytes
    receipt_path = solve / "RECEIPT.json"
    if receipt_path.is_file():
        solve_receipt, _ = _load_object(receipt_path)
        accounting = solve_receipt.get("byte_accounting")
        if isinstance(accounting, dict):
            fixed_bytes = int(accounting.get("fixed_nonexpert_bytes", 0))
            logical_payload_bytes = int(accounting.get("candidate_payload_bytes", payload_bytes))
    source_bindings = {
        source: {
            "root": str(destination),
            "identity": contract_path.name,
            "identity_sha256": _sha256(contract_raw),
            "identity_bytes": len(contract_raw),
            "basis_sha256": identity["basis_sha256"],
        }
        for source in sorted(source_counts)
    }
    manifest = {
        "schema": SCHEMA,
        "status": "PASS_LOGICAL_FULL_WIRE",
        "storage": {
            "kind": "mixed-v7-member-contract-v1",
            "tensor_payload_copy_bytes": 0,
            "source_roots_bound_once": True,
        },
        "basis_sha256": identity["basis_sha256"],
        "arm_name": "solve-mixed-v7",
        "assignment_map_sha256": _sha256(assignment_raw),
        "source_receipt": _evidence(solve / "identity.json", identity_raw),
        "source_bindings": source_bindings,
        "mixed_v7_member_contract": _evidence(contract_path, contract_raw),
        "mixed_v7_bind_receipt": bind_receipt,
        "runtime_binding": {
            "basis_sha256": identity["basis_sha256"],
            "virtual_manifest": str(destination / VIRTUAL_MANIFEST),
            "materialization_index": str(destination / INDEX_FILE),
            "mixed_v7_member_contract": str(contract_path),
        },
        "assignment": {
            "file": ASSIGNMENT_FILE,
            "sha256": _sha256(assignment_raw),
            "bytes": len(assignment_raw),
            "rows": len(index_rows),
        },
        "materialization_index": {
            "file": INDEX_FILE,
            "sha256": _sha256(index_raw),
            "bytes": len(index_raw),
            "rows": len(index_rows),
        },
        "tier_counts": dict(sorted(tier_counts.items())),
        "source_component_counts": dict(sorted(source_counts.items())),
        "byte_accounting": {
            "payload_bytes": logical_payload_bytes,
            "physical_payload_bytes": payload_bytes,
            "activation_bytes": 0,
            "assigned_expert_bytes": logical_payload_bytes,
            "fixed_nonexpert_bytes": fixed_bytes,
            "assigned_package_bytes": logical_payload_bytes + fixed_bytes,
            "tier_payload_bytes": {},
        },
        "activated_artifacts": [],
        "expert_parameter_denominator": 1,
        "expert_wire_bpw": float(payload_bytes * 8),
        "geometry": {"cells": len(index_rows)},
    }
    _atomic_write(destination / VIRTUAL_MANIFEST, _canonical(manifest))
    return verify_virtual_backpack(destination)


def materialize_virtual_backpack(
    source_receipt: str | Path,
    option_ledger: str | Path,
    source_bindings: str | Path | None = None,
    output: str | Path | None = None,
    *,
    arm_name: str | None = None,
    expected_assignment_sha256: str | None = None,
    expected_source_receipt_sha256: str | None = None,
    solve_input: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize canonical runs, while retaining the historical receipt adapter."""

    source = Path(source_receipt).expanduser().resolve()
    if source.is_dir() and (source / "PLAN.json").is_file():
        if source_bindings is not None or output is not None:
            raise TypeError("canonical run materialization accepts RUN_ROOT and OUTPUT only")
        return _materialize_canonical_backpack_run(source, option_ledger)
    if source_bindings is None or output is None or arm_name is None or expected_assignment_sha256 is None:
        raise TypeError("legacy virtual materialization requires all bound receipt inputs")
    return _materialize_legacy_virtual_backpack(
        source,
        option_ledger,
        source_bindings,
        output,
        arm_name=arm_name,
        expected_assignment_sha256=expected_assignment_sha256,
        expected_source_receipt_sha256=expected_source_receipt_sha256,
        solve_input=solve_input,
    )


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
    index_rows: list[dict[str, Any]] = []
    for line in (root_path / INDEX_FILE).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("virtual Backpack index row is invalid")
        index_rows.append(row)
    if len(index_rows) != index_descriptor.get("rows"):
        raise ValueError("virtual Backpack index row count mismatch")
    cells = [str(row.get("cell_id")) for row in index_rows]
    if len(set(cells)) != len(cells):
        raise ValueError("virtual Backpack index cells are not unique")
    payload_bytes = 0
    tier_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    activation_ids: set[str] = set()
    for row in index_rows:
        byte_count = row.get("physical_bytes")
        ids = row.get("activation_artifact_ids", [])
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
            raise ValueError("virtual Backpack index payload bytes are invalid")
        if not isinstance(ids, list) or any(
            not isinstance(value, str) or not value for value in ids
        ):
            raise ValueError("virtual Backpack index activation ids are invalid")
        payload_bytes += byte_count
        tier_counts[str(row.get("tier"))] += 1
        source_counts[str(row.get("source_key"))] += 1
        activation_ids.update(ids)
    declared_counts = {str(key): int(value) for key, value in manifest["tier_counts"].items()}
    if {key: tier_counts.get(key, 0) for key in declared_counts} != declared_counts:
        raise ValueError("virtual Backpack tier counts do not match its index")
    if dict(source_counts) != {
        str(key): int(value) for key, value in manifest["source_component_counts"].items()
    }:
        raise ValueError("virtual Backpack source counts do not match its index")
    artifacts = manifest.get("activated_artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("virtual Backpack activation registry is invalid")
    artifact_by_id = {str(row.get("artifact_id")): row for row in artifacts if isinstance(row, dict)}
    if set(artifact_by_id) != activation_ids:
        raise ValueError("virtual Backpack activation registry does not match its index")
    activation_bytes = sum(int(artifact_by_id[key]["bytes"]) for key in activation_ids)
    accounting = manifest.get("byte_accounting")
    if not isinstance(accounting, dict):
        raise ValueError("virtual Backpack byte accounting is missing")
    if accounting.get("physical_payload_bytes", accounting.get("payload_bytes")) != payload_bytes:
        raise ValueError("virtual Backpack physical payload accounting does not match its index")
    if accounting.get("activation_bytes") != activation_bytes:
        raise ValueError("virtual Backpack activation accounting does not match its index")
    logical_payload_bytes = accounting.get("payload_bytes")
    if not isinstance(logical_payload_bytes, int) or isinstance(logical_payload_bytes, bool) or logical_payload_bytes < payload_bytes:
        raise ValueError("virtual Backpack logical payload accounting is invalid")
    if accounting.get("assigned_expert_bytes") != logical_payload_bytes + activation_bytes:
        raise ValueError("virtual Backpack expert accounting is inconsistent")
    padding_bytes = accounting.get("padding_bytes", 0)
    if (
        isinstance(padding_bytes, bool)
        or not isinstance(padding_bytes, int)
        or padding_bytes < 0
    ):
        raise ValueError("virtual Backpack padding accounting is invalid")
    if accounting.get("assigned_package_bytes") != (
        logical_payload_bytes
        + activation_bytes
        + accounting.get("fixed_nonexpert_bytes", -1)
        + padding_bytes
    ):
        raise ValueError("virtual Backpack package accounting is inconsistent")

    ledger_descriptor = manifest.get("option_ledger")
    if isinstance(ledger_descriptor, dict):
        ledger_value = ledger_descriptor.get("path", ledger_descriptor.get("file"))
        if not isinstance(ledger_value, str) or not ledger_value:
            raise ValueError("virtual Backpack option-ledger path is invalid")
        ledger_path = Path(ledger_value).expanduser()
        if not ledger_path.is_absolute():
            ledger_path = root_path / ledger_path
        ledger_raw = ledger_path.read_bytes()
        if (
            _sha256(ledger_raw) != ledger_descriptor.get("sha256")
            or len(ledger_raw) != ledger_descriptor.get("bytes")
        ):
            raise ValueError("virtual Backpack option ledger drift")
        for line in ledger_raw.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            receipt_value = row.get("physical_receipt_path")
            expected = row.get("physical_receipt_sha256")
            if receipt_value is None:
                continue
            receipt_path = Path(str(receipt_value)).expanduser()
            if receipt_path.is_symlink() or not receipt_path.is_file():
                raise ValueError("virtual Backpack physical receipt is missing or unsafe")
            if _sha256(receipt_path.read_bytes()) != expected:
                raise ValueError("virtual Backpack physical receipt drift")
    for key, source in manifest["source_bindings"].items():
        source_root = Path(source["root"]).expanduser().resolve()
        relative = Path(source["identity"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"virtual Backpack source identity is unsafe: {key}")
        unresolved = source_root / relative
        identity = unresolved.resolve()
        try:
            identity.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(f"virtual Backpack source identity escapes its root: {key}") from exc
        if unresolved.is_symlink() or not unresolved.is_file():
            raise ValueError(f"virtual Backpack source identity is missing or unsafe: {key}")
        raw = unresolved.read_bytes()
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
