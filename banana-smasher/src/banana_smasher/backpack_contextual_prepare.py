from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .backpack_contextual import ContextualValuationError, _atomic_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ContextualValuationError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ContextualValuationError(
                f"expected JSON object at {path}:{line_number}"
            )
        rows.append(value)
    return rows


def _bound_path(
    descriptor: Mapping[str, Any],
    *,
    root: Path,
    role: str,
) -> Path:
    declared = descriptor.get("file", descriptor.get("path"))
    if not isinstance(declared, str) or not declared:
        raise ContextualValuationError(f"{role} descriptor path is invalid")
    path = Path(declared).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    expected_bytes = descriptor.get("bytes")
    expected_sha256 = descriptor.get("sha256")
    if not isinstance(expected_bytes, int) or expected_bytes < 0:
        raise ContextualValuationError(f"{role} descriptor bytes are invalid")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ContextualValuationError(f"{role} descriptor sha256 is invalid")
    if not path.is_file():
        raise FileNotFoundError(f"{role} input is missing: {path}")
    if path.stat().st_size != expected_bytes or _sha256(path) != expected_sha256:
        raise ContextualValuationError(f"{role} descriptor identity mismatch")
    return path


def _physical_source_key(row: Mapping[str, Any]) -> str:
    components = row.get("prediction_components")
    if isinstance(components, Mapping):
        explicit = components.get("physical_source_key")
        if isinstance(explicit, str) and explicit:
            return explicit
        qtip_k = components.get("qtip_k")
        if isinstance(qtip_k, int) and not isinstance(qtip_k, bool) and qtip_k > 0:
            return f"qtip{qtip_k}"
    source = row.get("source_key", row.get("tier"))
    if not isinstance(source, str) or not source:
        raise ContextualValuationError("option physical source key is invalid")
    return source


def _physical_identity(
    row: Mapping[str, Any],
    *,
    basis_sha256: str,
) -> str:
    cell = row.get("cell_id")
    byte_count = row.get("physical_bytes")
    activation_ids = row.get("activation_artifact_ids", [])
    if not isinstance(cell, str) or not cell:
        raise ContextualValuationError("physical option cell_id is invalid")
    if not isinstance(byte_count, int) or byte_count < 0:
        raise ContextualValuationError("physical option bytes are invalid")
    if (
        not isinstance(activation_ids, list)
        or any(not isinstance(value, str) or not value for value in activation_ids)
        or len(set(activation_ids)) != len(activation_ids)
    ):
        raise ContextualValuationError("physical option activation ids are invalid")
    identity = {
        "basis_sha256": basis_sha256,
        "cell_id": cell,
        "physical_source_key": _physical_source_key(row),
        "physical_bytes": byte_count,
        "activation_artifact_ids": sorted(activation_ids),
    }
    receipt_sha256 = row.get("physical_receipt_sha256")
    if receipt_sha256 is not None:
        if not isinstance(receipt_sha256, str) or len(receipt_sha256) != 64:
            raise ContextualValuationError("physical option receipt SHA is invalid")
        identity["physical_receipt_sha256"] = receipt_sha256
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _prepared_member_identity(member: Mapping[str, Any], *, basis_sha256: str) -> str:
    identity = {
        "basis_sha256": basis_sha256,
        "cell_id": member.get("cell"),
        "physical_source_key": member.get("physical_source_key"),
        "physical_bytes": member.get("payload_bytes"),
        "activation_artifact_ids": sorted(
            str(row.get("id"))
            for row in member.get("activations", [])
            if isinstance(row, Mapping)
        ),
    }
    if member.get("physical_receipt_sha256") is not None:
        identity["physical_receipt_sha256"] = member["physical_receipt_sha256"]
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _group_physical_identity(
    selection_group: str,
    option: str,
    members: list[Mapping[str, Any]],
    *,
    basis_sha256: str,
) -> str:
    value = {
        "basis_sha256": basis_sha256,
        "selection_group": selection_group,
        "option": option,
        "members": [
            {
                "cell": str(member["cell"]),
                "physical_identity": _prepared_member_identity(
                    member, basis_sha256=basis_sha256
                ),
            }
            for member in sorted(members, key=lambda row: str(row["cell"]))
        ],
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def prepare_contextual_iteration(
    virtual_manifest_path: str | Path,
    score_receipt_path: str | Path,
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    """Prepare a fresh contextual valuation iteration from scored Backpack artifacts."""

    virtual_path = Path(virtual_manifest_path).expanduser().resolve()
    score_path = Path(score_receipt_path).expanduser().resolve()
    virtual = _read_json(virtual_path)
    if (
        virtual.get("schema") != "banana-smasher-backpack-virtual-assignment-v1"
        or virtual.get("status") != "PASS_LOGICAL_FULL_WIRE"
    ):
        raise ContextualValuationError("virtual Backpack manifest must be PASS v1")
    basis_sha256 = virtual.get("basis_sha256")
    if not isinstance(basis_sha256, str) or len(basis_sha256) != 64:
        raise ContextualValuationError("virtual Backpack basis is invalid")

    descriptors: dict[str, Mapping[str, Any]] = {}
    paths: dict[str, Path] = {}
    for role in ("assignment", "materialization_index", "option_ledger", "solve_input"):
        descriptor = virtual.get(role)
        if not isinstance(descriptor, Mapping):
            raise ContextualValuationError(f"virtual Backpack {role} descriptor missing")
        descriptors[role] = descriptor
        paths[role] = _bound_path(descriptor, root=virtual_path.parent, role=role)
    if virtual.get("assignment_map_sha256") != descriptors["assignment"].get("sha256"):
        raise ContextualValuationError("virtual assignment SHA binding mismatch")

    solve_input = _read_json(paths["solve_input"])
    if (
        solve_input.get("schema")
        != "banana-smasher-backpack-exact-matched-full-wire-input-v2"
        or solve_input.get("status") != "PASS"
        or solve_input.get("basis_sha256") != basis_sha256
    ):
        raise ContextualValuationError("full-wire solve input must be basis-bound PASS v2")
    budget = solve_input.get("budget")
    if not isinstance(budget, Mapping):
        raise ContextualValuationError("full-wire solve budget is missing")
    fixed_bytes = budget.get("fixed_nonexpert_bytes")
    package_cap_bytes = budget.get("total_package_bytes")
    if not isinstance(fixed_bytes, int) or fixed_bytes < 0:
        raise ContextualValuationError("full-wire fixed bytes are invalid")
    if not isinstance(package_cap_bytes, int) or package_cap_bytes < fixed_bytes:
        raise ContextualValuationError("full-wire package cap is invalid")

    raw_artifacts = solve_input.get("activation_artifacts", [])
    if not isinstance(raw_artifacts, list):
        raise ContextualValuationError("activation artifacts must be an array")
    artifacts: dict[str, dict[str, Any]] = {}
    for raw in raw_artifacts:
        if not isinstance(raw, Mapping):
            raise ContextualValuationError("activation artifact is invalid")
        artifact_id = raw.get("artifact_id")
        byte_count = raw.get("bytes")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ContextualValuationError("activation artifact id is invalid")
        if not isinstance(byte_count, int) or byte_count < 0:
            raise ContextualValuationError("activation artifact bytes are invalid")
        if artifact_id in artifacts:
            raise ContextualValuationError("activation artifact ids must be unique")
        artifacts[artifact_id] = {"id": artifact_id, "bytes": byte_count}

    score = _read_json(score_path)
    if (
        score.get("schema") != "banana-smasher-backpack-exact64-terminal-v1"
        or score.get("status") != "PASS"
        or score.get("basis_sha256") != basis_sha256
    ):
        raise ContextualValuationError("score receipt must be basis-bound exact64 PASS")
    pack_files = [
        {
            "file": path.name,
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in (
            paths["assignment"],
            virtual_path,
            paths["materialization_index"],
        )
    ]
    pack_files.sort(key=lambda row: row["file"])
    pack_sha256 = hashlib.sha256(
        (json.dumps(pack_files, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    if score.get("pack_sha256") != pack_sha256:
        raise ContextualValuationError("exact64 score pack binding mismatch")

    def activation_rows(row: Mapping[str, Any]) -> list[dict[str, Any]]:
        ids = row.get("activation_artifact_ids", [])
        if not isinstance(ids, list):
            raise ContextualValuationError("activation_artifact_ids must be an array")
        result: list[dict[str, Any]] = []
        for artifact_id in ids:
            if artifact_id not in artifacts:
                raise ContextualValuationError(
                    f"undeclared activation artifact: {artifact_id!r}"
                )
            result.append(dict(artifacts[artifact_id]))
        return result

    def prepared_member(row: Mapping[str, Any]) -> dict[str, Any]:
        member = {
            "cell": row.get("cell_id"),
            "physical_source_key": _physical_source_key(row),
            "physical_identity": _physical_identity(row, basis_sha256=basis_sha256),
            "payload_bytes": row.get("physical_bytes"),
            "activations": activation_rows(row),
        }
        if row.get("physical_receipt_sha256") is not None:
            member["physical_receipt_sha256"] = row["physical_receipt_sha256"]
        if row.get("physical_receipt_path") is not None:
            member["physical_receipt_path"] = row["physical_receipt_path"]
        return member

    def prepared_group(
        selection_group: str, tier: str, rows: list[Mapping[str, Any]]
    ) -> dict[str, Any]:
        members = [prepared_member(row) for row in rows]
        members.sort(key=lambda row: str(row["cell"]))
        activations_by_id: dict[str, dict[str, Any]] = {}
        for member in members:
            for activation in member["activations"]:
                artifact_id = str(activation["id"])
                if (
                    artifact_id in activations_by_id
                    and activations_by_id[artifact_id] != activation
                ):
                    raise ContextualValuationError("group activation bytes conflict")
                activations_by_id[artifact_id] = activation
        return {
            "cell": selection_group,
            "selection_group": selection_group,
            "option": tier,
            "physical_identity": _group_physical_identity(
                selection_group, tier, members, basis_sha256=basis_sha256
            ),
            "payload_bytes": sum(int(member["payload_bytes"]) for member in members),
            "activations": [
                activations_by_id[key] for key in sorted(activations_by_id)
            ],
            "members": members,
        }

    materialization = _read_jsonl(paths["materialization_index"])
    cells: list[dict[str, Any]] = []
    seen_cells: set[str] = set()
    materialized_groups: dict[str, list[dict[str, Any]]] = {}
    for row in materialization:
        cell = row.get("cell_id")
        if not isinstance(cell, str) or not cell or cell in seen_cells:
            raise ContextualValuationError("materialization cells must be unique")
        seen_cells.add(cell)
        if row.get("basis_sha256", basis_sha256) != basis_sha256:
            raise ContextualValuationError("materialization basis mismatch")
        selection_group = row.get("selection_group", cell)
        if not isinstance(selection_group, str) or not selection_group:
            raise ContextualValuationError("materialization selection group is invalid")
        materialized_groups.setdefault(selection_group, []).append(row)
    group_cells = {
        group: {str(row["cell_id"]) for row in rows}
        for group, rows in materialized_groups.items()
    }
    for group, rows in sorted(materialized_groups.items()):
        tiers = {str(row.get("tier")) for row in rows}
        if len(tiers) != 1:
            raise ContextualValuationError("materialized selection group has mixed tiers")
        cells.append(prepared_group(group, tiers.pop(), rows))

    options: list[dict[str, Any]] = []
    option_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in _read_jsonl(paths["option_ledger"]):
        if (
            row.get("schema") != "banana-smasher-backpack-option-row-v1"
            or row.get("basis_sha256") != basis_sha256
        ):
            raise ContextualValuationError("option row schema or basis mismatch")
        cell = row.get("cell_id")
        if cell not in seen_cells:
            raise ContextualValuationError(f"option references unknown cell: {cell!r}")
        selection_group = row.get("selection_group", cell)
        tier = row.get("tier")
        if not isinstance(selection_group, str) or not selection_group:
            raise ContextualValuationError("option selection group is invalid")
        if not isinstance(tier, str) or not tier:
            raise ContextualValuationError("option tier is invalid")
        option_groups.setdefault((selection_group, tier), []).append(row)
    for (group, tier), rows in sorted(option_groups.items()):
        if {str(row["cell_id"]) for row in rows} != group_cells.get(group):
            raise ContextualValuationError(
                "option selection group does not cover the canonical member cells"
            )
        options.append(prepared_group(group, tier, rows))

    assignment_sha256 = descriptors["assignment"]["sha256"]
    terminal_receipt_sha256 = _sha256(score_path)
    physical_score_sha256 = score.get(
        "score_receipt_sha256", terminal_receipt_sha256
    )
    if not isinstance(physical_score_sha256, str) or len(physical_score_sha256) != 64:
        raise ContextualValuationError("exact64 physical score SHA is invalid")
    input_bindings = {
        "virtual_manifest_sha256": _sha256(virtual_path),
        "assignment_sha256": assignment_sha256,
        "materialization_index_sha256": descriptors["materialization_index"][
            "sha256"
        ],
        "option_ledger_sha256": descriptors["option_ledger"]["sha256"],
        "solve_input_sha256": descriptors["solve_input"]["sha256"],
        "terminal_receipt_sha256": terminal_receipt_sha256,
        "score_receipt_sha256": physical_score_sha256,
        "pack_sha256": pack_sha256,
    }
    anchor = {
        "schema": "banana-smasher-contextual-anchor-v1",
        "status": "PASS",
        "basis_sha256": basis_sha256,
        "assignment_sha256": assignment_sha256,
        "physical_score_receipt_sha256": physical_score_sha256,
        "fixed_bytes": fixed_bytes,
        "package_cap_bytes": package_cap_bytes,
        "baseline_metrics": {
            key: score.get(key)
            for key in (
                "windows",
                "positions",
                "support_width",
                "mean_kld",
                "top1_matches",
                "top1_agreement",
            )
            if key in score
        },
        "input_bindings": input_bindings,
        "cells": cells,
    }
    inventory = {
        "schema": "banana-smasher-contextual-option-inventory-v1",
        "status": "READY",
        "basis_sha256": basis_sha256,
        "anchor_assignment_sha256": assignment_sha256,
        "input_bindings": input_bindings,
        "options": options,
    }
    measurements = {
        "schema": "banana-smasher-contextual-measurement-manifest-v1",
        "status": "READY",
        "basis_sha256": basis_sha256,
        "anchor_assignment_sha256": assignment_sha256,
        "measurements": [],
    }

    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "anchor": output / "ANCHOR.json",
        "option_inventory": output / "OPTION_INVENTORY.json",
        "measurements": output / "MEASUREMENTS.json",
    }
    for role, value in (
        ("anchor", anchor),
        ("option_inventory", inventory),
        ("measurements", measurements),
    ):
        _atomic_json(outputs[role], value)
    return {
        "schema": "banana-smasher-contextual-prepare-receipt-v1",
        "status": "PASS",
        "basis_sha256": basis_sha256,
        "assignment_sha256": assignment_sha256,
        "cells": len(cells),
        "options": len(options),
        "outputs": {
            role: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for role, path in outputs.items()
        },
    }
