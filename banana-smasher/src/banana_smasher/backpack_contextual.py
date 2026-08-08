from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


class ContextualValuationError(ValueError):
    """A contextual valuation input is incomplete or inconsistent."""


def _json_input(path: str | Path, *, role: str) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    payload = source.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ContextualValuationError(f"{role} must contain one JSON object")
    return value, {
        "role": role,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _measurement_receipt_sha256(value: Mapping[str, Any]) -> str:
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    return hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def build_contextual_delta_ledger(
    anchor: Mapping[str, Any],
    options: Sequence[Mapping[str, Any]],
    measurements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build marginal option values against one physically scored assignment."""

    if anchor.get("schema") != "banana-smasher-contextual-anchor-v1":
        raise ContextualValuationError("unsupported contextual anchor schema")
    if anchor.get("status") != "PASS":
        raise ContextualValuationError("contextual anchor must be PASS")
    anchor_rows = anchor.get("cells")
    if not isinstance(anchor_rows, list) or not anchor_rows:
        raise ContextualValuationError("contextual anchor cells must be non-empty")
    by_cell = {row["cell"]: row for row in anchor_rows}
    if len(by_cell) != len(anchor_rows):
        raise ContextualValuationError("contextual anchor cells must be unique")

    measurement_by_physical_option: dict[tuple[str, str], Mapping[str, Any]] = {}
    for measurement in measurements:
        if (
            measurement.get("schema")
            != "banana-smasher-contextual-swap-measurement-v1"
            or measurement.get("status") != "PASS"
        ):
            raise ContextualValuationError("contextual swap measurement must be v1 PASS")
        if measurement.get("anchor_assignment_sha256") != anchor.get(
            "assignment_sha256"
        ):
            raise ContextualValuationError("contextual swap measurement anchor mismatch")
        if measurement.get("anchor_score_sha256") != anchor.get(
            "physical_score_receipt_sha256"
        ):
            raise ContextualValuationError("contextual swap measurement score mismatch")
        if measurement.get("receipt_sha256") != _measurement_receipt_sha256(
            measurement
        ):
            raise ContextualValuationError("contextual swap measurement receipt hash mismatch")
        for field in (
            "candidate_assignment_sha256",
            "candidate_score_sha256",
            "candidate_pack_sha256",
        ):
            value = measurement.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise ContextualValuationError(
                    f"contextual swap measurement {field} is invalid"
                )
        delta_mean_kld = measurement.get("delta_mean_kld")
        delta_top1 = measurement.get("delta_top1_matches")
        stderr = measurement.get("stderr_mean_kld")
        if (
            not isinstance(delta_mean_kld, (int, float))
            or isinstance(delta_mean_kld, bool)
            or not isinstance(stderr, (int, float))
            or isinstance(stderr, bool)
            or not isinstance(delta_top1, int)
            or isinstance(delta_top1, bool)
        ):
            raise ContextualValuationError("contextual swap measurement values are invalid")
        if not all(
            value == value and abs(value) != float("inf")
            for value in (float(delta_mean_kld), float(stderr))
        ) or stderr < 0:
            raise ContextualValuationError("contextual swap measurement values are invalid")
        change = measurement.get("change")
        if not isinstance(change, Mapping):
            raise ContextualValuationError("contextual swap measurement change is missing")
        change_cell = change.get("cell")
        change_identity = change.get("physical_identity")
        if not isinstance(change_cell, str) or not change_cell:
            raise ContextualValuationError("contextual swap measurement change is invalid")
        if not isinstance(change_identity, str) or not change_identity:
            raise ContextualValuationError("contextual swap measurement change is invalid")
        key = (change_cell, change_identity)
        if key in measurement_by_physical_option:
            raise ContextualValuationError(
                "duplicate contextual measurement; aggregate repeats before valuation"
            )
        measurement_by_physical_option[key] = measurement

    rows: list[dict[str, Any]] = []
    for option in options:
        cell = option.get("cell")
        if not isinstance(cell, str) or not cell:
            raise ContextualValuationError("option cell is invalid")
        incumbent = by_cell.get(cell)
        if incumbent is None:
            raise ContextualValuationError(f"option references unknown cell: {cell!r}")
        option_identity = option.get("physical_identity")
        if not isinstance(option_identity, str) or not option_identity:
            raise ContextualValuationError("option physical identity is invalid")
        members = option.get("members")
        if members is not None:
            from .backpack_contextual_prepare import (
                _group_physical_identity,
                _prepared_member_identity,
            )

            if not isinstance(members, list) or not members:
                raise ContextualValuationError("option group members are invalid")
            for member in members:
                if not isinstance(member, Mapping) or member.get(
                    "physical_identity"
                ) != _prepared_member_identity(
                    member, basis_sha256=str(anchor.get("basis_sha256"))
                ):
                    raise ContextualValuationError(
                        "option group member physical identity mismatch"
                    )
            if option_identity != _group_physical_identity(
                cell,
                str(option.get("option")),
                members,
                basis_sha256=str(anchor.get("basis_sha256")),
            ):
                raise ContextualValuationError("option group physical identity mismatch")
            if sum(int(member.get("payload_bytes", -1)) for member in members) != option.get(
                "payload_bytes"
            ):
                raise ContextualValuationError("option group payload bytes mismatch")
        if option_identity != incumbent.get("physical_identity"):
            measurement = measurement_by_physical_option.get(
                (cell, option_identity)
            )
            if measurement is not None:
                rows.append(
                    {
                        "cell": cell,
                        "option": option.get("option"),
                        "physical_identity": option.get("physical_identity"),
                        "payload_bytes": option.get("payload_bytes"),
                        **(
                            {"activations": option["activations"]}
                            if "activations" in option
                            else {}
                        ),
                        "eligible": True,
                        "delta_mean_kld": measurement.get("delta_mean_kld"),
                        "delta_top1_matches": measurement.get(
                            "delta_top1_matches"
                        ),
                        "stderr_mean_kld": measurement.get("stderr_mean_kld"),
                        "valuation_source": "physical-swap-receipt",
                        "measurement_receipt_sha256": measurement.get(
                            "receipt_sha256"
                        ),
                        "measurement_scope": measurement.get("scope"),
                        "measurement_windows": measurement.get("windows"),
                        "measurement_positions": measurement.get("positions"),
                        "measurement_support_width": measurement.get("support_width"),
                    }
                )
                continue
            rows.append(
                {
                    "cell": cell,
                    "option": option.get("option"),
                    "physical_identity": option.get("physical_identity"),
                    "payload_bytes": option.get("payload_bytes"),
                    **(
                        {"activations": option["activations"]}
                        if "activations" in option
                        else {}
                    ),
                    "eligible": False,
                    "delta_mean_kld": None,
                    "delta_top1_matches": None,
                    "valuation_source": "unmeasured",
                    "measurement_receipt_sha256": None,
                }
            )
            continue
        rows.append(
            {
                "cell": cell,
                "option": option.get("option"),
                "physical_identity": option.get("physical_identity"),
                "payload_bytes": option.get("payload_bytes"),
                **(
                    {"activations": option["activations"]}
                    if "activations" in option
                    else {}
                ),
                "eligible": True,
                "delta_mean_kld": 0.0,
                "delta_top1_matches": 0,
                "valuation_source": "physical-alias-invariance",
                "measurement_receipt_sha256": None,
            }
        )
    return {
        "schema": "banana-smasher-contextual-delta-ledger-v1",
        "status": "PASS",
        "anchor_assignment_sha256": anchor.get("assignment_sha256"),
        "anchor_score_receipt_sha256": anchor.get("physical_score_receipt_sha256"),
        "rows": rows,
    }


def run_contextual_value_update(
    anchor_path: str | Path,
    option_inventory_path: str | Path,
    measurement_manifest_path: str | Path,
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    """Build one deterministic contextual-value ledger from versioned artifacts."""

    anchor, anchor_binding = _json_input(anchor_path, role="anchor")
    inventory, inventory_binding = _json_input(
        option_inventory_path, role="option_inventory"
    )
    manifest, measurement_binding = _json_input(
        measurement_manifest_path, role="measurement_manifest"
    )
    if (
        inventory.get("schema")
        != "banana-smasher-contextual-option-inventory-v1"
        or inventory.get("status") != "READY"
        or not isinstance(inventory.get("options"), list)
    ):
        raise ContextualValuationError("option inventory must be v1 READY")
    if (
        manifest.get("schema")
        != "banana-smasher-contextual-measurement-manifest-v1"
        or manifest.get("status") != "READY"
        or not isinstance(manifest.get("measurements"), list)
    ):
        raise ContextualValuationError("measurement manifest must be v1 READY")
    result = build_contextual_delta_ledger(
        anchor, inventory["options"], manifest["measurements"]
    )
    result["input_bindings"] = [
        anchor_binding,
        inventory_binding,
        measurement_binding,
    ]
    output = Path(output_path).expanduser().resolve()
    _atomic_json(output, result)
    payload = output.read_bytes()
    return {
        "status": "PASS",
        "output": str(output),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "eligible_rows": sum(bool(row["eligible"]) for row in result["rows"]),
        "rows": len(result["rows"]),
    }


def solve_contextual_trust_region(
    anchor: Mapping[str, Any],
    ledger: Mapping[str, Any],
    *,
    max_changes: int,
    uncertainty_multiplier: float,
    time_limit_seconds: float,
) -> dict[str, Any]:
    """Solve measured contextual substitutions under bytes and Hamming radius."""

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "contextual trust solve requires scipy; install banana-smasher[backpack]"
        ) from exc
    if (
        anchor.get("schema") != "banana-smasher-contextual-anchor-v1"
        or anchor.get("status") != "PASS"
    ):
        raise ContextualValuationError("unsupported contextual anchor schema")
    if (
        ledger.get("schema") != "banana-smasher-contextual-delta-ledger-v1"
        or ledger.get("status") != "PASS"
    ):
        raise ContextualValuationError("unsupported contextual delta ledger schema")
    if ledger.get("anchor_assignment_sha256") != anchor.get("assignment_sha256"):
        raise ContextualValuationError("contextual ledger anchor mismatch")
    if not isinstance(max_changes, int) or isinstance(max_changes, bool) or max_changes < 0:
        raise ContextualValuationError("max_changes must be a non-negative integer")
    fixed_bytes = anchor.get("fixed_bytes")
    package_cap_bytes = anchor.get("package_cap_bytes")
    if not isinstance(fixed_bytes, int) or fixed_bytes < 0:
        raise ContextualValuationError("anchor fixed_bytes must be non-negative")
    if not isinstance(package_cap_bytes, int) or package_cap_bytes < fixed_bytes:
        raise ContextualValuationError("anchor package_cap_bytes is invalid")
    anchor_cells = anchor.get("cells")
    ledger_rows = ledger.get("rows")
    if not isinstance(anchor_cells, list) or not anchor_cells:
        raise ContextualValuationError("anchor cells must be non-empty")
    if not isinstance(ledger_rows, list):
        raise ContextualValuationError("contextual ledger rows must be an array")

    candidates_by_cell: dict[str, list[dict[str, Any]]] = {}
    incumbent_identity: dict[str, str] = {}
    for raw in anchor_cells:
        cell = raw.get("cell")
        identity = raw.get("physical_identity")
        payload_bytes = raw.get("payload_bytes")
        if not isinstance(cell, str) or not cell or cell in candidates_by_cell:
            raise ContextualValuationError("anchor cells must have unique string keys")
        if not isinstance(identity, str) or not identity:
            raise ContextualValuationError("anchor physical identity is invalid")
        if not isinstance(payload_bytes, int) or payload_bytes < 0:
            raise ContextualValuationError("anchor payload_bytes is invalid")
        incumbent = {
            **raw,
            "eligible": True,
            "delta_mean_kld": 0.0,
            "stderr_mean_kld": 0.0,
            "valuation_source": "incumbent",
        }
        candidates_by_cell[cell] = [incumbent]
        incumbent_identity[cell] = identity

    for raw in ledger_rows:
        if not raw.get("eligible"):
            continue
        cell = raw.get("cell")
        identity = raw.get("physical_identity")
        if cell not in candidates_by_cell:
            raise ContextualValuationError(f"ledger references unknown cell: {cell!r}")
        if not isinstance(identity, str) or not identity:
            raise ContextualValuationError("ledger physical identity is invalid")
        if identity == incumbent_identity[cell]:
            continue
        payload_bytes = raw.get("payload_bytes")
        delta = raw.get("delta_mean_kld")
        stderr = raw.get("stderr_mean_kld", 0.0)
        if not isinstance(payload_bytes, int) or payload_bytes < 0:
            raise ContextualValuationError("ledger payload_bytes is invalid")
        if not isinstance(delta, (int, float)) or not isinstance(stderr, (int, float)):
            raise ContextualValuationError("eligible ledger values must be numeric")
        existing = next(
            (
                candidate
                for candidate in candidates_by_cell[cell]
                if candidate["physical_identity"] == identity
            ),
            None,
        )
        if existing is not None:
            if any(
                existing.get(field) != raw.get(field)
                for field in ("payload_bytes", "delta_mean_kld", "stderr_mean_kld")
            ):
                raise ContextualValuationError(
                    "one physical identity has conflicting logical valuations"
                )
            continue
        candidates_by_cell[cell].append(dict(raw))

    candidates = [
        candidate
        for cell in candidates_by_cell
        for candidate in candidates_by_cell[cell]
    ]
    activation_bytes: dict[str, int] = {}
    activation_users: dict[str, list[int]] = {}
    for index, candidate in enumerate(candidates):
        plural = candidate.get("activations")
        singular = candidate.get("activation")
        if plural is not None and singular is not None:
            raise ContextualValuationError(
                "candidate cannot declare both activation and activations"
            )
        if plural is None:
            raw_activations = [] if singular is None else [singular]
        elif isinstance(plural, list):
            raw_activations = plural
        else:
            raise ContextualValuationError("activations must be an array")
        normalized: list[dict[str, Any]] = []
        seen_for_candidate: set[str] = set()
        for activation in raw_activations:
            if not isinstance(activation, Mapping):
                raise ContextualValuationError("activation must be an object")
            activation_id = activation.get("id")
            byte_count = activation.get("bytes")
            if not isinstance(activation_id, str) or not activation_id:
                raise ContextualValuationError("activation id is invalid")
            if not isinstance(byte_count, int) or byte_count < 0:
                raise ContextualValuationError("activation bytes is invalid")
            if activation_id in seen_for_candidate:
                raise ContextualValuationError("candidate activation ids must be unique")
            seen_for_candidate.add(activation_id)
            if (
                activation_id in activation_bytes
                and activation_bytes[activation_id] != byte_count
            ):
                raise ContextualValuationError("activation bytes conflict")
            activation_bytes[activation_id] = byte_count
            activation_users.setdefault(activation_id, []).append(index)
            normalized.append({"id": activation_id, "bytes": byte_count})
        candidate["_normalized_activations"] = normalized

    activation_ids = sorted(activation_bytes)
    activation_variable = {
        activation_id: len(candidates) + index
        for index, activation_id in enumerate(activation_ids)
    }
    variable_count = len(candidates) + len(activation_ids)
    objective = np.zeros(variable_count, dtype=np.float64)
    for index, candidate in enumerate(candidates):
        objective[index] = float(candidate.get("delta_mean_kld", 0.0)) + (
            float(uncertainty_multiplier)
            * float(candidate.get("stderr_mean_kld", 0.0))
        )

    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    candidate_index = {id(candidate): index for index, candidate in enumerate(candidates)}

    def add_constraint(coefficients: Mapping[int, float], lb: float, ub: float) -> None:
        row = len(lower)
        for column, value in coefficients.items():
            matrix_rows.append(row)
            matrix_columns.append(column)
            matrix_values.append(value)
        lower.append(lb)
        upper.append(ub)

    for cell_candidates in candidates_by_cell.values():
        add_constraint(
            {candidate_index[id(candidate)]: 1.0 for candidate in cell_candidates},
            1.0,
            1.0,
        )
    budget_coefficients = {
        index: float(candidate["payload_bytes"])
        for index, candidate in enumerate(candidates)
    }
    budget_coefficients.update(
        {
            activation_variable[activation_id]: float(byte_count)
            for activation_id, byte_count in activation_bytes.items()
        }
    )
    add_constraint(
        budget_coefficients,
        -np.inf,
        float(package_cap_bytes - fixed_bytes),
    )
    add_constraint(
        {
            index: 1.0
            for index, candidate in enumerate(candidates)
            if candidate["physical_identity"]
            != incumbent_identity[candidate["cell"]]
        },
        -np.inf,
        float(max_changes),
    )
    for activation_id, users in activation_users.items():
        y = activation_variable[activation_id]
        for x in users:
            add_constraint({x: 1.0, y: -1.0}, -np.inf, 0.0)
        add_constraint({y: 1.0, **{x: -1.0 for x in users}}, -np.inf, 0.0)

    matrix = coo_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(len(lower), variable_count),
    ).tocsr()
    solved = milp(
        objective,
        integrality=np.ones(variable_count, dtype=np.int8),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": float(time_limit_seconds)},
    )
    if solved.x is None:
        raise RuntimeError(f"contextual trust solve failed: {solved.message}")
    selected_indexes = [
        index for index in range(len(candidates)) if solved.x[index] >= 0.5
    ]
    selected = [candidates[index] for index in selected_indexes]
    selected_activations = {
        activation["id"]
        for candidate in selected
        for activation in candidate["_normalized_activations"]
    }
    package_bytes = fixed_bytes + sum(
        int(candidate["payload_bytes"]) for candidate in selected
    ) + sum(activation_bytes[activation_id] for activation_id in selected_activations)
    changed_cells = sum(
        candidate["physical_identity"] != incumbent_identity[candidate["cell"]]
        for candidate in selected
    )
    if package_bytes > package_cap_bytes or changed_cells > max_changes:
        raise RuntimeError("contextual trust solve verification failed")
    assignment = [
        {
            "cell": candidate["cell"],
            "option": candidate.get("option"),
            "physical_identity": candidate["physical_identity"],
            "payload_bytes": candidate["payload_bytes"],
            "activation": candidate.get("activation"),
            "activations": candidate["_normalized_activations"],
            "delta_mean_kld": float(candidate.get("delta_mean_kld", 0.0)),
            "valuation_source": candidate.get("valuation_source", "measured"),
        }
        for candidate in selected
    ]
    assignment.sort(key=lambda row: row["cell"])
    assignment_sha256 = hashlib.sha256(
        json.dumps(assignment, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": "banana-smasher-contextual-trust-solve-v1",
        "status": "PASS",
        "solver_status": int(solved.status),
        "solver_message": str(solved.message),
        "anchor_assignment_sha256": anchor.get("assignment_sha256"),
        "assignment_sha256": assignment_sha256,
        "assignment": assignment,
        "changed_cells": changed_cells,
        "max_changes": max_changes,
        "fixed_bytes": fixed_bytes,
        "package_cap_bytes": package_cap_bytes,
        "package_bytes": package_bytes,
        "predicted_delta_mean_kld": sum(
            float(candidate.get("delta_mean_kld", 0.0)) for candidate in selected
        ),
        "uncertainty_multiplier": float(uncertainty_multiplier),
    }


def run_contextual_trust_solve(
    anchor_path: str | Path,
    ledger_path: str | Path,
    *,
    output_path: str | Path,
    max_changes: int,
    uncertainty_multiplier: float,
    time_limit_seconds: float,
) -> dict[str, Any]:
    """Run a content-bound trust-region solve and atomically write its receipt."""

    anchor, anchor_binding = _json_input(anchor_path, role="anchor")
    ledger, ledger_binding = _json_input(ledger_path, role="ledger")
    result = solve_contextual_trust_region(
        anchor,
        ledger,
        max_changes=max_changes,
        uncertainty_multiplier=uncertainty_multiplier,
        time_limit_seconds=time_limit_seconds,
    )
    result["input_bindings"] = [anchor_binding, ledger_binding]
    output = Path(output_path).expanduser().resolve()
    _atomic_json(output, result)
    payload = output.read_bytes()
    return {
        "status": "PASS",
        "output": str(output),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "assignment_sha256": result["assignment_sha256"],
        "changed_cells": result["changed_cells"],
        "package_bytes": result["package_bytes"],
        "package_cap_bytes": result["package_cap_bytes"],
    }
