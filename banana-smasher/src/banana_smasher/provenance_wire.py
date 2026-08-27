"""Wire-accounting adapters for provenance-weighted option ledgers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .knapsack import solve_class_balanced_options

_D4_TIERS = {"d4_k2048", "d4_k4096"}


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _descriptor(path: Path, raw: bytes) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256(raw), "bytes": len(raw)}


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_full_wire_provenance_ledger(
    provenance_option_ledger: str | Path,
    full_wire_option_ledger: str | Path,
    activation_registry: str | Path,
    output_ledger: str | Path,
    output_receipt: str | Path,
    *,
    expected_provenance_sha256: str,
    expected_full_wire_sha256: str,
    expected_activation_registry_sha256: str,
) -> dict[str, Any]:
    """Replace D4 code-only prices with full selectable wire plus shared assets."""

    provenance_path = Path(provenance_option_ledger).expanduser().resolve()
    full_wire_path = Path(full_wire_option_ledger).expanduser().resolve()
    registry_path = Path(activation_registry).expanduser().resolve()
    destination = Path(output_ledger).expanduser().resolve()
    receipt_path = Path(output_receipt).expanduser().resolve()
    provenance_raw = provenance_path.read_bytes()
    full_wire_raw = full_wire_path.read_bytes()
    registry_raw = registry_path.read_bytes()
    for label, actual, expected in (
        ("provenance option ledger", _sha256(provenance_raw), expected_provenance_sha256),
        ("full-wire option ledger", _sha256(full_wire_raw), expected_full_wire_sha256),
        ("activation registry", _sha256(registry_raw), expected_activation_registry_sha256),
    ):
        if actual != expected:
            raise ValueError(f"{label} SHA-256 mismatch")

    registry_value = json.loads(registry_raw)
    raw_artifacts = registry_value.get("activation_artifacts")
    if not isinstance(raw_artifacts, list):
        raise ValueError("activation registry lacks activation_artifacts")
    artifacts: dict[str, dict[str, Any]] = {}
    for row in raw_artifacts:
        if not isinstance(row, dict):
            raise ValueError("activation registry row is not an object")
        artifact_id = str(row.get("artifact_id", row.get("id", "")))
        byte_count = row.get("bytes")
        if (
            not artifact_id
            or artifact_id in artifacts
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise ValueError("activation registry identity/bytes are invalid")
        artifacts[artifact_id] = {**row, "id": artifact_id, "bytes": byte_count}

    full_wire_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, line in enumerate(full_wire_raw.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or row.get("tier") not in _D4_TIERS:
            continue
        key = (str(row.get("cell_id")), str(row.get("tier")))
        if key in full_wire_rows:
            raise ValueError(f"duplicate D4 full-wire row at line {line_number}: {key}")
        ids = row.get("activation_artifact_ids")
        if not isinstance(ids, list) or any(str(value) not in artifacts for value in ids):
            raise ValueError(f"D4 full-wire row has an unknown activation artifact: {key}")
        full_wire_rows[key] = row

    source_descriptor = _descriptor(full_wire_path, full_wire_raw)
    output_rows: list[dict[str, Any]] = []
    seen_d4: set[tuple[str, str]] = set()
    tier_counts: dict[str, int] = {}
    for line_number, line in enumerate(provenance_raw.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"provenance option row {line_number} is not an object")
        tier = str(row.get("tier"))
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        upgraded = dict(row)
        if tier in _D4_TIERS:
            key = (str(row.get("cell_id")), tier)
            wire = full_wire_rows.get(key)
            if wire is None:
                raise ValueError(f"D4 provenance option lacks full-wire authority: {key}")
            ids = [str(value) for value in wire["activation_artifact_ids"]]
            upgraded["physical_bytes"] = int(wire["physical_bytes"])
            upgraded["activation_ids"] = ids
            upgraded["activation_artifacts"] = [artifacts[value] for value in ids]
            upgraded["physical_producer"] = {
                **source_descriptor,
                "artifact_sha256": _sha256(_canonical(wire)),
            }
            upgraded["physical_byte_definition"] = wire.get("byte_definition")
            seen_d4.add(key)
        else:
            # QTIP provenance prices are complete per-unit file sizes; no extra
            # shared charge is required. Native MXFP4 likewise has no activation.
            upgraded["activation_ids"] = []
            upgraded["activation_artifacts"] = []
        output_rows.append(upgraded)

    if seen_d4 != set(full_wire_rows):
        missing = sorted(set(full_wire_rows) - seen_d4)
        extra = sorted(seen_d4 - set(full_wire_rows))
        raise ValueError(
            f"D4 provenance/full-wire coverage mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )
    output_raw = b"".join(_canonical(row) for row in output_rows)
    if destination.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite full-wire provenance outputs")
    _atomic_write(destination, output_raw)
    receipt = {
        "schema": "banana-smasher-provenance-full-wire-ledger-receipt-v1",
        "status": "PASS",
        "inputs": {
            "provenance_option_ledger": _descriptor(provenance_path, provenance_raw),
            "full_wire_option_ledger": source_descriptor,
            "activation_registry": _descriptor(registry_path, registry_raw),
        },
        "output": {
            **_descriptor(destination, output_raw),
            "rows": len(output_rows),
            "tier_counts": dict(sorted(tier_counts.items())),
            "d4_rows": len(seen_d4),
        },
        "activation_registry": {
            "artifacts": len(artifacts),
            "bytes": sum(int(row["bytes"]) for row in artifacts.values()),
        },
    }
    _atomic_write(receipt_path, _canonical(receipt))
    return receipt


def run_full_wire_provenance_solve(
    option_ledger: str | Path,
    fixed_accounting: str | Path,
    output_assignment: str | Path,
    output_receipt: str | Path,
    *,
    expected_option_ledger_sha256: str,
    expected_fixed_accounting_sha256: str,
    shipping_bytes_cap: int,
    class_weights: dict[str, float],
    class_caps: dict[str, float] | None = None,
    allowed_tiers: tuple[str, ...] | None = None,
    exact_envelope: bool = False,
    padding_policy: str | None = None,
) -> dict[str, Any]:
    """Solve a full-wire provenance ledger, charging shared assets once."""

    ledger_path = Path(option_ledger).expanduser().resolve()
    fixed_path = Path(fixed_accounting).expanduser().resolve()
    assignment_path = Path(output_assignment).expanduser().resolve()
    receipt_path = Path(output_receipt).expanduser().resolve()
    ledger_raw = ledger_path.read_bytes()
    fixed_raw = fixed_path.read_bytes()
    if _sha256(ledger_raw) != expected_option_ledger_sha256:
        raise ValueError("full-wire provenance ledger SHA-256 mismatch")
    if _sha256(fixed_raw) != expected_fixed_accounting_sha256:
        raise ValueError("fixed accounting SHA-256 mismatch")
    if (
        isinstance(shipping_bytes_cap, bool)
        or not isinstance(shipping_bytes_cap, int)
        or shipping_bytes_cap <= 0
    ):
        raise ValueError("shipping_bytes_cap must be a positive integer")
    if padding_policy not in (None, "metadata_reserve"):
        raise ValueError("padding_policy must be metadata_reserve when provided")
    if exact_envelope and padding_policy is not None:
        raise ValueError("exact_envelope and padding_policy are mutually exclusive")
    fixed = json.loads(fixed_raw)
    components = fixed.get("components")
    if not isinstance(components, dict):
        raise ValueError("fixed accounting lacks components")
    dense = int(components["dense_nonrouted_bytes"])
    repair = int(components["repair_bytes"])
    metadata = int(components["metadata_bytes"])
    fixed_bytes = dense + repair + metadata
    envelope = shipping_bytes_cap - fixed_bytes
    if envelope < 0:
        raise ValueError("fixed accounting exceeds shipping cap")

    rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    identity: dict[str, Any] | None = None
    identity_keys = (
        "model_id",
        "model_revision",
        "basis_sha256",
        "bank_sha256",
        "teacher_sha256",
        "scorer_sha256",
    )
    for line_number, line in enumerate(ledger_raw.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"full-wire provenance row {line_number} is not an object")
        row_identity = {key: row.get(key) for key in identity_keys}
        if identity is None:
            identity = row_identity
        elif row_identity != identity:
            raise ValueError(f"full-wire provenance identity drift at line {line_number}")
        tier = str(row.get("tier"))
        if allowed_tiers is not None and tier not in allowed_tiers:
            continue
        key = (str(row.get("cell_id")), tier)
        if key in by_key:
            raise ValueError(f"duplicate full-wire provenance option: {key}")
        activations = row.get("activation_artifacts", [])
        activation_ids = row.get("activation_ids", [])
        if (
            not isinstance(activations, list)
            or not isinstance(activation_ids, list)
            or sorted(str(value) for value in activation_ids)
            != sorted(str(value.get("id")) for value in activations if isinstance(value, dict))
        ):
            raise ValueError(f"full-wire provenance activation binding mismatch: {key}")
        by_key[key] = row
        rows.append(row)
    if not rows or identity is None:
        raise ValueError("full-wire provenance ledger is empty")
    cells = sorted({key[0] for key in by_key})
    tiers = sorted({key[1] for key in by_key})
    expected_keys = {(cell, tier) for cell in cells for tier in tiers}
    if set(by_key) != expected_keys:
        raise ValueError("full-wire provenance ledger is not a complete cell/tier matrix")
    costs_by_option = {
        key: {
            str(name): float(value)
            for name, value in row["prediction_by_class"].items()
        }
        for key, row in by_key.items()
    }
    effective_caps = class_caps
    if effective_caps is None:
        classes = sorted(next(iter(costs_by_option.values())))
        effective_caps = {
            name: sum(
                max(costs_by_option[(cell, tier)][name] for tier in tiers)
                for cell in cells
            )
            for name in classes
        }
    result = solve_class_balanced_options(
        cells=cells,
        tiers=tiers,
        bytes_by_option={key: int(row["physical_bytes"]) for key, row in by_key.items()},
        class_costs_by_option=costs_by_option,
        activation_artifacts_by_option={
            key: tuple(dict(value) for value in row["activation_artifacts"])
            for key, row in by_key.items()
        },
        envelope_bytes=envelope,
        class_caps=effective_caps,
        class_weights=class_weights,
        exact_envelope=exact_envelope,
    )
    selected_expert_bytes = int(result["assigned_bytes"])
    unpadded_whole_bytes = fixed_bytes + selected_expert_bytes
    padding_bytes = (
        shipping_bytes_cap - unpadded_whole_bytes
        if padding_policy == "metadata_reserve"
        else 0
    )
    whole_bytes = unpadded_whole_bytes + padding_bytes
    predicted_totals = result.get(
        "predicted_class_totals", result.get("prediction_by_class")
    )
    if not isinstance(predicted_totals, dict):
        raise RuntimeError("class-balanced solve omitted predicted class totals")
    accounting = {
        "shipping_bytes_cap": shipping_bytes_cap,
        "expert_envelope_bytes": envelope,
        "selected_expert_bytes": selected_expert_bytes,
        "selected_cell_payload_bytes": int(result["cell_payload_bytes"]),
        "selected_activation_bytes": int(result["activation_bytes"]),
        "dense_nonrouted_bytes": dense,
        "repair_bytes": repair,
        "metadata_bytes": metadata,
        "fixed_nonexpert_bytes": fixed_bytes,
        "unpadded_whole_shipping_bytes": unpadded_whole_bytes,
        "padding_bytes": padding_bytes,
        "padding_policy": padding_policy,
        "whole_shipping_bytes": whole_bytes,
        "shipping_slack_bytes": shipping_bytes_cap - whole_bytes,
    }
    assignment = {
        "schema": "banana-smasher-provenance-weighted-assignment-v1",
        "status": "PASS_PREDICTION_ONLY",
        **identity,
        "source_option_ledger": _descriptor(ledger_path, ledger_raw),
        "source_fixed_accounting": _descriptor(fixed_path, fixed_raw),
        "whole_model_accounting": accounting,
        "activation_artifacts": result["activated_artifacts"],
        "predicted_class_totals": predicted_totals,
        "assignments": result["assignments"],
        "solver": result["solver"],
        "allowed_tiers": tiers,
        "exact_envelope": exact_envelope,
        "padding_policy": padding_policy,
        "exact_model_size": whole_bytes == shipping_bytes_cap,
    }
    assignment_raw = _canonical(assignment)
    if assignment_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite provenance solve outputs")
    _atomic_write(assignment_path, assignment_raw)
    receipt = {
        "schema": "banana-smasher-provenance-weighted-solve-receipt-v1",
        "status": "PASS",
        "option_ledger": _descriptor(ledger_path, ledger_raw),
        "fixed_accounting": _descriptor(fixed_path, fixed_raw),
        "assignment": _descriptor(assignment_path, assignment_raw),
        "whole_model_accounting": accounting,
        "activation_artifacts": result["activated_artifacts"],
        "predicted_class_totals": predicted_totals,
        "solver": result["solver"],
    }
    _atomic_write(receipt_path, _canonical(receipt))
    return receipt


def run_configured_provenance_solve(
    config: str | Path, *, output: str | Path
) -> dict[str, Any]:
    """Run a hash-bound tier-filtered provenance solve from one JSON config."""

    config_path = Path(config).expanduser().resolve()
    config_raw = config_path.read_bytes()
    value = json.loads(config_raw)
    if value.get("schema") != "banana-smasher-provenance-solve-config-v1":
        raise ValueError("provenance solve config schema mismatch")
    inputs = value.get("inputs")
    target = value.get("target")
    if not isinstance(inputs, dict) or not isinstance(target, dict):
        raise ValueError("provenance solve config lacks inputs/target")

    def configured_path(name: str) -> tuple[Path, str]:
        descriptor = inputs.get(name)
        if not isinstance(descriptor, dict):
            raise ValueError(f"provenance solve config lacks {name}")
        path = Path(str(descriptor.get("path", ""))).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        expected = str(descriptor.get("sha256", ""))
        if len(expected) != 64:
            raise ValueError(f"provenance solve config has invalid {name} SHA-256")
        return path.resolve(), expected

    ledger, ledger_sha = configured_path("option_ledger")
    fixed, fixed_sha = configured_path("fixed_accounting")
    tiers = value.get("allowed_tiers")
    if (
        not isinstance(tiers, list)
        or not tiers
        or any(not isinstance(tier, str) or not tier for tier in tiers)
        or len(set(tiers)) != len(tiers)
    ):
        raise ValueError("allowed_tiers must be a non-empty unique string list")
    class_weights = value.get("class_weights")
    if not isinstance(class_weights, dict) or not class_weights:
        raise ValueError("class_weights must be a non-empty object")
    class_caps = value.get("class_caps")
    if class_caps is not None and not isinstance(class_caps, dict):
        raise ValueError("class_caps must be an object when provided")
    cap = target.get("whole_model_bytes")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise ValueError("target.whole_model_bytes must be a positive integer")
    exact = target.get("exact", False)
    if not isinstance(exact, bool):
        raise ValueError("target.exact must be boolean")
    padding_policy = target.get("padding_policy")
    if padding_policy not in (None, "metadata_reserve"):
        raise ValueError("target.padding_policy must be metadata_reserve when provided")
    if padding_policy is not None and not exact:
        raise ValueError("target.padding_policy requires target.exact=true")

    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    receipt = run_full_wire_provenance_solve(
        ledger,
        fixed,
        destination / "ASSIGNMENT.json",
        destination / "RECEIPT.json",
        expected_option_ledger_sha256=ledger_sha,
        expected_fixed_accounting_sha256=fixed_sha,
        shipping_bytes_cap=cap,
        class_weights={str(name): float(weight) for name, weight in class_weights.items()},
        class_caps=(
            None
            if class_caps is None
            else {str(name): float(limit) for name, limit in class_caps.items()}
        ),
        allowed_tiers=tuple(tiers),
        exact_envelope=exact and padding_policy is None,
        padding_policy=padding_policy,
    )
    return {
        **receipt,
        "config": _descriptor(config_path, config_raw),
        "allowed_tiers": tiers,
        "exact": exact,
    }
