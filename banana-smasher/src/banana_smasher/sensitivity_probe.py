"""Build and score full-model single-cell sensitivity probes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Mapping


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _bound(root: Path, descriptor: Mapping[str, Any], role: str) -> Path:
    value = descriptor.get("file")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{role} descriptor has no file")
    path = root / value
    if (
        not path.is_file()
        or path.stat().st_size != descriptor.get("bytes")
        or _sha(path) != descriptor.get("sha256")
    ):
        raise ValueError(f"{role} descriptor identity mismatch")
    return path


def _ledger_option(path: Path, cell_id: str, tier: str, basis: str) -> dict[str, Any]:
    found = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("basis_sha256") != basis:
                raise ValueError(f"ledger basis mismatch at line {line_number}")
            if row.get("cell_id") == cell_id and row.get("tier") == tier:
                found.append(row)
    if len(found) != 1:
        raise ValueError(f"expected one target option for {(cell_id, tier)}, found {len(found)}")
    return found[0]


def materialize_sensitivity_candidate(
    baseline_manifest_path: str | Path,
    option_ledger_path: str | Path,
    probe: Mapping[str, Any],
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    """Create one diagnostic virtual pack with exactly one upgraded cell."""

    manifest_path = Path(baseline_manifest_path).expanduser().resolve()
    baseline_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != "banana-smasher-backpack-virtual-assignment-v1"
        or manifest.get("status") != "PASS_LOGICAL_FULL_WIRE"
    ):
        raise ValueError("baseline virtual manifest must be PASS v1")
    basis = manifest.get("basis_sha256")
    if not isinstance(basis, str) or len(basis) != 64:
        raise ValueError("baseline basis is invalid")
    assignment_path = _bound(baseline_root, manifest["assignment"], "assignment")
    index_path = _bound(baseline_root, manifest["materialization_index"], "index")
    raw_cell_id = probe.get("cell_id")
    raw_source_tier = probe.get("source_tier")
    raw_target_tier = probe.get("target_tier")
    if not all(
        isinstance(value, str) and value
        for value in (raw_cell_id, raw_source_tier, raw_target_tier)
    ):
        raise ValueError("probe cell/source/target is invalid")
    cell_id = str(raw_cell_id)
    source_tier = str(raw_source_tier)
    target_tier = str(raw_target_tier)
    if source_tier == target_tier:
        raise ValueError("sensitivity probe must change tier")
    target = _ledger_option(Path(option_ledger_path).resolve(), cell_id, target_tier, basis)
    producer = target.get("physical_producer")
    if not isinstance(producer, Mapping):
        raise ValueError("target option has no physical producer")

    assignment = json.loads(assignment_path.read_text())
    layer_s, expert_s, projection = cell_id.split(":")
    layer, expert = int(layer_s[1:]), int(expert_s[1:])
    if assignment[str(layer)][str(expert)][projection] != source_tier:
        raise ValueError("probe source tier does not match baseline assignment")
    assignment[str(layer)][str(expert)][projection] = target_tier

    rows = []
    matches = 0
    source_bytes = None
    for line in index_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("cell_id") == cell_id:
            if row.get("tier") != source_tier:
                raise ValueError("probe source tier does not match baseline index")
            source_bytes = int(row["physical_bytes"])
            row.update(
                tier=target_tier,
                source_key=target_tier,
                physical_bytes=int(target["physical_bytes"]),
                physical_artifact_sha256=str(producer["artifact_sha256"]),
                physical_receipt_path=str(producer["path"]),
                physical_receipt_sha256=str(producer["sha256"]),
            )
            matches += 1
        rows.append(row)
    if matches != 1 or source_bytes is None:
        raise ValueError("probe cell did not resolve exactly once")

    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty sensitivity candidate: {output}")
    output.mkdir(parents=True, exist_ok=True)
    assignment_raw = _canonical(assignment)
    index_raw = b"".join(_canonical(row) for row in sorted(rows, key=lambda row: row["cell_id"]))
    _atomic(output / "ASSIGNMENT.json", assignment_raw)
    _atomic(output / "MATERIALIZATION_INDEX.jsonl", index_raw)

    delta_bytes = int(target["physical_bytes"]) - source_bytes
    candidate = json.loads(json.dumps(manifest))
    candidate["arm_name"] = "sensitivity-single-cell-diagnostic"
    candidate["assignment_map_sha256"] = hashlib.sha256(assignment_raw).hexdigest()
    candidate["assignment"] = {
        "file": "ASSIGNMENT.json",
        "bytes": len(assignment_raw),
        "rows": len(rows),
        "sha256": candidate["assignment_map_sha256"],
    }
    candidate["materialization_index"] = {
        "file": "MATERIALIZATION_INDEX.jsonl",
        "bytes": len(index_raw),
        "rows": len(rows),
        "sha256": hashlib.sha256(index_raw).hexdigest(),
    }
    candidate["tier_counts"][source_tier] -= 1
    candidate["tier_counts"][target_tier] += 1
    candidate["byte_accounting"]["payload_bytes"] += delta_bytes
    candidate["byte_accounting"]["assigned_expert_bytes"] += delta_bytes
    candidate["byte_accounting"]["assigned_package_bytes"] += delta_bytes
    candidate["byte_accounting"]["tier_payload_bytes"][source_tier] -= source_bytes
    candidate["byte_accounting"]["tier_payload_bytes"][target_tier] += int(target["physical_bytes"])
    candidate["expert_wire_bpw"] = (
        candidate["byte_accounting"]["assigned_expert_bytes"] * 8
        / candidate["expert_parameter_denominator"]
    )
    accounting = candidate["whole_model_accounting"]
    accounting["expert_physical_wire_bytes"] += delta_bytes
    whole = (
        accounting["expert_physical_wire_bytes"]
        + accounting["fixed_nonexpert_bytes"]
        + accounting.get("padding_bytes", 0)
    )
    accounting["whole_shipping_bytes"] = whole
    accounting["shipping_slack_bytes"] = accounting["shipping_bytes_cap"] - whole
    numerator = whole * 8
    accounting["whole_model_bpw_numerator_bits"] = numerator
    accounting["whole_model_bpw_exact_ratio"] = f"{numerator}/{accounting['logical_base_parameters']}"
    with localcontext() as context:
        context.prec = 80
        accounting["whole_model_bpw_decimal"] = format(
            Decimal(numerator) / Decimal(accounting["logical_base_parameters"]), "f"
        )
    candidate["sensitivity_probe"] = {
        "probe_id": probe.get("probe_id"),
        "cell_id": cell_id,
        "source_tier": source_tier,
        "target_tier": target_tier,
        "predicted_delta_mean_kld": probe.get("predicted_delta_mean_kld"),
        "diagnostic_nonshipping": True,
        "shipping_delta_bytes": delta_bytes,
    }
    manifest_raw = _canonical(candidate)
    manifest_output = output / "BACKPACK_VIRTUAL_MANIFEST.json"
    _atomic(manifest_output, manifest_raw)
    terminal = {
        "schema": "banana-smasher-sensitivity-virtual-terminal-v1",
        "status": "PASS",
        "basis_sha256": basis,
        "virtual_manifest_path": str(manifest_output),
        "virtual_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "whole_model_accounting": accounting,
        "probe_id": probe.get("probe_id"),
        "cell_id": cell_id,
    }
    terminal_raw = _canonical(terminal)
    terminal_output = output / "SENSITIVITY_VIRTUAL_TERMINAL.json"
    _atomic(terminal_output, terminal_raw)
    return {
        "schema": "banana-smasher-sensitivity-candidate-receipt-v1",
        "status": "PASS",
        "root": str(output),
        "basis_sha256": basis,
        "probe_id": probe.get("probe_id"),
        "cell_id": cell_id,
        "source_tier": source_tier,
        "target_tier": target_tier,
        "target_root_map_path": producer.get("root_map_path"),
        "target_root_map_sha256": producer.get("root_map_sha256"),
        "shipping_delta_bytes": delta_bytes,
        "manifest_path": str(manifest_output),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "terminal_path": str(terminal_output),
        "terminal_sha256": hashlib.sha256(terminal_raw).hexdigest(),
        "index_path": str(output / "MATERIALIZATION_INDEX.jsonl"),
        "index_sha256": hashlib.sha256(index_raw).hexdigest(),
    }
