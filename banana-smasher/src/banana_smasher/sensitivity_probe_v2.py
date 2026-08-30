"""Generalized full-model sensitivity probe materializer (PROBE_MANIFEST_V2).

Extends :mod:`banana_smasher.sensitivity_probe` with exactly the three
capabilities PROBE_MANIFEST_V2 requires and v1 lacks:

1. ``null_control`` probes where ``source_tier == target_tier``. v1 hard-refused
   these ("sensitivity probe must change tier"). A null re-materializes a
   byte-identical pack so its measured delta isolates pure instrument noise.
2. ``additivity_joint`` probes that swap two cells in a single candidate.
3. Per-cell binding of the *target* root map, so the physical unit actually
   scored is the ledger's declared producer rather than the sealed baseline's
   incumbent payload. This is the exact defect that produced
   ``QTIP unit payload identity mismatch`` on run6989.

Every v1 invariant is preserved: fail-closed identity checks, canonical JSON,
atomic writes, and refusal to reuse a non-empty candidate directory.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

from banana_smasher.sensitivity_probe import _atomic, _bound, _canonical, _sha

NULL_ROLE = "null_control"


def _ledger_options(path: Path, wanted: set[tuple[str, str]], basis: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Single streaming pass over the 187 MB option ledger for every wanted key."""

    found: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("basis_sha256") != basis:
                raise ValueError(f"ledger basis mismatch at line {line_number}")
            key = (row.get("cell_id"), row.get("tier"))
            if key in wanted:
                if key in found:
                    raise ValueError(f"duplicate ledger option for {key}")
                found[key] = row
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"missing ledger options: {sorted(missing)}")
    return found


def probe_cell_specs(probe: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize a v1 single-cell or v2 multi-cell probe into per-cell specs."""

    cells = probe.get("cells") or [probe.get("cell_id")]
    producer = probe.get("target_physical_producer")
    producers: Sequence[Any] = producer if isinstance(producer, list) else [producer]
    if len(cells) != len(producers):
        raise ValueError(f"probe {probe.get('probe_id')}: cells/producers length mismatch")
    source_tier = str(probe["source_tier"])
    target_tier = str(probe["target_tier"])
    specs = []
    for cell_id, prod in zip(cells, producers):
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError("probe cell id is invalid")
        if not isinstance(prod, Mapping):
            raise ValueError(f"probe {probe.get('probe_id')}: cell {cell_id} has no physical producer")
        specs.append({
            "cell_id": cell_id,
            "source_tier": source_tier,
            "target_tier": target_tier,
            "producer": dict(prod),
        })
    return specs


def materialize_sensitivity_candidate_v2(
    baseline_manifest_path: str | Path,
    option_ledger_path: str | Path,
    probe: Mapping[str, Any],
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    """Create one diagnostic virtual pack with 1..N cells re-tiered."""

    role = str(probe.get("role", "treatment"))
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

    specs = probe_cell_specs(probe)
    is_null = all(s["source_tier"] == s["target_tier"] for s in specs)
    if is_null and role != NULL_ROLE:
        raise ValueError(f"probe {probe.get('probe_id')}: same-tier swap outside {NULL_ROLE}")
    if not is_null and any(s["source_tier"] == s["target_tier"] for s in specs):
        raise ValueError("mixed null/non-null cells in one probe")
    if len({s["cell_id"] for s in specs}) != len(specs):
        raise ValueError("probe repeats a cell")

    targets = _ledger_options(
        Path(option_ledger_path).resolve(),
        {(s["cell_id"], s["target_tier"]) for s in specs},
        basis,
    )
    for spec in specs:
        target = targets[(spec["cell_id"], spec["target_tier"])]
        ledger_producer = target.get("physical_producer")
        if not isinstance(ledger_producer, Mapping):
            raise ValueError("target option has no physical producer")
        # The manifest's producer must be exactly the ledger's producer. This is
        # the binding that run6989 lacked.
        for field in ("artifact_sha256", "sha256", "path"):
            if str(spec["producer"].get(field)) != str(ledger_producer.get(field)):
                raise ValueError(
                    f"probe {probe.get('probe_id')} cell {spec['cell_id']}: "
                    f"manifest/ledger producer {field} divergence"
                )
        spec["target"] = target
        spec["producer"] = dict(ledger_producer)

    assignment = json.loads(assignment_path.read_text())
    for spec in specs:
        layer_s, expert_s, projection = spec["cell_id"].split(":")
        layer, expert = int(layer_s[1:]), int(expert_s[1:])
        if assignment[str(layer)][str(expert)][projection] != spec["source_tier"]:
            raise ValueError(f"probe source tier mismatch in assignment at {spec['cell_id']}")
        assignment[str(layer)][str(expert)][projection] = spec["target_tier"]

    by_cell = {s["cell_id"]: s for s in specs}
    rows = []
    matched: set[str] = set()
    for line in index_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        spec = by_cell.get(row.get("cell_id"))
        if spec is not None:
            if row.get("tier") != spec["source_tier"]:
                raise ValueError(f"probe source tier mismatch in index at {spec['cell_id']}")
            spec["source_bytes"] = int(row["physical_bytes"])
            row.update(
                tier=spec["target_tier"],
                source_key=spec["target_tier"],
                physical_bytes=int(spec["target"]["physical_bytes"]),
                physical_artifact_sha256=str(spec["producer"]["artifact_sha256"]),
                physical_receipt_path=str(spec["producer"]["path"]),
                physical_receipt_sha256=str(spec["producer"]["sha256"]),
            )
            matched.add(spec["cell_id"])
        rows.append(row)
    if matched != set(by_cell):
        raise ValueError(f"probe cells did not resolve exactly once: {sorted(set(by_cell) - matched)}")

    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty sensitivity candidate: {output}")
    output.mkdir(parents=True, exist_ok=True)
    assignment_raw = _canonical(assignment)
    index_raw = b"".join(_canonical(row) for row in sorted(rows, key=lambda row: row["cell_id"]))
    _atomic(output / "ASSIGNMENT.json", assignment_raw)
    _atomic(output / "MATERIALIZATION_INDEX.jsonl", index_raw)

    delta_bytes = sum(int(s["target"]["physical_bytes"]) - int(s["source_bytes"]) for s in specs)
    candidate = json.loads(json.dumps(manifest))
    candidate["arm_name"] = "sensitivity-multi-cell-diagnostic" if len(specs) > 1 else "sensitivity-single-cell-diagnostic"
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
    for spec in specs:
        candidate["tier_counts"][spec["source_tier"]] -= 1
        candidate["tier_counts"][spec["target_tier"]] += 1
        candidate["byte_accounting"]["tier_payload_bytes"][spec["source_tier"]] -= int(spec["source_bytes"])
        candidate["byte_accounting"]["tier_payload_bytes"][spec["target_tier"]] += int(spec["target"]["physical_bytes"])
    candidate["byte_accounting"]["payload_bytes"] += delta_bytes
    candidate["byte_accounting"]["assigned_expert_bytes"] += delta_bytes
    candidate["byte_accounting"]["assigned_package_bytes"] += delta_bytes
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
        "role": role,
        "cell_ids": [s["cell_id"] for s in specs],
        "source_tier": specs[0]["source_tier"],
        "target_tier": specs[0]["target_tier"],
        "predicted_delta_mean_kld": probe.get("predicted_delta_mean_kld"),
        "diagnostic_nonshipping": True,
        "shipping_delta_bytes": delta_bytes,
        "is_null_control": is_null,
    }
    manifest_raw = _canonical(candidate)
    manifest_output = output / "BACKPACK_VIRTUAL_MANIFEST.json"
    _atomic(manifest_output, manifest_raw)
    terminal = {
        # The exact64 runtime pins this schema string for diagnostic_nonshipping
        # scoring; the contract is unchanged from v1, so the identifier must not move.
        "schema": "banana-smasher-sensitivity-virtual-terminal-v1",
        "status": "PASS",
        "basis_sha256": basis,
        "virtual_manifest_path": str(manifest_output),
        "virtual_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "whole_model_accounting": accounting,
        "probe_id": probe.get("probe_id"),
        "role": role,
        "cell_ids": [s["cell_id"] for s in specs],
    }
    terminal_raw = _canonical(terminal)
    terminal_output = output / "SENSITIVITY_VIRTUAL_TERMINAL.json"
    _atomic(terminal_output, terminal_raw)
    root_maps = sorted({
        (str(s["producer"].get("root_map_path")), str(s["producer"].get("root_map_sha256")))
        for s in specs
        if s["target_tier"] in {"qtip2", "qtip3"}
    })
    return {
        "schema": "banana-smasher-sensitivity-candidate-receipt-v2",
        "status": "PASS",
        "root": str(output),
        "basis_sha256": basis,
        "probe_id": probe.get("probe_id"),
        "role": role,
        "cell_ids": [s["cell_id"] for s in specs],
        "source_tier": specs[0]["source_tier"],
        "target_tier": specs[0]["target_tier"],
        "is_null_control": is_null,
        "target_root_maps": [{"path": p, "sha256": h} for p, h in root_maps],
        "target_units": [
            {
                "cell_id": s["cell_id"],
                "tier": s["target_tier"],
                "artifact_sha256": str(s["producer"]["artifact_sha256"]),
                "receipt_sha256": str(s["producer"]["sha256"]),
                "physical_bytes": int(s["target"]["physical_bytes"]),
            }
            for s in specs
        ],
        "shipping_delta_bytes": delta_bytes,
        "manifest_path": str(manifest_output),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "terminal_path": str(terminal_output),
        "terminal_sha256": hashlib.sha256(terminal_raw).hexdigest(),
        "index_path": str(output / "MATERIALIZATION_INDEX.jsonl"),
        "index_sha256": hashlib.sha256(index_raw).hexdigest(),
    }


def authenticate_target_units(
    candidate: Mapping[str, Any],
    *,
    root_map_path: Path,
    root_map_sha256: str,
) -> list[dict[str, Any]]:
    """Fail-closed: prove the *payload bytes* the decoder will read are the
    ledger's declared producer, before any GPU time is spent.

    run6989 bound the sealed baseline root map for a qtip2->qtip3 swap, so the
    QTIP3 decoder was handed the incumbent QTIP2 payload and refused it after
    the 441 s instrument gate had already run. This check moves that failure
    to t=0 and makes it explicit.
    """

    if not root_map_path.is_file() or _sha(root_map_path) != root_map_sha256:
        raise RuntimeError(f"TARGET_ROOT_MAP_IDENTITY_RED:{root_map_path}")
    root_map = json.loads(root_map_path.read_text())
    layer_roots = root_map.get("layer_roots") or {}
    cell_roots = root_map.get("cell_roots") or {}
    if not isinstance(cell_roots, Mapping):
        raise RuntimeError("TARGET_ROOT_MAP_CELL_OVERRIDES_RED")
    proofs = []
    for unit in candidate.get("target_units", []):
        if unit["tier"] not in {"qtip2", "qtip3"}:
            continue
        layer_s, expert_s, projection = unit["cell_id"].split(":")
        layer = int(layer_s[1:])
        root = cell_roots.get(unit["cell_id"], layer_roots.get(str(layer)))
        if not root:
            raise RuntimeError(f"TARGET_ROOT_MAP_LAYER_MISSING:{unit['cell_id']}")
        unit_path = Path(root) / layer_s / f"{expert_s}_{projection}" / "QTIP_UNIT.pt"
        receipt_path = unit_path.parent / "QTIP_SOLVE_RECEIPT.json"
        if not unit_path.is_file():
            raise RuntimeError(f"TARGET_UNIT_ABSENT:{unit_path}")
        observed_unit = _sha(unit_path)
        if observed_unit != unit["artifact_sha256"]:
            raise RuntimeError(
                f"TARGET_UNIT_PAYLOAD_RED:{unit['cell_id']}:{observed_unit}:{unit['artifact_sha256']}"
            )
        observed_receipt = _sha(receipt_path) if receipt_path.is_file() else None
        if observed_receipt != unit["receipt_sha256"]:
            raise RuntimeError(
                f"TARGET_RECEIPT_PAYLOAD_RED:{unit['cell_id']}:{observed_receipt}:{unit['receipt_sha256']}"
            )
        if unit_path.stat().st_size != unit["physical_bytes"]:
            raise RuntimeError(f"TARGET_UNIT_BYTES_RED:{unit['cell_id']}")
        proofs.append({
            "cell_id": unit["cell_id"],
            "tier": unit["tier"],
            "unit_path": str(unit_path),
            "unit_sha256": observed_unit,
            "receipt_sha256": observed_receipt,
            "physical_bytes": unit["physical_bytes"],
        })
    return proofs
