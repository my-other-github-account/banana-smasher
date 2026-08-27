from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any


CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")
CANDIDATE_LEDGER_SCHEMAS = frozenset(
    {
        "banana-smasher-dynamic-backpack-candidate-ledger-row-v1",
        "banana-smasher-dynamic-backpack-candidate-ledger-row-v2",
    }
)
DIMENSION_BINDING_SCHEMA = "banana-smasher-dynamic-backpack-dimension-binding-v1"
SENSITIVITY_ROW_SCHEMA = "banana-smasher-sensitivity-row-v1"
SENSITIVITY_TIER_NAMES = {"Q2": "qtip2", "QTIP3_V7": "qtip3"}


class DynamicDimensionsError(ValueError):
    """Raised when explicit dynamic Backpack dimensions are incomplete or inconsistent."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_field(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise DynamicDimensionsError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _read_json(path: Path, label: str) -> tuple[Any, bytes]:
    try:
        raw = path.read_bytes()
        return json.loads(raw), raw
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DynamicDimensionsError(f"cannot read valid {label} at {path}: {exc}") from exc


def _read_jsonl(path: Path, label: str) -> tuple[list[dict[str, Any]], bytes]:
    try:
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DynamicDimensionsError(f"cannot read valid {label} at {path}: {exc}") from exc
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise DynamicDimensionsError(f"{label} must contain non-empty JSON-object rows")
    return rows, raw


def _finite(value: object, label: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DynamicDimensionsError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0) or (nonnegative and result < 0.0):
        qualifier = "positive finite" if positive else "non-negative finite" if nonnegative else "finite"
        raise DynamicDimensionsError(f"{label} must be {qualifier}")
    return result


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n").encode()


def _canonical_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows)


def _write_once(path: Path, payload: bytes) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace different sealed output: {path}")
        return
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FileExistsError(f"refusing to replace different sealed output: {path}")
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _identity(row: dict[str, Any], label: str) -> tuple[int, int, str, str]:
    layer, expert = row.get("layer"), row.get("expert")
    projection, tier = row.get("projection"), row.get("tier")
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise DynamicDimensionsError(f"{label} layer must be a non-negative integer")
    if isinstance(expert, bool) or not isinstance(expert, int) or expert < 0:
        raise DynamicDimensionsError(f"{label} expert must be a non-negative integer")
    if projection not in {"down", "fused13", "13"}:
        raise DynamicDimensionsError(f"{label} projection must be down, fused13, or 13")
    if not isinstance(tier, str) or not tier:
        raise DynamicDimensionsError(f"{label} tier must be a non-empty string")
    return layer, expert, projection, tier


def _normalize_mixed_source_rows(
    rows: list[dict[str, Any]], basis: str, label: str
) -> list[dict[str, Any]]:
    """Losslessly encode explicit expert sensitivity rows as projection options.

    A sensitivity row prices and sizes one complete expert (down + fused13).
    The existing mixed solver aggregates projection rows into that same expert
    choice, so the aggregate bytes and scalar measured damage are carried by
    the canonical ``down`` row and the companion projection carries zero. This
    preserves the measured expert option exactly; it does not split or infer a
    projection-level value. The scalar objective is intentionally identical
    for all six class lanes because the source authority is class-neutral.
    """

    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if row.get("schema") != SENSITIVITY_ROW_SCHEMA:
            normalized.append(row)
            continue
        row_label = f"{label} sensitivity row {index}"
        if row.get("basis_sha256") != basis:
            raise DynamicDimensionsError(f"{row_label} basis mismatch")
        layer, expert = row.get("layer"), row.get("expert")
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or layer < 0
            or isinstance(expert, bool)
            or not isinstance(expert, int)
            or expert < 0
        ):
            raise DynamicDimensionsError(f"{row_label} layer/expert geometry is invalid")
        source_tier = row.get("tier")
        if not isinstance(source_tier, str):
            raise DynamicDimensionsError(f"{row_label} tier is unsupported")
        tier = SENSITIVITY_TIER_NAMES.get(source_tier)
        if tier is None:
            raise DynamicDimensionsError(f"{row_label} tier is unsupported")
        physical_bytes = row.get("physical_bytes", row.get("bytes"))
        if (
            isinstance(physical_bytes, bool)
            or not isinstance(physical_bytes, int)
            or physical_bytes < 0
        ):
            raise DynamicDimensionsError(f"{row_label} physical bytes is invalid")
        raw_activations = row.get("activation_artifacts", [])
        if not isinstance(raw_activations, list):
            raise DynamicDimensionsError(
                f"{row_label} activation_artifacts must be an array"
            )
        activation_artifacts: list[dict[str, Any]] = []
        seen_activation_ids: set[str] = set()
        for activation_index, activation in enumerate(raw_activations):
            activation_label = (
                f"{row_label} activation_artifacts[{activation_index}]"
            )
            if not isinstance(activation, dict):
                raise DynamicDimensionsError(f"{activation_label} must be an object")
            activation_id = activation.get("id")
            activation_bytes = activation.get("bytes")
            if (
                not isinstance(activation_id, str)
                or not activation_id
                or activation_id in seen_activation_ids
                or isinstance(activation_bytes, bool)
                or not isinstance(activation_bytes, int)
                or activation_bytes < 0
            ):
                raise DynamicDimensionsError(f"{activation_label} is invalid")
            seen_activation_ids.add(activation_id)
            activation_artifacts.append(dict(activation))
        damage = _finite(
            row.get("predicted_delta_contribution"),
            f"{row_label} predicted_delta_contribution",
            nonnegative=True,
        )
        routing_mass = _finite(
            row.get("routing_mass"), f"{row_label} routing_mass", nonnegative=True
        )
        projections = row.get("projection_terms") or row.get("projection_metrics")
        if routing_mass > 0.0 and (
            not isinstance(projections, dict)
            or not {"down", "fused13"}.issubset(projections)
        ):
            raise DynamicDimensionsError(
                f"{row_label} lacks explicit down/fused13 activation evidence"
            )
        if routing_mass == 0.0 and row.get("coverage_status") not in {
            "ZERO_ROUTING_MASS_NO_ACTIVATION_ROWS",
            None,
        }:
            raise DynamicDimensionsError(f"{row_label} zero-route status is invalid")
        authority = {
            "schema": SENSITIVITY_ROW_SCHEMA,
            "builder_sha256": row.get("builder_sha256"),
            "split_metrics_sha256": row.get("split_metrics_sha256"),
            "routing_mass": routing_mass,
            "predicted_delta_contribution": damage,
            "encoding": "expert-total-carried-by-down;class-neutral-scalar",
        }
        for projection in ("down", "fused13"):
            carrier = projection == "down"
            normalized.append(
                {
                    "schema": "banana-smasher-dynamic-backpack-candidate-ledger-row-v2",
                    "status": "ADMITTED_COMPLETE_ALLOCATION_ELIGIBLE",
                    "allocation_eligible": True,
                    "basis_sha256": basis,
                    "candidate_id": f"L{layer:03d}.E{expert:03d}.{projection}.{tier}",
                    "layer": layer,
                    "expert": expert,
                    "projection": projection,
                    "tier": tier,
                    "physical_bytes": physical_bytes if carrier else 0,
                    "six_class_predictions": {
                        name: damage if carrier else 0.0 for name in CLASSES
                    },
                    "activation_artifacts": (
                        activation_artifacts if carrier else []
                    ),
                    "sensitivity_authority": authority,
                }
            )
    return normalized


def bind_mixed_v7_physical_dimensions(
    *,
    sensitivity_ledger: str | Path,
    member_contract: str | Path,
    qtip3_terminals: str | Path,
    basis_sha256: str,
    output: str | Path,
    receipt: str | Path,
) -> dict[str, Any]:
    """Bind measured V7 payload, control, and shared-table bytes to solver rows.

    The materialized member contract is the runtime authority for physical
    bytes. Per-expert controls are charged with their QTIP3 expert option;
    deduplicated LUTs are emitted as activation artifacts and charged once by
    the existing class-balanced solver.
    """

    basis = _sha_field(basis_sha256, "basis_sha256")
    ledger_path = Path(sensitivity_ledger).expanduser().resolve()
    contract_path = Path(member_contract).expanduser().resolve()
    terminals_root = Path(qtip3_terminals).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    receipt_path = Path(receipt).expanduser().resolve()
    if output_path == receipt_path:
        raise DynamicDimensionsError("output and receipt paths must differ")
    rows, ledger_raw = _read_jsonl(ledger_path, "sensitivity ledger")
    contract, contract_raw = _read_json(contract_path, "mixed V7 member contract")
    if (
        not isinstance(contract, dict)
        or contract.get("schema") != "banana-smasher-mixed-v7-member-contract-v1"
        or contract.get("status") != "PASS_ADMISSION_READY"
        or contract.get("basis_sha256") != basis
    ):
        raise DynamicDimensionsError("mixed V7 member contract schema/status/basis mismatch")
    members = contract.get("members")
    if not isinstance(members, list) or not members:
        raise DynamicDimensionsError("mixed V7 member contract has no members")

    qtip3: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    shared: dict[tuple[str, str, str], dict[str, Any]] = {}
    qtip2_lut: dict[str, Any] | None = None
    for index, member in enumerate(members):
        label = f"mixed V7 member {index}"
        if not isinstance(member, dict):
            raise DynamicDimensionsError(f"{label} must be an object")
        tier, cell = member.get("tier"), member.get("cell_id")
        payload, metadata = member.get("payload"), member.get("unit_metadata")
        if tier not in {"qtip2", "qtip3"} or not isinstance(cell, str):
            raise DynamicDimensionsError(f"{label} identity is invalid")
        if not isinstance(payload, dict) or not isinstance(metadata, dict):
            raise DynamicDimensionsError(f"{label} payload/metadata is invalid")
        for descriptor_name, descriptor in [("payload", payload), *metadata.items()]:
            if (
                not isinstance(descriptor, dict)
                or set(descriptor) != {"path", "sha256", "bytes"}
                or not isinstance(descriptor["path"], str)
                or not descriptor["path"]
                or isinstance(descriptor["bytes"], bool)
                or not isinstance(descriptor["bytes"], int)
                or descriptor["bytes"] < 0
            ):
                raise DynamicDimensionsError(f"{label} {descriptor_name} is invalid")
            _sha_field(descriptor["sha256"], f"{label} {descriptor_name}.sha256")
        tlut = metadata.get("tlut")
        if not isinstance(tlut, dict):
            raise DynamicDimensionsError(f"{label} lacks tlut metadata")
        shared_key = (tier, tlut["path"], tlut["sha256"])
        prior_shared = shared.get(shared_key)
        if prior_shared is not None and prior_shared != tlut:
            raise DynamicDimensionsError(f"{label} shared tlut conflicts")
        shared[shared_key] = dict(tlut)
        if tier == "qtip2":
            if qtip2_lut is not None and qtip2_lut != tlut:
                raise DynamicDimensionsError("QTIP2 shared tlut conflicts")
            qtip2_lut = dict(tlut)
            continue
        match = re.fullmatch(r"L(\d{3})\.E(\d{3})\.(down|fused13)", cell)
        control = metadata.get("control")
        if match is None or not isinstance(control, dict):
            raise DynamicDimensionsError(f"{label} QTIP3 identity/control is invalid")
        key = (int(match.group(1)), int(match.group(2)))
        projection = match.group(3)
        projections = qtip3.setdefault(key, {})
        if projection in projections:
            raise DynamicDimensionsError(f"duplicate QTIP3 contract member {cell}")
        projections[projection] = {
            "payload": dict(payload),
            "control": dict(control),
            "tlut": dict(tlut),
        }

    if qtip2_lut is None:
        raise DynamicDimensionsError("mixed V7 member contract lacks QTIP2 shared tlut")
    for key, projections in qtip3.items():
        if set(projections) != {"down", "fused13"}:
            raise DynamicDimensionsError(f"incomplete QTIP3 projection contract for {key}")

    terminal_sources: list[dict[str, Any]] = []
    for terminal_path in sorted(terminals_root.glob("L*/LAYER_PRODUCT_TERMINAL.json")):
        terminal, terminal_raw = _read_json(terminal_path, "QTIP3 layer terminal")
        layer = terminal.get("layer") if isinstance(terminal, dict) else None
        terminal_members = terminal.get("members") if isinstance(terminal, dict) else None
        lut = terminal.get("method", {}).get("lut") if isinstance(terminal, dict) else None
        if (
            terminal.get("schema") != "banana-smasher-qtip3-full43-layer-product-v2"
            or terminal.get("status") != "PASS"
            or terminal.get("basis_sha256") != basis
            or isinstance(layer, bool)
            or not isinstance(layer, int)
            or terminal.get("cells") != 512
            or terminal.get("complete_members") != 512
            or terminal.get("gaps") != 0
            or terminal.get("duplicates") != 0
            or not isinstance(terminal_members, list)
            or len(terminal_members) != 512
            or not isinstance(lut, dict)
        ):
            raise DynamicDimensionsError(f"QTIP3 layer terminal is incomplete: {terminal_path}")
        terminal_sources.append(
            {"path": str(terminal_path), "sha256": _sha(terminal_raw), "bytes": len(terminal_raw), "layer": layer}
        )
        terminal_by_expert: dict[int, dict[str, dict[str, Any]]] = {}
        for member_index, member in enumerate(terminal_members):
            if not isinstance(member, dict):
                raise DynamicDimensionsError(f"invalid QTIP3 terminal member {member_index}")
            match = re.fullmatch(r"E(\d{3})_(down|fused13)", str(member.get("cell", "")))
            codes, control = member.get("codes"), member.get("control")
            if match is None or not isinstance(codes, dict) or not isinstance(control, dict):
                raise DynamicDimensionsError(f"invalid QTIP3 terminal member {member_index}")
            expert, projection = int(match.group(1)), match.group(2)
            terminal_by_expert.setdefault(expert, {})[projection] = {
                "payload": dict(codes), "control": dict(control), "tlut": dict(lut)
            }
        if set(terminal_by_expert) != set(range(256)) or any(
            set(projections) != {"down", "fused13"}
            for projections in terminal_by_expert.values()
        ):
            raise DynamicDimensionsError(f"QTIP3 layer terminal geometry mismatch: {terminal_path}")
        for expert, projections in terminal_by_expert.items():
            qtip3[(layer, expert)] = projections

    enriched: list[dict[str, Any]] = []
    qtip3_bound = 0
    qtip2_bound = 0
    for index, row in enumerate(rows):
        if row.get("schema") != SENSITIVITY_ROW_SCHEMA or row.get("basis_sha256") != basis:
            raise DynamicDimensionsError(f"sensitivity row {index} schema/basis mismatch")
        source_tier = row.get("tier")
        if not isinstance(source_tier, str):
            raise DynamicDimensionsError(f"sensitivity row {index} tier is unsupported")
        tier = SENSITIVITY_TIER_NAMES.get(source_tier)
        layer, expert = row.get("layer"), row.get("expert")
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or isinstance(expert, bool)
            or not isinstance(expert, int)
        ):
            raise DynamicDimensionsError(f"sensitivity row {index} geometry is invalid")
        updated = dict(row)
        if tier == "qtip2":
            updated["activation_artifacts"] = [
                {
                    "id": f"qtip2:tlut:{qtip2_lut['sha256']}",
                    "bytes": qtip2_lut["bytes"],
                    "path": qtip2_lut["path"],
                    "sha256": qtip2_lut["sha256"],
                }
            ]
            qtip2_bound += 1
        elif tier == "qtip3":
            key = (layer, expert)
            projections = qtip3.get(key)
            if projections is None:
                raise DynamicDimensionsError(f"QTIP3 sensitivity row {index} lacks physical contract")
            updated["physical_bytes"] = sum(
                part[name]["bytes"]
                for part in projections.values()
                for name in ("payload", "control")
            )
            tlut = projections["down"]["tlut"]
            updated["activation_artifacts"] = [
                {
                    "id": f"qtip3:L{key[0]:03d}:tlut:{tlut['sha256']}",
                    "bytes": tlut["bytes"],
                    "path": tlut["path"],
                    "sha256": tlut["sha256"],
                }
            ]
            qtip3_bound += 1
        else:
            raise DynamicDimensionsError(f"sensitivity row {index} tier is unsupported")
        updated["physical_dimension_authority"] = {
            "member_contract_path": str(contract_path),
            "member_contract_sha256": _sha(contract_raw),
        }
        enriched.append(updated)

    output_raw = _canonical_jsonl(enriched)
    receipt_value = {
        "schema": "banana-smasher-mixed-v7-physical-dimensions-receipt-v1",
        "status": "PASS_PHYSICAL_DIMENSIONS_BOUND",
        "basis_sha256": basis,
        "rows": len(enriched),
        "qtip2_rows": qtip2_bound,
        "qtip3_rows": qtip3_bound,
        "shared_activation_artifacts": len(shared),
        "sources": {
            "sensitivity_ledger": {"path": str(ledger_path), "sha256": _sha(ledger_raw), "bytes": len(ledger_raw)},
            "member_contract": {"path": str(contract_path), "sha256": _sha(contract_raw), "bytes": len(contract_raw)},
            "qtip3_terminals": terminal_sources,
        },
        "output": {"path": str(output_path), "sha256": _sha(output_raw), "bytes": len(output_raw)},
    }
    receipt_raw = _canonical_json(receipt_value)
    _write_once(output_path, output_raw)
    _write_once(receipt_path, receipt_raw)
    return {
        **receipt_value,
        "receipt": {"path": str(receipt_path), "sha256": _sha(receipt_raw), "bytes": len(receipt_raw)},
    }


def build_dynamic_dimensions(
    *,
    ledger: str | Path,
    dimensions: str | Path,
    class_ceilings: str | Path,
    basis_sha256: str,
    output: str | Path,
    receipt: str | Path,
) -> dict[str, Any]:
    """Join only explicit per-candidate dimensions; never infer cells from aggregates."""

    basis = _sha_field(basis_sha256, "basis_sha256")
    ledger_path = Path(ledger).expanduser().resolve()
    dimensions_path = Path(dimensions).expanduser().resolve()
    ceilings_path = Path(class_ceilings).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    receipt_path = Path(receipt).expanduser().resolve()
    if output_path == receipt_path:
        raise DynamicDimensionsError("output and receipt paths must differ")

    ledger_rows, ledger_raw = _read_jsonl(ledger_path, "candidate ledger")
    dimension_rows, dimension_raw = _read_jsonl(dimensions_path, "dimension ledger")
    ceilings_value, ceilings_raw = _read_json(ceilings_path, "class ceilings")
    if not isinstance(ceilings_value, dict) or ceilings_value.get("schema") != "banana-smasher-dynamic-backpack-class-ceilings-v1":
        raise DynamicDimensionsError("class ceilings must use banana-smasher-dynamic-backpack-class-ceilings-v1")
    if ceilings_value.get("basis_sha256") != basis or ceilings_value.get("status") not in {"PASS", "SEALED"}:
        raise DynamicDimensionsError("class ceilings basis/status mismatch")
    raw_ceilings = ceilings_value.get("six_class_ceilings")
    if not isinstance(raw_ceilings, dict) or set(raw_ceilings) != set(CLASSES):
        raise DynamicDimensionsError("six_class_ceilings must explicitly cover the six canonical classes")
    ceilings = {name: _finite(raw_ceilings[name], f"ceiling {name}", nonnegative=True) for name in CLASSES}

    candidates: dict[str, dict[str, Any]] = {}
    candidate_identities: dict[str, tuple[int, int, str, str]] = {}
    physical_bindings: dict[str, int] = {}
    for index, row in enumerate(ledger_rows):
        schema = row.get("schema")
        if schema in CANDIDATE_LEDGER_SCHEMAS:
            if row.get("basis_sha256") != basis:
                raise DynamicDimensionsError(f"candidate ledger row {index} basis mismatch")
            candidate_id = row.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id or candidate_id in candidates:
                raise DynamicDimensionsError(f"candidate ledger row {index} candidate_id must be unique")
            candidates[candidate_id] = row
            candidate_identities[candidate_id] = _identity(row, f"candidate {candidate_id}")
            continue
        if schema != DIMENSION_BINDING_SCHEMA:
            raise DynamicDimensionsError(f"candidate ledger row {index} schema mismatch")
        if row.get("basis_sha256") != basis:
            raise DynamicDimensionsError(f"dimension binding row {index} basis mismatch")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id not in candidates:
            raise DynamicDimensionsError(f"dimension binding row {index} names unknown candidate {candidate_id!r}")
        if _identity(row, f"dimension binding {candidate_id}") != candidate_identities[candidate_id]:
            raise DynamicDimensionsError(f"dimension binding identity mismatch for {candidate_id}")
        physical_bytes = row.get("physical_bytes")
        if isinstance(physical_bytes, bool) or not isinstance(physical_bytes, int) or physical_bytes < 0:
            raise DynamicDimensionsError(f"dimension binding physical_bytes must be a non-negative integer for {candidate_id}")
        source_sidecar = row.get("source_physical_sidecar_sha256")
        _sha_field(source_sidecar, f"dimension binding {candidate_id} source_physical_sidecar_sha256")
        candidate_physical_bytes = candidates[candidate_id].get("physical_bytes")
        if candidate_physical_bytes is not None and candidate_physical_bytes != physical_bytes:
            raise DynamicDimensionsError(f"dimension binding physical_bytes conflict for {candidate_id}")
        prior_physical_bytes = physical_bindings.get(candidate_id)
        if prior_physical_bytes is not None and prior_physical_bytes != physical_bytes:
            raise DynamicDimensionsError(f"conflicting dimension bindings for {candidate_id}")
        physical_bindings[candidate_id] = physical_bytes

    explicit: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(dimension_rows):
        if row.get("schema") != "banana-smasher-dynamic-backpack-explicit-dimension-row-v1":
            raise DynamicDimensionsError(f"dimension row {index} schema mismatch; aggregate inputs are forbidden")
        if row.get("basis_sha256") != basis:
            raise DynamicDimensionsError(f"dimension row {index} basis mismatch")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in explicit:
            raise DynamicDimensionsError(f"dimension row {index} candidate_id must be unique")
        if candidate_id not in candidates:
            raise DynamicDimensionsError(f"dimension row {index} names unknown candidate {candidate_id!r}")
        if _identity(row, f"dimension {candidate_id}") != candidate_identities[candidate_id]:
            raise DynamicDimensionsError(f"dimension identity mismatch for {candidate_id}")
        explicit[candidate_id] = row

    missing = sorted(set(candidates) - set(explicit))
    if missing:
        raise DynamicDimensionsError(
            f"missing explicit dimensions for {len(missing)} candidates; first={missing[0]!r}; allocation forbidden"
        )

    completed: list[dict[str, Any]] = []
    for candidate_id in sorted(candidates):
        candidate, dimension = candidates[candidate_id], explicit[candidate_id]
        physical_bytes = physical_bindings.get(candidate_id, candidate.get("physical_bytes"))
        if isinstance(physical_bytes, bool) or not isinstance(physical_bytes, int) or physical_bytes < 0:
            raise DynamicDimensionsError(f"physical_bytes must be a non-negative integer for {candidate_id}")
        if dimension.get("physical_bytes") != physical_bytes:
            raise DynamicDimensionsError(f"physical_bytes mismatch for {candidate_id}")
        predictions_raw = dimension.get("six_class_predictions")
        if not isinstance(predictions_raw, dict) or set(predictions_raw) != set(CLASSES):
            raise DynamicDimensionsError(
                f"six_class_predictions must explicitly cover six classes for {candidate_id}; aggregate inference forbidden"
            )
        predictions = {
            name: _finite(predictions_raw[name], f"{candidate_id} prediction {name}", nonnegative=True)
            for name in CLASSES
        }
        routing_importance = _finite(
            dimension.get("routing_importance"), f"{candidate_id} routing_importance", positive=True
        )
        source_importance = candidate.get("source_class_features", {}).get("routing_importance")
        if source_importance is not None and not math.isclose(
            routing_importance,
            _finite(source_importance, f"{candidate_id} source routing_importance", positive=True),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise DynamicDimensionsError(f"routing_importance mismatch for {candidate_id}")
        projection_weight = _finite(
            dimension.get("projection_weight"), f"{candidate_id} projection_weight", positive=True
        )
        projection_correction = _finite(
            dimension.get("projection_correction"), f"{candidate_id} projection_correction"
        )
        authority = dimension.get("authority")
        if not isinstance(authority, dict):
            raise DynamicDimensionsError(f"authority must be explicit for {candidate_id}")
        for field in (
            "six_class_predictions_sha256",
            "routing_importance_sha256",
            "projection_correction_sha256",
            "physical_bytes_sha256",
        ):
            _sha_field(authority.get(field), f"{candidate_id} authority.{field}")
        completed.append(
            {
                **candidate,
                "physical_bytes": physical_bytes,
                "six_class_predictions": predictions,
                "six_class_ceilings": ceilings,
                "routing_importance": routing_importance,
                "projection_weight": projection_weight,
                "projection_correction": projection_correction,
                "dimension_authority": authority,
                "missing_dimensions": [],
                "allocation_eligible": True,
                "status": "ADMITTED_COMPLETE_ALLOCATION_ELIGIBLE",
            }
        )

    output_raw = _canonical_jsonl(completed)
    receipt_value = {
        "schema": "banana-smasher-dynamic-backpack-dimensions-receipt-v1",
        "status": "PASS_EXPLICIT_DIMENSIONS_COMPLETE",
        "basis_sha256": basis,
        "candidate_count": len(completed),
        "classes": list(CLASSES),
        "allocation_eligible": True,
        "inference_policy": "explicit-per-candidate-only; aggregate-to-cell inference forbidden",
        "sources": {
            "ledger": {"path": str(ledger_path), "sha256": _sha(ledger_raw), "bytes": len(ledger_raw)},
            "dimensions": {"path": str(dimensions_path), "sha256": _sha(dimension_raw), "bytes": len(dimension_raw)},
            "class_ceilings": {"path": str(ceilings_path), "sha256": _sha(ceilings_raw), "bytes": len(ceilings_raw)},
        },
        "output": {"path": str(output_path), "sha256": _sha(output_raw), "bytes": len(output_raw)},
    }
    receipt_raw = _canonical_json(receipt_value)
    _write_once(output_path, output_raw)
    _write_once(receipt_path, receipt_raw)
    if output_path.read_bytes() != output_raw or receipt_path.read_bytes() != receipt_raw:
        raise RuntimeError("sealed dimension outputs changed during publication")
    return {
        "status": receipt_value["status"],
        "command": "backpack-dimensions",
        "basis_sha256": basis,
        "candidate_count": len(completed),
        "allocation_eligible": True,
        "output": receipt_value["output"],
        "receipt": {"path": str(receipt_path), "sha256": _sha(receipt_raw), "bytes": len(receipt_raw)},
    }


def _resolve_mixed_dimension_sources(
    config_path: Path,
    descriptor: object,
    basis: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
    list[dict[str, Any]],
]:
    if not isinstance(descriptor, dict):
        raise DynamicDimensionsError("dimensions must be an object")
    if set(descriptor) == {"path", "sha256"}:
        raw_sources = [descriptor]
    elif set(descriptor) == {"sources"} and isinstance(descriptor["sources"], list):
        raw_sources = descriptor["sources"]
        if not raw_sources:
            raise DynamicDimensionsError("dimensions.sources must be non-empty")
    else:
        raise DynamicDimensionsError(
            "dimensions must contain path and sha256, or a non-empty sources array"
        )

    rows: list[dict[str, Any]] = []
    admitted: list[dict[str, Any]] = []
    pending: list[str] = []
    physical_members: list[dict[str, Any]] = []
    physical_member_keys: set[tuple[str, str]] = set()
    for index, source in enumerate(raw_sources):
        if not isinstance(source, dict):
            raise DynamicDimensionsError(f"dimensions source {index} must be an object")
        resolved = source
        locator_path: Path | None = None
        locator_raw = b""
        if set(source) == {"locator_path"}:
            locator_path = Path(str(source["locator_path"])).expanduser()
            if not locator_path.is_absolute():
                locator_path = config_path.parent / locator_path
            locator_path = locator_path.resolve()
            if not locator_path.exists():
                pending.append(str(locator_path))
                continue
            locator, locator_raw = _read_json(locator_path, f"dimensions locator {index}")
            if (
                not isinstance(locator, dict)
                or locator.get("status") not in {"PASS", "SEALED"}
                or locator.get("basis_sha256") != basis
            ):
                raise DynamicDimensionsError(
                    f"dimensions locator {index} schema/status/basis mismatch"
                )
            locator_schema = locator.get("schema")
            if locator_schema == "banana-smasher-mixed-backpack-physical-locator-v1":
                manifest_descriptor = locator.get("physical_manifest")
                if (
                    not isinstance(manifest_descriptor, dict)
                    or set(manifest_descriptor) != {"path", "sha256"}
                ):
                    raise DynamicDimensionsError(
                        f"physical locator {index} lacks physical_manifest descriptor"
                    )
                manifest_path = Path(str(manifest_descriptor["path"])).expanduser()
                if not manifest_path.is_absolute():
                    manifest_path = locator_path.parent / manifest_path
                manifest_path = manifest_path.resolve()
                expected_manifest_sha = _sha_field(
                    manifest_descriptor["sha256"],
                    f"physical locator {index}.physical_manifest.sha256",
                )
                manifest, manifest_raw = _read_json(
                    manifest_path, f"physical manifest {index}"
                )
                if _sha(manifest_raw) != expected_manifest_sha:
                    raise DynamicDimensionsError(
                        f"physical manifest {index} SHA-256 mismatch"
                    )
                members_expected = (
                    manifest.get("members_expected")
                    if isinstance(manifest, dict)
                    else None
                )
                members = manifest.get("members") if isinstance(manifest, dict) else None
                if (
                    not isinstance(manifest, dict)
                    or manifest.get("status") not in {"PASS", "SEALED"}
                    or manifest.get("basis_sha256") != basis
                    or manifest.get("gaps") != 0
                    or manifest.get("duplicates") != 0
                    or isinstance(members_expected, bool)
                    or not isinstance(members_expected, int)
                    or members_expected <= 0
                    or manifest.get("members_complete") != members_expected
                    or not isinstance(members, list)
                    or len(members) != members_expected
                ):
                    raise DynamicDimensionsError(
                        f"physical manifest {index} is not a complete basis-bound inventory"
                    )
                if manifest.get("schema") == "banana-smasher-mixed-backpack-physical-members-v1":
                    for member_index, member in enumerate(members):
                        member_label = f"physical manifest {index} member {member_index}"
                        if not isinstance(member, dict) or set(member) != {
                            "cell_id",
                            "tier",
                            "artifact",
                        }:
                            raise DynamicDimensionsError(
                                f"{member_label} must contain cell_id, tier, and artifact"
                            )
                        cell_id = member["cell_id"]
                        tier = member["tier"]
                        artifact = member["artifact"]
                        if (
                            not isinstance(cell_id, str)
                            or not re.fullmatch(
                                r"L\d{3}\.E\d{3}\.(?:down|fused13)", cell_id
                            )
                            or not isinstance(tier, str)
                            or not tier
                            or not isinstance(artifact, dict)
                            or set(artifact) != {"host", "path", "sha256", "bytes"}
                            or not isinstance(artifact["host"], str)
                            or not artifact["host"]
                            or not isinstance(artifact["path"], str)
                            or not artifact["path"]
                            or isinstance(artifact["bytes"], bool)
                            or not isinstance(artifact["bytes"], int)
                            or artifact["bytes"] <= 0
                        ):
                            raise DynamicDimensionsError(f"{member_label} is invalid")
                        _sha_field(
                            artifact["sha256"], f"{member_label}.artifact.sha256"
                        )
                        key = (cell_id, tier)
                        if key in physical_member_keys:
                            raise DynamicDimensionsError(
                                f"duplicate physical member binding {key!r}"
                            )
                        physical_member_keys.add(key)
                        physical_members.append(dict(member))
                admitted.append(
                    {
                        "kind": "physical_inventory",
                        "path": str(manifest_path),
                        "sha256": expected_manifest_sha,
                        "bytes": len(manifest_raw),
                        "rows": 0,
                        "members": manifest["members_complete"],
                        "locator": {
                            "path": str(locator_path),
                            "sha256": _sha(locator_raw),
                            "bytes": len(locator_raw),
                        },
                    }
                )
                continue
            if locator_schema != "banana-smasher-mixed-backpack-dimensions-locator-v1":
                raise DynamicDimensionsError(
                    f"dimensions locator {index} schema/status/basis mismatch"
                )
            resolved = locator.get("dimensions")
            if not isinstance(resolved, dict):
                raise DynamicDimensionsError(
                    f"dimensions locator {index} lacks dimensions descriptor"
                )
        elif set(source) != {"path", "sha256"}:
            raise DynamicDimensionsError(
                f"dimensions source {index} must contain path and sha256, or locator_path"
            )

        if not isinstance(resolved, dict) or set(resolved) != {"path", "sha256"}:
            raise DynamicDimensionsError(
                f"dimensions source {index} descriptor must contain path and sha256"
            )
        dimensions_path = Path(str(resolved["path"])).expanduser()
        if not dimensions_path.is_absolute():
            dimensions_path = (locator_path or config_path).parent / dimensions_path
        dimensions_path = dimensions_path.resolve()
        expected_sha = _sha_field(
            resolved["sha256"], f"dimensions source {index}.sha256"
        )
        source_rows, source_raw = _read_jsonl(
            dimensions_path, f"mixed Backpack dimensions source {index}"
        )
        if _sha(source_raw) != expected_sha:
            raise DynamicDimensionsError(
                f"mixed Backpack dimensions source {index} SHA-256 mismatch"
            )
        rows.extend(
            _normalize_mixed_source_rows(
                source_rows, basis, f"mixed Backpack dimensions source {index}"
            )
        )
        source_receipt = {
            "path": str(dimensions_path),
            "sha256": expected_sha,
            "bytes": len(source_raw),
            "rows": len(source_rows),
        }
        if locator_path is not None:
            source_receipt["locator"] = {
                "path": str(locator_path),
                "sha256": _sha(locator_raw),
                "bytes": len(locator_raw),
            }
        admitted.append(source_receipt)
    return rows, admitted, pending, physical_members


def preflight_mixed_backpack_config(config: str | Path) -> dict[str, Any]:
    """Admit available dimension shards and report exact pending coverage."""

    config_path = Path(config).expanduser().resolve()
    value, config_raw = _read_json(config_path, "mixed Backpack config")
    if not isinstance(value, dict) or value.get("schema") != "banana-smasher-mixed-backpack-config-v1":
        raise DynamicDimensionsError(
            "mixed Backpack config must use banana-smasher-mixed-backpack-config-v1"
        )
    basis = _sha_field(value.get("basis_sha256"), "basis_sha256")
    tiers = value.get("allowed_tiers")
    if not isinstance(tiers, list) or not tiers or any(
        not isinstance(tier, str) or not tier for tier in tiers
    ):
        raise DynamicDimensionsError("allowed_tiers must be a non-empty string array")
    fallback = value.get("fallback_tier")
    if fallback not in tiers:
        raise DynamicDimensionsError("fallback_tier must be present in allowed_tiers")
    topology = value.get("topology")
    if not isinstance(topology, dict) or set(topology) != {
        "layers",
        "experts_per_layer",
        "projections",
    }:
        raise DynamicDimensionsError(
            "topology must contain layers, experts_per_layer, and projections"
        )
    layers = topology["layers"]
    experts = topology["experts_per_layer"]
    projections = topology["projections"]
    if (
        not isinstance(layers, list)
        or not layers
        or len(layers) != len(set(layers))
        or any(isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 for layer in layers)
        or isinstance(experts, bool)
        or not isinstance(experts, int)
        or experts <= 0
        or not isinstance(projections, list)
        or not projections
        or len(projections) != len(set(projections))
        or any(projection not in {"down", "fused13"} for projection in projections)
    ):
        raise DynamicDimensionsError("topology geometry is invalid")

    rows, admitted, pending, _physical_members = _resolve_mixed_dimension_sources(
        config_path, value.get("dimensions"), basis
    )
    available: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        if row.get("basis_sha256") != basis:
            raise DynamicDimensionsError(f"mixed dimension row {index} basis mismatch")
        if row.get("allocation_eligible") is not True or row.get("status") != "ADMITTED_COMPLETE_ALLOCATION_ELIGIBLE":
            raise DynamicDimensionsError(f"mixed dimension row {index} is not allocation eligible")
        layer, expert, projection, tier = _identity(row, f"mixed dimension row {index}")
        projection = "fused13" if projection == "13" else projection
        key = (f"L{layer:03d}.E{expert:03d}.{projection}", tier)
        if key in available:
            raise DynamicDimensionsError(f"duplicate mixed option {key!r}")
        available.add(key)
    expected = [
        f"L{layer:03d}.E{expert:03d}.{projection}"
        for layer in sorted(layers)
        for expert in range(experts)
        for projection in projections
    ]
    coverage = {
        tier: {
            "available_projection_cells": sum((cell, tier) in available for cell in expected),
            "missing_layers": [
                layer
                for layer in sorted(layers)
                if not any(
                    (f"L{layer:03d}.E{expert:03d}.{projection}", tier) in available
                    for expert in range(experts)
                    for projection in projections
                )
            ],
        }
        for tier in tiers
    }
    missing_fallback = [cell for cell in expected if (cell, fallback) not in available]
    ready = not pending and not missing_fallback
    return {
        "schema": "banana-smasher-mixed-backpack-preflight-v1",
        "status": "READY_TO_SOLVE" if ready else "WAITING_FOR_DIMENSION_LOCATORS",
        "ready_to_solve": ready,
        "basis_sha256": basis,
        "config": {"path": str(config_path), "sha256": _sha(config_raw), "bytes": len(config_raw)},
        "sources": {"admitted": len(admitted), "pending": len(pending)},
        "admitted_sources": admitted,
        "pending_locators": pending,
        "coverage": coverage,
        "missing_fallback_projection_cells": missing_fallback,
    }


def solve_mixed_backpack_config(
    config: str | Path, *, output: str | Path
) -> dict[str, Any]:
    """Solve a basis-bound sparse tier inventory from one declarative config.

    The inventory may omit a tier for a cell.  Such rows are disabled in the
    shared class-balanced solver rather than synthesized; the configured
    fallback must be physically present for every cell.
    """

    from .knapsack import solve_class_balanced_options

    config_path = Path(config).expanduser().resolve()
    value, config_raw = _read_json(config_path, "mixed Backpack config")
    if not isinstance(value, dict) or value.get("schema") != "banana-smasher-mixed-backpack-config-v1":
        raise DynamicDimensionsError(
            "mixed Backpack config must use banana-smasher-mixed-backpack-config-v1"
        )
    allowed_fields = {
        "schema",
        "basis_sha256",
        "target",
        "allowed_tiers",
        "fallback_tier",
        "topology",
        "dimensions",
        "class_caps",
        "class_weights",
    }
    unknown = sorted(set(value) - allowed_fields)
    if unknown:
        raise DynamicDimensionsError(f"mixed Backpack config has unknown fields: {unknown}")
    basis = _sha_field(value.get("basis_sha256"), "basis_sha256")
    tiers = value.get("allowed_tiers")
    if (
        not isinstance(tiers, list)
        or not tiers
        or len(tiers) != len(set(tiers))
        or any(not isinstance(tier, str) or not tier for tier in tiers)
    ):
        raise DynamicDimensionsError("allowed_tiers must be a non-empty unique string array")
    fallback = value.get("fallback_tier")
    if fallback not in tiers:
        raise DynamicDimensionsError("fallback_tier must be present in allowed_tiers")

    target = value.get("target")
    if not isinstance(target, dict) or set(target) != {
        "whole_model_bytes",
        "fixed_nonexpert_bytes",
        "exact",
    }:
        raise DynamicDimensionsError(
            "target must contain whole_model_bytes, fixed_nonexpert_bytes, and exact"
        )
    whole_target = target["whole_model_bytes"]
    fixed_bytes = target["fixed_nonexpert_bytes"]
    if (
        isinstance(whole_target, bool)
        or not isinstance(whole_target, int)
        or isinstance(fixed_bytes, bool)
        or not isinstance(fixed_bytes, int)
        or whole_target <= fixed_bytes
        or fixed_bytes < 0
    ):
        raise DynamicDimensionsError("target byte accounting is invalid")
    if not isinstance(target["exact"], bool):
        raise DynamicDimensionsError("target.exact must be boolean")

    rows, dimension_sources, pending_locators, physical_members = _resolve_mixed_dimension_sources(
        config_path, value.get("dimensions"), basis
    )
    if pending_locators:
        raise DynamicDimensionsError(
            f"mixed Backpack dimensions have {len(pending_locators)} pending locators; "
            f"first={pending_locators[0]!r}"
        )
    if value.get("topology") is not None:
        preflight = preflight_mixed_backpack_config(config_path)
        if not preflight["ready_to_solve"]:
            missing = preflight["missing_fallback_projection_cells"]
            raise DynamicDimensionsError(
                f"fallback tier {fallback!r} is missing for {len(missing)} expected projection cells; "
                f"first={missing[0]!r}"
            )

    raw_caps = value.get("class_caps")
    if not isinstance(raw_caps, dict) or set(raw_caps) != set(CLASSES):
        raise DynamicDimensionsError("class_caps must cover the six canonical classes")
    class_caps = {
        name: _finite(raw_caps[name], f"class_caps.{name}", nonnegative=True)
        for name in CLASSES
    }
    raw_weights = value.get("class_weights")
    class_weights = None
    if raw_weights is not None:
        if not isinstance(raw_weights, dict) or set(raw_weights) != set(CLASSES):
            raise DynamicDimensionsError("class_weights must cover the six canonical classes")
        class_weights = {
            name: _finite(raw_weights[name], f"class_weights.{name}", nonnegative=True)
            for name in CLASSES
        }

    projection_inventory: dict[tuple[str, str], dict[str, Any]] = {}
    projection_geometry: dict[str, tuple[int, int, str]] = {}
    for index, row in enumerate(rows):
        if row.get("basis_sha256") != basis:
            raise DynamicDimensionsError(f"mixed dimension row {index} basis mismatch")
        if row.get("allocation_eligible") is not True or row.get("status") != "ADMITTED_COMPLETE_ALLOCATION_ELIGIBLE":
            raise DynamicDimensionsError(f"mixed dimension row {index} is not allocation eligible")
        layer, expert, projection, tier = _identity(row, f"mixed dimension row {index}")
        if tier not in tiers:
            continue
        projection = "fused13" if projection == "13" else projection
        cell_id = f"L{layer:03d}.E{expert:03d}.{projection}"
        key = (cell_id, tier)
        if key in projection_inventory:
            raise DynamicDimensionsError(f"duplicate mixed option {key!r}")
        predictions = row.get("six_class_predictions")
        if not isinstance(predictions, dict) or set(predictions) != set(CLASSES):
            raise DynamicDimensionsError(f"mixed option {key!r} lacks six-class predictions")
        physical_bytes = row.get("physical_bytes")
        if isinstance(physical_bytes, bool) or not isinstance(physical_bytes, int) or physical_bytes < 0:
            raise DynamicDimensionsError(f"mixed option {key!r} physical_bytes is invalid")
        activations = row.get("activation_artifacts", [])
        if not isinstance(activations, list) or not all(isinstance(item, dict) for item in activations):
            raise DynamicDimensionsError(f"mixed option {key!r} activation_artifacts is invalid")
        projection_inventory[key] = {
            "bytes": physical_bytes,
            "predictions": {
                name: _finite(predictions[name], f"{key!r}.{name}", nonnegative=True)
                for name in CLASSES
            },
            "activations": tuple(dict(item) for item in activations),
            "candidate_id": row.get("candidate_id"),
        }
        projection_geometry[cell_id] = (layer, expert, projection)
    projection_cells = sorted(projection_geometry)
    if not projection_cells:
        raise DynamicDimensionsError("mixed Backpack inventory has no allowed options")
    missing_fallback = [
        cell for cell in projection_cells if (cell, fallback) not in projection_inventory
    ]
    if missing_fallback:
        raise DynamicDimensionsError(
            f"fallback tier {fallback!r} is missing for {len(missing_fallback)} cells; first={missing_fallback[0]!r}"
        )

    # Runtime tier maps select one family per layer/expert, shared by down and
    # fused13.  Aggregate the measured projection rows into that physical
    # selection unit so the optimizer cannot emit an unmaterializable split.
    projections_by_cell: dict[str, list[str]] = {}
    cell_geometry: dict[str, tuple[int, int]] = {}
    for projection_cell, (layer, expert, _projection) in projection_geometry.items():
        cell_id = f"L{layer:03d}.E{expert:03d}"
        projections_by_cell.setdefault(cell_id, []).append(projection_cell)
        cell_geometry[cell_id] = (layer, expert)
    cells = sorted(projections_by_cell)
    for cell in cells:
        projections_by_cell[cell].sort(
            key=lambda value: (0 if value.endswith(".down") else 1, value)
        )

    expected_options = {(cell, tier) for cell in cells for tier in tiers}
    available = {
        (cell, tier)
        for cell in cells
        for tier in tiers
        if all(
            (projection_cell, tier) in projection_inventory
            for projection_cell in projections_by_cell[cell]
        )
    }
    bytes_by_option: dict[tuple[str, str], int] = {}
    costs_by_option: dict[tuple[str, str], dict[str, float]] = {}
    activations_by_option: dict[tuple[str, str], tuple[dict[str, Any], ...]] = {}
    fallback_cost = {name: 0.0 for name in CLASSES}
    for key in expected_options:
        cell, tier = key
        if key not in available:
            bytes_by_option[key] = 0
            costs_by_option[key] = dict(fallback_cost)
            activations_by_option[key] = ()
            continue
        options = [
            projection_inventory[(projection_cell, tier)]
            for projection_cell in projections_by_cell[cell]
        ]
        bytes_by_option[key] = sum(int(option["bytes"]) for option in options)
        costs_by_option[key] = {
            name: math.fsum(float(option["predictions"][name]) for option in options)
            for name in CLASSES
        }
        activation_by_id: dict[str, dict[str, Any]] = {}
        for option in options:
            for activation in option["activations"]:
                artifact_id = activation.get("id")
                prior = activation_by_id.get(str(artifact_id))
                if prior is not None and prior != activation:
                    raise DynamicDimensionsError(
                        f"activation artifact {artifact_id!r} conflicts across projections for {key!r}"
                    )
                activation_by_id[str(artifact_id)] = dict(activation)
        activations_by_option[key] = tuple(
            activation_by_id[name] for name in sorted(activation_by_id)
        )

    envelope = whole_target - fixed_bytes
    solved = solve_class_balanced_options(
        cells=cells,
        tiers=list(tiers),
        bytes_by_option=bytes_by_option,
        class_costs_by_option=costs_by_option,
        envelope_bytes=envelope,
        class_caps=class_caps,
        class_weights=class_weights,
        exact_envelope=target["exact"],
        available_options=available,
        activation_artifacts_by_option=activations_by_option,
    )
    assignment = {row["cell_id"]: row["tier"] for row in solved["assignments"]}
    assignment_sha = _sha(_canonical_json(assignment))
    layers: dict[int, dict[str, int]] = {}
    for cell_id, tier in assignment.items():
        layer = cell_geometry[cell_id][0]
        layer_counts = layers.setdefault(layer, {})
        layer_counts[tier] = layer_counts.get(tier, 0) + 1
    all_layers = sorted({geometry[0] for geometry in cell_geometry.values()})
    coverage = {
        tier: {
            "available_cells": sum((cell, tier) in available for cell in cells),
            "missing_layers": [
                layer
                for layer in all_layers
                if not any(
                    geometry[0] == layer and (cell, tier) in available
                    for cell, geometry in cell_geometry.items()
                )
            ],
        }
        for tier in tiers
    }
    output_root = Path(output).expanduser().resolve()
    physical_member_map = {
        (member["cell_id"], member["tier"]): member for member in physical_members
    }
    bound_tiers = {member["tier"] for member in physical_members}
    selected_projection_keys = {
        (projection_cell, assignment[cell])
        for cell in cells
        for projection_cell in projections_by_cell[cell]
    }
    missing_selected_members = sorted(
        key
        for key in selected_projection_keys
        if key[1] in bound_tiers and key not in physical_member_map
    )
    if missing_selected_members:
        raise DynamicDimensionsError(
            f"missing physical member binding {missing_selected_members[0]!r}"
        )
    selected_members = sorted(
        (
            member
            for member in physical_members
            if (member["cell_id"], member["tier"]) in selected_projection_keys
        ),
        key=lambda member: (member["cell_id"], member["tier"]),
    )
    selected_member_descriptor: dict[str, Any] | None = None
    if physical_members:
        selected_member_document = {
            "schema": "banana-smasher-mixed-backpack-selected-members-v1",
            "status": "PRE_REPAIR_SELECTED",
            "basis_sha256": basis,
            "assignment_sha256": assignment_sha,
            "members_expected": len(selected_members),
            "members_complete": len(selected_members),
            "gaps": 0,
            "duplicates": 0,
            "members": selected_members,
        }
        selected_member_raw = _canonical_json(selected_member_document)
        selected_member_path = output_root / "SELECTED_PHYSICAL_MEMBERS.json"
        _write_once(selected_member_path, selected_member_raw)
        selected_member_descriptor = {
            "path": "SELECTED_PHYSICAL_MEMBERS.json",
            "sha256": _sha(selected_member_raw),
            "bytes": len(selected_member_raw),
            "members": len(selected_members),
        }
    identity = {
        "schema": "banana-smasher-mixed-backpack-identity-v1",
        "status": "PRE_REPAIR_SOLVED",
        "basis_sha256": basis,
        "assignment_sha256": assignment_sha,
        "assignment": assignment,
        "allowed_tiers": list(tiers),
        "fallback_tier": fallback,
        "coverage": coverage,
        "composition": {
            "kind": "mixed-per-layer-per-expert",
            "layers": [
                {"layer": layer, "tiers": dict(sorted(layers[layer].items()))}
                for layer in sorted(layers)
            ],
        },
    }
    if selected_member_descriptor is not None:
        identity["selected_physical_members"] = selected_member_descriptor
    whole_bytes = fixed_bytes + solved["assigned_bytes"]
    byte_accounting = {
        "fixed_nonexpert_bytes": fixed_bytes,
        "candidate_payload_bytes": solved["assigned_bytes"],
        "whole_model_bytes": whole_bytes,
        "target_whole_model_bytes": whole_target,
        "slack_bytes": whole_target - whole_bytes,
    }
    assignment_document = {
        "schema": "banana-smasher-mixed-backpack-assignment-v1",
        "status": solved["status"],
        "basis_sha256": basis,
        "assignment_sha256": assignment_sha,
        "assignments": solved["assignments"],
        "materialization_assignments": [
            {"cell_id": projection_cell, "tier": assignment[cell]}
            for cell in cells
            for projection_cell in projections_by_cell[cell]
        ],
        "activated_artifacts": solved["activated_artifacts"],
        "byte_accounting": byte_accounting,
        "prediction_by_class": solved["prediction_by_class"],
        "objective": solved["objective"],
        "solver": solved["solver"],
    }
    assignment_raw = _canonical_json(assignment_document)
    identity_raw = _canonical_json(identity)
    _write_once(output_root / "ASSIGNMENT.json", assignment_raw)
    _write_once(output_root / "identity.json", identity_raw)
    receipt = {
        "schema": "banana-smasher-mixed-backpack-solve-receipt-v1",
        "status": "PASS_PRE_REPAIR_MIX_SOLVED",
        "basis_sha256": basis,
        "config": {
            "path": str(config_path),
            "sha256": _sha(config_raw),
            "bytes": len(config_raw),
        },
        "dimensions": {"sources": dimension_sources},
        "sources": {"admitted": len(dimension_sources), "pending": 0},
        "assignment": {
            "path": str(output_root / "ASSIGNMENT.json"),
            "sha256": _sha(assignment_raw),
            "bytes": len(assignment_raw),
        },
        "identity": {
            "path": str(output_root / "identity.json"),
            "sha256": _sha(identity_raw),
            "bytes": len(identity_raw),
        },
        "coverage": coverage,
        "byte_accounting": byte_accounting,
    }
    if selected_member_descriptor is not None:
        receipt["selected_physical_members"] = selected_member_descriptor
    receipt_raw = _canonical_json(receipt)
    _write_once(output_root / "RECEIPT.json", receipt_raw)
    return {
        **receipt,
        "receipt": {
            "path": str(output_root / "RECEIPT.json"),
            "sha256": _sha(receipt_raw),
            "bytes": len(receipt_raw),
        },
    }
