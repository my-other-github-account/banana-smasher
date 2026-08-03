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
SOLVED_CELL_RE = re.compile(
    r"L(?P<layer>[0-9]{3})\.E(?P<expert>[0-9]{3})\.(?P<projection>P2|P13)"
)


class DynamicDimensionsError(ValueError):
    """Raised when explicit dynamic Backpack dimensions are incomplete or inconsistent."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_field(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise DynamicDimensionsError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _read_json(path: Path, label: str) -> tuple[Any, bytes]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise DynamicDimensionsError(f"duplicate JSON key {key!r} in {label} at {path}")
            value[key] = item
        return value

    try:
        raw = path.read_bytes()
        return json.loads(raw, object_pairs_hook=reject_duplicate_keys), raw
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


def build_solved_dimension_sidecars(
    *,
    handoff: str | Path,
    handoff_sha256: str,
    basis_sha256: str,
    layers: list[int],
    output_dir: str | Path,
    authority_expectations: str | Path | None = None,
) -> dict[str, Any]:
    """Publish authenticated solve bindings and explicit blockers without inference."""

    basis = _sha_field(basis_sha256, "basis_sha256")
    expected_handoff_sha = _sha_field(handoff_sha256, "handoff_sha256")
    handoff_path = Path(handoff).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    handoff_value, handoff_raw = _read_json(handoff_path, "producer handoff")
    if _sha(handoff_raw) != expected_handoff_sha:
        raise DynamicDimensionsError("handoff SHA-256 mismatch")
    if not isinstance(handoff_value, dict):
        raise DynamicDimensionsError("producer handoff must be a JSON object")
    if handoff_value.get("basis_sha256") != basis:
        raise DynamicDimensionsError("producer handoff basis mismatch")
    if handoff_value.get("status") != "PASS_ADMISSION_READY_NOT_QUARANTINED":
        raise DynamicDimensionsError("producer handoff is not admission-ready")
    if not layers or any(isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 for layer in layers):
        raise DynamicDimensionsError("layers must contain non-negative integers")
    expected_layers = sorted(set(layers))
    handoff_layers = handoff_value.get("layers")
    if (
        expected_layers != layers
        or not isinstance(handoff_layers, list)
        or any(isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 for layer in handoff_layers)
        or handoff_layers != expected_layers
    ):
        raise DynamicDimensionsError("producer handoff layer mismatch")

    tier = handoff_value.get("tier")
    if not isinstance(tier, str) or not tier:
        raise DynamicDimensionsError("producer handoff tier is missing")
    layer_rows = handoff_value.get("layer_rows")
    if not isinstance(layer_rows, list) or len(layer_rows) != len(expected_layers):
        raise DynamicDimensionsError("producer handoff layer rows are incomplete")
    layer_row_ids = [row.get("layer") if isinstance(row, dict) else None for row in layer_rows]
    if (
        any(isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 for layer in layer_row_ids)
        or layer_row_ids != expected_layers
    ):
        raise DynamicDimensionsError("producer handoff layer rows must cover requested layers exactly")

    candidates: list[dict[str, Any]] = []
    physical: list[dict[str, Any]] = []
    routing: list[dict[str, Any]] = []
    source_members: list[dict[str, Any]] = []
    for layer_entry in layer_rows:
        if (
            not isinstance(layer_entry, dict)
            or isinstance(layer_entry.get("layer"), bool)
            or not isinstance(layer_entry.get("layer"), int)
            or layer_entry.get("layer") not in expected_layers
        ):
            raise DynamicDimensionsError("invalid producer handoff layer row")
        layer = layer_entry["layer"]
        members = layer_entry.get("members")
        if not isinstance(members, dict):
            raise DynamicDimensionsError(f"layer {layer} members are missing")
        loaded: dict[str, tuple[Any, bytes, Path, str]] = {}
        for name in ("OBJECTIVE", "PROFILE_ROWS"):
            member = members.get(name)
            if not isinstance(member, dict):
                raise DynamicDimensionsError(f"layer {layer} {name} member is missing")
            member_path_value = member.get("path")
            if not isinstance(member_path_value, str) or not member_path_value:
                raise DynamicDimensionsError(
                    f"layer {layer} {name} path must be a non-empty string"
                )
            member_path = Path(member_path_value).expanduser().resolve()
            member_sha = _sha_field(member.get("sha256"), f"layer {layer} {name} sha256")
            if name == "PROFILE_ROWS":
                value, raw = _read_jsonl(member_path, f"layer {layer} profile rows")
            else:
                value, raw = _read_json(member_path, f"layer {layer} objective")
            if _sha(raw) != member_sha:
                raise DynamicDimensionsError(f"layer {layer} {name} SHA-256 mismatch")
            loaded[name] = value, raw, member_path, member_sha
            source_members.append(
                {
                    "layer": layer,
                    "role": name,
                    "path": str(member_path),
                    "sha256": member_sha,
                    "bytes": len(raw),
                }
            )

        objective = loaded["OBJECTIVE"][0]
        assignments = objective.get("assignment") if isinstance(objective, dict) else None
        if not isinstance(assignments, dict) or not assignments:
            raise DynamicDimensionsError(f"layer {layer} objective assignment is missing")
        profile_by_expert: dict[int, dict[str, Any]] = {}
        for row in loaded["PROFILE_ROWS"][0]:
            row_layer = row.get("layer")
            expert = row.get("expert")
            if (
                isinstance(row_layer, bool)
                or not isinstance(row_layer, int)
                or row_layer != layer
                or isinstance(expert, bool)
                or not isinstance(expert, int)
                or expert in profile_by_expert
            ):
                raise DynamicDimensionsError(f"layer {layer} profile identities are invalid")
            routed_rows = row.get("routed_rows")
            if isinstance(routed_rows, bool) or not isinstance(routed_rows, int) or routed_rows < 0:
                raise DynamicDimensionsError(f"layer {layer} expert {expert} routed_rows is invalid")
            profile_by_expert[expert] = row

        observed_cells: set[tuple[int, str]] = set()
        for cell, assignment in assignments.items():
            if not isinstance(cell, str) or not isinstance(assignment, dict):
                raise DynamicDimensionsError(f"layer {layer} objective assignment row is invalid")
            match = SOLVED_CELL_RE.fullmatch(cell)
            if match is None:
                raise DynamicDimensionsError(f"invalid objective cell identity {cell!r}")
            cell_layer = int(match.group("layer"))
            expert = int(match.group("expert"))
            projection_token = match.group("projection")
            projection = {"P2": "down", "P13": "13"}.get(projection_token)
            if cell_layer != layer or projection is None or expert not in profile_by_expert:
                raise DynamicDimensionsError(f"objective cell identity is inconsistent: {cell}")
            observed_cells.add((expert, projection_token))
            count = assignment.get("codeword_assignment_count")
            dtype = assignment.get("codeword_assignment_dtype")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0 or dtype != "int16-le":
                raise DynamicDimensionsError(f"objective assignment wire layout is invalid: {cell}")
            assignment_sha = _sha_field(
                assignment.get("codeword_assignment_sha256"),
                f"{cell} codeword_assignment_sha256",
            )
            if assignment.get("tier") != tier:
                raise DynamicDimensionsError(f"objective assignment tier mismatch: {cell}")
            candidate_id = f"{tier}:L{layer:03d}:E{expert:03d}:{projection}"
            identity = {
                "schema": "banana-smasher-dynamic-backpack-candidate-identity-v1",
                "basis_sha256": basis,
                "candidate_id": candidate_id,
                "layer": layer,
                "expert": expert,
                "projection": projection,
                "tier": tier,
                "source_cell": cell,
            }
            candidates.append(identity)
            physical.append(
                {
                    **identity,
                    "schema": "banana-smasher-dynamic-backpack-packed-wire-bytes-v1",
                    "packed_wire_bytes": count * 2,
                    "codeword_assignment_count": count,
                    "codeword_assignment_dtype": dtype,
                    "codeword_assignment_sha256": assignment_sha,
                    "authority_path": str(loaded["OBJECTIVE"][2]),
                    "authority_sha256": loaded["OBJECTIVE"][3],
                    "status": "PASS_EXACT_INTEGER_PACKED_WIRE_BYTES",
                }
            )
            routing.append(
                {
                    **identity,
                    "schema": "banana-smasher-dynamic-backpack-expert-routing-importance-v1",
                    "routing_importance": profile_by_expert[expert]["routed_rows"],
                    "routing_importance_unit": "routed_rows",
                    "authority_path": str(loaded["PROFILE_ROWS"][2]),
                    "authority_sha256": loaded["PROFILE_ROWS"][3],
                    "status": "PASS_EXPLICIT_EXPERT_ROUTING_IMPORTANCE",
                }
            )
        expected_cells = {
            (expert, projection)
            for expert in profile_by_expert
            for projection in ("P2", "P13")
        }
        if observed_cells != expected_cells:
            missing = sorted(expected_cells - observed_cells)
            extra = sorted(observed_cells - expected_cells)
            raise DynamicDimensionsError(
                f"layer {layer} objective assignment coverage mismatch: "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )

    candidates.sort(key=lambda row: row["candidate_id"])
    physical.sort(key=lambda row: row["candidate_id"])
    routing.sort(key=lambda row: row["candidate_id"])
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise DynamicDimensionsError("duplicate candidate identities")

    sidecar_values = {
        "CANDIDATE_IDENTITIES.jsonl": candidates,
        "PACKED_WIRE_BYTES.jsonl": physical,
        "EXPERT_ROUTING_IMPORTANCE.jsonl": routing,
    }
    sidecars: list[dict[str, Any]] = []
    publications: list[tuple[Path, bytes]] = []
    for name, rows in sidecar_values.items():
        raw = _canonical_jsonl(rows)
        path = output_root / name
        publications.append((path, raw))
        sidecars.append(
            {"path": str(path), "sha256": _sha(raw), "bytes": len(raw), "rows": len(rows)}
        )

    blocked_groups = [
        "projection_correction",
        "projection_weight",
        "six_class_ceilings",
        "six_class_predictions",
    ]
    expectation_source = None
    if authority_expectations is None:
        blocked_details = {
            name: {
                "required_key": name,
                "expected_authority_path": None,
                "expected_authority_sha256": None,
                "searched_sources": [],
                "reason": "no explicit authenticated authority supplied; inference/default/substitution forbidden",
            }
            for name in blocked_groups
        }
    else:
        expectations_path = Path(authority_expectations).expanduser().resolve()
        expectations, expectations_raw = _read_json(expectations_path, "authority expectations")
        if (
            not isinstance(expectations, dict)
            or expectations.get("schema")
            != "banana-smasher-dynamic-backpack-authority-expectations-v1"
            or expectations.get("basis_sha256") != basis
        ):
            raise DynamicDimensionsError("authority expectations schema/basis mismatch")
        blocked_details = expectations.get("blocked_groups")
        if not isinstance(blocked_details, dict) or list(blocked_details) != blocked_groups:
            raise DynamicDimensionsError(
                "authority expectations must cover the exact blocked groups in canonical order"
            )
        for group, detail in blocked_details.items():
            if not isinstance(detail, dict) or detail.get("required_key") != group:
                raise DynamicDimensionsError(f"invalid authority expectation for {group}")
            expected_path = detail.get("expected_authority_path")
            if not isinstance(expected_path, str) or not expected_path:
                raise DynamicDimensionsError(f"expected authority path must be explicit for {group}")
            expected_authority_sha = detail.get("expected_authority_sha256")
            if expected_authority_sha is not None:
                _sha_field(expected_authority_sha, f"{group} expected_authority_sha256")
            searched = detail.get("searched_sources")
            if not isinstance(searched, list) or not searched:
                raise DynamicDimensionsError(f"searched_sources must be explicit for {group}")
            for index, source in enumerate(searched):
                if (
                    not isinstance(source, dict)
                    or not isinstance(source.get("path"), str)
                    or not source["path"]
                ):
                    raise DynamicDimensionsError(f"invalid searched source {index} for {group}")
                _sha_field(source.get("sha256"), f"{group} searched source {index} sha256")
        expectation_source = {
            "path": str(expectations_path),
            "sha256": _sha(expectations_raw),
            "bytes": len(expectations_raw),
        }
    blocker_value = {
        "schema": "banana-smasher-dynamic-backpack-authority-blockers-v1",
        "status": "PASS_EXACT_AUTHORITY_BLOCKERS_ALLOCATION_FORBIDDEN",
        "basis_sha256": basis,
        "candidate_count": len(candidates),
        "blocked_groups": blocked_details,
        "authority_expectations": expectation_source,
        "qtip2_substitution": False,
        "fixed_quota_substitution": False,
        "allocation_finalized": False,
    }
    blocker_raw = _canonical_json(blocker_value)
    blocker_path = output_root / "AUTHORITATIVE_PRODUCER_BLOCKERS.json"
    manifest_value = {
        "schema": "banana-smasher-dynamic-backpack-solved-sidecar-manifest-v1",
        "status": "PASS_NON_INFERRED_BINDINGS_PLUS_EXACT_BLOCKERS",
        "basis_sha256": basis,
        "layers": expected_layers,
        "tier": tier,
        "candidate_count": len(candidates),
        "bound_groups": [
            "candidate_identity",
            "expert_routing_importance",
            "packed_wire_bytes",
        ],
        "blocked_groups": blocked_groups,
        "packed_wire_bytes": sum(row["packed_wire_bytes"] for row in physical),
        "allocation_finalized": False,
        "solve_or_replay_invoked": False,
        "qtip2_substitution": False,
        "fixed_quota_substitution": False,
        "source_handoff": {
            "path": str(handoff_path),
            "sha256": expected_handoff_sha,
            "bytes": len(handoff_raw),
        },
        "source_members": source_members,
        "sidecars": sidecars,
        "blocker": {
            "path": str(blocker_path),
            "sha256": _sha(blocker_raw),
            "bytes": len(blocker_raw),
        },
    }
    manifest_raw = _canonical_json(manifest_value)
    manifest_path = output_root / "DIMENSION_SIDECAR_MANIFEST.json"
    publications.extend(((blocker_path, blocker_raw), (manifest_path, manifest_raw)))
    for path, payload in publications:
        if path.exists() and path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace different sealed output: {path}")
    for path, payload in publications:
        _write_once(path, payload)
    return {
        "status": manifest_value["status"],
        "command": "backpack-solved-sidecars",
        "basis_sha256": basis,
        "candidate_count": len(candidates),
        "packed_wire_bytes": manifest_value["packed_wire_bytes"],
        "allocation_finalized": False,
        "manifest": {
            "path": str(manifest_path),
            "sha256": _sha(manifest_raw),
            "bytes": len(manifest_raw),
        },
    }
