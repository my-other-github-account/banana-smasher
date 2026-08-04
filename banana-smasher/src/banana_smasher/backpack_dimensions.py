from __future__ import annotations

import hashlib
import json
import math
import os
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
