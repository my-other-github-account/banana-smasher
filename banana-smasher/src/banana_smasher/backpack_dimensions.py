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
SENSITIVITY_ROW_SCHEMA = "banana-smasher-sensitivity-row-v1"
SENSITIVITY_TIER_NAMES = {"Q2": "qtip2", "QTIP3_V7": "qtip3"}


class DynamicDimensionsError(ValueError):
    """Raised when explicit dynamic Backpack dimensions are incomplete or inconsistent."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_field(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise DynamicDimensionsError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _read_json(path: Path, label: str) -> tuple[Any, bytes]:
    try:
        raw = path.read_bytes()
        return json.loads(raw), raw
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DynamicDimensionsError(
            f"cannot read valid {label} at {path}: {exc}"
        ) from exc


def _read_jsonl(path: Path, label: str) -> tuple[list[dict[str, Any]], bytes]:
    try:
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DynamicDimensionsError(
            f"cannot read valid {label} at {path}: {exc}"
        ) from exc
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise DynamicDimensionsError(f"{label} must contain non-empty JSON-object rows")
    return rows, raw


def _finite(
    value: object, label: str, *, positive: bool = False, nonnegative: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DynamicDimensionsError(f"{label} must be numeric")
    result = float(value)
    if (
        not math.isfinite(result)
        or (positive and result <= 0.0)
        or (nonnegative and result < 0.0)
    ):
        qualifier = (
            "positive finite"
            if positive
            else "non-negative finite"
            if nonnegative
            else "finite"
        )
        raise DynamicDimensionsError(f"{label} must be {qualifier}")
    return result


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
    ).encode()


def _canonical_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def _write_once(path: Path, payload: bytes) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(
                f"refusing to replace different sealed output: {path}"
            )
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
                raise FileExistsError(
                    f"refusing to replace different sealed output: {path}"
                )
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
            raise DynamicDimensionsError(
                f"{row_label} layer/expert geometry is invalid"
            )
        source_tier = row.get("tier")
        if not isinstance(source_tier, str):
            raise DynamicDimensionsError(f"{row_label} tier is unsupported")
        tier = SENSITIVITY_TIER_NAMES.get(source_tier)
        if tier is None:
            raise DynamicDimensionsError(f"{row_label} tier is unsupported")
        physical_bytes = row.get("bytes")
        if (
            isinstance(physical_bytes, bool)
            or not isinstance(physical_bytes, int)
            or physical_bytes < 0
        ):
            raise DynamicDimensionsError(f"{row_label} bytes is invalid")
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
                    "activation_artifacts": [],
                    "sensitivity_authority": authority,
                }
            )
    return normalized


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
    if (
        not isinstance(ceilings_value, dict)
        or ceilings_value.get("schema")
        != "banana-smasher-dynamic-backpack-class-ceilings-v1"
    ):
        raise DynamicDimensionsError(
            "class ceilings must use banana-smasher-dynamic-backpack-class-ceilings-v1"
        )
    if ceilings_value.get("basis_sha256") != basis or ceilings_value.get(
        "status"
    ) not in {"PASS", "SEALED"}:
        raise DynamicDimensionsError("class ceilings basis/status mismatch")
    raw_ceilings = ceilings_value.get("six_class_ceilings")
    if not isinstance(raw_ceilings, dict) or set(raw_ceilings) != set(CLASSES):
        raise DynamicDimensionsError(
            "six_class_ceilings must explicitly cover the six canonical classes"
        )
    ceilings = {
        name: _finite(raw_ceilings[name], f"ceiling {name}", nonnegative=True)
        for name in CLASSES
    }

    candidates: dict[str, dict[str, Any]] = {}
    candidate_identities: dict[str, tuple[int, int, str, str]] = {}
    physical_bindings: dict[str, int] = {}
    for index, row in enumerate(ledger_rows):
        schema = row.get("schema")
        if schema in CANDIDATE_LEDGER_SCHEMAS:
            if row.get("basis_sha256") != basis:
                raise DynamicDimensionsError(
                    f"candidate ledger row {index} basis mismatch"
                )
            candidate_id = row.get("candidate_id")
            if (
                not isinstance(candidate_id, str)
                or not candidate_id
                or candidate_id in candidates
            ):
                raise DynamicDimensionsError(
                    f"candidate ledger row {index} candidate_id must be unique"
                )
            candidates[candidate_id] = row
            candidate_identities[candidate_id] = _identity(
                row, f"candidate {candidate_id}"
            )
            continue
        if schema != DIMENSION_BINDING_SCHEMA:
            raise DynamicDimensionsError(
                f"candidate ledger row {index} schema mismatch"
            )
        if row.get("basis_sha256") != basis:
            raise DynamicDimensionsError(
                f"dimension binding row {index} basis mismatch"
            )
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id not in candidates:
            raise DynamicDimensionsError(
                f"dimension binding row {index} names unknown candidate {candidate_id!r}"
            )
        if (
            _identity(row, f"dimension binding {candidate_id}")
            != candidate_identities[candidate_id]
        ):
            raise DynamicDimensionsError(
                f"dimension binding identity mismatch for {candidate_id}"
            )
        physical_bytes = row.get("physical_bytes")
        if (
            isinstance(physical_bytes, bool)
            or not isinstance(physical_bytes, int)
            or physical_bytes < 0
        ):
            raise DynamicDimensionsError(
                f"dimension binding physical_bytes must be a non-negative integer for {candidate_id}"
            )
        source_sidecar = row.get("source_physical_sidecar_sha256")
        _sha_field(
            source_sidecar,
            f"dimension binding {candidate_id} source_physical_sidecar_sha256",
        )
        candidate_physical_bytes = candidates[candidate_id].get("physical_bytes")
        if (
            candidate_physical_bytes is not None
            and candidate_physical_bytes != physical_bytes
        ):
            raise DynamicDimensionsError(
                f"dimension binding physical_bytes conflict for {candidate_id}"
            )
        prior_physical_bytes = physical_bindings.get(candidate_id)
        if prior_physical_bytes is not None and prior_physical_bytes != physical_bytes:
            raise DynamicDimensionsError(
                f"conflicting dimension bindings for {candidate_id}"
            )
        physical_bindings[candidate_id] = physical_bytes

    explicit: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(dimension_rows):
        if (
            row.get("schema")
            != "banana-smasher-dynamic-backpack-explicit-dimension-row-v1"
        ):
            raise DynamicDimensionsError(
                f"dimension row {index} schema mismatch; aggregate inputs are forbidden"
            )
        if row.get("basis_sha256") != basis:
            raise DynamicDimensionsError(f"dimension row {index} basis mismatch")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in explicit:
            raise DynamicDimensionsError(
                f"dimension row {index} candidate_id must be unique"
            )
        if candidate_id not in candidates:
            raise DynamicDimensionsError(
                f"dimension row {index} names unknown candidate {candidate_id!r}"
            )
        if (
            _identity(row, f"dimension {candidate_id}")
            != candidate_identities[candidate_id]
        ):
            raise DynamicDimensionsError(
                f"dimension identity mismatch for {candidate_id}"
            )
        explicit[candidate_id] = row

    missing = sorted(set(candidates) - set(explicit))
    if missing:
        raise DynamicDimensionsError(
            f"missing explicit dimensions for {len(missing)} candidates; first={missing[0]!r}; allocation forbidden"
        )

    completed: list[dict[str, Any]] = []
    for candidate_id in sorted(candidates):
        candidate, dimension = candidates[candidate_id], explicit[candidate_id]
        physical_bytes = physical_bindings.get(
            candidate_id, candidate.get("physical_bytes")
        )
        if (
            isinstance(physical_bytes, bool)
            or not isinstance(physical_bytes, int)
            or physical_bytes < 0
        ):
            raise DynamicDimensionsError(
                f"physical_bytes must be a non-negative integer for {candidate_id}"
            )
        if dimension.get("physical_bytes") != physical_bytes:
            raise DynamicDimensionsError(f"physical_bytes mismatch for {candidate_id}")
        predictions_raw = dimension.get("six_class_predictions")
        if not isinstance(predictions_raw, dict) or set(predictions_raw) != set(
            CLASSES
        ):
            raise DynamicDimensionsError(
                f"six_class_predictions must explicitly cover six classes for {candidate_id}; aggregate inference forbidden"
            )
        predictions = {
            name: _finite(
                predictions_raw[name],
                f"{candidate_id} prediction {name}",
                nonnegative=True,
            )
            for name in CLASSES
        }
        routing_importance = _finite(
            dimension.get("routing_importance"),
            f"{candidate_id} routing_importance",
            positive=True,
        )
        source_importance = candidate.get("source_class_features", {}).get(
            "routing_importance"
        )
        if source_importance is not None and not math.isclose(
            routing_importance,
            _finite(
                source_importance,
                f"{candidate_id} source routing_importance",
                positive=True,
            ),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise DynamicDimensionsError(
                f"routing_importance mismatch for {candidate_id}"
            )
        projection_weight = _finite(
            dimension.get("projection_weight"),
            f"{candidate_id} projection_weight",
            positive=True,
        )
        projection_correction = _finite(
            dimension.get("projection_correction"),
            f"{candidate_id} projection_correction",
        )
        authority = dimension.get("authority")
        if not isinstance(authority, dict):
            raise DynamicDimensionsError(
                f"authority must be explicit for {candidate_id}"
            )
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
            "ledger": {
                "path": str(ledger_path),
                "sha256": _sha(ledger_raw),
                "bytes": len(ledger_raw),
            },
            "dimensions": {
                "path": str(dimensions_path),
                "sha256": _sha(dimension_raw),
                "bytes": len(dimension_raw),
            },
            "class_ceilings": {
                "path": str(ceilings_path),
                "sha256": _sha(ceilings_raw),
                "bytes": len(ceilings_raw),
            },
        },
        "output": {
            "path": str(output_path),
            "sha256": _sha(output_raw),
            "bytes": len(output_raw),
        },
    }
    receipt_raw = _canonical_json(receipt_value)
    _write_once(output_path, output_raw)
    _write_once(receipt_path, receipt_raw)
    if (
        output_path.read_bytes() != output_raw
        or receipt_path.read_bytes() != receipt_raw
    ):
        raise RuntimeError("sealed dimension outputs changed during publication")
    return {
        "status": receipt_value["status"],
        "command": "backpack-dimensions",
        "basis_sha256": basis,
        "candidate_count": len(completed),
        "allocation_eligible": True,
        "output": receipt_value["output"],
        "receipt": {
            "path": str(receipt_path),
            "sha256": _sha(receipt_raw),
            "bytes": len(receipt_raw),
        },
    }


def _resolve_mixed_dimension_sources(
    config_path: Path,
    descriptor: object,
    basis: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
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
            locator, locator_raw = _read_json(
                locator_path, f"dimensions locator {index}"
            )
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
                if not isinstance(manifest_descriptor, dict) or set(
                    manifest_descriptor
                ) != {"path", "sha256"}:
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
                members = (
                    manifest.get("members") if isinstance(manifest, dict) else None
                )
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
    return rows, admitted, pending


def preflight_mixed_backpack_config(config: str | Path) -> dict[str, Any]:
    """Admit available dimension shards and report exact pending coverage."""

    config_path = Path(config).expanduser().resolve()
    value, config_raw = _read_json(config_path, "mixed Backpack config")
    if (
        not isinstance(value, dict)
        or value.get("schema") != "banana-smasher-mixed-backpack-config-v1"
    ):
        raise DynamicDimensionsError(
            "mixed Backpack config must use banana-smasher-mixed-backpack-config-v1"
        )
    basis = _sha_field(value.get("basis_sha256"), "basis_sha256")
    tiers = value.get("allowed_tiers")
    if (
        not isinstance(tiers, list)
        or not tiers
        or any(not isinstance(tier, str) or not tier for tier in tiers)
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
        or any(
            isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
            for layer in layers
        )
        or isinstance(experts, bool)
        or not isinstance(experts, int)
        or experts <= 0
        or not isinstance(projections, list)
        or not projections
        or len(projections) != len(set(projections))
        or any(projection not in {"down", "fused13"} for projection in projections)
    ):
        raise DynamicDimensionsError("topology geometry is invalid")

    rows, admitted, pending = _resolve_mixed_dimension_sources(
        config_path, value.get("dimensions"), basis
    )
    available: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        if row.get("basis_sha256") != basis:
            raise DynamicDimensionsError(f"mixed dimension row {index} basis mismatch")
        if (
            row.get("allocation_eligible") is not True
            or row.get("status") != "ADMITTED_COMPLETE_ALLOCATION_ELIGIBLE"
        ):
            raise DynamicDimensionsError(
                f"mixed dimension row {index} is not allocation eligible"
            )
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
            "available_projection_cells": sum(
                (cell, tier) in available for cell in expected
            ),
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
        "config": {
            "path": str(config_path),
            "sha256": _sha(config_raw),
            "bytes": len(config_raw),
        },
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
    if (
        not isinstance(value, dict)
        or value.get("schema") != "banana-smasher-mixed-backpack-config-v1"
    ):
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
        raise DynamicDimensionsError(
            f"mixed Backpack config has unknown fields: {unknown}"
        )
    basis = _sha_field(value.get("basis_sha256"), "basis_sha256")
    tiers = value.get("allowed_tiers")
    if (
        not isinstance(tiers, list)
        or not tiers
        or len(tiers) != len(set(tiers))
        or any(not isinstance(tier, str) or not tier for tier in tiers)
    ):
        raise DynamicDimensionsError(
            "allowed_tiers must be a non-empty unique string array"
        )
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

    rows, dimension_sources, pending_locators = _resolve_mixed_dimension_sources(
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
            raise DynamicDimensionsError(
                "class_weights must cover the six canonical classes"
            )
        class_weights = {
            name: _finite(raw_weights[name], f"class_weights.{name}", nonnegative=True)
            for name in CLASSES
        }

    projection_inventory: dict[tuple[str, str], dict[str, Any]] = {}
    projection_geometry: dict[str, tuple[int, int, str]] = {}
    for index, row in enumerate(rows):
        if row.get("basis_sha256") != basis:
            raise DynamicDimensionsError(f"mixed dimension row {index} basis mismatch")
        if (
            row.get("allocation_eligible") is not True
            or row.get("status") != "ADMITTED_COMPLETE_ALLOCATION_ELIGIBLE"
        ):
            raise DynamicDimensionsError(
                f"mixed dimension row {index} is not allocation eligible"
            )
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
            raise DynamicDimensionsError(
                f"mixed option {key!r} lacks six-class predictions"
            )
        physical_bytes = row.get("physical_bytes")
        if (
            isinstance(physical_bytes, bool)
            or not isinstance(physical_bytes, int)
            or physical_bytes < 0
        ):
            raise DynamicDimensionsError(
                f"mixed option {key!r} physical_bytes is invalid"
            )
        activations = row.get("activation_artifacts", [])
        if not isinstance(activations, list) or not all(
            isinstance(item, dict) for item in activations
        ):
            raise DynamicDimensionsError(
                f"mixed option {key!r} activation_artifacts is invalid"
            )
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
        cell
        for cell in projection_cells
        if (cell, fallback) not in projection_inventory
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
    output_root = Path(output).expanduser().resolve()
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
