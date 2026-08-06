from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class KnapsackValidationError(ValueError):
    """Raised when manifest-bound knapsack inputs are incomplete or inconsistent."""


_MAX_LAYER_RANGE_SPAN = 1_000_000


@dataclass(frozen=True)
class _Source:
    path: Path
    relative_path: str
    sha256: str
    byte_count: int
    producer_command: str
    remedy_command: str
    payload: dict[str, Any]


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise KnapsackValidationError(f"cannot read {label} {path}: {exc}") from exc
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KnapsackValidationError(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise KnapsackValidationError(f"{label} {path} must contain a JSON object")
    return value, payload


def _local_path(root: Path, raw_path: object, *, label: str) -> tuple[Path, str]:
    if not isinstance(raw_path, str) or not raw_path:
        raise KnapsackValidationError(f"{label} path must be a non-empty string")
    relative = Path(raw_path)
    if relative.is_absolute():
        raise KnapsackValidationError(f"{label} path must be local to run root: {raw_path}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise KnapsackValidationError(f"{label} path escapes run root: {raw_path}") from exc
    return resolved, relative.as_posix()


def _sha_field(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise KnapsackValidationError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _producer(descriptor: dict[str, Any], *, fallback: str) -> str:
    value = descriptor.get("producer_command", fallback)
    if not isinstance(value, str) or not value.strip():
        raise KnapsackValidationError("producer_command must be a non-empty string")
    return value


def _preflight_source(
    *,
    root: Path,
    descriptor: object,
    label: str,
    missing_message: str,
    fallback_producer: str,
) -> _Source:
    if not isinstance(descriptor, dict):
        raise KnapsackValidationError(f"{label} descriptor must be an object")
    producer_command = _producer(descriptor, fallback=fallback_producer)
    path, relative_path = _local_path(root, descriptor.get("path"), label=label)
    expected_sha = _sha_field(descriptor.get("sha256"), label=f"{label} sha256")
    if not path.is_file():
        raise KnapsackValidationError(
            f"{missing_message}: {path}; required producer: {fallback_producer}"
        )
    payload = path.read_bytes()
    actual_sha = _sha256(payload)
    if actual_sha != expected_sha:
        raise KnapsackValidationError(
            f"{label} SHA-256 mismatch: {path}; expected {expected_sha}, got {actual_sha}; "
            f"required producer: {fallback_producer}"
        )
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KnapsackValidationError(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise KnapsackValidationError(f"{label} {path} must contain a JSON object")
    return _Source(
        path=path,
        relative_path=relative_path,
        sha256=actual_sha,
        byte_count=len(payload),
        producer_command=producer_command,
        remedy_command=fallback_producer,
        payload=value,
    )


def _intended_tiers(manifest: dict[str, Any]) -> list[str]:
    value = manifest.get("intended_tiers")
    if not isinstance(value, list) or not value:
        raise KnapsackValidationError("run manifest intended_tiers must be a non-empty list")
    tiers: list[str] = []
    for index, tier in enumerate(value):
        if not isinstance(tier, str) or not tier:
            raise KnapsackValidationError(
                f"run manifest intended_tiers[{index}] must be a non-empty string"
            )
        if tier in tiers:
            raise KnapsackValidationError(f"duplicate intended tier {tier!r}")
        tiers.append(tier)
    return tiers


def _basis(manifest: dict[str, Any]) -> str:
    candidates: list[str] = []
    for field in ("intended_basis_sha256", "basis_sha256"):
        if field in manifest:
            candidates.append(_sha_field(manifest[field], label=f"run manifest {field}"))
    intended = manifest.get("intended_basis")
    if isinstance(intended, dict) and "model_index_sha256" in intended:
        candidates.append(
            _sha_field(
                intended["model_index_sha256"],
                label="run manifest intended_basis.model_index_sha256",
            )
        )
    if not candidates:
        raise KnapsackValidationError(
            "run manifest must declare intended_basis_sha256, basis_sha256, or "
            "intended_basis.model_index_sha256"
        )
    if len(set(candidates)) != 1:
        raise KnapsackValidationError(
            f"run manifest basis mismatch within declarations: {sorted(set(candidates))}"
        )
    return candidates[0]


def _source_basis(source: _Source, *, label: str, intended_basis: str) -> None:
    actual = source.payload.get("basis_sha256")
    if actual is None and isinstance(source.payload.get("intended_basis"), dict):
        actual = source.payload["intended_basis"].get("model_index_sha256")
    actual_sha = _sha_field(actual, label=f"{label} basis_sha256")
    if actual_sha != intended_basis:
        raise KnapsackValidationError(
            f"{label} basis mismatch: expected {intended_basis}, got {actual_sha} at {source.path}"
        )


def _anchor_cells(source: _Source, *, tier: str, intended_basis: str) -> dict[str, int]:
    value = source.payload
    if value.get("tier") != tier:
        raise KnapsackValidationError(
            f"anchor manifest tier mismatch at {source.path}: expected {tier!r}, "
            f"got {value.get('tier')!r}"
        )
    status = value.get("status")
    if not isinstance(status, str) or not (status == "SEALED" or status.startswith("PASS")):
        raise KnapsackValidationError(
            f"anchor manifest for tier {tier!r} is not sealed/PASS at {source.path}: {status!r}"
        )
    _source_basis(source, label=f"anchor manifest for tier {tier!r}", intended_basis=intended_basis)
    rows = value.get("cells")
    if not isinstance(rows, list) or not rows:
        raise KnapsackValidationError(
            f"anchor manifest for tier {tier!r} must contain a non-empty cells list"
        )
    cells: dict[str, int] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise KnapsackValidationError(f"anchor {tier!r} cells[{index}] must be an object")
        cell_id = row.get("cell_id")
        byte_count = row.get("bytes")
        if not isinstance(cell_id, str) or not cell_id:
            raise KnapsackValidationError(
                f"anchor {tier!r} cells[{index}].cell_id must be a non-empty string"
            )
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise KnapsackValidationError(
                f"anchor {tier!r} cell {cell_id!r} bytes must be a non-negative integer"
            )
        if cell_id in cells:
            raise KnapsackValidationError(
                f"duplicate cell {cell_id!r} in anchor manifest for tier {tier!r}"
            )
        cells[cell_id] = byte_count
    return cells


def _damage_values(
    source: _Source,
    *,
    cells: list[str],
    tiers: list[str],
    intended_basis: str,
) -> dict[tuple[str, str], float]:
    _source_basis(source, label="damage rows", intended_basis=intended_basis)
    rows = source.payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise KnapsackValidationError("damage rows manifest must contain a non-empty rows list")
    allowed = {(cell, tier) for cell in cells for tier in tiers}
    values: dict[tuple[str, str], float] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise KnapsackValidationError(f"damage rows[{index}] must be an object")
        key = (row.get("cell_id"), row.get("tier"))
        if key not in allowed:
            raise KnapsackValidationError(
                f"damage rows[{index}] identifies undeclared cell/tier pair {key!r}"
            )
        raw_damage = row.get("damage")
        if isinstance(raw_damage, bool) or not isinstance(raw_damage, (int, float)):
            raise KnapsackValidationError(
                f"damage for cell {key[0]!r}, tier {key[1]!r} must be numeric"
            )
        damage = float(raw_damage)
        if not math.isfinite(damage):
            raise KnapsackValidationError(
                f"damage for cell {key[0]!r}, tier {key[1]!r} must be finite"
            )
        if key in values:
            raise KnapsackValidationError(f"duplicate damage row for {key!r}")
        values[key] = damage
    missing = sorted(allowed - values.keys())
    if missing:
        cell, tier = missing[0]
        raise KnapsackValidationError(
            f"missing damage row for cell {cell!r}, intended tier {tier!r}; "
            f"required producer: {source.remedy_command}"
        )
    return values


def _write_once(path: Path, value: object) -> tuple[str, int]:
    payload = _canonical_json(value)
    digest = _sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != payload:
            raise FileExistsError(f"refusing to replace different sealed output: {path}")
        return digest, len(payload)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FileExistsError(f"refusing to replace different sealed output: {path}")
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return digest, len(payload)


def _preflight_write_once(path: Path, value: object) -> None:
    """Refuse conflicting sealed outputs before publishing either paired output."""

    payload = _canonical_json(value)
    if path.exists() and path.read_bytes() != payload:
        raise FileExistsError(f"refusing to replace different sealed output: {path}")


def _output_path(root: Path, value: Path | None, *, default: str, label: str) -> Path:
    if value is None:
        value = Path(default)
    if value.is_absolute():
        resolved = value.resolve()
    else:
        resolved = (root / value).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise KnapsackValidationError(f"{label} must be local to run root: {resolved}") from exc
    return resolved


def _layer_range_label(value: object, *, index: int) -> str:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(layer, bool) or not isinstance(layer, int) for layer in value)
        or value[0] < 0
        or value[1] < value[0]
    ):
        raise KnapsackValidationError(
            f"knapsack input manifest missing_inputs[{index}].layers must be "
            "an ascending pair of non-negative integers"
        )
    first, last = value
    if last - first + 1 > _MAX_LAYER_RANGE_SPAN:
        raise KnapsackValidationError(
            f"knapsack input manifest missing_inputs[{index}] layer range exceeds safe bound "
            f"of {_MAX_LAYER_RANGE_SPAN} layers"
        )
    if first == last:
        return f"L{first:03d}"
    return f"L{first:03d}-L{last:03d}"


def preflight_export_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Validate export completeness before inspecting source tensors or metadata."""

    path = Path(manifest_path).expanduser().resolve()
    manifest, _ = _read_object(path, label="knapsack input manifest")
    expected_schema = "banana-smasher-knapsack-input-index-v1"
    if manifest.get("schema") != expected_schema:
        raise KnapsackValidationError(
            "knapsack input manifest schema mismatch: "
            f"expected {expected_schema!r}, got {manifest.get('schema')!r}"
        )
    basis = _basis(manifest)
    tiers = _intended_tiers(manifest)
    envelope_bytes = manifest.get("envelope_bytes")
    if (
        isinstance(envelope_bytes, bool)
        or not isinstance(envelope_bytes, int)
        or envelope_bytes < 0
    ):
        raise KnapsackValidationError(
            "knapsack input manifest envelope_bytes must be a non-negative integer"
        )
    missing_inputs = manifest.get("missing_inputs")
    if not isinstance(missing_inputs, list):
        raise KnapsackValidationError(
            "knapsack input manifest missing_inputs must be a list"
        )

    gaps: list[str] = []
    for index, row in enumerate(missing_inputs):
        if not isinstance(row, dict):
            raise KnapsackValidationError(
                f"knapsack input manifest missing_inputs[{index}] must be an object"
            )
        tier = row.get("tier")
        if not isinstance(tier, str) or not tier:
            raise KnapsackValidationError(
                f"knapsack input manifest missing_inputs[{index}].tier must be "
                "a non-empty string"
            )
        if tier not in tiers:
            raise KnapsackValidationError(
                f"knapsack input manifest missing_inputs[{index}] names undeclared "
                f"tier {tier!r}"
            )
        layer_label = _layer_range_label(row.get("layers"), index=index)
        state = row.get("state")
        if not isinstance(state, str) or not state:
            raise KnapsackValidationError(
                f"knapsack input manifest missing_inputs[{index}].state must be "
                "a non-empty string"
            )
        gaps.append(f"{tier}/{layer_label} ({state})")

    expected_status = "PRELIM_NOT_DECISION_GRADE" if gaps else "PASS"
    if manifest.get("status") != expected_status:
        raise KnapsackValidationError(
            "knapsack input manifest status/missing_inputs mismatch: "
            f"expected {expected_status!r}, got {manifest.get('status')!r}"
        )
    if gaps:
        raise KnapsackValidationError(
            "knapsack input manifest is incomplete; missing inputs: "
            f"{', '.join(gaps)}; required producer: smash solve"
        )
    return {
        "schema": expected_schema,
        "basis_sha256": basis,
        "envelope_bytes": envelope_bytes,
        "intended_tiers": tiers,
    }


def _metadata_nodes(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    pending = [receipt]
    while pending:
        current = pending.pop()
        nodes.append(current)
        children: list[dict[str, Any]] = []
        for field in ("verified_sources", "sealed_shards", "sources"):
            value = current.get(field, [])
            if not isinstance(value, list):
                raise KnapsackValidationError(f"receipt {field} must be a list")
            for index, node in enumerate(value):
                if not isinstance(node, dict):
                    raise KnapsackValidationError(f"receipt {field}[{index}] must be an object")
                children.append(node)
        pending.extend(reversed(children))
    return nodes


def _node_tiers(node: dict[str, Any]) -> list[str]:
    values: object | None = None
    intended = node.get("intended_tiers")
    if isinstance(intended, list) and intended:
        values = intended
    else:
        identity = node.get("identity_coverage")
        if isinstance(identity, dict) and "tiers" in identity:
            values = identity["tiers"]
        elif "tiers" in node:
            values = node["tiers"]
        elif "tier" in node:
            values = [node["tier"]]
    if values is None:
        return []
    if not isinstance(values, list):
        raise KnapsackValidationError("receipt tiers metadata must be a list")
    result: list[str] = []
    for index, tier in enumerate(values):
        if not isinstance(tier, str) or not tier:
            raise KnapsackValidationError(
                f"receipt tiers metadata[{index}] must be a non-empty string"
            )
        result.append(tier)
    return result


def _receipt_basis(receipt: dict[str, Any], *, path: Path) -> str:
    candidates: list[str] = []
    for node in _metadata_nodes(receipt):
        for field in ("basis_sha256", "intended_basis_sha256"):
            if field in node:
                candidates.append(_sha_field(node[field], label=f"{path} {field}"))
        intended = node.get("intended_basis")
        if isinstance(intended, str):
            candidates.append(_sha_field(intended, label=f"{path} intended_basis"))
        elif isinstance(intended, dict) and "model_index_sha256" in intended:
            candidates.append(
                _sha_field(
                    intended["model_index_sha256"],
                    label=f"{path} intended_basis.model_index_sha256",
                )
            )
    if not candidates:
        raise KnapsackValidationError(f"receipt does not declare a basis SHA-256: {path}")
    if len(set(candidates)) != 1:
        raise KnapsackValidationError(f"receipt basis mismatch within {path}: {sorted(set(candidates))}")
    return candidates[0]


def _validate_index_descriptor(descriptor: object, *, label: str) -> None:
    """Validate a solver descriptor without requiring its payload to exist yet."""

    if not isinstance(descriptor, dict):
        raise KnapsackValidationError(f"{label} must be an object")
    raw_path = descriptor.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise KnapsackValidationError(f"{label} path must be a non-empty string")
    relative = Path(raw_path)
    if relative.is_absolute():
        raise KnapsackValidationError(f"{label} path must be relative")
    descriptor_root = Path("/descriptor-root")
    resolved = (descriptor_root / relative).resolve()
    try:
        resolved.relative_to(descriptor_root)
    except ValueError as exc:
        raise KnapsackValidationError(
            f"{label} path escapes descriptor root: {raw_path}"
        ) from exc
    _sha_field(descriptor.get("sha256"), label=f"{label} sha256")
    if "producer_command" in descriptor:
        producer = descriptor["producer_command"]
        if not isinstance(producer, str) or not producer.strip():
            raise KnapsackValidationError(
                f"{label} producer_command must be a non-empty string"
            )


def _merge_descriptor(
    target: dict[str, Any], *, key: str, descriptor: object, label: str
) -> None:
    if not isinstance(descriptor, dict):
        raise KnapsackValidationError(f"{label} descriptor for {key!r} must be an object")
    existing = target.get(key)
    if existing is not None and existing != descriptor:
        raise KnapsackValidationError(f"conflicting {label} descriptors for {key!r}")
    target[key] = descriptor


def _sealed_status(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return value.upper() in {"PASS", "PASS_MERGEABLE", "SEALED", "MERGEABLE"}


def _mergeable_input_status(value: object) -> bool:
    return _sealed_status(value) or (
        isinstance(value, str) and value.upper() == "PRELIM_PASS_WITH_MISSING_INPUTS"
    )


def _node_layers(node: dict[str, Any]) -> set[int]:
    candidates: list[object] = [node.get("layers")]
    for field in ("coverage", "identity_coverage"):
        nested = node.get(field)
        if isinstance(nested, dict):
            candidates.append(nested.get("layers"))
    declared: list[set[int]] = []
    for candidate in candidates:
        if candidate is None:
            continue
        if not isinstance(candidate, list) or any(
            isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
            for layer in candidate
        ):
            raise KnapsackValidationError("sealed source layers must be non-negative integers")
        if len(set(candidate)) != len(candidate):
            raise KnapsackValidationError("sealed source layers must be unique")
        declared.append(set(candidate))

    expected = node.get("expected")
    observed = node.get("observed")
    if isinstance(expected, dict) and isinstance(observed, dict):
        layer_range = expected.get("layer_range")
        missing_layers = observed.get("missing_layers")
        if observed.get("coverage_complete") is True and missing_layers == []:
            if (
                not isinstance(layer_range, list)
                or len(layer_range) != 2
                or any(
                    isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
                    for layer in layer_range
                )
                or layer_range[1] < layer_range[0]
            ):
                raise KnapsackValidationError(
                    "complete sealed source expected.layer_range must be an ascending pair"
                )
            if layer_range[1] - layer_range[0] + 1 > _MAX_LAYER_RANGE_SPAN:
                raise KnapsackValidationError(
                    "complete sealed source layer range exceeds safe bound "
                    f"of {_MAX_LAYER_RANGE_SPAN} layers"
                )
            declared.append(set(range(layer_range[0], layer_range[1] + 1)))
    if not declared:
        return set()
    if any(layers != declared[0] for layers in declared[1:]):
        raise KnapsackValidationError("sealed source has contradictory layer declarations")
    return declared[0]


def _sealed_coverage(receipt: dict[str, Any]) -> dict[str, set[int]]:
    coverage: dict[str, set[int]] = {}
    for node in _metadata_nodes(receipt):
        if not _sealed_status(node.get("status", receipt.get("status"))):
            continue
        layers = _node_layers(node)
        if not layers:
            continue
        for tier in _node_tiers(node):
            coverage.setdefault(tier, set()).update(layers)
    return coverage


def _missing_rows(value: object, *, tiers: list[str], label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise KnapsackValidationError(f"{label} must be a list")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise KnapsackValidationError(f"{label}[{index}] must be an object")
        tier = row.get("tier")
        if tier not in tiers:
            raise KnapsackValidationError(
                f"{label}[{index}] names undeclared tier {tier!r}"
            )
        _layer_range_label(row.get("layers"), index=index)
        state = row.get("state")
        if not isinstance(state, str) or not state:
            raise KnapsackValidationError(
                f"{label}[{index}].state must be a non-empty string"
            )
        rows.append(dict(row))
    return rows


def _receipt_missing_rows(
    receipt: dict[str, Any],
    *,
    tiers: list[str],
    coverage: dict[str, set[int]],
    label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node_index, node in enumerate(_metadata_nodes(receipt)):
        for field in ("missing_set", "missing_inputs"):
            if field in node:
                rows.extend(
                    _missing_rows(
                        node[field],
                        tiers=tiers,
                        label=f"{label} metadata[{node_index}].{field}",
                    )
                )
    for row in rows:
        first, last = row["layers"]
        if any(
            layer in coverage.get(row["tier"], set())
            for layer in range(first, last + 1)
        ):
            raise KnapsackValidationError(
                f"{label} marks the same tier/layer both covered and missing: "
                f"{row['tier']} L{first:03d}-L{last:03d}"
            )
    if rows and _sealed_status(receipt.get("status")):
        raise KnapsackValidationError(
            f"{label} sealed/PASS status is inconsistent with missing rows"
        )
    return rows


def _subtract_covered_missing(
    rows: list[dict[str, Any]], coverage: dict[str, set[int]]
) -> list[dict[str, Any]]:
    remaining: list[dict[str, Any]] = []
    for row in rows:
        first, last = row["layers"]
        uncovered = [
            layer for layer in range(first, last + 1) if layer not in coverage.get(row["tier"], set())
        ]
        if not uncovered:
            continue
        start = previous = uncovered[0]
        for layer in uncovered[1:] + [None]:
            if layer is not None and layer == previous + 1:
                previous = layer
                continue
            fragment = dict(row)
            fragment["layers"] = [start, previous]
            remaining.append(fragment)
            if layer is not None:
                start = previous = layer
    unique = {_canonical_json(row): row for row in remaining}
    return [unique[key] for key in sorted(unique)]


def _merge_receipt_descriptors(
    receipt: dict[str, Any],
    *,
    anchor_manifests: dict[str, Any],
    damage_rows: dict[str, Any] | None,
    intended_tiers: list[str] | None = None,
) -> dict[str, Any] | None:
    for node in _metadata_nodes(receipt):
        descriptors = node.get("anchor_manifests")
        if descriptors is not None:
            if not isinstance(descriptors, dict):
                raise KnapsackValidationError("receipt anchor_manifests must be an object")
            for tier, descriptor in descriptors.items():
                if not isinstance(tier, str) or not tier:
                    raise KnapsackValidationError(
                        "anchor manifest tier keys must be non-empty strings"
                    )
                if intended_tiers is not None and tier not in intended_tiers:
                    raise KnapsackValidationError(
                        f"anchor manifest descriptor names undeclared tier {tier!r}"
                    )
                _merge_descriptor(
                    anchor_manifests,
                    key=tier,
                    descriptor=descriptor,
                    label="anchor manifest",
                )
        descriptor = node.get("anchor_manifest")
        if descriptor is not None:
            tiers = _node_tiers(node)
            if len(tiers) != 1:
                raise KnapsackValidationError(
                    "anchor_manifest metadata requires exactly one declared tier"
                )
            if intended_tiers is not None and tiers[0] not in intended_tiers:
                raise KnapsackValidationError(
                    f"anchor manifest descriptor names undeclared tier {tiers[0]!r}"
                )
            _merge_descriptor(
                anchor_manifests,
                key=tiers[0],
                descriptor=descriptor,
                label="anchor manifest",
            )
        current_damage = node.get("damage_rows")
        if current_damage is not None:
            if not isinstance(current_damage, dict):
                raise KnapsackValidationError("receipt damage_rows must be an object")
            if damage_rows is not None and damage_rows != current_damage:
                raise KnapsackValidationError("conflicting damage_rows descriptors")
            damage_rows = current_damage
    return damage_rows


def merge_knapsack_input_index(
    *,
    run_root: str | Path,
    input_index: str | Path,
    source_receipts: list[str | Path],
    selection_receipt: str | Path,
    basis_sha256: str,
    envelope_bytes: int,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Merge later sealed sources into an existing incomplete input index."""

    root = Path(run_root).expanduser().resolve()
    expected_basis = _sha_field(basis_sha256, label="basis_sha256")
    if isinstance(envelope_bytes, bool) or not isinstance(envelope_bytes, int) or envelope_bytes < 0:
        raise KnapsackValidationError("envelope_bytes must be a non-negative integer")
    input_path = Path(input_index).expanduser().resolve()
    index, input_payload = _read_object(input_path, label="knapsack input index")
    normalized_input = index.get("schema") == "banana-smasher-knapsack-input-index-v1"
    if normalized_input:
        input_basis = _basis(index)
    else:
        if not _mergeable_input_status(index.get("status")):
            raise KnapsackValidationError(
                f"input index is not sealed/PASS at {input_path}: {index.get('status')!r}"
            )
        input_basis = _receipt_basis(index, path=input_path)
    if input_basis != expected_basis:
        raise KnapsackValidationError(
            f"input index basis mismatch: expected {expected_basis}, got {input_basis}"
        )
    if index.get("envelope_bytes") != envelope_bytes:
        raise KnapsackValidationError(
            "input index envelope_bytes mismatch: "
            f"expected {envelope_bytes}, got {index.get('envelope_bytes')}"
        )

    anchor_manifests: dict[str, Any] = {}
    damage_rows: dict[str, Any] | None = None
    if normalized_input:
        tiers = _intended_tiers(index)
        damage_rows = _merge_receipt_descriptors(
            index,
            anchor_manifests=anchor_manifests,
            damage_rows=damage_rows,
            intended_tiers=tiers,
        )
        missing = _missing_rows(
            index.get("missing_inputs"), tiers=tiers, label="input index missing_inputs"
        )
        expected_input_status = "PRELIM_NOT_DECISION_GRADE" if missing else "PASS"
        if index.get("status") != expected_input_status:
            raise KnapsackValidationError(
                "normalized input index status is inconsistent with missing_inputs: "
                f"expected {expected_input_status!r}, got {index.get('status')!r}"
            )
        source_rows = index.get("source_receipts", [])
        if not isinstance(source_rows, list) or not all(
            isinstance(row, dict) for row in source_rows
        ):
            raise KnapsackValidationError(
                "input index source_receipts must be a list of objects"
            )
        merged_source_rows = [dict(row) for row in source_rows]
    else:
        damage_rows = _merge_receipt_descriptors(
            index, anchor_manifests=anchor_manifests, damage_rows=damage_rows
        )
        tier_names = {
            tier for node in _metadata_nodes(index) for tier in _node_tiers(node)
        }
        tier_names.update(anchor_manifests)
        if not tier_names:
            raise KnapsackValidationError("input index metadata declares no tiers")
        tiers = sorted(tier_names)
        missing = []
        for field in ("missing_set", "missing_inputs"):
            if field in index:
                missing.extend(
                    _missing_rows(index[field], tiers=tiers, label=f"input index {field}")
                )
        merged_source_rows = []
    coverage: dict[str, set[int]] = {}

    for raw_path in source_receipts:
        path = Path(raw_path).expanduser().resolve()
        source, payload = _read_object(path, label="sealed source receipt")
        if not _sealed_status(source.get("status")):
            raise KnapsackValidationError(
                f"source receipt is not sealed/PASS at {path}: {source.get('status')!r}"
            )
        source_basis = _receipt_basis(source, path=path)
        if source_basis != expected_basis:
            raise KnapsackValidationError(
                f"source receipt basis mismatch: expected {expected_basis}, "
                f"got {source_basis} at {path}"
            )
        source_tiers = {
            tier for node in _metadata_nodes(source) for tier in _node_tiers(node)
        }
        undeclared = sorted(source_tiers - set(tiers))
        if undeclared:
            raise KnapsackValidationError(
                f"source receipt declares tiers outside input index: {undeclared}"
            )
        source_coverage = _sealed_coverage(source)
        for tier, layers in source_coverage.items():
            if tier not in tiers:
                raise KnapsackValidationError(
                    f"source coverage names undeclared tier {tier!r}"
                )
            coverage.setdefault(tier, set()).update(layers)
        source_missing = _receipt_missing_rows(
            source,
            tiers=tiers,
            coverage=source_coverage,
            label=f"source receipt at {path}",
        )
        missing.extend(source_missing)
        damage_rows = _merge_receipt_descriptors(
            source,
            anchor_manifests=anchor_manifests,
            damage_rows=damage_rows,
            intended_tiers=tiers,
        )
        merged_source_rows.append(
            {
                "path": str(path),
                "sha256": _sha256(payload),
                "bytes": len(payload),
                "schema": source.get("schema"),
                "status": source.get("status"),
            }
        )

    missing = _subtract_covered_missing(missing, coverage)
    if not missing:
        missing_anchor_tiers = [tier for tier in tiers if tier not in anchor_manifests]
        if missing_anchor_tiers:
            raise KnapsackValidationError(
                "missing anchor_manifests descriptors for intended tiers "
                f"{missing_anchor_tiers}; required producer: smash anchor"
            )
        if damage_rows is None:
            raise KnapsackValidationError(
                "missing damage_rows descriptor; required producer: smash anchor"
            )
        for tier in tiers:
            _validate_index_descriptor(
                anchor_manifests[tier],
                label=f"anchor manifest descriptor for {tier!r}",
            )
        _validate_index_descriptor(damage_rows, label="damage_rows descriptor")
    merged_source_rows.append(
        {
            "path": str(input_path),
            "sha256": _sha256(input_payload),
            "bytes": len(input_payload),
            "schema": index.get("schema"),
            "status": index.get("status"),
        }
    )
    unique_sources = {_canonical_json(row): row for row in merged_source_rows}
    merged_source_rows = [unique_sources[key] for key in sorted(unique_sources)]
    status = "PRELIM_NOT_DECISION_GRADE" if missing else "PASS"
    output_path = _output_path(
        root,
        Path(output) if output is not None else None,
        default="MANIFEST.json",
        label="output",
    )
    selection_path = _output_path(
        root,
        Path(selection_receipt),
        default="knapsack/INDEX_RECEIPT.json",
        label="selection receipt",
    )
    if output_path == selection_path:
        raise KnapsackValidationError("output and selection receipt paths must differ")
    index_value: dict[str, Any] = {
        "schema": "banana-smasher-knapsack-input-index-v1",
        "status": status,
        "intended_basis_sha256": expected_basis,
        "intended_tiers": tiers,
        "envelope_bytes": envelope_bytes,
        "source_receipts": merged_source_rows,
        "missing_inputs": missing,
    }
    if anchor_manifests:
        index_value["anchor_manifests"] = {
            tier: anchor_manifests[tier] for tier in sorted(anchor_manifests)
        }
    if damage_rows is not None:
        index_value["damage_rows"] = damage_rows
    index_payload = _canonical_json(index_value)
    selection_value = {
        "schema": "banana-smasher-knapsack-index-receipt-v1",
        "status": status,
        "basis_sha256": expected_basis,
        "selected_tiers": tiers,
        "byte_accounting": {"envelope_bytes": envelope_bytes},
        "missing_inputs": missing,
        "source_receipts": merged_source_rows,
        "input_index": {
            "path": str(output_path),
            "sha256": _sha256(index_payload),
            "bytes": len(index_payload),
        },
    }
    _preflight_write_once(output_path, index_value)
    _preflight_write_once(selection_path, selection_value)
    index_sha, index_bytes = _write_once(output_path, index_value)
    selection_sha, selection_bytes = _write_once(selection_path, selection_value)
    return {
        "status": status,
        "command": "knapsack-index",
        "basis_sha256": expected_basis,
        "selected_tiers": tiers,
        "byte_accounting": {"envelope_bytes": envelope_bytes},
        "missing_inputs": missing,
        "input_index": {"path": str(output_path), "sha256": index_sha, "bytes": index_bytes},
        "receipt": {
            "path": str(selection_path),
            "sha256": selection_sha,
            "bytes": selection_bytes,
        },
    }


def build_knapsack_input_index(
    *,
    receipts: list[str | Path],
    output: str | Path,
    selection_receipt: str | Path,
    envelope_bytes: int,
) -> dict[str, Any]:
    """Build a deterministic open-tier knapsack index from sealed receipt metadata."""

    if not receipts:
        raise KnapsackValidationError("at least one sealed receipt is required")
    if isinstance(envelope_bytes, bool) or not isinstance(envelope_bytes, int) or envelope_bytes < 0:
        raise KnapsackValidationError("envelope_bytes must be a non-negative integer")
    output_path = Path(output).expanduser().resolve()
    selection_path = Path(selection_receipt).expanduser().resolve()
    if output_path == selection_path:
        raise KnapsackValidationError("output and selection receipt paths must differ")

    source_rows: list[dict[str, Any]] = []
    tier_names: set[str] = set()
    bases: set[str] = set()
    missing_inputs: list[dict[str, Any]] = []
    anchor_manifests: dict[str, Any] = {}
    damage_rows: dict[str, Any] | None = None
    for raw_path in receipts:
        path = Path(raw_path).expanduser().resolve()
        value, payload = _read_object(path, label="sealed anchor receipt")
        status = value.get("status")
        if not isinstance(status, str) or not any(
            marker in status.upper() for marker in ("PASS", "SEALED", "MERGEABLE")
        ):
            raise KnapsackValidationError(f"anchor receipt is not sealed/PASS at {path}: {status!r}")
        basis = _receipt_basis(value, path=path)
        bases.add(basis)
        declared_envelope = value.get("envelope_bytes")
        if declared_envelope is not None and declared_envelope != envelope_bytes:
            raise KnapsackValidationError(
                f"receipt envelope_bytes mismatch at {path}: "
                f"expected {envelope_bytes}, got {declared_envelope}"
            )
        for node in _metadata_nodes(value):
            tier_names.update(_node_tiers(node))
            descriptors = node.get("anchor_manifests")
            if descriptors is not None:
                if not isinstance(descriptors, dict):
                    raise KnapsackValidationError("receipt anchor_manifests must be an object")
                for tier, descriptor in descriptors.items():
                    if not isinstance(tier, str) or not tier:
                        raise KnapsackValidationError("anchor manifest tier keys must be non-empty strings")
                    tier_names.add(tier)
                    _merge_descriptor(
                        anchor_manifests,
                        key=tier,
                        descriptor=descriptor,
                        label="anchor manifest",
                    )
            descriptor = node.get("anchor_manifest")
            node_tiers = _node_tiers(node)
            if descriptor is not None:
                if len(node_tiers) != 1:
                    raise KnapsackValidationError(
                        "anchor_manifest metadata requires exactly one declared tier"
                    )
                _merge_descriptor(
                    anchor_manifests,
                    key=node_tiers[0],
                    descriptor=descriptor,
                    label="anchor manifest",
                )
            current_damage = node.get("damage_rows")
            if current_damage is not None:
                if not isinstance(current_damage, dict):
                    raise KnapsackValidationError("receipt damage_rows must be an object")
                if damage_rows is not None and damage_rows != current_damage:
                    raise KnapsackValidationError("conflicting damage_rows descriptors")
                damage_rows = current_damage
        for field in ("missing_set", "missing_inputs"):
            rows = value.get(field, [])
            if not isinstance(rows, list):
                raise KnapsackValidationError(f"receipt {field} must be a list")
            for index, row in enumerate(rows):
                if not isinstance(row, dict):
                    raise KnapsackValidationError(f"receipt {field}[{index}] must be an object")
                missing_inputs.append(row)
        source_rows.append(
            {
                "path": str(path),
                "sha256": _sha256(payload),
                "bytes": len(payload),
                "schema": value.get("schema"),
                "status": status,
            }
        )

    if len(bases) != 1:
        raise KnapsackValidationError(f"receipt basis mismatch: {sorted(bases)}")
    if not tier_names:
        raise KnapsackValidationError("sealed receipt metadata declares no tiers")
    selected_tiers = sorted(tier_names)
    if not missing_inputs:
        missing_anchor_tiers = [
            tier for tier in selected_tiers if tier not in anchor_manifests
        ]
        if missing_anchor_tiers:
            raise KnapsackValidationError(
                "missing anchor_manifests descriptors for intended tiers "
                f"{missing_anchor_tiers}; required producer: smash anchor"
            )
        if damage_rows is None:
            raise KnapsackValidationError(
                "missing damage_rows descriptor; required producer: smash anchor"
            )
        for tier in selected_tiers:
            _validate_index_descriptor(
                anchor_manifests[tier],
                label=f"anchor manifest descriptor for {tier!r}",
            )
        _validate_index_descriptor(damage_rows, label="damage_rows descriptor")
    source_rows.sort(key=lambda row: (row["sha256"], row["path"]))
    missing_inputs.sort(key=lambda row: _canonical_json(row))
    status = "PRELIM_NOT_DECISION_GRADE" if missing_inputs else "PASS"
    basis = next(iter(bases))
    index_value: dict[str, Any] = {
        "schema": "banana-smasher-knapsack-input-index-v1",
        "status": status,
        "intended_basis_sha256": basis,
        "intended_tiers": selected_tiers,
        "envelope_bytes": envelope_bytes,
        "source_receipts": source_rows,
        "missing_inputs": missing_inputs,
    }
    if anchor_manifests:
        index_value["anchor_manifests"] = {
            tier: anchor_manifests[tier] for tier in sorted(anchor_manifests)
        }
    if damage_rows is not None:
        index_value["damage_rows"] = damage_rows

    index_payload = _canonical_json(index_value)
    selection_value = {
        "schema": "banana-smasher-knapsack-index-receipt-v1",
        "status": status,
        "basis_sha256": basis,
        "selected_tiers": selected_tiers,
        "byte_accounting": {"envelope_bytes": envelope_bytes},
        "missing_inputs": missing_inputs,
        "source_receipts": source_rows,
        "input_index": {
            "path": str(output_path),
            "sha256": _sha256(index_payload),
            "bytes": len(index_payload),
        },
    }
    _preflight_write_once(output_path, index_value)
    _preflight_write_once(selection_path, selection_value)
    index_sha, index_bytes = _write_once(output_path, index_value)
    selection_sha, selection_bytes = _write_once(selection_path, selection_value)
    return {
        "status": status,
        "command": "knapsack-index",
        "basis_sha256": basis,
        "selected_tiers": selected_tiers,
        "byte_accounting": {"envelope_bytes": envelope_bytes},
        "missing_inputs": missing_inputs,
        "input_index": {
            "path": str(output_path),
            "sha256": index_sha,
            "bytes": index_bytes,
        },
        "receipt": {
            "path": str(selection_path),
            "sha256": selection_sha,
            "bytes": selection_bytes,
        },
    }


def solve_class_balanced_options(
    *,
    cells: list[str],
    tiers: list[str],
    bytes_by_option: dict[tuple[str, str], int],
    class_costs_by_option: dict[tuple[str, str], dict[str, float]],
    envelope_bytes: int,
    class_caps: dict[str, float],
    class_weights: dict[str, float] | None = None,
    class_floors: dict[str, float] | None = None,
    exact_envelope: bool = False,
) -> dict[str, Any]:
    """Select one tier per cell under exact bytes and aggregate class ceilings.

    This is the class-aware primitive used by the dynamic Backpack policy.  Its
    outputs are predictions, never measured KLD.  Integer byte accounting is
    rechecked in Python after the MILP solve.
    """

    if not cells or len(cells) != len(set(cells)):
        raise KnapsackValidationError("cells must be a non-empty unique list")
    if not tiers or len(tiers) != len(set(tiers)):
        raise KnapsackValidationError("tiers must be a non-empty unique list")
    if isinstance(envelope_bytes, bool) or not isinstance(envelope_bytes, int) or envelope_bytes < 0:
        raise KnapsackValidationError("envelope_bytes must be a non-negative integer")
    if not class_caps:
        raise KnapsackValidationError("class_caps must be a non-empty object")
    classes = sorted(class_caps)
    caps: dict[str, float] = {}
    for name in classes:
        cap = float(class_caps[name])
        if not math.isfinite(cap) or cap < 0.0:
            raise KnapsackValidationError(f"class cap must be finite and non-negative for {name!r}")
        caps[name] = cap
    if class_floors is None:
        floors = {name: 0.0 for name in classes}
    else:
        if set(class_floors) != set(classes):
            raise KnapsackValidationError("class_floors keys must exactly match class_caps")
        floors = {name: float(class_floors[name]) for name in classes}
        if any(
            not math.isfinite(floors[name])
            or floors[name] < 0.0
            or floors[name] > caps[name]
            for name in classes
        ):
            raise KnapsackValidationError(
                "class floors must be finite, non-negative, and no greater than class caps"
            )
    if class_weights is None:
        weights = {name: 1.0 / len(classes) for name in classes}
    else:
        if set(class_weights) != set(classes):
            raise KnapsackValidationError("class_weights keys must exactly match class_caps")
        raw_weights = {name: float(class_weights[name]) for name in classes}
        if any(not math.isfinite(value) or value < 0.0 for value in raw_weights.values()):
            raise KnapsackValidationError("class weights must be finite and non-negative")
        total_weight = math.fsum(raw_weights.values())
        if total_weight <= 0.0:
            raise KnapsackValidationError("class weights must have positive total mass")
        weights = {name: raw_weights[name] / total_weight for name in classes}

    expected_options = {(cell, tier) for cell in cells for tier in tiers}
    if set(bytes_by_option) != expected_options:
        missing = sorted(expected_options - set(bytes_by_option))
        extra = sorted(set(bytes_by_option) - expected_options)
        raise KnapsackValidationError(
            f"bytes_by_option must cover every cell/tier exactly: missing={missing[:3]}, extra={extra[:3]}"
        )
    if set(class_costs_by_option) != expected_options:
        missing = sorted(expected_options - set(class_costs_by_option))
        extra = sorted(set(class_costs_by_option) - expected_options)
        raise KnapsackValidationError(
            f"class_costs_by_option must cover every cell/tier exactly: missing={missing[:3]}, extra={extra[:3]}"
        )

    costs: dict[tuple[str, str], int] = {}
    predictions: dict[tuple[str, str], dict[str, float]] = {}
    for key in sorted(expected_options):
        byte_count = bytes_by_option[key]
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise KnapsackValidationError(f"option bytes must be a non-negative integer for {key!r}")
        costs[key] = byte_count
        raw = class_costs_by_option[key]
        if set(raw) != set(classes):
            raise KnapsackValidationError(f"class costs must exactly match class_caps for {key!r}")
        parsed = {name: float(raw[name]) for name in classes}
        if any(not math.isfinite(value) for value in parsed.values()):
            raise KnapsackValidationError(f"class costs must be finite for {key!r}")
        predictions[key] = parsed

    minimum_required_bytes = sum(min(costs[(cell, tier)] for tier in tiers) for cell in cells)
    if minimum_required_bytes > envelope_bytes:
        raise KnapsackValidationError(
            f"envelope infeasible: minimum required {minimum_required_bytes} bytes exceeds {envelope_bytes}"
        )

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except ImportError as exc:  # pragma: no cover - installation contract
        raise RuntimeError("class-balanced exact solver requires scipy") from exc

    baseline_by_cell = {cell: min(costs[(cell, tier)] for tier in tiers) for cell in cells}
    remaining_envelope = envelope_bytes - minimum_required_bytes
    byte_deltas = {
        (cell, tier): costs[(cell, tier)] - baseline_by_cell[cell]
        for cell in cells
        for tier in tiers
    }
    positive = [delta for delta in byte_deltas.values() if 0 < delta <= remaining_envelope]
    byte_divisor = math.gcd(*positive) if positive else 1
    scaled_capacity = remaining_envelope // byte_divisor
    scaled_deltas = {
        key: delta // byte_divisor for key, delta in byte_deltas.items() if delta <= remaining_envelope
    }
    maximum_scaled_use = sum(
        max(scaled_deltas.get((cell, tier), 0) for tier in tiers) for cell in cells
    )
    enforce_bytes = exact_envelope or scaled_capacity < maximum_scaled_use
    if enforce_bytes and (
        scaled_capacity > 2**53 or any(delta > 2**53 for delta in scaled_deltas.values())
    ):
        raise KnapsackValidationError("exact byte row exceeds float64 integer range after GCD normalization")

    variable_count = len(cells) * len(tiers)
    row_count = len(cells) + int(enforce_bytes) + len(classes)
    objective = np.empty(variable_count, dtype=np.float64)
    variable_upper = np.ones(variable_count, dtype=np.float64)
    row_indices: list[int] = []
    column_indices: list[int] = []
    coefficients: list[float] = []
    for cell_index, cell in enumerate(cells):
        seen_equal_options: set[tuple[int, tuple[float, ...]]] = set()
        for tier_index, tier in enumerate(tiers):
            variable_index = cell_index * len(tiers) + tier_index
            key = (cell, tier)
            equal_option_key = (
                costs[key],
                tuple(predictions[key][name] for name in classes),
            )
            if equal_option_key in seen_equal_options:
                variable_upper[variable_index] = 0.0
            else:
                seen_equal_options.add(equal_option_key)
            objective[variable_index] = math.fsum(
                weights[name] * predictions[key][name] for name in classes
            )
            row_indices.append(cell_index)
            column_indices.append(variable_index)
            coefficients.append(1.0)
            delta = byte_deltas[key]
            if delta > remaining_envelope:
                variable_upper[variable_index] = 0.0
            elif enforce_bytes:
                row_indices.append(len(cells))
                column_indices.append(variable_index)
                coefficients.append(float(scaled_deltas[key]))
            class_row_offset = len(cells) + int(enforce_bytes)
            for class_index, name in enumerate(classes):
                row_indices.append(class_row_offset + class_index)
                column_indices.append(variable_index)
                coefficients.append(predictions[key][name])
    matrix = coo_matrix(
        (coefficients, (row_indices, column_indices)), shape=(row_count, variable_count)
    ).tocsr()
    lower = np.empty(row_count, dtype=np.float64)
    upper = np.empty(row_count, dtype=np.float64)
    lower[: len(cells)] = 1.0
    upper[: len(cells)] = 1.0
    cursor = len(cells)
    if enforce_bytes:
        lower[cursor] = float(scaled_capacity) if exact_envelope else -np.inf
        upper[cursor] = float(scaled_capacity)
        cursor += 1
    for name in classes:
        lower[cursor] = floors[name]
        upper[cursor] = caps[name]
        cursor += 1
    solution = milp(
        c=objective,
        integrality=np.ones(variable_count, dtype=np.int8),
        bounds=Bounds(np.zeros(variable_count), variable_upper),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"presolve": True, "mip_rel_gap": 0.0},
    )
    if not solution.success or solution.x is None or int(solution.status) != 0:
        raise RuntimeError(
            f"class-balanced exact solve failed: status={solution.status}, message={solution.message}"
        )
    if float(getattr(solution, "mip_gap", math.inf)) != 0.0:
        raise RuntimeError("class-balanced exact solve returned a nonzero MIP gap")
    rounded = np.rint(solution.x).astype(np.int8)
    if not np.allclose(solution.x, rounded, rtol=0.0, atol=1e-6):
        raise RuntimeError("class-balanced exact solver returned a non-integral assignment")

    assignments: list[dict[str, Any]] = []
    predicted = {name: 0.0 for name in classes}
    for cell_index, cell in enumerate(cells):
        offset = cell_index * len(tiers)
        selected = np.flatnonzero(rounded[offset : offset + len(tiers)])
        if len(selected) != 1:
            raise RuntimeError(f"class-balanced solver selected {len(selected)} tiers for {cell!r}")
        tier = tiers[int(selected[0])]
        key = (cell, tier)
        for name in classes:
            predicted[name] += predictions[key][name]
        assignments.append(
            {
                "cell_id": cell,
                "tier": tier,
                "bytes": costs[key],
                "prediction_by_class": predictions[key],
            }
        )
    assigned_bytes = sum(row["bytes"] for row in assignments)
    if exact_envelope and assigned_bytes != envelope_bytes:
        raise RuntimeError(
            "class-balanced solver violated exact envelope: "
            f"{assigned_bytes} != {envelope_bytes}"
        )
    if assigned_bytes > envelope_bytes:
        raise RuntimeError(f"class-balanced solver violated envelope: {assigned_bytes} > {envelope_bytes}")
    if any(
        predicted[name] < floors[name] - 1e-10
        or predicted[name] > caps[name] + 1e-10
        for name in classes
    ):
        raise RuntimeError("class-balanced solver violated aggregate class bounds")
    objective_value = math.fsum(weights[name] * predicted[name] for name in classes)
    return {
        "status": "PASS_PREDICTION_ONLY",
        "assignments": assignments,
        "assigned_bytes": assigned_bytes,
        "envelope_bytes": envelope_bytes,
        "slack_bytes": envelope_bytes - assigned_bytes,
        "prediction_by_class": predicted,
        "class_caps": caps,
        "class_floors": floors,
        "objective": {
            "name": "uniform_mean_per_class_predicted_damage" if class_weights is None else "weighted_mean_per_class_predicted_damage",
            "value": objective_value,
            "normalized_class_weights": weights,
        },
        "solver": {
            "backend": "scipy.optimize.milp/HiGHS",
            "status": int(solution.status),
            "mip_gap": float(getattr(solution, "mip_gap", 0.0)),
            "byte_gcd_divisor": byte_divisor,
            "equal_option_tie_breaker": "first_manifest_tier",
        },
    }


def run_knapsack(
    *,
    run_root: str | Path,
    envelope_bytes: int,
    output: str | Path | None = None,
    receipt: str | Path | None = None,
) -> dict[str, Any]:
    """Solve a manifest-bound multiple-choice integer knapsack exactly with HiGHS."""

    if isinstance(envelope_bytes, bool) or not isinstance(envelope_bytes, int) or envelope_bytes < 0:
        raise KnapsackValidationError("envelope_bytes must be a non-negative integer")
    root = Path(run_root).expanduser().resolve()
    manifest_path = root / "MANIFEST.json"
    manifest, manifest_bytes = _read_object(manifest_path, label="run manifest")
    tiers = _intended_tiers(manifest)
    intended_basis = _basis(manifest)
    descriptors = manifest.get("anchor_manifests")
    if not isinstance(descriptors, dict):
        raise KnapsackValidationError("run manifest anchor_manifests must be an object")

    # Complete local+SHA preflight for every declared tier before parsing or solving any one tier.
    anchor_sources: dict[str, _Source] = {}
    anchor_producer = f"smash anchor --run-root {root}"
    for tier in tiers:
        descriptor = descriptors.get(tier)
        if descriptor is None:
            raise KnapsackValidationError(
                f"missing intended anchor manifest descriptor for tier {tier!r}; "
                f"required producer: {anchor_producer}"
            )
        anchor_sources[tier] = _preflight_source(
            root=root,
            descriptor=descriptor,
            label=f"anchor manifest for tier {tier!r}",
            missing_message=f"missing intended anchor manifest for tier {tier!r}",
            fallback_producer=anchor_producer,
        )

    damage_descriptor = manifest.get("damage_rows")
    if damage_descriptor is None:
        raise KnapsackValidationError(
            "missing damage rows manifest descriptor; "
            f"required producer: {anchor_producer}"
        )
    damage_source = _preflight_source(
        root=root,
        descriptor=damage_descriptor,
        label="damage rows manifest",
        missing_message="missing damage rows manifest",
        fallback_producer=anchor_producer,
    )

    costs: dict[str, dict[str, int]] = {}
    expected_cells: set[str] | None = None
    for tier in tiers:
        costs[tier] = _anchor_cells(
            anchor_sources[tier], tier=tier, intended_basis=intended_basis
        )
        current_cells = set(costs[tier])
        if expected_cells is None:
            expected_cells = current_cells
        elif current_cells != expected_cells:
            missing = sorted(expected_cells - current_cells)
            extra = sorted(current_cells - expected_cells)
            raise KnapsackValidationError(
                f"anchor cell-set mismatch for tier {tier!r}: missing={missing[:3]}, extra={extra[:3]}"
            )
    cells = sorted(expected_cells or ())
    damages = _damage_values(
        damage_source, cells=cells, tiers=tiers, intended_basis=intended_basis
    )
    minimum_required_bytes = sum(min(costs[tier][cell] for tier in tiers) for cell in cells)
    if minimum_required_bytes > envelope_bytes:
        raise KnapsackValidationError(
            f"envelope infeasible: minimum required {minimum_required_bytes} bytes exceeds "
            f"--envelope-bytes {envelope_bytes}"
        )

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except ImportError as exc:  # pragma: no cover - exercised by installation smoke tests
        raise RuntimeError(
            "exact knapsack solver unavailable; install banana-smasher[knapsack] "
            "(requires scipy)"
        ) from exc

    # HiGHS accepts only float64 coefficients. Subtract the mandatory per-cell
    # baseline in Python integers, divide the remaining byte deltas by their
    # exact GCD, and refuse any still-binding row outside exact float64 range.
    baseline_by_cell = {
        cell: min(costs[tier][cell] for tier in tiers) for cell in cells
    }
    remaining_envelope = envelope_bytes - minimum_required_bytes
    byte_deltas = {
        (cell, tier): costs[tier][cell] - baseline_by_cell[cell]
        for cell in cells
        for tier in tiers
    }
    feasible_positive_deltas = [
        delta for delta in byte_deltas.values() if 0 < delta <= remaining_envelope
    ]
    byte_divisor = math.gcd(*feasible_positive_deltas) if feasible_positive_deltas else 1
    scaled_capacity = remaining_envelope // byte_divisor
    scaled_deltas = {
        key: delta // byte_divisor
        for key, delta in byte_deltas.items()
        if delta <= remaining_envelope
    }
    maximum_scaled_use = sum(
        max(scaled_deltas.get((cell, tier), 0) for tier in tiers) for cell in cells
    )
    exact_float_integer_max = 2**53
    enforce_byte_constraint = scaled_capacity < maximum_scaled_use
    if enforce_byte_constraint and (
        scaled_capacity > exact_float_integer_max
        or any(delta > exact_float_integer_max for delta in scaled_deltas.values())
    ):
        raise KnapsackValidationError(
            "exact byte constraint remains outside float64 integer range after "
            f"baseline/GCD normalization: capacity={scaled_capacity}, "
            f"divisor={byte_divisor}"
        )

    variable_count = len(cells) * len(tiers)
    objective = np.empty(variable_count, dtype=np.float64)
    variable_upper = np.ones(variable_count, dtype=np.float64)
    row_indices: list[int] = []
    column_indices: list[int] = []
    coefficients: list[float] = []
    for cell_index, cell in enumerate(cells):
        seen_equal_options: set[tuple[int, float]] = set()
        for tier_index, tier in enumerate(tiers):
            variable_index = cell_index * len(tiers) + tier_index
            equal_option_key = (costs[tier][cell], damages[(cell, tier)])
            if equal_option_key in seen_equal_options:
                variable_upper[variable_index] = 0.0
            else:
                seen_equal_options.add(equal_option_key)
            objective[variable_index] = damages[(cell, tier)]
            row_indices.append(cell_index)
            column_indices.append(variable_index)
            coefficients.append(1.0)
            delta = byte_deltas[(cell, tier)]
            if delta > remaining_envelope:
                variable_upper[variable_index] = 0.0
            elif enforce_byte_constraint:
                row_indices.append(len(cells))
                column_indices.append(variable_index)
                coefficients.append(float(scaled_deltas[(cell, tier)]))
    constraint_count = len(cells) + int(enforce_byte_constraint)
    matrix = coo_matrix(
        (coefficients, (row_indices, column_indices)),
        shape=(constraint_count, variable_count),
    ).tocsr()
    lower = np.ones(constraint_count)
    upper = np.ones(constraint_count)
    if enforce_byte_constraint:
        lower[-1] = -np.inf
        upper[-1] = float(scaled_capacity)
    solution = milp(
        c=objective,
        integrality=np.ones(variable_count, dtype=np.int8),
        bounds=Bounds(np.zeros(variable_count), variable_upper),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"presolve": True, "mip_rel_gap": 0.0},
    )
    if not solution.success or solution.x is None or int(solution.status) != 0:
        raise RuntimeError(
            f"exact knapsack solve failed: status={solution.status}, message={solution.message}"
        )
    if float(getattr(solution, "mip_gap", math.inf)) != 0.0:
        raise RuntimeError(
            "exact knapsack solve returned a nonzero MIP gap: "
            f"{getattr(solution, 'mip_gap', None)}"
        )
    rounded = np.rint(solution.x).astype(np.int8)
    if not np.allclose(solution.x, rounded, rtol=0.0, atol=1e-6):
        raise RuntimeError("exact knapsack solver returned a non-integral assignment")

    assignments: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(cells):
        offset = cell_index * len(tiers)
        selected = np.flatnonzero(rounded[offset : offset + len(tiers)])
        if len(selected) != 1:
            raise RuntimeError(
                f"exact knapsack solver selected {len(selected)} tiers for cell {cell!r}"
            )
        tier = tiers[int(selected[0])]
        assignments.append(
            {
                "cell_id": cell,
                "tier": tier,
                "bytes": costs[tier][cell],
                "damage": damages[(cell, tier)],
            }
        )
    assigned_bytes = sum(item["bytes"] for item in assignments)
    if assigned_bytes > envelope_bytes:
        raise RuntimeError(
            f"exact knapsack solver violated envelope: {assigned_bytes} > {envelope_bytes}"
        )
    total_damage = math.fsum(item["damage"] for item in assignments)

    output_path = _output_path(
        root, Path(output) if output is not None else None, default="knapsack/ASSIGNMENT.json", label="output"
    )
    receipt_path = _output_path(
        root, Path(receipt) if receipt is not None else None, default="knapsack/RECEIPT.json", label="receipt"
    )
    if output_path == receipt_path:
        raise KnapsackValidationError("output and receipt paths must differ")
    assignment_value = {
        "schema": "banana-smasher-knapsack-assignment-v1",
        "status": "PASS",
        "basis_sha256": intended_basis,
        "tiers": tiers,
        "objective": {"name": "min_total_damage", "total_damage": total_damage},
        "byte_accounting": {
            "assigned_bytes": assigned_bytes,
            "envelope_bytes": envelope_bytes,
            "slack_bytes": envelope_bytes - assigned_bytes,
        },
        "assignments": assignments,
    }
    assignment_payload = _canonical_json(assignment_value)
    assignment_sha = _sha256(assignment_payload)
    assignment_bytes = len(assignment_payload)
    receipt_value = {
        "schema": "banana-smasher-knapsack-receipt-v1",
        "status": "PASS",
        "run_root": str(root),
        "basis_sha256": intended_basis,
        "tiers": tiers,
        "cell_count": len(cells),
        "objective": assignment_value["objective"],
        "byte_accounting": assignment_value["byte_accounting"],
        "run_manifest": {
            "path": "MANIFEST.json",
            "sha256": _sha256(manifest_bytes),
            "bytes": len(manifest_bytes),
        },
        "anchor_manifests": [
            {
                "tier": tier,
                "path": anchor_sources[tier].relative_path,
                "sha256": anchor_sources[tier].sha256,
                "bytes": anchor_sources[tier].byte_count,
                "producer_command": anchor_sources[tier].producer_command,
            }
            for tier in tiers
        ],
        "damage_rows": {
            "path": damage_source.relative_path,
            "sha256": damage_source.sha256,
            "bytes": damage_source.byte_count,
            "producer_command": damage_source.producer_command,
        },
        "assignment": {
            "path": output_path.relative_to(root).as_posix(),
            "sha256": assignment_sha,
            "bytes": assignment_bytes,
        },
        "solver": {
            "backend": "scipy.optimize.milp/HiGHS",
            "status": int(solution.status),
            "message": str(solution.message),
            "mip_gap": float(getattr(solution, "mip_gap", 0.0)),
            "equal_option_tie_breaker": "first_manifest_tier",
            "byte_normalization": {
                "baseline_bytes": minimum_required_bytes,
                "remaining_envelope_bytes": remaining_envelope,
                "gcd_divisor": byte_divisor,
                "scaled_capacity": scaled_capacity,
                "constraint_required": enforce_byte_constraint,
            },
        },
    }
    # Validate both immutable destinations before publishing either half of the
    # assignment/receipt pair. The receipt is fully staged from validated input
    # bytes and the canonical assignment payload before any PASS file appears.
    _preflight_write_once(output_path, assignment_value)
    _preflight_write_once(receipt_path, receipt_value)
    written_assignment_sha, written_assignment_bytes = _write_once(
        output_path, assignment_value
    )
    if (written_assignment_sha, written_assignment_bytes) != (
        assignment_sha,
        assignment_bytes,
    ):
        raise RuntimeError("canonical assignment changed during pair publication")
    receipt_sha, receipt_bytes = _write_once(receipt_path, receipt_value)
    return {
        "status": "PASS",
        "command": "knapsack",
        "run_root": str(root),
        "tiers": tiers,
        "cell_count": len(cells),
        "objective": assignment_value["objective"],
        "byte_accounting": assignment_value["byte_accounting"],
        "assignment": {
            "path": str(output_path),
            "sha256": assignment_sha,
            "bytes": assignment_bytes,
        },
        "receipt": {
            "path": str(receipt_path),
            "sha256": receipt_sha,
            "bytes": receipt_bytes,
        },
    }
