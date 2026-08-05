from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

BANK_MANIFEST_SCHEMA = "banana-smasher-bank-manifest-v1"
BANK_ROLES = frozenset(
    {
        "train_balanced64",
        "train512",
        "holdout_balanced64",
        "holdout512",
    }
)
_REQUIRED_IDENTITIES = frozenset({"corpus", "tokenizer", "teacher", "scorer"})
_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "bank_id",
        "role",
        "parent_corpus",
        "windows",
        "class_counts",
        "identities",
        "dataset_fields",
        "split_lineage",
        "creation",
        "relationships",
        "content_hashes",
    }
)
_HEX = frozenset("0123456789abcdef")


class AnchorEvaluationError(ValueError):
    """A fail-closed anchor evaluation contract violation."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and set(value) <= _HEX
    )


def _window_key(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise AnchorEvaluationError("window id must be a string or integer")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _validate_identity(name: str, value: Any) -> None:
    if not isinstance(value, Mapping):
        raise AnchorEvaluationError(f"identity {name!r} must be an object")
    status = value.get("status")
    if status == "resolved":
        if set(value) != {"status", "sha256", "uri"}:
            raise AnchorEvaluationError(
                f"resolved identity {name!r} must contain exactly status, sha256 and uri"
            )
        if not _is_sha256(value.get("sha256")):
            raise AnchorEvaluationError(
                f"resolved identity {name!r} requires lowercase SHA-256"
            )
        if not isinstance(value.get("uri"), str) or not value["uri"]:
            raise AnchorEvaluationError(f"resolved identity {name!r} requires uri")
    elif status == "unresolved":
        if set(value) != {"status", "reason"}:
            raise AnchorEvaluationError(
                f"unresolved identity {name!r} must contain exactly status and reason"
            )
        if not isinstance(value.get("reason"), str) or not value["reason"]:
            raise AnchorEvaluationError(
                f"unresolved identity {name!r} requires an explicit reason"
            )
        if "sha256" in value or "uri" in value:
            raise AnchorEvaluationError(
                f"unresolved identity {name!r} cannot imply sha256 or uri"
            )
    else:
        raise AnchorEvaluationError(
            f"identity {name!r} status must be 'resolved' or 'unresolved'"
        )


def _content_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    windows = manifest["windows"]
    payload = {key: value for key, value in manifest.items() if key != "content_hashes"}
    return {
        "membership_sha256": _sha256_bytes(
            _canonical_bytes([window["id"] for window in windows])
        ),
        "class_map_sha256": _sha256_bytes(_canonical_bytes(windows)),
        "manifest_payload_sha256": _sha256_bytes(_canonical_bytes(payload)),
    }


def build_bank_manifest(
    *,
    bank_id: str,
    role: str,
    windows: Sequence[Mapping[str, Any]],
    parent_corpus: Mapping[str, Any],
    identities: Mapping[str, Mapping[str, Any]],
    split_lineage: Mapping[str, Any],
    creation: Mapping[str, Any],
    relationships: Sequence[Mapping[str, Any]],
    dataset_fields: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build and validate a deterministic bank manifest.

    Window order is semantic and contributes to every content hash. Identity
    objects may be explicitly unresolved, but they may never contain guessed
    hashes or locations.
    """

    normalized_windows = [
        {"id": window.get("id"), "class": window.get("class")} for window in windows
    ]
    counts = Counter(window["class"] for window in normalized_windows)
    manifest: dict[str, Any] = {
        "schema": BANK_MANIFEST_SCHEMA,
        "bank_id": bank_id,
        "role": role,
        "parent_corpus": dict(parent_corpus),
        "windows": normalized_windows,
        "class_counts": dict(sorted(counts.items())),
        "identities": {key: dict(value) for key, value in sorted(identities.items())},
        "dataset_fields": dict(
            dataset_fields or {"window_id": "window_id", "class": "class"}
        ),
        "split_lineage": dict(split_lineage),
        "creation": dict(creation),
        "relationships": [dict(value) for value in relationships],
    }
    manifest["content_hashes"] = _content_hashes(manifest)
    validate_bank_manifest(manifest)
    return manifest


def validate_bank_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise AnchorEvaluationError("manifest must be an object")
    if set(manifest) != _MANIFEST_FIELDS:
        raise AnchorEvaluationError(
            "manifest fields mismatch: "
            f"missing={sorted(_MANIFEST_FIELDS - set(manifest))}, "
            f"unexpected={sorted(set(manifest) - _MANIFEST_FIELDS)}"
        )
    if manifest.get("schema") != BANK_MANIFEST_SCHEMA:
        raise AnchorEvaluationError(
            f"schema must be {BANK_MANIFEST_SCHEMA!r}, got {manifest.get('schema')!r}"
        )
    if not isinstance(manifest.get("bank_id"), str) or not manifest["bank_id"]:
        raise AnchorEvaluationError("bank_id must be a non-empty string")
    if manifest.get("role") not in BANK_ROLES:
        raise AnchorEvaluationError(
            f"role must be one of {sorted(BANK_ROLES)}, got {manifest.get('role')!r}"
        )

    windows = manifest.get("windows")
    if not isinstance(windows, list) or not windows:
        raise AnchorEvaluationError("windows must be a non-empty ordered list")
    seen: set[str] = set()
    actual_counts: Counter[str] = Counter()
    for index, window in enumerate(windows):
        if not isinstance(window, Mapping) or set(window) != {"id", "class"}:
            raise AnchorEvaluationError(
                f"windows[{index}] must contain exactly id and class"
            )
        key = _window_key(window["id"])
        if key in seen:
            raise AnchorEvaluationError(f"duplicate window id {window['id']!r}")
        seen.add(key)
        label = window["class"]
        if not isinstance(label, str) or not label:
            raise AnchorEvaluationError(f"windows[{index}].class must be non-empty")
        actual_counts[label] += 1

    declared_counts = manifest.get("class_counts")
    expected_counts = dict(sorted(actual_counts.items()))
    if declared_counts != expected_counts:
        raise AnchorEvaluationError(
            f"class_counts mismatch: expected {expected_counts}, got {declared_counts}"
        )

    _validate_identity("parent_corpus", manifest.get("parent_corpus"))
    identities = manifest.get("identities")
    if not isinstance(identities, Mapping):
        raise AnchorEvaluationError("identities must be an object")
    if set(identities) != _REQUIRED_IDENTITIES:
        raise AnchorEvaluationError(
            "identities fields mismatch: "
            f"missing={sorted(_REQUIRED_IDENTITIES - set(identities))}, "
            f"unexpected={sorted(set(identities) - _REQUIRED_IDENTITIES)}"
        )
    for name, value in identities.items():
        _validate_identity(str(name), value)

    fields = manifest.get("dataset_fields")
    if not isinstance(fields, Mapping) or set(fields) != {"window_id", "class"}:
        raise AnchorEvaluationError(
            "dataset_fields must contain exactly window_id and class"
        )
    if any(not isinstance(value, str) or not value for value in fields.values()):
        raise AnchorEvaluationError("dataset_fields values must be non-empty strings")

    lineage = manifest.get("split_lineage")
    if not isinstance(lineage, Mapping) or not isinstance(lineage.get("split"), str):
        raise AnchorEvaluationError("split_lineage requires a string split")
    creation = manifest.get("creation")
    if (
        not isinstance(creation, Mapping)
        or not isinstance(creation.get("method"), str)
        or not isinstance(creation.get("config"), Mapping)
    ):
        raise AnchorEvaluationError("creation requires method and config")
    relationships = manifest.get("relationships")
    if not isinstance(relationships, list):
        raise AnchorEvaluationError("relationships must be a list")
    for index, relationship in enumerate(relationships):
        if (
            not isinstance(relationship, Mapping)
            or not isinstance(relationship.get("bank_id"), str)
            or not relationship["bank_id"]
            or not isinstance(relationship.get("relation"), str)
            or not relationship["relation"]
        ):
            raise AnchorEvaluationError(
                f"relationships[{index}] requires non-empty bank_id and relation"
            )

    expected_hashes = _content_hashes(manifest)
    if manifest.get("content_hashes") != expected_hashes:
        raise AnchorEvaluationError(
            "content hashes mismatch; expected "
            f"{expected_hashes}, got {manifest.get('content_hashes')}"
        )
    unresolved = sorted(
        name for name, value in identities.items() if value.get("status") == "unresolved"
    )
    if manifest["parent_corpus"].get("status") == "unresolved":
        unresolved.insert(0, "parent_corpus")
    return {
        "schema": "banana-smasher-bank-validation-receipt-v1",
        "status": "PASS",
        "bank_id": manifest["bank_id"],
        "role": manifest["role"],
        "window_count": len(windows),
        "class_counts": expected_counts,
        "content_hashes": expected_hashes,
        "unresolved_provenance": unresolved,
    }


def resolve_bank_identities(
    manifest: Mapping[str, Any], updates: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Return a new manifest with exact identity updates and fresh hashes."""
    validate_bank_manifest(manifest)
    if not isinstance(updates, Mapping) or not updates:
        raise AnchorEvaluationError("identity updates must be a non-empty object")
    allowed = {"corpus", "tokenizer", "teacher", "scorer", "parent_corpus"}
    unknown = sorted(set(updates) - allowed)
    if unknown:
        raise AnchorEvaluationError(f"unknown identity update keys: {unknown}")

    identities = json.loads(json.dumps(manifest["identities"]))
    parent_corpus = json.loads(json.dumps(manifest["parent_corpus"]))
    for name, identity in updates.items():
        _validate_identity(f"identity update {name}", identity)
        if identity["status"] != "resolved":
            raise AnchorEvaluationError(
                f"identity update {name} must be resolved with exact sha256 and uri"
            )
        if name == "parent_corpus":
            parent_corpus = dict(identity)
        else:
            identities[name] = dict(identity)

    return build_bank_manifest(
        bank_id=manifest["bank_id"],
        role=manifest["role"],
        windows=manifest["windows"],
        parent_corpus=parent_corpus,
        identities=identities,
        split_lineage=manifest["split_lineage"],
        creation=manifest["creation"],
        relationships=manifest["relationships"],
        dataset_fields=manifest["dataset_fields"],
    )


def _parse_jsonl(payload: bytes, path: Path) -> list[dict[str, Any]]:
    try:
        text = payload.decode()
    except UnicodeDecodeError as exc:
        raise AnchorEvaluationError(f"{path}: producer dataset must be UTF-8") from exc
    if text.lstrip().startswith("["):
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AnchorEvaluationError(f"{path}: invalid JSON array: {exc}") from exc
        if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
            raise AnchorEvaluationError(f"{path}: JSON array must contain only objects")
        return value
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnchorEvaluationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise AnchorEvaluationError(f"{path}:{line_number}: row must be an object")
        rows.append(value)
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        payload = path.read_bytes()
    except FileNotFoundError as exc:
        raise AnchorEvaluationError(
            f"missing producer dataset {path}; materialize or import the declared source"
        ) from exc
    return _parse_jsonl(payload, path)


def _index_parent(
    manifest: Mapping[str, Any], parent_path: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    validate_bank_manifest(manifest)
    identity = manifest["parent_corpus"]
    if identity["status"] != "resolved":
        raise AnchorEvaluationError(
            "parent_corpus is unresolved; provide a resolved manifest before materialization"
        )
    try:
        payload = parent_path.read_bytes()
    except FileNotFoundError as exc:
        raise AnchorEvaluationError(f"missing declared parent dataset {parent_path}") from exc
    actual = _sha256_bytes(payload)
    if actual != identity["sha256"]:
        raise AnchorEvaluationError(
            f"parent corpus SHA-256 mismatch: expected {identity['sha256']}, got {actual}"
        )
    rows = _parse_jsonl(payload, parent_path)
    id_field = manifest["dataset_fields"]["window_id"]
    class_field = manifest["dataset_fields"]["class"]
    index: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, 1):
        if id_field not in row:
            raise AnchorEvaluationError(
                f"parent row {row_number} requires window-id field {id_field!r}"
            )
        key = _window_key(row[id_field])
        if key in index:
            raise AnchorEvaluationError(f"duplicate parent window id {row[id_field]!r}")
        index[key] = row
    return rows, index


def create_balanced_subset(
    parent_manifest: Mapping[str, Any],
    parent_path: Path | str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if parent_manifest.get("role") != "train512":
        raise AnchorEvaluationError("balanced training subsets require a train512 parent")
    parent_path = Path(parent_path)
    rows, _ = _index_parent(parent_manifest, parent_path)
    if config.get("role") != "train_balanced64":
        raise AnchorEvaluationError("balanced subset role must be train_balanced64")
    quotas = config.get("quotas")
    if not isinstance(quotas, Mapping) or not quotas:
        raise AnchorEvaluationError("selection config requires non-empty quotas")
    if any(
        not isinstance(label, str)
        or not isinstance(quota, int)
        or isinstance(quota, bool)
        or quota <= 0
        for label, quota in quotas.items()
    ):
        raise AnchorEvaluationError("selection quotas require positive integer values")
    seed = config.get("seed")
    if not isinstance(seed, (str, int)) or isinstance(seed, bool):
        raise AnchorEvaluationError("selection seed must be a string or integer")

    id_field = parent_manifest["dataset_fields"]["window_id"]
    class_field = parent_manifest["dataset_fields"]["class"]
    ranking_field = config.get("ranking_field")
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[class_field]].append(row)

    selected: list[dict[str, Any]] = []
    for label, quota in quotas.items():
        candidates = grouped.get(label, [])
        if len(candidates) < quota:
            raise AnchorEvaluationError(
                f"class {label!r} has {len(candidates)} rows but quota is {quota}"
            )

        def key(row: Mapping[str, Any]) -> tuple[Any, str]:
            tie = _sha256_bytes(
                _canonical_bytes({"seed": seed, "window_id": row[id_field]})
            )
            if ranking_field is None:
                return (0, tie)
            rank = row.get(ranking_field)
            if not isinstance(rank, (int, float)) or isinstance(rank, bool) or not math.isfinite(rank):
                raise AnchorEvaluationError(
                    f"ranking field {ranking_field!r} must be finite numeric for every row"
                )
            return (rank, tie)

        chosen = sorted(candidates, key=key)[:quota]
        selected.extend({"id": row[id_field], "class": label} for row in chosen)

    creation_config = dict(config)
    return build_bank_manifest(
        bank_id=str(config.get("bank_id", "")),
        role="train_balanced64",
        windows=selected,
        parent_corpus=parent_manifest["parent_corpus"],
        identities=parent_manifest["identities"],
        dataset_fields=parent_manifest["dataset_fields"],
        split_lineage={
            "split": parent_manifest["split_lineage"]["split"],
            "parent_bank_id": parent_manifest["bank_id"],
        },
        creation={"method": "deterministic_balanced_subset", "config": creation_config},
        relationships=[
            {
                "bank_id": parent_manifest["bank_id"],
                "relation": "ordered_subset_of",
            }
        ],
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def materialize_bank(
    manifest: Mapping[str, Any],
    parent_path: Path | str,
    output_path: Path | str,
    *,
    disjoint_manifests: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    receipt = validate_bank_manifest(manifest)
    parent_path = Path(parent_path)
    output_path = Path(output_path)
    _, parent = _index_parent(manifest, parent_path)

    requested = {_window_key(window["id"]): window for window in manifest["windows"]}
    for other in disjoint_manifests:
        validate_bank_manifest(other)
        overlap = sorted(
            (window["id"] for window in other["windows"] if _window_key(window["id"]) in requested),
            key=str,
        )
        if overlap:
            raise AnchorEvaluationError(
                f"requested disjointness failed; overlapping window ids: {overlap}"
            )

    id_field = manifest["dataset_fields"]["window_id"]
    class_field = manifest["dataset_fields"]["class"]
    selected: list[dict[str, Any]] = []
    for window in manifest["windows"]:
        row = parent.get(_window_key(window["id"]))
        if row is None:
            raise AnchorEvaluationError(
                f"window {window['id']!r} is not a member of the declared parent"
            )
        if class_field in row and row[class_field] != window["class"]:
            raise AnchorEvaluationError(
                f"class-map mismatch for window {window['id']!r}: "
                f"manifest={window['class']!r}, parent={row[class_field]!r}"
            )
        selected_row = dict(row)
        selected_row.setdefault(class_field, window["class"])
        selected.append(selected_row)
    payload = b"".join(_canonical_bytes(row) for row in selected)
    resumed = False
    if output_path.exists():
        if output_path.read_bytes() != payload:
            raise AnchorEvaluationError(
                f"refusing to overwrite non-identical materialized bank {output_path}"
            )
        resumed = True
    else:
        _atomic_write(output_path, payload)
    return {
        "schema": "banana-smasher-bank-materialization-receipt-v1",
        "status": "PASS",
        "bank_id": manifest["bank_id"],
        "role": manifest["role"],
        "expected_count": len(manifest["windows"]),
        "materialized_count": len(selected),
        "unique_count": len({_window_key(row[id_field]) for row in selected}),
        "class_counts": receipt["class_counts"],
        "output_sha256": _sha256_bytes(payload),
        "resumed": resumed,
    }


def _producer_rows(path: Path, label: str) -> tuple[list[dict[str, Any]], str, bytes]:
    if not path.is_file():
        raise AnchorEvaluationError(
            f"missing {label} producer {path}; materialize or import the declared "
            f"{label} rows for this exact bank"
        )
    payload = path.read_bytes()
    return _parse_jsonl(payload, path), _sha256_bytes(payload), payload


def _producer_index(
    rows: Sequence[Mapping[str, Any]], label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row_number, row in enumerate(rows, 1):
        if "window_id" not in row:
            raise AnchorEvaluationError(
                f"{label} producer row {row_number} requires window_id"
            )
        key = _window_key(row["window_id"])
        if key in result:
            raise AnchorEvaluationError(
                f"duplicate {label} producer window id {row['window_id']!r}"
            )
        result[key] = row
    return result


def _probabilities(
    row: Mapping[str, Any], label: str, window_id: Any
) -> list[list[float]]:
    has_logits = "logits" in row
    has_probabilities = "probabilities" in row
    if has_logits == has_probabilities:
        raise AnchorEvaluationError(
            f"{label} window {window_id!r} requires exactly one of logits or probabilities"
        )
    value = row["logits"] if has_logits else row["probabilities"]
    if not isinstance(value, list) or not value:
        raise AnchorEvaluationError(
            f"{label} window {window_id!r} requires a non-empty vector or position matrix"
        )
    nested = isinstance(value[0], list)
    if any(isinstance(item, list) != nested for item in value):
        raise AnchorEvaluationError(
            f"{label} window {window_id!r} cannot mix vectors and position rows"
        )
    vectors = value if nested else [value]
    probabilities: list[list[float]] = []
    vocabulary_size: int | None = None
    for position, vector in enumerate(vectors):
        if len(vector) < 2:
            raise AnchorEvaluationError(
                f"{label} window {window_id!r} position {position} requires at least two values"
            )
        if vocabulary_size is None:
            vocabulary_size = len(vector)
        elif len(vector) != vocabulary_size:
            raise AnchorEvaluationError(
                f"{label} window {window_id!r} position matrix must be rectangular"
            )
        values: list[float] = []
        for item in vector:
            if not isinstance(item, (int, float)) or isinstance(item, bool):
                raise AnchorEvaluationError(
                    f"{label} window {window_id!r} position matrix must be numeric"
                )
            numeric = float(item)
            if not math.isfinite(numeric):
                raise AnchorEvaluationError(
                    f"{label} window {window_id!r} position matrix must be finite"
                )
            values.append(numeric)
        if has_logits:
            maximum = max(values)
            exponents = [math.exp(item - maximum) for item in values]
            total = math.fsum(exponents)
            probabilities.append([item / total for item in exponents])
            continue
        if any(item < 0 for item in values):
            raise AnchorEvaluationError(
                f"{label} window {window_id!r} probabilities cannot be negative"
            )
        total = math.fsum(values)
        if not math.isfinite(total) or total <= 0:
            raise AnchorEvaluationError(
                f"{label} window {window_id!r} probabilities require positive mass"
            )
        probabilities.append([item / total for item in values])
    return probabilities


def _kld(teacher: Sequence[float], candidate: Sequence[float], window_id: Any) -> float:
    if len(teacher) != len(candidate):
        raise AnchorEvaluationError(
            f"producer dimension mismatch for window {window_id!r}: "
            f"teacher={len(teacher)}, candidate={len(candidate)}"
        )
    terms: list[float] = []
    for teacher_probability, candidate_probability in zip(teacher, candidate, strict=True):
        if teacher_probability == 0:
            continue
        if candidate_probability <= 0:
            raise AnchorEvaluationError(
                f"candidate probability is zero where teacher is positive for window {window_id!r}"
            )
        terms.append(
            teacher_probability
            * math.log(teacher_probability / candidate_probability)
        )
    value = math.fsum(terms)
    if value < -1e-12 or not math.isfinite(value):
        raise AnchorEvaluationError(f"invalid KLD {value!r} for window {window_id!r}")
    return max(0.0, value)


def _mean_position_kld(
    teacher: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
    window_id: Any,
) -> float:
    if len(teacher) != len(candidate):
        raise AnchorEvaluationError(
            f"producer position-count mismatch for window {window_id!r}: "
            f"teacher={len(teacher)}, candidate={len(candidate)}"
        )
    values = [
        _kld(teacher_row, candidate_row, window_id)
        for teacher_row, candidate_row in zip(teacher, candidate, strict=True)
    ]
    return math.fsum(values) / len(values)


def score_bank(
    manifest: Mapping[str, Any],
    teacher_path: Path | str,
    candidate_path: Path | str,
    output_path: Path | str,
    *,
    candidate_id: str,
    candidate_identity: Mapping[str, Any],
    teacher_identity: Mapping[str, Any],
    basis_sha256: str,
) -> dict[str, Any]:
    """Score exact teacher/candidate producer rows with idempotent resume."""

    validate_bank_manifest(manifest)
    unresolved_required = [
        name
        for name in ("corpus", "tokenizer", "scorer")
        if manifest["identities"][name]["status"] != "resolved"
    ]
    if manifest["parent_corpus"]["status"] != "resolved":
        unresolved_required.append("parent_corpus")
    if unresolved_required:
        raise AnchorEvaluationError(
            "resolve bank identities before scoring: "
            + ", ".join(unresolved_required)
        )
    if not isinstance(candidate_id, str) or not candidate_id:
        raise AnchorEvaluationError("candidate_id must be a non-empty string")
    _validate_identity("candidate", candidate_identity)
    _validate_identity("teacher_producer", teacher_identity)
    if candidate_identity["status"] != "resolved":
        raise AnchorEvaluationError("candidate identity must be resolved before scoring")
    if teacher_identity["status"] != "resolved":
        raise AnchorEvaluationError("teacher identity must be resolved before scoring")
    if not _is_sha256(basis_sha256):
        raise AnchorEvaluationError("basis_sha256 must be a lowercase SHA-256")

    teacher_path = Path(teacher_path)
    candidate_path = Path(candidate_path)
    output_path = Path(output_path)
    teacher_rows, teacher_producer_sha, _ = _producer_rows(teacher_path, "teacher")
    candidate_rows, candidate_producer_sha, _ = _producer_rows(
        candidate_path, "candidate"
    )
    declared_teacher = manifest["identities"]["teacher"]
    if (
        declared_teacher["status"] == "resolved"
        and declared_teacher["sha256"] != teacher_identity["sha256"]
    ):
        raise AnchorEvaluationError(
            "teacher identity differs from the bank manifest; create a new bank manifest"
        )
    teacher = _producer_index(teacher_rows, "teacher")
    candidate = _producer_index(candidate_rows, "candidate")

    expected_keys = [_window_key(window["id"]) for window in manifest["windows"]]
    unexpected_teacher = sorted(set(teacher) - set(expected_keys))
    unexpected_candidate = sorted(set(candidate) - set(expected_keys))
    if unexpected_teacher or unexpected_candidate:
        raise AnchorEvaluationError(
            "producer rows contain unexpected window ids; "
            f"teacher={unexpected_teacher}, candidate={unexpected_candidate}"
        )
    missing_teacher = [
        window["id"]
        for window, key in zip(manifest["windows"], expected_keys, strict=True)
        if key not in teacher
    ]
    missing_candidate = [
        window["id"]
        for window, key in zip(manifest["windows"], expected_keys, strict=True)
        if key not in candidate
    ]
    if missing_teacher or missing_candidate:
        raise AnchorEvaluationError(
            "producer coverage incomplete; materialize missing exact rows: "
            f"teacher={missing_teacher}, candidate={missing_candidate}"
        )

    bindings = {
        "basis_sha256": basis_sha256,
        "bank_content_sha256": manifest["content_hashes"]["manifest_payload_sha256"],
        "teacher_sha256": teacher_identity["sha256"],
        "teacher_producer_sha256": teacher_producer_sha,
        "candidate_sha256": candidate_identity["sha256"],
        "candidate_producer_sha256": candidate_producer_sha,
        "scorer_sha256": manifest["identities"]["scorer"]["sha256"],
    }
    existing: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        for row in _read_jsonl(output_path):
            if row.get("schema") != "banana-smasher-anchor-window-score-v1":
                raise AnchorEvaluationError("resume output has an unsupported score schema")
            key = _window_key(row.get("window_id"))
            if key in existing:
                raise AnchorEvaluationError(
                    f"resume output has duplicate window id {row.get('window_id')!r}"
                )
            if row.get("bindings") != bindings or row.get("candidate_id") != candidate_id:
                raise AnchorEvaluationError(
                    "resume output bindings differ from the requested same-work score"
                )
            existing[key] = row

    rows: list[dict[str, Any]] = []
    new_rows = 0
    for window, key in zip(manifest["windows"], expected_keys, strict=True):
        if key in existing:
            row = existing[key]
            if row.get("class") != window["class"]:
                raise AnchorEvaluationError(
                    f"resume class mismatch for window {window['id']!r}"
                )
            value = row.get("kld")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise AnchorEvaluationError(
                    f"resume KLD is invalid for window {window['id']!r}"
                )
            rows.append(row)
            continue
        teacher_probabilities = _probabilities(
            teacher[key], "teacher", window["id"]
        )
        candidate_probabilities = _probabilities(
            candidate[key], "candidate", window["id"]
        )
        rows.append(
            {
                "schema": "banana-smasher-anchor-window-score-v1",
                "bank_id": manifest["bank_id"],
                "bank_role": manifest["role"],
                "candidate_id": candidate_id,
                "window_id": window["id"],
                "class": window["class"],
                "position_count": len(teacher_probabilities),
                "kld": _mean_position_kld(
                    teacher_probabilities, candidate_probabilities, window["id"]
                ),
                "bindings": bindings,
            }
        )
        new_rows += 1
    payload = b"".join(_canonical_bytes(row) for row in rows)
    _atomic_write(output_path, payload)
    return {
        "schema": "banana-smasher-anchor-score-receipt-v1",
        "status": "PASS",
        "bank_id": manifest["bank_id"],
        "bank_role": manifest["role"],
        "candidate_id": candidate_id,
        "coverage": f"{len(rows)}/{len(manifest['windows'])}",
        "new_rows": new_rows,
        "resumed_rows": len(rows) - new_rows,
        "raw_sha256": _sha256_bytes(payload),
        "bindings": bindings,
    }


def aggregate_scores(
    manifest: Mapping[str, Any],
    raw_path: Path | str,
    output_path: Path | str,
    *,
    candidate_id: str,
    calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_bank_manifest(manifest)
    raw_path = Path(raw_path)
    rows = _read_jsonl(raw_path)
    expected = {_window_key(window["id"]): window for window in manifest["windows"]}
    by_key: dict[str, Mapping[str, Any]] = {}
    by_class: defaultdict[str, list[float]] = defaultdict(list)
    bindings: Mapping[str, Any] | None = None
    for row in rows:
        key = _window_key(row.get("window_id"))
        if key not in expected:
            raise AnchorEvaluationError(
                f"raw score contains unexpected window id {row.get('window_id')!r}"
            )
        if key in by_key:
            raise AnchorEvaluationError(
                f"raw score contains duplicate window id {row.get('window_id')!r}"
            )
        if row.get("candidate_id") != candidate_id:
            raise AnchorEvaluationError("raw score candidate_id mismatch")
        if row.get("class") != expected[key]["class"]:
            raise AnchorEvaluationError(
                f"raw score class mismatch for window {row.get('window_id')!r}"
            )
        value = row.get("kld")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise AnchorEvaluationError(
                f"raw score KLD is invalid for window {row.get('window_id')!r}"
            )
        if bindings is None:
            bindings = row.get("bindings")
        elif bindings != row.get("bindings"):
            raise AnchorEvaluationError("raw score rows have mixed provenance bindings")
        by_key[key] = row
        by_class[row["class"]].append(float(value))
    missing = [window["id"] for key, window in expected.items() if key not in by_key]
    if missing:
        raise AnchorEvaluationError(
            f"raw score coverage incomplete; missing window ids: {missing}"
        )
    ordered_values = [float(by_key[_window_key(window["id"])]["kld"]) for window in manifest["windows"]]
    per_class = {
        label: math.fsum(values) / len(values)
        for label, values in sorted(by_class.items())
    }
    result: dict[str, Any] = {
        "schema": "banana-smasher-anchor-aggregate-v1",
        "status": "PASS",
        "bank_id": manifest["bank_id"],
        "bank_role": manifest["role"],
        "candidate_id": candidate_id,
        "coverage": f"{len(rows)}/{len(manifest['windows'])}",
        "evaluation_contract": {
            "dimensions": manifest["creation"]["config"].get("dimensions"),
            "tier_menus": manifest["creation"]["config"].get("tier_menus"),
            "basis_sha256": (bindings or {}).get("basis_sha256"),
            "candidate_sha256": (bindings or {}).get("candidate_sha256"),
            "teacher_sha256": (bindings or {}).get("teacher_sha256"),
            "scorer_sha256": (bindings or {}).get("scorer_sha256"),
        },
        "measured": {
            "label": "measured_on_bank",
            "global_mean_kld": math.fsum(ordered_values) / len(ordered_values),
            "per_class_mean_kld": per_class,
        },
        "bindings": dict(bindings or {}),
        "raw_sha256": _sha256_bytes(raw_path.read_bytes()),
    }
    if calibration is not None:
        if calibration.get("schema") != "banana-smasher-anchor-calibration-v1":
            raise AnchorEvaluationError("unsupported calibration schema")
        _validate_identity("calibration source", calibration.get("source"))
        factors = calibration.get("correction_factors")
        parent_counts = calibration.get("parent_class_counts")
        classes = set(per_class)
        if not isinstance(factors, Mapping) or set(factors) != classes:
            raise AnchorEvaluationError(
                "calibration correction_factors must exactly match measured classes"
            )
        if not isinstance(parent_counts, Mapping) or set(parent_counts) != classes:
            raise AnchorEvaluationError(
                "calibration parent_class_counts must exactly match measured classes"
            )
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
            for value in factors.values()
        ):
            raise AnchorEvaluationError("calibration factors must be positive and finite")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in parent_counts.values()
        ):
            raise AnchorEvaluationError("parent class counts must be positive integers")
        estimated = {
            label: per_class[label] * float(factors[label]) for label in sorted(classes)
        }
        total = sum(parent_counts.values())
        result["parent_estimates"] = {
            "label": "estimated_parent_not_measured",
            "per_class_mean_kld": estimated,
            "global_mean_kld": math.fsum(
                estimated[label] * parent_counts[label] for label in classes
            )
            / total,
            "parent_class_counts": dict(sorted(parent_counts.items())),
            "calibration_sha256": _sha256_bytes(_canonical_bytes(calibration)),
            "calibration_source": dict(calibration["source"]),
        }
    _atomic_write(Path(output_path), _canonical_bytes(result))
    return result


def _relative_error_pct(measured: float, reference: float) -> float:
    if reference == 0:
        if measured == 0:
            return 0.0
        raise AnchorEvaluationError("relative error is undefined for a zero parent value")
    return (measured - reference) / reference * 100.0


def compare_training_rails(
    panel: Mapping[str, Any],
    parent: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    if panel.get("schema") != "banana-smasher-anchor-aggregate-v1":
        raise AnchorEvaluationError("panel aggregate schema mismatch")
    if parent.get("schema") != "banana-smasher-anchor-aggregate-v1":
        raise AnchorEvaluationError("parent aggregate schema mismatch")
    if panel.get("bank_role") != "train_balanced64":
        raise AnchorEvaluationError("panel aggregate must use train_balanced64")
    if parent.get("bank_role") != "train512":
        raise AnchorEvaluationError("parent aggregate must use train512")
    if panel.get("candidate_id") != parent.get("candidate_id"):
        raise AnchorEvaluationError("training rail candidate identities differ")
    if panel.get("evaluation_contract") != parent.get("evaluation_contract"):
        raise AnchorEvaluationError(
            "training rail comparison requires the same dimensions and tier menus"
        )
    global_threshold = thresholds.get("max_abs_global_relative_pct")
    class_threshold = thresholds.get("max_abs_class_relative_pct")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
        for value in (global_threshold, class_threshold)
    ):
        raise AnchorEvaluationError("comparison thresholds must be finite and non-negative")
    panel_classes = panel["measured"]["per_class_mean_kld"]
    parent_classes = parent["measured"]["per_class_mean_kld"]
    if set(panel_classes) != set(parent_classes):
        raise AnchorEvaluationError("training rail class sets differ")
    per_class: dict[str, Any] = {}
    class_pass = True
    for label in sorted(panel_classes):
        absolute = float(panel_classes[label]) - float(parent_classes[label])
        relative = _relative_error_pct(
            float(panel_classes[label]), float(parent_classes[label])
        )
        passed = abs(relative) <= float(class_threshold)
        class_pass &= passed
        per_class[label] = {
            "absolute_error": absolute,
            "relative_error_pct": relative,
            "passed": passed,
        }
    panel_global = float(panel["measured"]["global_mean_kld"])
    parent_global = float(parent["measured"]["global_mean_kld"])
    global_relative = _relative_error_pct(panel_global, parent_global)
    global_pass = abs(global_relative) <= float(global_threshold)
    return {
        "schema": "banana-smasher-anchor-rail-comparison-v1",
        "status": "PASS",
        "candidate_id": panel["candidate_id"],
        "panel_bank_id": panel["bank_id"],
        "parent_bank_id": parent["bank_id"],
        "thresholds": dict(thresholds),
        "evaluation_contract": dict(panel["evaluation_contract"]),
        "global": {
            "absolute_error": panel_global - parent_global,
            "relative_error_pct": global_relative,
            "passed": global_pass,
        },
        "per_class": per_class,
        "policy_result": (
            "retain_panel" if global_pass and class_pass else "escalate_to_full_parent"
        ),
    }


def emit_solver_row(
    manifest: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    output_path: Path | str,
    *,
    diagnostic_override: bool = False,
) -> dict[str, Any]:
    validate_bank_manifest(manifest)
    if aggregate.get("schema") != "banana-smasher-anchor-aggregate-v1":
        raise AnchorEvaluationError("aggregate schema mismatch")
    if (
        aggregate.get("bank_id") != manifest["bank_id"]
        or aggregate.get("bank_role") != manifest["role"]
    ):
        raise AnchorEvaluationError("aggregate does not match the bank manifest")
    holdout = manifest["role"].startswith("holdout")
    if holdout and not diagnostic_override:
        raise AnchorEvaluationError(
            "holdout-role outputs cannot be used as solver inputs; "
            "use diagnostic_override only for a clearly labeled non-fitting diagnostic"
        )
    row: dict[str, Any] = {
        "schema": "banana-smasher-solver-anchor-row-v1",
        "status": "PASS",
        "candidate_id": aggregate["candidate_id"],
        "bank_id": manifest["bank_id"],
        "bank_role": manifest["role"],
        "diagnostic_only": holdout,
        "evaluation_contract": aggregate.get("evaluation_contract"),
        "measured_global_mean_kld": aggregate["measured"]["global_mean_kld"],
        "measured_per_class_mean_kld": aggregate["measured"]["per_class_mean_kld"],
        "bank_content_sha256": manifest["content_hashes"]["manifest_payload_sha256"],
    }
    if "parent_estimates" in aggregate:
        row["estimated_parent_not_measured"] = aggregate["parent_estimates"]
    _atomic_write(Path(output_path), _canonical_bytes(row))
    return row


def _safe_component(value: str, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise AnchorEvaluationError(
            f"{label} must be a path-safe identifier containing letters, digits, '.', '_' or '-'"
        )
    return value


def _initialize_run_root(run_root: Path) -> None:
    for relative in (
        "manifests",
        "banks",
        "producers/teacher",
        "producers/candidate",
        "imports",
        "scores",
        "aggregates",
        "comparisons",
        "solver_rows",
    ):
        (run_root / relative).mkdir(parents=True, exist_ok=True)


def register_bank(
    run_root: Path | str, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    validation = validate_bank_manifest(manifest)
    run_root = Path(run_root)
    _initialize_run_root(run_root)
    bank_id = _safe_component(manifest["bank_id"], "bank_id")
    path = run_root / "manifests" / f"{bank_id}.json"
    payload = _canonical_bytes(manifest)
    resumed = False
    if path.exists():
        if path.read_bytes() != payload:
            raise AnchorEvaluationError(
                f"run root already contains a different manifest for {bank_id!r}"
            )
        resumed = True
    else:
        _atomic_write(path, payload)
    return {
        "schema": "banana-smasher-bank-registration-receipt-v1",
        "status": "PASS",
        "bank_id": bank_id,
        "role": manifest["role"],
        "relative_path": path.relative_to(run_root).as_posix(),
        "manifest_sha256": _sha256_bytes(payload),
        "resumed": resumed,
        "unresolved_provenance": validation["unresolved_provenance"],
    }


def load_registered_bank(run_root: Path | str, bank_id: str) -> dict[str, Any]:
    run_root = Path(run_root)
    bank_id = _safe_component(bank_id, "bank_id")
    path = run_root / "manifests" / f"{bank_id}.json"
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise AnchorEvaluationError(
            f"bank {bank_id!r} is not registered under {run_root}; run anchor register"
        ) from exc
    except json.JSONDecodeError as exc:
        raise AnchorEvaluationError(f"registered manifest {path} is invalid JSON: {exc}") from exc
    validate_bank_manifest(manifest)
    return manifest


def import_producer(
    run_root: Path | str,
    manifest: Mapping[str, Any],
    source_path: Path | str,
    *,
    kind: str,
    expected_sha256: str,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    validate_bank_manifest(manifest)
    if kind not in {"teacher", "candidate"}:
        raise AnchorEvaluationError("producer kind must be teacher or candidate")
    if not _is_sha256(expected_sha256):
        raise AnchorEvaluationError("expected producer SHA-256 is invalid")
    run_root = Path(run_root)
    _initialize_run_root(run_root)
    source_path = Path(source_path)
    rows, actual_sha256, payload = _producer_rows(source_path, kind)
    if actual_sha256 != expected_sha256:
        raise AnchorEvaluationError(
            f"{kind} producer SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    indexed = _producer_index(rows, kind)
    expected = {_window_key(window["id"]) for window in manifest["windows"]}
    missing = sorted(expected - set(indexed))
    unexpected = sorted(set(indexed) - expected)
    if missing or unexpected:
        raise AnchorEvaluationError(
            f"{kind} producer membership mismatch; missing={missing}, unexpected={unexpected}"
        )
    bank_id = _safe_component(manifest["bank_id"], "bank_id")
    if kind == "teacher":
        destination = run_root / "producers" / "teacher" / f"{bank_id}.jsonl"
        receipt_path = run_root / "imports" / f"teacher--{bank_id}.json"
        bound_candidate = None
    else:
        if candidate_id is None:
            raise AnchorEvaluationError("candidate producer import requires candidate_id")
        bound_candidate = _safe_component(candidate_id, "candidate_id")
        destination = (
            run_root / "producers" / "candidate" / bound_candidate / f"{bank_id}.jsonl"
        )
        receipt_path = (
            run_root / "imports" / f"candidate--{bound_candidate}--{bank_id}.json"
        )
    resumed = False
    if destination.exists():
        if destination.read_bytes() != payload:
            raise AnchorEvaluationError(
                f"refusing to replace different imported producer {destination}"
            )
        resumed = True
    else:
        _atomic_write(destination, payload)
    receipt = {
        "schema": "banana-smasher-producer-import-receipt-v1",
        "status": "PASS",
        "kind": kind,
        "bank_id": bank_id,
        "candidate_id": bound_candidate,
        "coverage": f"{len(rows)}/{len(manifest['windows'])}",
        "sha256": actual_sha256,
        "relative_path": destination.relative_to(run_root).as_posix(),
        "resumed": resumed,
    }
    receipt_payload = _canonical_bytes(receipt)
    if receipt_path.exists() and receipt_path.read_bytes() != receipt_payload:
        raise AnchorEvaluationError(
            f"import receipt conflict at {receipt_path}; use a new run root or candidate id"
        )
    _atomic_write(receipt_path, receipt_payload)
    return receipt


def materialize_candidate_producer(
    run_root: Path | str,
    manifest: Mapping[str, Any],
    *,
    candidate_id: str,
    model_root: Path | str,
    producer_config: Path | str,
    basis_sha256: str,
    execution_mode: str = "auto",
    chunk_size: int = 8,
) -> dict[str, Any]:
    """Run one selected producer backend and import its exact 64 rows."""

    validate_bank_manifest(manifest)
    if len(manifest["windows"]) != 64:
        raise AnchorEvaluationError(
            "candidate materialization requires an exact 64-window bank"
        )
    if not _is_sha256(basis_sha256):
        raise AnchorEvaluationError("basis_sha256 must be a lowercase SHA-256")
    if execution_mode not in {"auto", "vllm", "offline-layerwise"}:
        raise AnchorEvaluationError(
            "execution_mode must be auto, vllm, or offline-layerwise"
        )
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise AnchorEvaluationError("chunk_size must be a positive integer")
    run_root = Path(run_root).resolve()
    candidate_id = _safe_component(candidate_id, "candidate_id")
    bank_id = _safe_component(manifest["bank_id"], "bank_id")
    model_root = Path(model_root).expanduser().resolve()
    producer_config = Path(producer_config).expanduser().resolve()
    if not model_root.is_dir():
        raise AnchorEvaluationError(f"candidate model root is missing: {model_root}")
    pack_manifest_path = model_root / "BANANA_PACK_MANIFEST.json"
    if not pack_manifest_path.is_file():
        raise AnchorEvaluationError(
            f"candidate model is not an exported bs-pack: {pack_manifest_path}"
        )
    from .contract import PackValidationError, load_manifest, verify_pack

    try:
        pack_verification = verify_pack(model_root)
        pack_manifest = load_manifest(model_root)
    except (OSError, ValueError, PackValidationError) as exc:
        raise AnchorEvaluationError(
            f"candidate model pack verification failed: {exc}"
        ) from exc
    if not isinstance(pack_verification, Mapping) or pack_verification.get("status") != "PASS":
        raise AnchorEvaluationError("candidate model pack verification did not return PASS")
    reusable_pack_verification = {
        "schema": "banana-smasher-pack-verification-receipt-v1",
        "status": "PASS",
        "model_root": str(model_root),
        "basis_sha256": basis_sha256,
        "manifest_sha256": _sha256_bytes(pack_manifest_path.read_bytes()),
        "verification": dict(pack_verification),
    }
    declared_layers = pack_manifest.get("layers")
    if not isinstance(declared_layers, list) or not all(
        isinstance(layer, int) and not isinstance(layer, bool)
        for layer in declared_layers
    ):
        raise AnchorEvaluationError("candidate model pack has invalid layers")
    layer_receipts = sorted((model_root / "provenance").glob("layer_*/LAYER_RECEIPT.json"))
    if not layer_receipts:
        single = model_root / "provenance" / "LAYER_RECEIPT.json"
        if single.is_file():
            layer_receipts = [single]
    if not layer_receipts:
        raise AnchorEvaluationError("candidate model has no fixed-D4 layer receipts")
    receipt_layers: set[int] = set()
    for path in layer_receipts:
        try:
            receipt = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise AnchorEvaluationError(f"invalid fixed-D4 layer receipt {path}: {exc}") from exc
        receipt_layer = receipt.get("layer")
        if not isinstance(receipt_layer, int) or isinstance(receipt_layer, bool):
            raise AnchorEvaluationError(f"candidate model has invalid layer receipt {path}")
        receipt_layers.add(receipt_layer)
        if receipt.get("tier") not in {"d4_k2048", "d4_k4096"} or receipt.get(
            "basis_sha256"
        ) != basis_sha256:
            raise AnchorEvaluationError(
                f"candidate model fixed-D4 basis mismatch in {path}"
            )
    if receipt_layers != set(declared_layers):
        raise AnchorEvaluationError(
            "candidate model fixed-D4 receipts do not match the pack layer set"
        )

    try:
        config_payload = producer_config.read_bytes()
        config = json.loads(config_payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise AnchorEvaluationError(f"invalid candidate producer config {producer_config}: {exc}") from exc
    command = config.get("command") if isinstance(config, Mapping) else None
    external_command = command if isinstance(command, list) else []
    configured_producer = config.get("producer") if isinstance(config, Mapping) else None
    builtin = (
        isinstance(config, Mapping)
        and config.get("schema") == "banana-smasher-candidate-producer-v1"
        and configured_producer in {"fixed-d4-vllm", "fixed-d4-offline-layerwise"}
        and set(config) == {"schema", "producer", "parameters"}
    )
    external = (
        isinstance(config, Mapping)
        and config.get("schema") == "banana-smasher-candidate-producer-v1"
        and bool(external_command)
        and all(isinstance(value, str) and value for value in external_command)
    )
    if not builtin and not external:
        raise AnchorEvaluationError(
            "candidate producer config requires schema banana-smasher-candidate-producer-v1 "
            "and either a fixed-d4-vllm/fixed-d4-offline-layerwise producer or "
            "a non-empty string command array"
        )
    configured_mode = (
        "offline-layerwise"
        if configured_producer == "fixed-d4-offline-layerwise"
        else "vllm"
    )
    selected_mode = configured_mode if execution_mode == "auto" else execution_mode
    if selected_mode != configured_mode:
        raise AnchorEvaluationError(
            f"execution mode {selected_mode!r} does not match producer mode {configured_mode!r}"
        )

    bank_path = run_root / "banks" / f"{bank_id}.jsonl"
    if not bank_path.is_file():
        raise AnchorEvaluationError(
            f"materialized bank is missing: {bank_path}; run anchor materialize"
        )
    staging_root = run_root / "materializations" / "candidate" / candidate_id
    staging_root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{bank_id}.", suffix=".jsonl", dir=staging_root
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if configured_producer == "fixed-d4-offline-layerwise":
            from .fixed_d4 import produce_fixed_d4_layerwise_logits

            produce_fixed_d4_layerwise_logits(
                model_root,
                producer_config,
                bank_path,
                temporary,
                basis_sha256=basis_sha256,
                verified_pack_receipt=reusable_pack_verification,
            )
            producer_backend = "fixed-d4-offline-layerwise"
        elif builtin:
            from .fixed_d4 import produce_fixed_d4_logits

            produce_fixed_d4_logits(
                model_root,
                producer_config,
                bank_path,
                temporary,
                basis_sha256=basis_sha256,
            )
            producer_backend = "fixed-d4-vllm"
        else:
            completed = subprocess.run(
                [
                    *external_command,
                    "--model",
                    str(model_root),
                    "--config",
                    str(producer_config),
                    "--bank",
                    str(bank_path),
                    "--output",
                    str(temporary),
                    "--basis-sha256",
                    basis_sha256,
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise AnchorEvaluationError(
                    f"candidate producer failed with exit {completed.returncode}: {detail}"
                )
            producer_backend = "external-command"
        payload = temporary.read_bytes()
        producer_sha256 = _sha256_bytes(payload)
        imported = import_producer(
            run_root,
            manifest,
            temporary,
            kind="candidate",
            expected_sha256=producer_sha256,
            candidate_id=candidate_id,
        )
    finally:
        temporary.unlink(missing_ok=True)

    materialization_receipt = {
        "schema": "banana-smasher-candidate-materialization-receipt-v1",
        "status": "PASS",
        "candidate_id": candidate_id,
        "bank_id": bank_id,
        "coverage": imported["coverage"],
        "basis_sha256": basis_sha256,
        "producer_backend": producer_backend,
        "execution_mode": selected_mode,
        "resumed_windows": 0,
        "computed_windows": len(manifest["windows"]),
        "model_manifest_sha256": _sha256_bytes(pack_manifest_path.read_bytes()),
        "producer_config_sha256": _sha256_bytes(config_payload),
        "producer_sha256": producer_sha256,
        "relative_path": imported["relative_path"],
    }
    receipt_path = (
        run_root / "imports" / f"candidate-materialization--{candidate_id}--{bank_id}.json"
    )
    _atomic_write(receipt_path, _canonical_bytes(materialization_receipt))
    return materialization_receipt


def _coverage_for_path(
    path: Path,
    expected: set[str],
    *,
    id_field: str = "window_id",
    score: bool = False,
) -> tuple[str, bool]:
    if not path.is_file():
        return "MISSING", False
    try:
        rows = _read_jsonl(path)
        keys = {_window_key(row.get(id_field)) for row in rows}
    except AnchorEvaluationError:
        return "INVALID", False
    complete = len(rows) == len(expected) and keys == expected
    if score and any(row.get("schema") != "banana-smasher-anchor-window-score-v1" for row in rows):
        complete = False
    return f"{len(keys)}/{len(expected)}", complete


def status_report(run_root: Path | str) -> dict[str, Any]:
    run_root = Path(run_root)
    if not (run_root / "manifests").is_dir():
        raise AnchorEvaluationError(
            f"run root {run_root} has no manifests directory; register a bank first"
        )
    banks: list[dict[str, Any]] = []
    for manifest_path in sorted((run_root / "manifests").glob("*.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise AnchorEvaluationError(
                f"registered manifest {manifest_path} is invalid JSON: {exc}"
            ) from exc
        validation = validate_bank_manifest(manifest)
        bank_id = manifest["bank_id"]
        expected = {_window_key(window["id"]) for window in manifest["windows"]}
        bank_coverage, _ = _coverage_for_path(
            run_root / "banks" / f"{bank_id}.jsonl",
            expected,
            id_field=manifest["dataset_fields"]["window_id"],
        )
        teacher_coverage, _ = _coverage_for_path(
            run_root / "producers" / "teacher" / f"{bank_id}.jsonl", expected
        )
        candidate_ids: set[str] = set()
        for root_name in ("producers/candidate", "scores", "aggregates"):
            root = run_root / root_name
            if root.is_dir():
                candidate_ids.update(path.name for path in root.iterdir() if path.is_dir())
        candidate_details: list[dict[str, Any]] = []
        score_complete = 0
        aggregate_complete = 0
        for candidate_id in sorted(candidate_ids):
            candidate_coverage, candidate_ok = _coverage_for_path(
                run_root
                / "producers"
                / "candidate"
                / candidate_id
                / f"{bank_id}.jsonl",
                expected,
            )
            score_coverage, score_ok = _coverage_for_path(
                run_root / "scores" / candidate_id / bank_id / "raw.jsonl",
                expected,
                score=True,
            )
            aggregate_path = run_root / "aggregates" / candidate_id / f"{bank_id}.json"
            aggregate_ok = False
            if aggregate_path.is_file():
                try:
                    aggregate = json.loads(aggregate_path.read_text())
                    aggregate_ok = (
                        aggregate.get("schema") == "banana-smasher-anchor-aggregate-v1"
                        and aggregate.get("bank_id") == bank_id
                        and aggregate.get("candidate_id") == candidate_id
                    )
                except json.JSONDecodeError:
                    aggregate_ok = False
            score_complete += int(score_ok)
            aggregate_complete += int(aggregate_ok)
            candidate_details.append(
                {
                    "candidate_id": candidate_id,
                    "coverage": candidate_coverage,
                    "coverage_complete": candidate_ok,
                    "scoring": score_coverage,
                    "scoring_complete": score_ok,
                    "aggregation_complete": aggregate_ok,
                }
            )
        unresolved = validation["unresolved_provenance"]
        banks.append(
            {
                "bank_id": bank_id,
                "role": manifest["role"],
                "bank_production": bank_coverage,
                "teacher_coverage": teacher_coverage,
                "candidate_coverage": f"{len(candidate_ids)} candidates",
                "scoring": f"{score_complete} complete",
                "aggregation": f"{aggregate_complete} complete",
                "provenance": (
                    f"UNRESOLVED: {','.join(unresolved)}" if unresolved else "RESOLVED"
                ),
                "candidate_details": candidate_details,
            }
        )
    return {
        "schema": "banana-smasher-anchor-status-v1",
        "status": "PASS",
        "run_root": ".",
        "bank_count": len(banks),
        "banks": banks,
    }


def format_status(status: Mapping[str, Any]) -> str:
    headers = [
        "BANK",
        "ROLE",
        "PRODUCTION",
        "TEACHER",
        "CANDIDATES",
        "SCORING",
        "AGGREGATION",
        "PROVENANCE",
    ]
    rows = [
        [
            row["bank_id"],
            row["role"],
            row["bank_production"],
            row["teacher_coverage"],
            row["candidate_coverage"],
            row["scoring"],
            row["aggregation"],
            row["provenance"],
        ]
        for row in status["banks"]
    ]
    widths = [
        max(len(str(value)) for value in [header, *(row[index] for row in rows)])
        for index, header in enumerate(headers)
    ]
    lines = [" | ".join(header.ljust(width) for header, width in zip(headers, widths, strict=True))]
    lines.append("-+-".join("-" * width for width in widths))
    lines.extend(
        " | ".join(str(value).ljust(width) for value, width in zip(row, widths, strict=True))
        for row in rows
    )
    return "\n".join(lines) + "\n"
