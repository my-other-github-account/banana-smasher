from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from .backpack_preview import CLASSES
from .repair import (
    REPAIR_FORMAT,
    REPAIR_MECHANISM,
    CodebookRepair,
    RepairBundle,
    apply_residual_update,
    load_repair_bundle,
)

PLAN_SCHEMA = "banana-smasher-backpack-plan-v1"
MODEL_SCHEMA = "banana-smasher-backpack-model-v1"
STAGE_SCHEMA = "banana-smasher-backpack-stage-receipt-v1"
FINAL_SCHEMA = "banana-smasher-backpack-final-receipt-v1"
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,126}[A-Za-z0-9]|[A-Za-z0-9]")
STAGES = (
    "inspect",
    "candidates",
    "candidate_anchor",
    "pred",
    "solve_materialize",
    "pre_repair_anchor",
    "repair",
    "final_score",
)
QTIP_BACKENDS = frozenset({"packaged_qtip", "fixture_reference"})
QTIP_PROJECTIONS = ("fused13", "down")
_PROJECTION_ORDER = {name: index for index, name in enumerate(QTIP_PROJECTIONS)}
REPAIR_FIXTURE_METHODS = frozenset({"residual", "fixture_residual"})
REPAIR_BUNDLE_METHOD = "repair_bundle"
REUSABLE_STAGE_IMPORTS = frozenset({"candidates", "candidate_anchor"})


class BackpackPlanError(ValueError):
    """A declarative Backpack plan is incomplete or internally inconsistent."""


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(path, _canonical_bytes(value))


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BackpackPlanError(f"{label} must be an object")
    return dict(value)


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise BackpackPlanError(f"{label} has unknown fields: {', '.join(unknown)}")


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BackpackPlanError(f"{label} must be a non-empty string")
    return value


def _safe_id(value: object, label: str) -> str:
    result = _nonempty(value, label)
    if _ID_RE.fullmatch(result) is None or "/" in result or "\\" in result:
        raise BackpackPlanError(
            f"{label} must be a safe path component containing only letters, digits, '.', '_', or '-'"
        )
    return result


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BackpackPlanError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise BackpackPlanError(f"{label} must be positive and finite")
    return result


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BackpackPlanError(f"{label} must be a positive integer")
    return value


def _path(value: object, label: str, *, base_dir: Path) -> str:
    raw = Path(_nonempty(value, label)).expanduser()
    candidate = base_dir / raw if not raw.is_absolute() else raw
    return os.path.abspath(candidate)


@dataclass(frozen=True)
class BackpackPlan:
    schema: str
    model: dict[str, Any]
    target: dict[str, Any]
    tiers: tuple[dict[str, Any], ...]
    anchor: dict[str, Any]
    prediction: dict[str, Any]
    repair: dict[str, Any]
    output: dict[str, Any]
    reuse_receipts: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, base_dir: str | Path | None = None
    ) -> "BackpackPlan":
        if not isinstance(value, Mapping):
            raise BackpackPlanError("Backpack plan must be an object")
        if value.get("schema") != PLAN_SCHEMA:
            raise BackpackPlanError(f"plan schema must be {PLAN_SCHEMA}")
        _reject_unknown(
            value,
            {
                "schema",
                "model",
                "target",
                "tiers",
                "anchor",
                "prediction",
                "repair",
                "output",
                "reuse_receipts",
            },
            "plan",
        )
        base = Path.cwd() if base_dir is None else Path(base_dir).expanduser().resolve()

        model = _object(value.get("model"), "model")
        _reject_unknown(model, {"root", "manifest", "revision"}, "model")
        model = {
            **model,
            "root": _path(model.get("root"), "model.root", base_dir=base),
            "revision": _nonempty(model.get("revision"), "model.revision"),
        }
        if "manifest" in model:
            model["manifest"] = _path(model["manifest"], "model.manifest", base_dir=base)

        target = _object(value.get("target"), "target")
        _reject_unknown(target, {"exact_bytes", "whole_model_bpw"}, "target")
        target_fields = [name for name in ("exact_bytes", "whole_model_bpw") if name in target]
        if len(target_fields) != 1:
            raise BackpackPlanError(
                "target must provide exactly one of exact_bytes or whole_model_bpw"
            )
        if "exact_bytes" in target:
            _positive_int(target["exact_bytes"], "target.exact_bytes")
        else:
            _positive_number(target["whole_model_bpw"], "target.whole_model_bpw")

        raw_tiers = value.get("tiers")
        if not isinstance(raw_tiers, Sequence) or isinstance(raw_tiers, (str, bytes)) or not raw_tiers:
            raise BackpackPlanError("tiers must be a non-empty array")
        tiers: list[dict[str, Any]] = []
        seen: set[str] = set()
        seen_casefolded: set[str] = set()
        for index, raw in enumerate(raw_tiers):
            tier = _object(raw, f"tiers[{index}]")
            tier_id = _safe_id(tier.get("id"), f"tiers[{index}].id")
            if tier_id in seen:
                raise BackpackPlanError(f"duplicate tier id {tier_id!r}")
            if tier_id.casefold() in seen_casefolded:
                raise BackpackPlanError(
                    f"colliding tier id {tier_id!r} on a case-insensitive filesystem"
                )
            seen.add(tier_id)
            seen_casefolded.add(tier_id.casefold())
            family = tier.get("family")
            if family == "vector_vq":
                _reject_unknown(
                    tier,
                    {"id", "family", "dimension", "bits", "codebook_size", "bpw"},
                    f"tiers[{index}]",
                )
                dimension = tier.get("dimension")
                if dimension not in {4, 8}:
                    raise BackpackPlanError(
                        f"tiers[{index}] vector_vq dimension must be 4 or 8"
                    )
                geometry_fields = [
                    name for name in ("bits", "codebook_size", "bpw") if name in tier
                ]
                if len(geometry_fields) != 1:
                    raise BackpackPlanError(
                        f"tiers[{index}] vector_vq requires exactly one of bits, codebook_size, or bpw"
                    )
                if "bits" in tier:
                    bits = _positive_int(tier["bits"], f"tiers[{index}].bits")
                    if bits > 16:
                        raise BackpackPlanError("vector_vq bits must be in 1..16")
                elif "codebook_size" in tier:
                    size = _positive_int(
                        tier["codebook_size"], f"tiers[{index}].codebook_size"
                    )
                    if size < 2 or size > 65536 or size & (size - 1):
                        raise BackpackPlanError(
                            "vector_vq codebook_size must be a power of two in 2..65536"
                        )
                else:
                    bpw = _positive_number(tier["bpw"], f"tiers[{index}].bpw")
                    implied_bits = bpw * dimension
                    if not float(implied_bits).is_integer():
                        raise BackpackPlanError(
                            "vector_vq requested bpw must imply an integer index width"
                        )
                    if not 1 <= int(implied_bits) <= 16:
                        raise BackpackPlanError(
                            "vector_vq requested bpw must imply an index width in 1..16"
                        )
            elif family == "qtip":
                _reject_unknown(
                    tier,
                    {"id", "family", "bpw", "backend", "source_root"},
                    f"tiers[{index}]",
                )
                bpw = Decimal(str(_positive_number(tier.get("bpw"), f"tiers[{index}].bpw")))
                if (
                    bpw < Decimal("1.00")
                    or bpw > Decimal("4.00")
                    or bpw * 4 != (bpw * 4).to_integral_value()
                ):
                    raise BackpackPlanError(
                        "QTIP bpw must use supported 0.25 increments from 1.00 through 4.00"
                    )
                if any(name in tier for name in ("dimension", "bits", "codebook_size")):
                    raise BackpackPlanError(
                        "QTIP descriptors cannot declare vector-VQ geometry"
                    )
                backend = tier.get("backend", "packaged_qtip")
                if backend not in QTIP_BACKENDS:
                    raise BackpackPlanError(
                        "QTIP backend must be packaged_qtip or fixture_reference"
                    )
                tier["backend"] = backend
                if backend == "packaged_qtip":
                    tier["source_root"] = _path(
                        tier.get("source_root"),
                        f"tiers[{index}].source_root",
                        base_dir=base,
                    )
                elif "source_root" in tier:
                    raise BackpackPlanError(
                        f"tiers[{index}].source_root is only valid for packaged_qtip"
                    )
            else:
                raise BackpackPlanError(
                    f"tiers[{index}].family must be vector_vq or qtip"
                )
            tiers.append(tier)

        anchor = _object(value.get("anchor"), "anchor")
        _reject_unknown(anchor, {"bank", "teacher"}, "anchor")
        anchor["bank"] = _path(anchor.get("bank"), "anchor.bank", base_dir=base)
        if anchor.get("teacher") != "model":
            anchor["teacher"] = _path(
                anchor.get("teacher"), "anchor.teacher", base_dir=base
            )

        prediction = _object(value.get("prediction"), "prediction")
        _reject_unknown(prediction, {"class_caps"}, "prediction")
        caps = prediction.get("class_caps")
        if not isinstance(caps, Mapping) or set(caps) != set(CLASSES):
            raise BackpackPlanError("prediction.class_caps must cover the six classes")
        prediction["class_caps"] = {
            name: _positive_number(caps[name], f"prediction.class_caps.{name}")
            for name in CLASSES
        }

        repair = _object(value.get("repair"), "repair")
        repair_method = _nonempty(repair.get("method"), "repair.method")
        if repair_method in REPAIR_FIXTURE_METHODS:
            _reject_unknown(repair, {"method", "strength"}, "repair")
            strength = _positive_number(repair.get("strength"), "repair.strength")
            if strength > 1.0:
                raise BackpackPlanError("repair.strength must not exceed 1")
            repair = {"method": "fixture_residual", "strength": strength}
        elif repair_method == REPAIR_BUNDLE_METHOD:
            _reject_unknown(
                repair,
                {
                    "method",
                    "checkpoint",
                    "checkpoint_sha256",
                    "active_overlay",
                    "active_overlay_sha256",
                    "assignment",
                    "assignment_sha256",
                    "update",
                },
                "repair",
            )
            repair = {
                "method": REPAIR_BUNDLE_METHOD,
                "checkpoint": _path(
                    repair.get("checkpoint"), "repair.checkpoint", base_dir=base
                ),
                "checkpoint_sha256": _nonempty(
                    repair.get("checkpoint_sha256"), "repair.checkpoint_sha256"
                ),
                "active_overlay": _path(
                    repair.get("active_overlay"), "repair.active_overlay", base_dir=base
                ),
                "active_overlay_sha256": _nonempty(
                    repair.get("active_overlay_sha256"), "repair.active_overlay_sha256"
                ),
                "assignment": _path(
                    repair.get("assignment"), "repair.assignment", base_dir=base
                ),
                "assignment_sha256": _nonempty(
                    repair.get("assignment_sha256"), "repair.assignment_sha256"
                ),
                "update": _positive_int(repair.get("update"), "repair.update"),
            }
            for field in (
                "checkpoint_sha256",
                "active_overlay_sha256",
                "assignment_sha256",
            ):
                if re.fullmatch(r"[0-9a-f]{64}", repair[field]) is None:
                    raise BackpackPlanError(f"repair.{field} must be a lowercase SHA-256")
        elif repair_method == "none":
            _reject_unknown(repair, {"method"}, "repair")
        else:
            raise BackpackPlanError(
                "repair.method must be none, fixture_residual, or repair_bundle"
            )

        output = _object(value.get("output"), "output")
        _reject_unknown(output, {"pack", "model_id", "instance_id"}, "output")
        output = {
            **output,
            "pack": _path(output.get("pack"), "output.pack", base_dir=base),
            "model_id": _nonempty(output.get("model_id"), "output.model_id"),
            "instance_id": _nonempty(output.get("instance_id"), "output.instance_id"),
        }
        raw_reuse = value.get("reuse_receipts", [])
        if not isinstance(raw_reuse, Sequence) or isinstance(raw_reuse, (str, bytes)):
            raise BackpackPlanError("reuse_receipts must be an array")
        reuse_receipts: list[dict[str, Any]] = []
        reuse_roles: set[str] = set()
        for index, raw in enumerate(raw_reuse):
            row = _object(raw, f"reuse_receipts[{index}]")
            _reject_unknown(
                row,
                {"role", "path", "sha256", "admission", "schema", "stage"},
                f"reuse_receipts[{index}]",
            )
            role = _safe_id(row.get("role"), f"reuse_receipts[{index}].role")
            if role in reuse_roles:
                raise BackpackPlanError(f"duplicate reuse receipt role {role!r}")
            reuse_roles.add(role)
            admission = row.get("admission", "admitted")
            if admission not in {"admitted", "evidence_only"}:
                raise BackpackPlanError(
                    f"reuse_receipts[{index}].admission must be admitted or evidence_only"
                )
            sha256 = _nonempty(row.get("sha256"), f"reuse_receipts[{index}].sha256")
            if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
                raise BackpackPlanError(
                    f"reuse_receipts[{index}].sha256 must be a lowercase SHA-256"
                )
            reuse_receipts.append(
                {
                    "role": role,
                    "path": _path(
                        row.get("path"), f"reuse_receipts[{index}].path", base_dir=base
                    ),
                    "sha256": sha256,
                    "admission": admission,
                    **(
                        {
                            "schema": _nonempty(
                                row.get("schema"), f"reuse_receipts[{index}].schema"
                            )
                        }
                        if "schema" in row
                        else {}
                    ),
                    **(
                        {
                            "stage": _nonempty(
                                row.get("stage"), f"reuse_receipts[{index}].stage"
                            )
                        }
                        if "stage" in row
                        else {}
                    ),
                }
            )
            if "stage" in reuse_receipts[-1] and reuse_receipts[-1]["stage"] not in STAGES:
                raise BackpackPlanError(
                    f"reuse_receipts[{index}].stage must be one of {', '.join(STAGES)}"
                )
            if "stage" in reuse_receipts[-1] and "schema" not in reuse_receipts[-1]:
                raise BackpackPlanError(
                    f"reuse_receipts[{index}].schema is required when stage is supplied"
                )
            if (
                "stage" in reuse_receipts[-1]
                and "schema" in reuse_receipts[-1]
                and reuse_receipts[-1]["schema"] != STAGE_SCHEMA
            ):
                raise BackpackPlanError(
                    f"reuse_receipts[{index}].schema must be {STAGE_SCHEMA} when stage is supplied"
                )
        return cls(
            schema=PLAN_SCHEMA,
            model=model,
            target=target,
            tiers=tuple(tiers),
            anchor=anchor,
            prediction=prediction,
            repair=repair,
            output=output,
            reuse_receipts=tuple(reuse_receipts),
        )

    def as_mapping(self) -> dict[str, Any]:
        result = {
            "schema": self.schema,
            "model": self.model,
            "target": self.target,
            "tiers": list(self.tiers),
            "anchor": self.anchor,
            "prediction": self.prediction,
            "repair": self.repair,
            "output": self.output,
        }
        if self.reuse_receipts:
            result["reuse_receipts"] = list(self.reuse_receipts)
        return result


def _execution_plan_mapping(plan: BackpackPlan) -> dict[str, Any]:
    result = plan.as_mapping()
    result.pop("reuse_receipts", None)
    return result


def _execution_plan_sha(plan: BackpackPlan) -> str:
    return _sha(_canonical_bytes(_execution_plan_mapping(plan)))


def pack_indices(indices: np.ndarray, *, bits: int) -> bytes:
    """Pack unsigned assignment indices densely, least-significant bit first."""

    values = np.asarray(indices)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError("indices must be one integer vector")
    if isinstance(bits, bool) or not isinstance(bits, int) or bits <= 0 or bits > 31:
        raise ValueError("bits must be an integer in 1..31")
    maximum = (1 << bits) - 1
    if np.any(values < 0) or np.any(values > maximum):
        raise ValueError("assignment index exceeds bit width")
    output = bytearray((values.size * bits + 7) // 8)
    bit_offset = 0
    for raw in values:
        value = int(raw)
        for bit in range(bits):
            if value & (1 << bit):
                offset = bit_offset + bit
                output[offset // 8] |= 1 << (offset % 8)
        bit_offset += bits
    return bytes(output)


def _tier_geometry(tier: Mapping[str, Any]) -> tuple[int, int]:
    dimension = int(tier["dimension"])
    if "bits" in tier:
        bits = int(tier["bits"])
    elif "codebook_size" in tier:
        bits = int(math.log2(int(tier["codebook_size"])))
    else:
        bits = int(round(float(tier["bpw"]) * dimension))
    return dimension, bits


def quantize_vector_cell(
    weights: np.ndarray, *, dimension: int, bits: int
) -> dict[str, np.ndarray | bytes | int]:
    """Deterministic D4/D8 nearest-codeword quantization with dense bit packing."""

    flat = np.asarray(weights, dtype=np.float32).reshape(-1)
    if dimension not in {4, 8}:
        raise ValueError("vector dimension must be 4 or 8")
    if flat.size == 0 or flat.size % dimension:
        raise ValueError(f"weight count {flat.size} is not divisible by D{dimension}")
    if bits <= 0 or bits > 16:
        raise ValueError("vector index width must be in 1..16")
    vectors = np.ascontiguousarray(flat.reshape(-1, dimension))
    k = 1 << bits
    unique, first = np.unique(vectors, axis=0, return_index=True)
    ordered = unique[np.argsort(first)]
    if ordered.shape[0] >= k:
        codebook = ordered[:k]
    else:
        repeats = np.resize(np.arange(ordered.shape[0]), k)
        codebook = ordered[repeats]
    distances = np.sum(
        (vectors[:, None, :] - codebook[None, :, :]) ** 2, axis=2, dtype=np.float64
    )
    assignments = np.argmin(distances, axis=1).astype(np.int32)
    wire_codebook = np.ascontiguousarray(codebook, dtype=np.float16)
    decoded = np.ascontiguousarray(
        wire_codebook[assignments].reshape(-1), dtype=np.float32
    )
    packed = pack_indices(assignments, bits=bits)
    return {
        "vectors": vectors,
        "codebook": wire_codebook,
        "assignments": assignments,
        "decoded": decoded,
        "packed": packed,
        "bits": bits,
    }


def _byte_string_array(values: Sequence[str], *, width: int) -> np.ndarray:
    encoded = [value.encode("utf-8") for value in values]
    if any(len(value) > width for value in encoded):
        raise BackpackPlanError(f"metadata label exceeds {width} UTF-8 bytes")
    output = np.zeros((len(encoded), width), dtype=np.uint8)
    for index, value in enumerate(encoded):
        output[index, : len(value)] = np.frombuffer(value, dtype=np.uint8)
    return output


def _payload_offsets(
    record_sizes: Sequence[tuple[int, int, int]],
) -> np.ndarray:
    offsets = np.zeros((len(record_sizes) + 1, 3), dtype=np.int64)
    if record_sizes:
        offsets[1:] = np.cumsum(np.asarray(record_sizes, dtype=np.int64), axis=0)
    return offsets


def _repeat_payload(array: np.ndarray, *, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("record repeat count must be positive")
    if array.ndim == 1:
        return np.tile(array, count)
    if array.ndim == 2:
        return np.stack([array] * count, axis=0)
    return np.concatenate([array] * count, axis=0)


def _quantize_qtip_reference_record(
    weights: np.ndarray, *, geometry: tuple[int, int, int]
) -> dict[str, Any]:
    """Deterministic CPU fixture backend for one explicit QTIP geometry."""

    _L, K, V = geometry
    flat = np.asarray(weights, dtype=np.float32).reshape(-1)
    if flat.size == 0 or flat.size % V:
        raise ValueError(
            f"weight count {flat.size} is not divisible by fixture QTIP V={V}"
        )
    signs = np.where(np.arange(flat.size) % 2, -1.0, 1.0).astype(np.float32)
    transformed = flat * signs
    maximum = float(np.max(np.abs(transformed)))
    lattice = (
        np.zeros(1 << K, dtype=np.float32)
        if maximum == 0.0
        else np.linspace(-maximum, maximum, 1 << K, dtype=np.float32)
    )
    states = np.empty(flat.size, dtype=np.int32)
    for index, value in enumerate(transformed):
        state = int(np.argmin(np.abs(lattice - value)))
        states[index] = state
    wire_lattice = np.asarray(lattice, dtype=np.float16)
    decoded = np.asarray(wire_lattice[states], dtype=np.float32) * signs
    packed = pack_indices(states.astype(np.int64), bits=K)
    return {
        "states": states,
        "decoded": np.ascontiguousarray(decoded),
        "packed": packed,
        "scale": np.asarray([maximum], dtype=np.float32),
        "lattice": wire_lattice,
        "transform": "fixture-qtip-reference-lattice",
        "geometry": geometry,
    }


def _model_manifest(plan: BackpackPlan) -> tuple[Path, dict[str, Any]]:
    root = Path(plan.model["root"])
    path = Path(plan.model.get("manifest", root / "BACKPACK_MODEL.json"))
    if path.is_symlink() or not path.is_file():
        raise BackpackPlanError(f"model geometry manifest must be a regular file: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BackpackPlanError(f"cannot read model geometry manifest {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != MODEL_SCHEMA:
        raise BackpackPlanError(f"model geometry manifest must use {MODEL_SCHEMA}")
    return path, value


def _fixed_artifacts(
    plan: BackpackPlan, manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    root = Path(plan.model["root"])
    raw_rows = manifest.get("fixed_artifacts", [])
    if not isinstance(raw_rows, list):
        raise BackpackPlanError("model fixed_artifacts must be a list")
    totals = {"dense": 0, "metadata": 0, "repair": 0}
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise BackpackPlanError(f"fixed_artifacts[{index}] must be an object")
        role = raw.get("role")
        relative = raw.get("path")
        if role not in totals or not isinstance(relative, str):
            raise BackpackPlanError(f"invalid fixed_artifacts[{index}] role/path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise BackpackPlanError(f"unsafe fixed artifact path: {relative}")
        source = root / relative_path
        if source.is_symlink() or not source.is_file():
            raise BackpackPlanError(f"fixed artifact must be a regular file: {source}")
        try:
            source.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise BackpackPlanError(f"fixed artifact escapes model root: {source}") from exc
        actual_bytes = source.stat().st_size
        actual_sha = _sha_file(source)
        if raw.get("bytes") != actual_bytes or raw.get("sha256") != actual_sha:
            raise BackpackPlanError(f"fixed artifact identity mismatch: {source}")
        totals[str(role)] += actual_bytes
        records.append(
            {
                "role": role,
                "source": str(source),
                "bytes": actual_bytes,
                "sha256": actual_sha,
            }
        )
    for role, field in (
        ("dense", "dense_bytes"),
        ("metadata", "metadata_bytes"),
        ("repair", "repair_bytes"),
    ):
        if role == "repair" and plan.repair["method"] == REPAIR_BUNDLE_METHOD:
            if totals[role]:
                raise BackpackPlanError(
                    "repair_bundle cannot be combined with manifest-bound fixed repair artifacts"
                )
            continue
        if totals[role] != manifest.get(field, 0):
            raise BackpackPlanError(
                f"model {field}={manifest.get(field, 0)} does not match "
                f"manifest-bound fixed artifacts={totals[role]}"
            )
    return records


def _load_cells(plan: BackpackPlan) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _manifest_path, manifest = _model_manifest(plan)
    root = Path(plan.model["root"])
    rows = manifest.get("cells")
    if not isinstance(rows, list) or not rows:
        raise BackpackPlanError("model geometry manifest cells must be non-empty")
    cells: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_casefolded: set[str] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise BackpackPlanError(f"model cell {index} must be an object")
        cell_id = _safe_id(raw.get("cell_id"), f"model cells[{index}].cell_id")
        if cell_id in seen:
            raise BackpackPlanError(f"duplicate model cell {cell_id!r}")
        if cell_id.casefold() in seen_casefolded:
            raise BackpackPlanError(
                f"colliding model cell {cell_id!r} on a case-insensitive filesystem"
            )
        seen.add(cell_id)
        seen_casefolded.add(cell_id.casefold())
        relative = Path(_nonempty(raw.get("path"), f"model cells[{index}].path"))
        if relative.is_absolute() or ".." in relative.parts:
            raise BackpackPlanError(f"model cell path must remain inside model root: {relative}")
        input_path = root / relative
        if input_path.is_symlink() or not input_path.is_file():
            raise BackpackPlanError(f"model cell must be a regular NPY file: {input_path}")
        path = input_path.resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise BackpackPlanError(f"model cell path escapes model root: {relative}") from exc
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.float32 or array.size == 0 or not np.isfinite(array).all():
            raise BackpackPlanError(f"model cell {cell_id} must be finite non-empty float32 NPY")
        feature_slice = raw.get("feature_slice")
        if (
            not isinstance(feature_slice, list)
            or len(feature_slice) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in feature_slice)
            or feature_slice[0] < 0
            or feature_slice[1] - feature_slice[0] != array.size
        ):
            raise BackpackPlanError(f"model cell {cell_id} feature_slice/weight shape mismatch")
        layer = raw.get("layer")
        expert_ids = raw.get("expert_ids")
        projection = raw.get("projection", "down")
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise BackpackPlanError(f"model cell {cell_id} layer must be a non-negative integer")
        if projection not in QTIP_PROJECTIONS:
            raise BackpackPlanError(
                f"model cell {cell_id} projection must be one of {', '.join(QTIP_PROJECTIONS)}"
            )
        if (
            not isinstance(expert_ids, list)
            or not expert_ids
            or any(
                isinstance(expert, bool)
                or not isinstance(expert, int)
                or expert < 0
                or expert >= 256
                for expert in expert_ids
            )
            or len(expert_ids) != len(set(expert_ids))
        ):
            raise BackpackPlanError(
                f"model cell {cell_id} expert_ids must be unique integers in 0..255"
            )
        cells.append(
            {
                "cell_id": cell_id,
                "path": str(path),
                "relative_path": relative.as_posix(),
                "feature_slice": feature_slice,
                "layer": layer,
                "projection": projection,
                "expert_ids": expert_ids,
                "weights": np.ascontiguousarray(array.reshape(-1)),
            }
        )
    for layer in sorted({int(cell["layer"]) for cell in cells}):
        partitions_by_projection: dict[str, set[tuple[int, ...]]] = {}
        for projection in QTIP_PROJECTIONS:
            routed = [
                int(expert)
                for cell in cells
                if int(cell["layer"]) == layer and cell["projection"] == projection
                for expert in cell["expert_ids"]
            ]
            if len(routed) != 256 or set(routed) != set(range(256)):
                raise BackpackPlanError(
                    f"model cells for layer {layer} projection {projection} "
                    "must partition expert IDs 0..255 exactly"
                )
            partitions_by_projection[projection] = {
                tuple(int(expert) for expert in cell["expert_ids"])
                for cell in cells
                if int(cell["layer"]) == layer and cell["projection"] == projection
            }
        reference = partitions_by_projection[QTIP_PROJECTIONS[0]]
        if any(partitions_by_projection[projection] != reference for projection in QTIP_PROJECTIONS[1:]):
            raise BackpackPlanError(
                f"model cells for layer {layer} must use identical expert partitions "
                "across fused13 and down"
            )
        for group_index, expert_partition in enumerate(sorted(reference)):
            for cell in cells:
                if (
                    int(cell["layer"]) == layer
                    and tuple(int(expert) for expert in cell["expert_ids"])
                    == expert_partition
                ):
                    cell["selection_group"] = f"layer-{layer}-experts-{group_index}"
    return manifest, cells


def _load_anchor(plan: BackpackPlan, *, weight_count: int) -> tuple[np.ndarray, np.ndarray]:
    path = Path(plan.anchor["bank"])
    if path.is_symlink() or not path.is_file():
        raise BackpackPlanError(f"anchor.bank must be a regular NPZ file: {path}")
    with np.load(path, allow_pickle=False) as bank:
        if set(bank.files) != {"features", "classes"}:
            raise BackpackPlanError("anchor bank must contain exactly features and classes")
        features = np.asarray(bank["features"], dtype=np.float32)
        classes = np.asarray(bank["classes"]).astype(str)
    if features.shape != (64, weight_count):
        raise BackpackPlanError(
            f"Anchor64 features must have shape [64,{weight_count}], got {features.shape}"
        )
    if classes.shape != (64,) or set(classes) != set(CLASSES):
        raise BackpackPlanError("Anchor64 classes must be 64 labels covering all six classes")
    return features, classes


def _anchor_metrics(
    features: np.ndarray,
    classes: np.ndarray,
    teacher_weights: np.ndarray,
    candidate_weights: np.ndarray,
) -> dict[str, Any]:
    teacher_logits = features @ teacher_weights
    candidate_logits = features @ candidate_weights
    p = 1.0 / (1.0 + np.exp(-np.clip(teacher_logits, -30.0, 30.0)))
    q = 1.0 / (1.0 + np.exp(-np.clip(candidate_logits, -30.0, 30.0)))
    epsilon = 1e-12
    kld = p * np.log((p + epsilon) / (q + epsilon)) + (1.0 - p) * np.log(
        (1.0 - p + epsilon) / (1.0 - q + epsilon)
    )
    top1 = (teacher_logits >= 0) == (candidate_logits >= 0)

    def summary(mask: np.ndarray) -> dict[str, Any]:
        return {
            "windows": int(mask.sum()),
            "kld": float(np.mean(kld[mask], dtype=np.float64)),
            "top1": float(np.mean(top1[mask], dtype=np.float64)),
            "top1_matches": int(np.sum(top1[mask])),
        }

    all_rows = np.ones(64, dtype=bool)
    return {
        "schema": "banana-smasher-backpack-anchor64-score-v1",
        "status": "PASS",
        "windows": 64,
        "overall": summary(all_rows),
        "by_class": {name: summary(classes == name) for name in CLASSES},
    }


def _stage_path(root: Path, index: int, stage: str) -> Path:
    return root / "stages" / f"{index:02d}-{stage.replace('_', '-')}.json"


def _candidate_root(root: Path, tier: str, cell: str) -> Path:
    candidates = root / "candidates"
    tier_root = candidates / tier
    destination = tier_root / cell
    for path in (root, candidates, tier_root, destination):
        if path.is_symlink():
            raise BackpackPlanError(f"candidate output path must not be a symlink: {path}")
    return destination


def _stage_inspect(plan: BackpackPlan, root: Path) -> dict[str, Any]:
    manifest_path, manifest = _model_manifest(plan)
    if manifest.get("revision") != plan.model["revision"]:
        raise BackpackPlanError("model revision does not match geometry manifest")
    _same_manifest, cells = _load_cells(plan)
    weight_count = manifest.get("weight_count")
    if (
        isinstance(weight_count, bool)
        or not isinstance(weight_count, int)
        or weight_count != sum(cell["weights"].size for cell in cells)
    ):
        raise BackpackPlanError("model weight_count does not match cell geometry")
    features, classes = _load_anchor(plan, weight_count=weight_count)
    del features, classes
    _teacher_weights(plan, cells)
    fixed = {}
    for field in ("dense_bytes", "metadata_bytes", "repair_bytes"):
        value = manifest.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BackpackPlanError(f"model {field} must be a non-negative integer")
        fixed[field] = value
    fixed_artifacts = _fixed_artifacts(plan, manifest)
    fixed["routing_bytes"] = 256 * len({int(cell["layer"]) for cell in cells})
    target_bytes = (
        int(plan.target["exact_bytes"])
        if "exact_bytes" in plan.target
        else math.floor(weight_count * float(plan.target["whole_model_bpw"]) / 8.0)
    )
    fixed_bytes = sum(fixed.values())
    if target_bytes < fixed_bytes:
        raise BackpackPlanError(
            f"whole-model target {target_bytes} is below fixed bytes {fixed_bytes}"
        )
    reuse = (
        reuse_backpack_receipts(
            plan.reuse_receipts,
            output=root / "REUSED_RECEIPTS.json",
        )
        if plan.reuse_receipts
        else None
    )
    result = {
        "model_manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": _sha_file(manifest_path),
            "revision": manifest["revision"],
        },
        "weight_count": weight_count,
        "cell_ids": [cell["cell_id"] for cell in cells],
        "cell_artifacts": [
            {
                "cell_id": cell["cell_id"],
                "path": cell["path"],
                "bytes": Path(cell["path"]).stat().st_size,
                "sha256": _sha_file(Path(cell["path"])),
            }
            for cell in cells
        ],
        "fixed_bytes": fixed,
        "fixed_artifacts": fixed_artifacts,
        "fixed_total_bytes": fixed_bytes,
        "target_whole_model_bytes": target_bytes,
        "payload_envelope_bytes": target_bytes - fixed_bytes,
        "target_whole_model_bpw": target_bytes * 8.0 / weight_count,
        "anchor_bank": {
            "path": plan.anchor["bank"],
            "bytes": Path(plan.anchor["bank"]).stat().st_size,
            "sha256": _sha_file(Path(plan.anchor["bank"])),
        },
        "teacher": (
            {"kind": "model", "model_manifest_sha256": _sha_file(manifest_path)}
            if plan.anchor["teacher"] == "model"
            else {
                "kind": "npy",
                "path": plan.anchor["teacher"],
                "bytes": Path(plan.anchor["teacher"]).stat().st_size,
                "sha256": _sha_file(Path(plan.anchor["teacher"])),
            }
        ),
    }
    if reuse is not None:
        result["receipt_reuse"] = reuse
    return result


def _write_candidate_artifact(
    root: Path,
    *,
    tier: Mapping[str, Any],
    cell: Mapping[str, Any],
    decoded: np.ndarray,
    packed: bytes,
    extra_arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    destination = _candidate_root(root, str(tier["id"]), str(cell["cell_id"]))
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "decoded.npy", decoded.astype(np.float32), allow_pickle=False)
    _atomic_bytes(destination / "wire.bin", packed)
    array_rows = []
    for name, value in extra_arrays.items():
        path = destination / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        array_rows.append(
            {"name": name, "path": str(path), "bytes": path.stat().st_size, "sha256": _sha_file(path)}
        )
    receipt = {
        "schema": "banana-smasher-backpack-candidate-cell-v1",
        "status": "PASS",
        "tier": tier["id"],
        "family": tier["family"],
        "cell_id": cell["cell_id"],
        "weight_count": int(decoded.size),
        "physical_bytes": len(packed)
        + sum(int(value.nbytes) for value in extra_arrays.values()),
        "wire": {
            "path": str(destination / "wire.bin"),
            "bytes": len(packed),
            "sha256": _sha(packed),
        },
        "decoded": {
            "path": str(destination / "decoded.npy"),
            "sha256": _sha_file(destination / "decoded.npy"),
        },
        "arrays": array_rows,
        **metadata,
    }
    _atomic_json(destination / "RECEIPT.json", receipt)
    return receipt


def _record_rows(
    *,
    expert_ids: Sequence[int],
    projection: str,
    tier: str,
    geometries: Sequence[tuple[int, int, int]],
    geometry_fields: tuple[str, str, str],
) -> list[dict[str, Any]]:
    return [
        {
            "expert_id": int(expert_id),
            "projection": projection,
            "tier": tier,
            "geometry": {
                key: int(value)
                for key, value in zip(geometry_fields, geometry, strict=True)
            },
        }
        for expert_id, geometry in zip(expert_ids, geometries, strict=True)
    ]


def _cell_identities(cell: Mapping[str, Any]) -> list[tuple[int, int, str]]:
    projection = str(cell["projection"])
    layer = int(cell["layer"])
    return [
        (layer, int(expert_id), projection)
        for expert_id in cell["expert_ids"]
    ]


def _exact_qtip_geometries_for_cell(
    tier: Mapping[str, Any], cell: Mapping[str, Any]
) -> dict[tuple[int, int, str], tuple[int, int, int]]:
    from .qtip_rings import assign_ring_geometries, resolve_qtip_ring

    ring = resolve_qtip_ring(tier["bpw"])
    count = len(cell["expert_ids"])
    if any((count * component.quarters) % 4 for component in ring.components):
        raise BackpackPlanError(
            f"cell {cell['cell_id']} expert count {count} cannot represent "
            f"QTIP {ring.canonical_bpw} component quarters exactly"
        )
    return assign_ring_geometries(ring, _cell_identities(cell))


def _exact_qtip_geometries(
    tier: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]
) -> dict[tuple[int, int, str], tuple[int, int, int]]:
    from .qtip_rings import assign_ring_geometries, resolve_qtip_ring

    ring = resolve_qtip_ring(tier["bpw"])
    grouped: dict[tuple[int, str], list[tuple[int, int, str]]] = {}
    for cell in cells:
        grouped.setdefault(
            (int(cell["layer"]), str(cell["projection"])), []
        ).extend(_cell_identities(cell))
    assigned: dict[tuple[int, int, str], tuple[int, int, int]] = {}
    for identities in grouped.values():
        if len(identities) != 256 or len(set(identities)) != 256:
            raise BackpackPlanError(
                "QTIP geometry assignment requires all 256 unique experts per layer/projection"
            )
        assigned.update(assign_ring_geometries(ring, identities))
    return assigned


def _candidate_receipt_map(result: Mapping[str, Any]) -> dict[tuple[str, str], Path]:
    rows = result.get("candidate_tiers")
    if not isinstance(rows, list):
        raise BackpackPlanError("candidate receipt map requires candidate_tiers")
    mapped: dict[tuple[str, str], Path] = {}
    for tier_row in rows:
        if not isinstance(tier_row, Mapping):
            raise BackpackPlanError("candidate tier row must be an object")
        tier = _nonempty(tier_row.get("tier"), "candidate_tiers[].tier")
        cell_rows = tier_row.get("cells")
        if not isinstance(cell_rows, list):
            raise BackpackPlanError(f"candidate tier {tier!r} must contain cells")
        for cell_row in cell_rows:
            if not isinstance(cell_row, Mapping):
                raise BackpackPlanError(f"candidate tier {tier!r} cell row must be an object")
            cell_id = _nonempty(cell_row.get("cell_id"), f"candidate tier {tier!r} cell_id")
            receipt = Path(
                _nonempty(cell_row.get("receipt"), f"candidate tier {tier!r} receipt")
            )
            mapped[(tier, cell_id)] = receipt
    return mapped


def candidate_artifact_root(
    candidates: Mapping[str, Any], *, tier: str, cell_id: str
) -> Path:
    try:
        receipt = _candidate_receipt_map(candidates)[(tier, cell_id)]
    except KeyError as exc:
        raise BackpackPlanError(
            f"candidate receipt is missing for tier={tier!r} cell={cell_id!r}"
        ) from exc
    return receipt.parent


def _expert_weight_rows(
    cell: Mapping[str, Any], weights: np.ndarray | None = None
) -> np.ndarray:
    expert_count = len(cell["expert_ids"])
    source = np.asarray(
        cell["weights"] if weights is None else weights,
        dtype=np.float32,
    ).reshape(-1)
    if source.size % expert_count:
        raise BackpackPlanError(
            f"cell {cell['cell_id']} weight count {source.size} is not divisible by "
            f"its {expert_count} experts"
        )
    return np.ascontiguousarray(source.reshape(expert_count, -1))


def generate_vector_vq_backpack_candidate(
    run_root: str | Path,
    *,
    tier: Mapping[str, Any],
    cell: Mapping[str, Any],
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Generate one public Backpack vector-VQ candidate artifact."""

    dimension, bits = _tier_geometry(tier)
    expert_ids = [int(expert_id) for expert_id in cell["expert_ids"]]
    record_count = len(expert_ids)
    packed_parts: list[bytes] = []
    codebooks: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    decoded_rows: list[np.ndarray] = []
    payload_sizes: list[tuple[int, int, int]] = []
    for source_weights in _expert_weight_rows(cell, weights):
        quantized = quantize_vector_cell(
            source_weights,
            dimension=dimension,
            bits=bits,
        )
        packed = bytes(quantized["packed"])
        codebook = np.asarray(quantized["codebook"])
        scale = np.ones(1, dtype=np.float16)
        packed_parts.append(packed)
        codebooks.append(codebook)
        scales.append(scale)
        decoded_rows.append(np.asarray(quantized["decoded"], dtype=np.float32))
        payload_sizes.append((len(packed), scale.nbytes, codebook.nbytes))
    geometry = (dimension, bits, 1 << bits)
    return _write_candidate_artifact(
        Path(run_root),
        tier=tier,
        cell=cell,
        decoded=np.concatenate(decoded_rows),
        packed=b"".join(packed_parts),
        extra_arrays={
            "codebooks": np.stack(codebooks),
            "scales": np.concatenate(scales),
            "expert_ids": np.asarray(expert_ids, dtype=np.int16),
            "tensor_offsets": _payload_offsets(payload_sizes),
            "record_tiers": _byte_string_array(
                [str(tier["id"])] * record_count, width=32
            ),
            "record_geometry": np.asarray([geometry] * record_count, dtype=np.int32),
            "record_projections": _byte_string_array(
                [str(cell["projection"])] * record_count,
                width=8,
            ),
        },
        metadata={
            "algorithm": "nearest-vector-codeword",
            "projection": str(cell["projection"]),
            "dimension": dimension,
            "bits": bits,
            "codebook_size": 1 << bits,
            "record_geometry_fields": ["dimension", "bits", "codebook_size"],
            "records": _record_rows(
                expert_ids=expert_ids,
                projection=str(cell["projection"]),
                tier=str(tier["id"]),
                geometries=[geometry] * record_count,
                geometry_fields=("dimension", "bits", "codebook_size"),
            ),
        },
    )


def _packaged_qtip_record(
    source_root: Path,
    *,
    identity: tuple[int, int, str],
    geometry: tuple[int, int, int],
    weight_count: int,
) -> dict[str, Any]:
    """Load one hash-bound public QTIP solve unit without regenerating it."""

    import torch

    layer, expert, projection = identity
    sealed_root = source_root.resolve()
    config = sealed_root / f"L{layer:03d}" / f"E{expert:03d}_{projection}.json"
    receipt_path = (
        sealed_root
        / "solve"
        / f"L{layer:03d}"
        / f"E{expert:03d}_{projection}"
        / "QTIP_SOLVE_RECEIPT.json"
    )
    for label, path in (("config", config), ("receipt", receipt_path)):
        if path.is_symlink() or not path.is_file():
            raise BackpackPlanError(f"packaged QTIP {label} must be a regular file: {path}")
        try:
            path.resolve().relative_to(sealed_root)
        except (OSError, ValueError) as exc:
            raise BackpackPlanError(f"packaged QTIP {label} escapes source_root: {path}") from exc
    try:
        config_payload = json.loads(config.read_text())
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BackpackPlanError(f"cannot read packaged QTIP unit {identity!r}: {exc}") from exc
    expected_geometry = {key: value for key, value in zip(("L", "K", "V"), geometry)}
    if (
        not isinstance(config_payload, Mapping)
        or config_payload.get("layer") != layer
        or config_payload.get("expert") != expert
        or config_payload.get("projection") != projection
        or config_payload.get("geometry") != expected_geometry
        or not isinstance(receipt, Mapping)
        or receipt.get("schema") != "banana-smasher-qtip-solve-v1"
        or receipt.get("status") != "PASS"
        or receipt.get("layer") != layer
        or receipt.get("expert") != expert
        or receipt.get("projection") != projection
        or receipt.get("config_sha256") != _sha_file(config)
    ):
        raise BackpackPlanError(f"packaged QTIP unit identity mismatch: {identity!r}")
    artifact_value = receipt.get("artifact")
    if not isinstance(artifact_value, str):
        raise BackpackPlanError(f"packaged QTIP unit lacks artifact path: {identity!r}")
    artifact = Path(artifact_value)
    if not artifact.is_absolute():
        artifact = receipt_path.parent / artifact
    if artifact.is_symlink() or not artifact.is_file():
        raise BackpackPlanError(f"packaged QTIP artifact must be a regular file: {artifact}")
    try:
        artifact.resolve().relative_to(sealed_root)
    except (OSError, ValueError) as exc:
        raise BackpackPlanError(f"packaged QTIP artifact escapes source_root: {artifact}") from exc
    artifact_sha256 = _sha_file(artifact)
    if receipt.get("artifact_sha256") != artifact_sha256:
        raise BackpackPlanError(f"packaged QTIP artifact hash drift: {artifact}")
    try:
        payload = torch.load(
            artifact,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise BackpackPlanError(f"cannot load packaged QTIP artifact {artifact}: {exc}") from exc
    tensor_names = ("trellis", "SU", "SV", "Wscale", "tlut")
    payload_geometry = payload.get("geometry") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema")
        not in {"banana-smasher-qtip-unit-v1", "ds4-qtip-hyb-bounded36-unit-v1"}
        or not isinstance(payload_geometry, Mapping)
        or tuple(payload_geometry.get(key) for key in ("L", "K", "V")) != geometry
        or any(not isinstance(payload.get(name), torch.Tensor) for name in tensor_names)
    ):
        raise BackpackPlanError(f"packaged QTIP artifact schema mismatch: {artifact}")

    def tensor_bytes(name: str) -> bytes:
        return payload[name].detach().cpu().contiguous().numpy().tobytes(order="C")

    codes = tensor_bytes("trellis")
    if receipt.get("assignment_sha256") != _sha(codes):
        raise BackpackPlanError(f"packaged QTIP assignment hash drift: {artifact}")
    scale_parts = [tensor_bytes(name) for name in ("SU", "SV", "Wscale")]
    decoded_tensor = payload.get("reconstructed_weight")
    if not isinstance(decoded_tensor, torch.Tensor):
        if not torch.cuda.is_available():
            raise BackpackPlanError(
                "packaged QTIP artifact requires its bound CUDA decoder when it lacks "
                f"reconstructed_weight evidence: {artifact}"
            )
        try:
            from .solver_qtip_profile import (
                _config_path,
                _declared_public_qtip_runner,
                _load_public_qtip_runner,
            )

            bound_config = dict(config_payload)
            runner_path, runner_sha256 = _declared_public_qtip_runner(bound_config)
            runner = _load_public_qtip_runner(runner_path, runner_sha256)
            setattr(runner, "QTIP", _config_path(bound_config, "qtip_root"))
            _bitshift, _ldlq, _math_utils, kernel_decode = runner.load_official_qtip()
            device = torch.device("cuda")
            unit_geometry = payload["geometry"]
            codebook_l = int(unit_geometry["L"])
            codebook_k = int(unit_geometry["K"])
            codebook_v = int(unit_geometry["V"])
            tlut_bits = int(unit_geometry["tlut_bits"])
            rows, columns = [int(value) for value in payload["shape"]]
            tlut = payload["tlut"].float().to(device)
            index = torch.arange(1 << codebook_l, device=device)
            quadratic = (index + 1) * index
            sign_flip = 1 - ((quadratic >> (codebook_l - 1)) & 1) * 2
            lut_index = (quadratic >> (codebook_l - tlut_bits - 1)) & (
                (1 << tlut_bits) - 1
            )
            expanded = tlut[lut_index]
            expanded[:, 0] *= sign_flip
            raw = kernel_decode.decode_compressed(
                codebook_l,
                tlut_bits,
                codebook_k,
                codebook_v - 1,
                rows,
                columns,
                payload["trellis"].to(device).reshape(-1),
                expanded,
            )
            decoded_tensor = raw * payload["Wscale"].to(device)
            decoded_tensor = (
                runner.fwht(decoded_tensor.T).T
                * payload["SV"].float().to(device)[:, None]
            )
            decoded_tensor = runner.fwht(decoded_tensor) * payload["SU"].float().to(
                device
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise BackpackPlanError(
                f"cannot decode packaged QTIP artifact {artifact}: {exc}"
            ) from exc
    decoded = np.asarray(
        decoded_tensor.detach().cpu().contiguous().numpy(), dtype=np.float32
    ).reshape(-1)
    if decoded.size != weight_count or not np.isfinite(decoded).all():
        raise BackpackPlanError(
            f"packaged QTIP reconstructed weight size mismatch: {decoded.size} != {weight_count}"
        )
    return {
        "codes": codes,
        "scales": b"".join(scale_parts),
        "codebooks": tensor_bytes("tlut"),
        "decoded": np.ascontiguousarray(decoded),
        "source_unit": {
            "receipt": str(receipt_path),
            "receipt_sha256": _sha_file(receipt_path),
            "artifact_sha256": artifact_sha256,
        },
    }


def generate_qtip_backpack_candidate(
    run_root: str | Path,
    *,
    tier: Mapping[str, Any],
    cell: Mapping[str, Any],
    geometry_by_identity: Mapping[tuple[int, int, str], tuple[int, int, int]],
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Generate one public Backpack QTIP candidate artifact."""

    from .qtip_rings import qtip_ring_manifest, resolve_qtip_ring
    from .qtip_materialize import require_qtip_ring_manifest

    backend = tier.get("backend")
    source_path: Path | None = None
    if backend != "fixture_reference":
        source_root = tier.get("source_root")
        if not isinstance(source_root, str):
            raise BackpackPlanError("packaged_qtip requires source_root")
        source_path = Path(source_root).expanduser()
        if source_path.is_symlink() or not source_path.is_dir():
            raise BackpackPlanError(
                f"packaged_qtip source_root must be a regular directory: {source_path}"
            )
        require_qtip_ring_manifest(source_path, tier["bpw"])
        if weights is not None:
            raise BackpackPlanError(
                "packaged_qtip consumes sealed unit artifacts and cannot accept replacement weights"
            )
    ring = resolve_qtip_ring(tier["bpw"])
    identities = _cell_identities(cell)
    expert_ids = [identity[1] for identity in identities]
    projection = str(cell["projection"])
    packed_parts: list[bytes] = []
    codebooks: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    geometries: list[tuple[int, int, int]] = []
    decoded_rows: list[np.ndarray] = []
    payload_sizes: list[tuple[int, int, int]] = []
    source_rows = _expert_weight_rows(cell, weights)
    source_units: list[dict[str, Any]] = []
    for identity, source_weights in zip(identities, source_rows, strict=True):
        geometry = geometry_by_identity[identity]
        if backend == "fixture_reference":
            quantized = _quantize_qtip_reference_record(
                source_weights,
                geometry=geometry,
            )
            packed = bytes(quantized["packed"])
            codebook = np.asarray(quantized["lattice"])
            scale = np.asarray(quantized["scale"])
            decoded = np.asarray(quantized["decoded"], dtype=np.float32)
        else:
            assert source_path is not None
            packaged = _packaged_qtip_record(
                source_path,
                identity=identity,
                geometry=geometry,
                weight_count=source_weights.size,
            )
            packed = bytes(packaged["codes"])
            codebook = np.frombuffer(packaged["codebooks"], dtype=np.uint8).copy()
            scale = np.frombuffer(packaged["scales"], dtype=np.uint8).copy()
            decoded = np.asarray(packaged["decoded"], dtype=np.float32)
            source_units.append(dict(packaged["source_unit"]))
        packed_parts.append(packed)
        codebooks.append(codebook)
        scales.append(scale)
        geometries.append(geometry)
        decoded_rows.append(decoded)
        payload_sizes.append(
            (
                len(packed),
                int(scale.nbytes),
                int(codebook.nbytes),
            )
        )
    decoded = np.concatenate(decoded_rows)
    return _write_candidate_artifact(
        Path(run_root),
        tier=tier,
        cell=cell,
        decoded=np.asarray(decoded, dtype=np.float32),
        packed=b"".join(packed_parts),
        extra_arrays={
            "codebooks": np.concatenate(codebooks),
            "scales": np.concatenate(scales),
            "expert_ids": np.asarray(expert_ids, dtype=np.int16),
            "tensor_offsets": _payload_offsets(payload_sizes),
            "record_tiers": _byte_string_array([ring.tier] * len(expert_ids), width=32),
            "record_geometry": np.asarray(geometries, dtype=np.int32),
            "record_projections": _byte_string_array(
                [projection] * len(expert_ids),
                width=8,
            ),
        },
        metadata={
            "algorithm": (
                "qtip-fixture-reference"
                if backend == "fixture_reference"
                else "qtip-packaged-v1"
            ),
            "backend": backend,
            "bpw": float(tier["bpw"]),
            "projection": projection,
            "ring": qtip_ring_manifest(ring),
            **({"source_units": source_units} if source_units else {}),
            "record_geometry_fields": ["L", "K", "V"],
            "records": _record_rows(
                expert_ids=expert_ids,
                projection=projection,
                tier=ring.tier,
                geometries=geometries,
                geometry_fields=("L", "K", "V"),
            ),
        },
    )


def _stage_candidates(plan: BackpackPlan, root: Path, _prior: dict[str, Any]) -> dict[str, Any]:
    _manifest, cells = _load_cells(plan)
    tiers: list[dict[str, Any]] = []
    for tier in plan.tiers:
        qtip_geometries = (
            _exact_qtip_geometries(tier, cells)
            if tier["family"] == "qtip"
            else None
        )
        cell_rows = []
        for cell in cells:
            if tier["family"] == "vector_vq":
                receipt = generate_vector_vq_backpack_candidate(
                    root,
                    tier=tier,
                    cell=cell,
                )
            else:
                assert qtip_geometries is not None
                receipt = generate_qtip_backpack_candidate(
                    root,
                    tier=tier,
                    cell=cell,
                    geometry_by_identity=qtip_geometries,
                )
            cell_rows.append(
                {
                    "cell_id": cell["cell_id"],
                    "physical_bytes": receipt["physical_bytes"],
                    "projection": cell["projection"],
                    "receipt": str(
                        _candidate_root(root, str(tier["id"]), str(cell["cell_id"]))
                        / "RECEIPT.json"
                    ),
                }
            )
        tiers.append(
            {
                "tier": tier["id"],
                "family": tier["family"],
                **(
                    {"dimension": tier["dimension"]}
                    if tier["family"] == "vector_vq"
                    else {"bpw": float(tier["bpw"])}
                ),
                "cells": cell_rows,
            }
        )
    return {"candidate_tiers": tiers}


def _teacher_weights(
    plan: BackpackPlan, cells: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    model_weights = np.concatenate(
        [np.asarray(cell["weights"], dtype=np.float32) for cell in cells]
    )
    if plan.anchor["teacher"] == "model":
        return model_weights
    path = Path(plan.anchor["teacher"])
    if path.is_symlink() or not path.is_file():
        raise BackpackPlanError(f"anchor.teacher must be a regular NPY file: {path}")
    teacher = np.load(path, allow_pickle=False)
    if (
        teacher.dtype != np.float32
        or teacher.shape != model_weights.shape
        or not np.isfinite(teacher).all()
    ):
        raise BackpackPlanError(
            f"anchor.teacher must be finite float32{model_weights.shape}, got "
            f"{teacher.dtype}{teacher.shape}"
        )
    return np.ascontiguousarray(teacher)


def _candidate_weights(
    candidates: Mapping[str, Any], tier: str, cells: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    return np.concatenate(
        [
            np.load(
                candidate_artifact_root(
                    candidates, tier=tier, cell_id=str(cell["cell_id"])
                )
                / "decoded.npy"
            )
            for cell in cells
        ]
    ).astype(np.float32)


def _stage_candidate_anchor(
    plan: BackpackPlan, root: Path, _prior: dict[str, Any]
) -> dict[str, Any]:
    manifest, cells = _load_cells(plan)
    features, classes = _load_anchor(plan, weight_count=int(manifest["weight_count"]))
    teacher = _teacher_weights(plan, cells)
    rows = []
    for tier in plan.tiers:
        metrics = _anchor_metrics(
            features,
            classes,
            teacher,
            _candidate_weights(_prior["candidates"], str(tier["id"]), cells),
        )
        path = root / "anchors" / f"{tier['id']}.json"
        _atomic_json(path, metrics)
        rows.append(
            {
                "tier": tier["id"],
                "family": tier["family"],
                "overall": metrics["overall"],
                "by_class": metrics["by_class"],
                "receipt": str(path),
                "receipt_bytes": path.stat().st_size,
                "receipt_sha256": _sha_file(path),
            }
        )
    return {"anchors": rows, "same_instrument": True, "windows": 64}


def _stage_pred(plan: BackpackPlan, root: Path, _prior: dict[str, Any]) -> dict[str, Any]:
    manifest, cells = _load_cells(plan)
    features, classes = _load_anchor(plan, weight_count=int(manifest["weight_count"]))
    teacher = _teacher_weights(plan, cells)
    rows = []
    candidates = _prior["candidates"]
    for cell_index, cell in enumerate(cells):
        for tier in plan.tiers:
            pieces = [np.asarray(row["weights"], dtype=np.float32) for row in cells]
            artifact_root = candidate_artifact_root(
                candidates,
                tier=str(tier["id"]),
                cell_id=str(cell["cell_id"]),
            )
            pieces[cell_index] = np.load(artifact_root / "decoded.npy")
            metrics = _anchor_metrics(
                features, classes, teacher, np.concatenate(pieces).astype(np.float32)
            )
            candidate_receipt = json.loads((artifact_root / "RECEIPT.json").read_text())
            rows.append(
                {
                    "cell_id": cell["cell_id"],
                    "tier": tier["id"],
                    "family": tier["family"],
                    "physical_bytes": candidate_receipt["physical_bytes"],
                    "prediction_by_class": {
                        name: metrics["by_class"][name]["kld"] for name in CLASSES
                    },
                }
            )
    path = root / "pred" / "rows.json"
    _atomic_json(
        path,
        {"schema": "banana-smasher-backpack-pred-v1", "status": "PASS", "rows": rows},
    )
    return {
        "rows": rows,
        "receipt": str(path),
        "receipt_bytes": path.stat().st_size,
        "receipt_sha256": _sha_file(path),
    }


def materialize_backpack_source(
    source: Path,
    *,
    plan: BackpackPlan,
    cells: Sequence[Mapping[str, Any]],
    assignment: Sequence[Mapping[str, Any]],
    artifact_roots: Mapping[str, Path],
) -> None:
    if any(path.is_symlink() for path in (source, *source.parents)):
        raise BackpackPlanError(
            f"materializer destination must not contain a symlink: {source}"
        )
    if source.exists():
        shutil.rmtree(source)
    source.mkdir(parents=True)
    _atomic_json(
        source / "config.json",
        {
            "_name_or_path": plan.output["model_id"],
            "model_type": "deepseek_v4",
            "backpack_plan_schema": PLAN_SCHEMA,
        },
    )
    selected = {str(row["cell_id"]): row for row in assignment}
    expected_cells = {str(cell["cell_id"]) for cell in cells}
    if len(selected) != len(assignment) or set(selected) != expected_cells:
        raise BackpackPlanError(
            "materializer assignment must cover every model cell exactly once"
        )
    for selection_group in {str(cell["selection_group"]) for cell in cells}:
        group_tiers = {
            str(selected[str(cell["cell_id"])]["tier"])
            for cell in cells
            if str(cell["selection_group"]) == selection_group
        }
        if len(group_tiers) != 1:
            raise BackpackPlanError(
                f"materializer assignment disagrees across projections for {selection_group}"
            )
    tier_descriptors = {str(row["id"]): row for row in plan.tiers}
    from .contract import TIER_CODES

    tier_maps = {
        int(layer): np.full(256, 255, dtype=np.uint8)
        for layer in {int(cell["layer"]) for cell in cells}
    }
    payloads: dict[tuple[int, str], dict[str, list[np.ndarray]]] = {}
    ordered_cells = sorted(
        cells,
        key=lambda cell: (
            int(cell["layer"]),
            _PROJECTION_ORDER[str(cell["projection"])],
            str(cell["cell_id"]),
        ),
    )
    for cell in ordered_cells:
        cell_id = str(cell["cell_id"])
        tier = str(selected[cell_id]["tier"])
        descriptor = tier_descriptors[tier]
        if descriptor["family"] == "vector_vq":
            family = f"truevq_d{descriptor['dimension']}"
        else:
            family = "qtip2" if float(descriptor["bpw"]) < 3.0 else "qtip3"
        layer = int(cell["layer"])
        expert_ids = np.asarray(cell["expert_ids"], dtype=np.uint8)
        tier_maps[layer][expert_ids] = TIER_CODES[family]
        artifact_root = artifact_roots[cell_id]
        bucket = payloads.setdefault(
            (layer, family),
            {
                name: []
                for name in (
                    "codes",
                    "codebooks",
                    "scales",
                    "expert_ids",
                    "tensor_offsets",
                    "record_tiers",
                    "record_geometry",
                    "record_projections",
                    "record_boundaries",
                )
            },
        )
        prior_bytes = np.asarray(
            [
                sum(array.nbytes for array in bucket[name])
                for name in ("codes", "scales", "codebooks")
            ],
            dtype=np.int64,
        )
        prior_records = sum(array.size for array in bucket["expert_ids"])
        if bucket["expert_ids"]:
            bucket["record_boundaries"].append(
                np.full((1, 3), prior_records, dtype=np.int64)
            )
        codes = np.frombuffer((artifact_root / "wire.bin").read_bytes(), dtype=np.uint8)
        bucket["codes"].append(codes)
        for name in (
            "codebooks",
            "scales",
            "expert_ids",
            "record_tiers",
            "record_geometry",
            "record_projections",
        ):
            value = np.asarray(np.load(artifact_root / f"{name}.npy", allow_pickle=False))
            bucket[name].append(
                value.reshape(value.shape[0], -1)
                if name in {"record_geometry", "record_tiers", "record_projections"}
                else value if name == "codebooks" else value.reshape(-1)
            )
        offsets = np.asarray(
            np.load(artifact_root / "tensor_offsets.npy", allow_pickle=False), dtype=np.int64
        ).reshape(-1, 3)
        adjusted = offsets + prior_bytes
        bucket["tensor_offsets"].append(
            adjusted if not bucket["tensor_offsets"] else adjusted[1:]
        )

    for layer, tier_map in tier_maps.items():
        if np.any(tier_map == 255):
            raise BackpackPlanError(f"layer {layer} tier map is incomplete")
        destination = source / "layers" / f"layer_{layer:03d}" / "experts"
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / "tier_map.npy", tier_map, allow_pickle=False)
    for (layer, family), fields in payloads.items():
        destination = source / "layers" / f"layer_{layer:03d}" / family
        destination.mkdir(parents=True, exist_ok=True)
        for name, arrays in fields.items():
            if arrays:
                if name == "codebooks" and not all(
                    array.ndim == arrays[0].ndim
                    and array.shape[1:] == arrays[0].shape[1:]
                    for array in arrays
                ):
                    value = np.concatenate([array.reshape(-1) for array in arrays])
                else:
                    value = np.concatenate(arrays)
                np.save(destination / f"{name}.npy", value, allow_pickle=False)


def _target_whole_model_bytes(plan: BackpackPlan, manifest: Mapping[str, Any]) -> int:
    if "exact_bytes" in plan.target:
        return int(plan.target["exact_bytes"])
    return math.floor(
        int(manifest["weight_count"]) * float(plan.target["whole_model_bpw"]) / 8.0
    )


def _repair_state_bytes(pack_manifest: Mapping[str, Any]) -> int:
    rows = pack_manifest.get("files")
    if not isinstance(rows, list):
        raise BackpackPlanError("pack manifest files must be an array")
    repair_rows = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("role") == "repair_state"
    ]
    if len(repair_rows) > 1:
        raise BackpackPlanError("pack contains multiple repair_state files")
    if not repair_rows:
        return 0
    value = repair_rows[0].get("bytes")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BackpackPlanError("pack repair_state bytes must be a non-negative integer")
    return value


def _backpack_accounting(
    plan: BackpackPlan,
    pack_manifest: Mapping[str, Any],
    fixed: Sequence[Mapping[str, Any]],
    *,
    include_repair: bool,
) -> dict[str, Any]:
    tensor_index = pack_manifest.get("tensor_index")
    if not isinstance(tensor_index, Mapping):
        raise BackpackPlanError("pack manifest tensor_index must be an object")
    _model_path, model_manifest = _model_manifest(plan)
    tensor_bytes = sum(int(row["data_bytes"]) for row in tensor_index.values())
    fixed_bytes = sum(int(record["bytes"]) for record in fixed)
    repair_state_bytes = _repair_state_bytes(pack_manifest)
    declared_repair_bytes = (
        int(model_manifest.get("repair_bytes", 0))
        if plan.repair["method"] == REPAIR_BUNDLE_METHOD
        else 0
    )
    expected_repair_bytes = declared_repair_bytes if include_repair else 0
    if repair_state_bytes != expected_repair_bytes:
        raise BackpackPlanError(
            "pack repair-state byte accounting mismatch: "
            f"{repair_state_bytes} != {expected_repair_bytes}"
        )
    whole_model_bytes = tensor_bytes + fixed_bytes + repair_state_bytes
    target_bytes = _target_whole_model_bytes(plan, model_manifest)
    materialization_target_bytes = target_bytes - (
        declared_repair_bytes if not include_repair else 0
    )
    if whole_model_bytes != materialization_target_bytes:
        raise BackpackPlanError(
            f"materialized whole-model bytes {whole_model_bytes} "
            f"!= target {materialization_target_bytes}"
        )
    return {
        "tensor_bytes": tensor_bytes,
        "fixed_bytes": fixed_bytes,
        "repair_state_bytes": repair_state_bytes,
        "whole_model_bytes": whole_model_bytes,
        "target_whole_model_bytes": target_bytes,
        "materialization_target_bytes": materialization_target_bytes,
        "fixed_artifacts": list(fixed),
    }


def _attach_fixed_artifacts(
    plan: BackpackPlan, output: Path, *, include_repair: bool
) -> None:
    from .contract import MANIFEST_NAME

    _model_path, model_manifest = _model_manifest(plan)
    fixed = _fixed_artifacts(plan, model_manifest)
    pack_manifest_path = output / MANIFEST_NAME
    pack_manifest = json.loads(pack_manifest_path.read_text())
    file_rows = pack_manifest.get("files")
    tensor_index = pack_manifest.get("tensor_index")
    if not isinstance(file_rows, list) or not isinstance(tensor_index, Mapping):
        raise BackpackPlanError(f"invalid exported pack manifest: {pack_manifest_path}")
    for index, record in enumerate(fixed):
        source = Path(record["source"])
        relative = Path("backpack-fixed") / str(record["role"]) / f"{index:04d}-{source.name}"
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        file_rows.append(
            {
                "path": relative.as_posix(),
                "role": f"backpack_fixed_{record['role']}",
                "bytes": int(record["bytes"]),
                "sha256": str(record["sha256"]),
            }
        )
    pack_manifest["backpack_byte_accounting"] = _backpack_accounting(
        plan,
        pack_manifest,
        fixed,
        include_repair=include_repair,
    )
    _atomic_json(pack_manifest_path, pack_manifest)


def _verify_backpack_accounting(
    plan: BackpackPlan, output: Path, *, include_repair: bool
) -> None:
    from .contract import load_manifest

    pack_manifest = load_manifest(output)
    _model_path, model_manifest = _model_manifest(plan)
    expected_fixed = _fixed_artifacts(plan, model_manifest)
    accounting = pack_manifest.get("backpack_byte_accounting")
    if not isinstance(accounting, Mapping):
        raise BackpackPlanError(f"pack lacks Backpack byte accounting: {output}")
    expected = _backpack_accounting(
        plan,
        pack_manifest,
        expected_fixed,
        include_repair=include_repair,
    )
    if dict(accounting) != expected:
        raise BackpackPlanError(f"pack Backpack byte accounting mismatch: {output}")


def _configured_repair_bundle(plan: BackpackPlan) -> RepairBundle | None:
    if plan.repair["method"] != REPAIR_BUNDLE_METHOD:
        return None
    return load_repair_bundle(
        checkpoint=plan.repair["checkpoint"],
        checkpoint_sha256=plan.repair["checkpoint_sha256"],
        active_overlay=plan.repair["active_overlay"],
        active_overlay_sha256=plan.repair["active_overlay_sha256"],
        assignment=plan.repair["assignment"],
        assignment_sha256=plan.repair["assignment_sha256"],
        update=int(plan.repair["update"]),
    )


def _fixture_lifecycle_repair_bundle(
    plan: BackpackPlan, root: Path, serving_model_root: str | Path
) -> RepairBundle:
    """Bind the fixture repair to the same load-time replacement ABI as production."""

    pre_root = root / "materialized" / "pre-repair-source"
    final_root = root / "materialized" / "final-source"
    grouped: dict[str, list[tuple[str, int, np.ndarray]]] = {}
    for source_path in sorted(pre_root.rglob("codebooks.npy")):
        relative = source_path.relative_to(pre_root)
        final_path = final_root / relative
        source = np.asarray(np.load(source_path, allow_pickle=False))
        replacement = np.asarray(np.load(final_path, allow_pickle=False))
        if source.shape != replacement.shape or source.dtype != np.dtype("float16"):
            continue
        if source.ndim == 2:
            source_slices = source[None, ...]
            replacement_slices = replacement[None, ...]
        elif source.ndim == 3:
            source_slices = source
            replacement_slices = replacement
        else:
            continue
        for index, (source_slice, replacement_slice) in enumerate(
            zip(source_slices, replacement_slices, strict=True)
        ):
            if np.array_equal(source_slice, replacement_slice):
                continue
            source_sha256 = _sha(np.ascontiguousarray(source_slice).tobytes(order="C"))
            grouped.setdefault(source_sha256, []).append(
                (relative.as_posix(), index, np.ascontiguousarray(replacement_slice))
            )

    if not grouped:
        strength = np.float16(float(plan.repair["strength"]))
        for source_path in sorted(pre_root.rglob("codebooks.npy")):
            relative = source_path.relative_to(pre_root)
            source = np.asarray(np.load(source_path, allow_pickle=False))
            if source.dtype != np.dtype("float16") or source.ndim not in {2, 3}:
                continue
            source_slices = source[None, ...] if source.ndim == 2 else source
            for index, source_slice in enumerate(source_slices):
                replacement = np.ascontiguousarray(source_slice + strength)
                if not bool(np.isfinite(replacement).all()):
                    continue
                source_sha256 = _sha(
                    np.ascontiguousarray(source_slice).tobytes(order="C")
                )
                grouped.setdefault(source_sha256, []).append(
                    (relative.as_posix(), index, replacement)
                )

    selected: tuple[str, list[tuple[str, int, np.ndarray]]] | None = None
    for source_sha256, rows in sorted(grouped.items()):
        if all(np.array_equal(rows[0][2], row[2]) for row in rows[1:]):
            selected = (source_sha256, rows)
            break
    if selected is None:
        raise BackpackPlanError(
            "fixture lifecycle repair produced no codebook replacement compatible with RepairBundle"
        )
    source_sha256, replacement_rows = selected
    replacement = replacement_rows[0][2]

    serving_root = Path(serving_model_root).expanduser().resolve()
    index_path = serving_root / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text())
        weight_map = index["weight_map"]
    except Exception as exc:
        raise BackpackPlanError(
            f"fixture lifecycle repair requires a valid serving weight index: {exc}"
        ) from exc
    norm_name = next(
        (name for name in sorted(weight_map) if name.endswith("norm.weight")), None
    )
    output_weight = next(
        (
            name
            for name in sorted(weight_map)
            if re.fullmatch(r"model\.layers\.\d+\.self_attn\.o_b_proj\.weight", name)
        ),
        None,
    )
    if norm_name is None or output_weight is None:
        raise BackpackPlanError(
            "fixture lifecycle repair requires one norm and one attention o_b_proj weight"
        )
    from safetensors import safe_open

    with safe_open(serving_root / weight_map[norm_name], framework="np") as handle:
        norm = np.asarray(handle.get_tensor(norm_name), dtype=np.float32)
    strength = float(plan.repair["strength"])
    norm_replacement = np.ascontiguousarray(norm * np.float32(1.0 + strength))
    output_gain_name = output_weight.removesuffix(".weight") + ".output_log_gain"
    output_gain = np.asarray(np.log1p(strength), dtype=np.float32)

    descriptor = root / "repair" / "FIXTURE_LIFECYCLE_BUNDLE.json"
    descriptor_payload = {
        "schema": "banana-smasher-fixture-lifecycle-repair-v1",
        "status": "PASS",
        "codebook": {
            "source_wire_sha256": source_sha256,
            "replacement_wire_sha256": _sha(replacement.tobytes(order="C")),
            "matched_slices": len(replacement_rows),
        },
        "norm": norm_name,
        "output_gain": output_gain_name,
        "strength": strength,
    }
    _atomic_json(descriptor, descriptor_payload)
    active_overlay = root / "anchors" / "pre-repair-backpack.json"
    assignment = root / "materialized" / "ASSIGNMENT.json"
    return RepairBundle(
        checkpoint_path=descriptor,
        checkpoint_sha256=_sha_file(descriptor),
        active_overlay_path=active_overlay,
        active_overlay_sha256=_sha_file(active_overlay),
        assignment_path=assignment,
        assignment_sha256=_sha_file(assignment),
        checkpoint_format=REPAIR_FORMAT,
        mechanism=REPAIR_MECHANISM,
        update=1,
        codebooks={
            source_sha256: CodebookRepair(
                checkpoint_key=f"fixture/codebook_{source_sha256}",
                source_wire_sha256=source_sha256,
                array=replacement,
            )
        },
        dense_tensors={
            f"norms/{norm_name}": norm_replacement,
            f"outputs/{output_gain_name}": output_gain,
        },
        norm_count=1,
        output_count=1,
    )


def _verify_pack_repair_contract(
    output: Path, *, plan: BackpackPlan, repair: RepairBundle | None
) -> None:
    from .contract import load_manifest

    manifest = load_manifest(output)
    summary = manifest.get("repair")
    if repair is None:
        if plan.repair["method"] != "none" and summary is not None:
            raise BackpackPlanError(
                f"pack unexpectedly contains a repair payload for {plan.repair['method']}"
            )
        return
    if not isinstance(summary, Mapping):
        raise BackpackPlanError("pack is missing the configured repair payload")
    expected = {
        "checkpoint_sha256": repair.checkpoint_sha256,
        "active_overlay_sha256": repair.active_overlay_sha256,
        "assignment_sha256": repair.assignment_sha256,
        "update": repair.update,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise BackpackPlanError(f"pack repair payload mismatch for {key}")


def _export_verified_pack(
    source: Path,
    output: Path,
    *,
    plan: BackpackPlan,
    suffix: str,
    repair: RepairBundle | None = None,
) -> dict[str, Any]:
    from .contract import export_pack, verify_pack

    if output.is_symlink():
        raise BackpackPlanError(f"output pack must not be a direct symlink: {output}")
    if output.exists():
        verification = verify_pack(output)
        expected_instance = f"{plan.output['instance_id']}-{suffix}"
        if verification.get("instance_id") != expected_instance:
            raise BackpackPlanError(
                f"existing output pack has instance {verification.get('instance_id')!r}, "
                f"expected {expected_instance!r}: {output}"
            )
        _verify_backpack_accounting(
            plan, output, include_repair=repair is not None
        )
        _verify_pack_repair_contract(output, plan=plan, repair=repair)
        return verification
    export_pack(
        source_root=source,
        output=output,
        model_id=plan.output["model_id"],
        instance_id=f"{plan.output['instance_id']}-{suffix}",
        link_mode="copy",
        repair=repair,
        runtime_floor_bytes=0,
    )
    _attach_fixed_artifacts(plan, output, include_repair=repair is not None)
    verification = verify_pack(output)
    _verify_backpack_accounting(plan, output, include_repair=repair is not None)
    _verify_pack_repair_contract(output, plan=plan, repair=repair)
    return verification


def _stage_solve_materialize(
    plan: BackpackPlan, root: Path, prior: dict[str, Any]
) -> dict[str, Any]:
    inspect = prior["inspect"]
    pred_rows = prior["pred"]["rows"]
    tiers = [str(tier["id"]) for tier in plan.tiers]
    _manifest, model_cells = _load_cells(plan)
    groups = sorted({str(cell["selection_group"]) for cell in model_cells})
    cell_group = {
        str(cell["cell_id"]): str(cell["selection_group"]) for cell in model_cells
    }
    rows_by_option = {
        (str(row["cell_id"]), str(row["tier"])): row for row in pred_rows
    }
    bytes_by_option = {
        (group, tier): sum(
            int(rows_by_option[(cell_id, tier)]["physical_bytes"])
            for cell_id, candidate_group in cell_group.items()
            if candidate_group == group
        )
        for group in groups
        for tier in tiers
    }
    class_costs = {
        (group, tier): {
            name: math.fsum(
                float(rows_by_option[(cell_id, tier)]["prediction_by_class"][name])
                for cell_id, candidate_group in cell_group.items()
                if candidate_group == group
            )
            for name in CLASSES
        }
        for group in groups
        for tier in tiers
    }
    from .knapsack import solve_class_balanced_options

    solved = solve_class_balanced_options(
        cells=groups,
        tiers=tiers,
        bytes_by_option=bytes_by_option,
        class_costs_by_option=class_costs,
        envelope_bytes=int(inspect["payload_envelope_bytes"]),
        class_caps=dict(plan.prediction["class_caps"]),
        exact_envelope=True,
    )
    selected_by_group = {
        str(row["cell_id"]): str(row["tier"]) for row in solved["assignments"]
    }
    assignment = [
        {
            "cell_id": str(cell["cell_id"]),
            "selection_group": str(cell["selection_group"]),
            "tier": selected_by_group[str(cell["selection_group"])],
            "bytes": int(
                rows_by_option[
                    (
                        str(cell["cell_id"]),
                        selected_by_group[str(cell["selection_group"])],
                    )
                ]["physical_bytes"]
            ),
        }
        for cell in model_cells
    ]
    if sum(int(row["bytes"]) for row in assignment) != int(solved["assigned_bytes"]):
        raise RuntimeError("coupled assignment byte accounting drift")
    artifact_roots = {
        str(row["cell_id"]): candidate_artifact_root(
            prior["candidates"],
            tier=str(row["tier"]),
            cell_id=str(row["cell_id"]),
        )
        for row in assignment
    }
    source = root / "materialized" / "pre-repair-source"
    materialize_backpack_source(
        source,
        plan=plan,
        cells=model_cells,
        assignment=assignment,
        artifact_roots=artifact_roots,
    )
    pre_pack = root / "pre-repair-pack"
    verification = _export_verified_pack(source, pre_pack, plan=plan, suffix="pre-repair")
    fixed = int(inspect["fixed_total_bytes"])
    whole = fixed + int(solved["assigned_bytes"])
    if whole > int(inspect["target_whole_model_bytes"]):
        raise RuntimeError("exact solver violated whole-model envelope")
    assignment_path = root / "materialized" / "ASSIGNMENT.json"
    _atomic_json(
        assignment_path,
        {
            "schema": "banana-smasher-backpack-assignment-v1",
            "status": "PASS",
            "assignments": assignment,
            "byte_accounting": {
                "candidate_payload_bytes": solved["assigned_bytes"],
                "fixed_bytes": fixed,
                "whole_model_bytes": whole,
                "target_whole_model_bytes": inspect["target_whole_model_bytes"],
                "slack_bytes": inspect["target_whole_model_bytes"] - whole,
            },
        },
    )
    return {
        "assignment": assignment,
        "assignment_receipt": str(assignment_path),
        "byte_accounting": {
            "candidate_payload_bytes": solved["assigned_bytes"],
            "fixed_bytes": fixed,
            "whole_model_bytes": whole,
            "target_whole_model_bytes": inspect["target_whole_model_bytes"],
            "slack_bytes": inspect["target_whole_model_bytes"] - whole,
        },
        "pre_repair_pack": str(pre_pack),
        "pre_repair_pack_manifest_sha256": _sha_file(
            pre_pack / "BANANA_PACK_MANIFEST.json"
        ),
        "pack_verification": verification,
        "solver": solved["solver"],
    }


def _selected_weights(
    candidates: Mapping[str, Any],
    assignment: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    selected = {
        str(row["cell_id"]): np.load(
            candidate_artifact_root(
                candidates,
                tier=str(row["tier"]),
                cell_id=str(row["cell_id"]),
            )
            / "decoded.npy"
        ).astype(np.float32)
        for row in assignment
    }
    return selected, np.concatenate([selected[str(cell["cell_id"])] for cell in cells])


def _unpack_backpack_indices(packed: bytes, *, bits: int, count: int) -> np.ndarray:
    expected_bytes = (bits * count + 7) // 8
    if len(packed) != expected_bytes:
        raise BackpackPlanError(
            f"packed record has {len(packed)} bytes; expected {expected_bytes}"
        )
    values = np.empty(count, dtype=np.int32)
    for index in range(count):
        bit_offset = index * bits
        value = 0
        for bit in range(bits):
            offset = bit_offset + bit
            value |= ((packed[offset // 8] >> (offset % 8)) & 1) << bit
        values[index] = value
    return values


def _record_payload_bytes(
    array: np.ndarray, offsets: np.ndarray, *, record: int, column: int
) -> bytes:
    start = int(offsets[record, column])
    stop = int(offsets[record + 1, column])
    return bytes(np.asarray(array).view(np.uint8).reshape(-1)[start:stop])


def _identity_bound_packaged_qtip_cell(
    candidates: Mapping[str, Any],
    *,
    tier: str,
    cell: Mapping[str, Any],
    final_arrays: Mapping[str, np.ndarray],
    final_offsets: np.ndarray,
    final_records: Sequence[int],
) -> np.ndarray:
    """Use packaged decoded evidence only when every exported record is byte-identical."""

    root = candidate_artifact_root(
        candidates,
        tier=tier,
        cell_id=str(cell["cell_id"]),
    )
    candidate_arrays: dict[str, np.ndarray] = {
        name: np.load(root / f"{name}.npy", allow_pickle=False)
        for name in (
            "scales",
            "codebooks",
            "expert_ids",
            "tensor_offsets",
            "record_tiers",
            "record_geometry",
            "record_projections",
        )
    }
    candidate_arrays["codes"] = np.frombuffer(
        (root / "wire.bin").read_bytes(), dtype=np.uint8
    )
    candidate_projections = _candidate_label_rows(
        candidate_arrays["record_projections"], width=8
    )
    final_projections = _candidate_label_rows(
        final_arrays["record_projections"], width=8
    )
    candidate_tiers = _candidate_label_rows(candidate_arrays["record_tiers"], width=32)
    final_tiers = _candidate_label_rows(final_arrays["record_tiers"], width=32)
    if any(
        value is None
        for value in (candidate_projections, final_projections, candidate_tiers, final_tiers)
    ):
        raise BackpackPlanError("packaged QTIP record labels are invalid")
    assert candidate_projections is not None
    assert final_projections is not None
    assert candidate_tiers is not None
    assert final_tiers is not None
    candidate_experts = np.asarray(candidate_arrays["expert_ids"], dtype=np.int16).reshape(-1)
    final_experts = np.asarray(final_arrays["expert_ids"], dtype=np.int16).reshape(-1)
    candidate_geometry = np.asarray(candidate_arrays["record_geometry"], dtype=np.int32)
    final_geometry = np.asarray(final_arrays["record_geometry"], dtype=np.int32)
    candidate_offsets = np.asarray(candidate_arrays["tensor_offsets"], dtype=np.int64)
    candidate_by_identity = {
        (int(expert), str(projection)): index
        for index, (expert, projection) in enumerate(
            zip(candidate_experts, candidate_projections, strict=True)
        )
    }
    if len(candidate_by_identity) != len(candidate_experts):
        raise BackpackPlanError("packaged QTIP candidate record identities are not unique")
    for final_record in final_records:
        identity = (int(final_experts[final_record]), str(final_projections[final_record]))
        candidate_record = candidate_by_identity.get(identity)
        if candidate_record is None:
            raise BackpackPlanError(
                f"packaged QTIP candidate lacks exported record {identity!r}"
            )
        if (
            not np.array_equal(
                candidate_geometry[candidate_record], final_geometry[final_record]
            )
            or candidate_tiers[candidate_record] != final_tiers[final_record]
        ):
            raise BackpackPlanError(
                f"packaged QTIP candidate metadata does not match exported record {identity!r}"
            )
        for field, column in (("codes", 0), ("scales", 1), ("codebooks", 2)):
            candidate_payload = _record_payload_bytes(
                candidate_arrays[field],
                candidate_offsets,
                record=candidate_record,
                column=column,
            )
            final_payload = _record_payload_bytes(
                final_arrays[field],
                final_offsets,
                record=final_record,
                column=column,
            )
            if candidate_payload != final_payload:
                raise BackpackPlanError(
                    f"packaged QTIP candidate {field} does not match exported record {identity!r}"
                )
    decoded = np.load(root / "decoded.npy", allow_pickle=False).astype(np.float32)
    if decoded.size != int(np.asarray(cell["weights"]).size):
        raise BackpackPlanError(
            f"packaged QTIP decoded evidence size mismatch for cell {cell['cell_id']}"
        )
    return decoded


def _final_pack_weights(
    plan: BackpackPlan,
    output: Path,
    assignment: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Any],
) -> np.ndarray:
    """Decode the exported wire artifact used by final Anchor64 scoring."""

    from .loader import PackLoader

    selected_tiers = {str(row["cell_id"]): str(row["tier"]) for row in assignment}
    descriptors = {str(row["id"]): row for row in plan.tiers}
    loader = PackLoader(output)
    decoded_cells: dict[str, np.ndarray] = {}
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault(int(cell["layer"]), []).append(cell)
    for layer, layer_cells in grouped.items():
        with loader.open_layer(layer, framework="np") as view:
            family_arrays: dict[str, dict[str, np.ndarray]] = {}
            for cell in layer_cells:
                cell_id = str(cell["cell_id"])
                descriptor = descriptors[selected_tiers[cell_id]]
                family = (
                    f"truevq_d{descriptor['dimension']}"
                    if descriptor["family"] == "vector_vq"
                    else "qtip2" if float(descriptor["bpw"]) < 3.0 else "qtip3"
                )
                arrays = family_arrays.setdefault(
                    family,
                    {
                        name: np.asarray(value)
                        for name, value in view.family(family).items()
                    },
                )
                required = {
                    "codes",
                    "scales",
                    "codebooks",
                    "expert_ids",
                    "tensor_offsets",
                    "record_tiers",
                    "record_geometry",
                    "record_projections",
                }
                if not required <= set(arrays):
                    raise BackpackPlanError(
                        f"final pack lacks scoring tensors for layer {layer} {family}"
                    )
                projections = _candidate_label_rows(arrays["record_projections"], width=8)
                if projections is None:
                    raise BackpackPlanError("final pack contains invalid projection metadata")
                expert_ids = np.asarray(arrays["expert_ids"], dtype=np.int16).reshape(-1)
                expected_experts = {int(value) for value in cell["expert_ids"]}
                record_indexes = [
                    index
                    for index, (expert, projection) in enumerate(
                        zip(expert_ids, projections, strict=True)
                    )
                    if int(expert) in expected_experts
                    and projection == str(cell["projection"])
                ]
                if (
                    len(record_indexes) != len(expected_experts)
                    or {int(expert_ids[index]) for index in record_indexes}
                    != expected_experts
                ):
                    raise BackpackPlanError(
                        f"final pack record routing mismatch for cell {cell_id}"
                    )
                offsets = np.asarray(arrays["tensor_offsets"], dtype=np.int64)
                geometry = np.asarray(arrays["record_geometry"], dtype=np.int32)
                if (
                    descriptor["family"] == "qtip"
                    and descriptor.get("backend") == "packaged_qtip"
                ):
                    decoded_cells[cell_id] = _identity_bound_packaged_qtip_cell(
                        candidates,
                        tier=selected_tiers[cell_id],
                        cell=cell,
                        final_arrays=arrays,
                        final_offsets=offsets,
                        final_records=record_indexes,
                    )
                    continue
                record_weights: list[np.ndarray] = []
                cell_weight_count = int(np.asarray(cell["weights"]).size)
                expert_count = len(cell["expert_ids"])
                if cell_weight_count % expert_count:
                    raise BackpackPlanError(
                        f"cell {cell_id} weights cannot be divided across its experts"
                    )
                weight_count = cell_weight_count // expert_count
                for record in record_indexes:
                    codes = _record_payload_bytes(
                        arrays["codes"], offsets, record=record, column=0
                    )
                    codebooks = _record_payload_bytes(
                        arrays["codebooks"], offsets, record=record, column=2
                    )
                    if descriptor["family"] == "vector_vq":
                        dimension, bits, size = (
                            int(value) for value in geometry[record]
                        )
                        if weight_count % dimension:
                            raise BackpackPlanError(
                                f"final pack vector geometry does not divide cell {cell_id}"
                            )
                        codebook = np.frombuffer(codebooks, dtype=np.float16)
                        if codebook.size != size * dimension:
                            raise BackpackPlanError(
                                f"final pack codebook geometry mismatch for cell {cell_id}"
                            )
                        indices = _unpack_backpack_indices(
                            codes, bits=bits, count=weight_count // dimension
                        )
                        if np.any(indices >= size):
                            raise BackpackPlanError(
                                f"final pack code index exceeds its codebook for cell {cell_id}"
                            )
                        record_weights.append(
                            np.asarray(
                                codebook.reshape(size, dimension)[indices].reshape(-1),
                                dtype=np.float32,
                            )
                        )
                    else:
                        _length, bits, _vector = (
                            int(value) for value in geometry[record]
                        )
                        lattice = np.frombuffer(codebooks, dtype=np.float16)
                        if lattice.size != 1 << bits:
                            raise BackpackPlanError(
                                f"final pack QTIP lattice mismatch for cell {cell_id}"
                            )
                        states = _unpack_backpack_indices(
                            codes, bits=bits, count=weight_count
                        )
                        signs = np.where(
                            np.arange(weight_count) % 2, -1.0, 1.0
                        ).astype(np.float32)
                        record_weights.append(
                            np.asarray(lattice[states], dtype=np.float32) * signs
                        )
                decoded_cells[cell_id] = np.concatenate(record_weights)
    return np.concatenate([decoded_cells[str(cell["cell_id"])] for cell in cells])


def _stage_pre_repair_anchor(
    plan: BackpackPlan, root: Path, prior: dict[str, Any]
) -> dict[str, Any]:
    manifest, cells = _load_cells(plan)
    features, classes = _load_anchor(plan, weight_count=int(manifest["weight_count"]))
    _selected, weights = _selected_weights(
        prior["candidates"], prior["solve_materialize"]["assignment"], cells
    )
    metrics = _anchor_metrics(features, classes, _teacher_weights(plan, cells), weights)
    path = root / "anchors" / "pre-repair-backpack.json"
    _atomic_json(path, metrics)
    return {
        "metrics": metrics,
        "receipt": str(path),
        "receipt_bytes": path.stat().st_size,
        "receipt_sha256": _sha_file(path),
    }


def _stage_repair(plan: BackpackPlan, root: Path, prior: dict[str, Any]) -> dict[str, Any]:
    _manifest, cells = _load_cells(plan)
    assignment = prior["solve_materialize"]["assignment"]
    selected, _weights = _selected_weights(prior["candidates"], assignment, cells)
    strength = float(plan.repair.get("strength", 0.0))
    repaired = {}
    artifact_roots: dict[str, Path] = {}
    tier_descriptors = {str(row["id"]): row for row in plan.tiers}
    qtip_geometries = {
        str(row["id"]): _exact_qtip_geometries(row, cells)
        for row in plan.tiers
        if row["family"] == "qtip"
    }
    selected_tiers = {str(row["cell_id"]): str(row["tier"]) for row in assignment}
    for cell in cells:
        cell_id = str(cell["cell_id"])
        tier = tier_descriptors[selected_tiers[cell_id]]
        if plan.repair["method"] in {"none", REPAIR_BUNDLE_METHOD}:
            repaired[cell_id] = selected[cell_id].astype(np.float32)
            artifact_roots[cell_id] = candidate_artifact_root(
                prior["candidates"],
                tier=selected_tiers[cell_id],
                cell_id=cell_id,
            )
            continue
        updated = apply_residual_update(
            np.asarray(cell["weights"], dtype=np.float32),
            selected[cell_id],
            strength=strength,
        )
        if tier["family"] == "vector_vq":
            generate_vector_vq_backpack_candidate(
                root / "repair",
                tier=tier,
                cell=cell,
                weights=updated,
            )
        else:
            generate_qtip_backpack_candidate(
                root / "repair",
                tier=tier,
                cell=cell,
                geometry_by_identity=qtip_geometries[str(tier["id"])],
                weights=updated,
            )
        artifact_roots[cell_id] = _candidate_root(
            root / "repair", selected_tiers[cell_id], cell_id
        )
        repaired[cell_id] = np.load(artifact_roots[cell_id] / "decoded.npy").astype(
            np.float32
        )
    source = root / "materialized" / "final-source"
    materialize_backpack_source(
        source,
        plan=plan,
        cells=cells,
        assignment=assignment,
        artifact_roots=artifact_roots,
    )
    output = Path(plan.output["pack"])
    bundle = _configured_repair_bundle(plan)
    verification = _export_verified_pack(
        source,
        output,
        plan=plan,
        suffix="final",
        repair=bundle,
    )
    arrays = root / "repair" / "cells"
    arrays.mkdir(parents=True, exist_ok=True)
    for cell_id, value in repaired.items():
        np.save(arrays / f"{cell_id}.npy", value, allow_pickle=False)
    receipt = {
        "schema": "banana-smasher-backpack-repair-v1",
        "status": "PASS",
        "method": plan.repair["method"],
        "strength": strength,
        "pack": str(output),
        "pack_manifest_sha256": _sha_file(output / "BANANA_PACK_MANIFEST.json"),
        "pack_verification": verification,
        "cells": len(repaired),
        **(
            {
                "repair_bundle": {
                    "checkpoint_sha256": bundle.checkpoint_sha256,
                    "active_overlay_sha256": bundle.active_overlay_sha256,
                    "assignment_sha256": bundle.assignment_sha256,
                    "update": bundle.update,
                }
            }
            if bundle is not None
            else {}
        ),
    }
    path = root / "repair" / "RECEIPT.json"
    _atomic_json(path, receipt)
    return {**receipt, "receipt": str(path)}


def _stage_final_score(plan: BackpackPlan, root: Path, prior: dict[str, Any]) -> dict[str, Any]:
    manifest, cells = _load_cells(plan)
    features, classes = _load_anchor(plan, weight_count=int(manifest["weight_count"]))
    final_weights = _final_pack_weights(
        plan,
        Path(plan.output["pack"]),
        prior["solve_materialize"]["assignment"],
        cells,
        prior["candidates"],
    )
    metrics = _anchor_metrics(features, classes, _teacher_weights(plan, cells), final_weights)
    candidate_table = [
        {
            "tier": row["tier"],
            "family": row["family"],
            "kld": row["overall"]["kld"],
            "top1": row["overall"]["top1"],
        }
        for row in prior["candidate_anchor"]["anchors"]
    ]
    receipt = {
        "schema": FINAL_SCHEMA,
        "status": "PASS",
        "plan_schema": PLAN_SCHEMA,
        "model_revision": plan.model["revision"],
        "candidate_tiers": prior["candidates"]["candidate_tiers"],
        "assignment": prior["solve_materialize"]["assignment"],
        "byte_accounting": prior["solve_materialize"]["byte_accounting"],
        "pre_repair_anchor": prior["pre_repair_anchor"]["metrics"],
        "repair": prior["repair"],
        "final_anchor": metrics,
        "final_pack": plan.output["pack"],
        "scored_pack_manifest_sha256": prior["repair"]["pack_manifest_sha256"],
        "table": {
            "candidates": candidate_table,
            "pre_repair": prior["pre_repair_anchor"]["metrics"]["overall"],
            "final": metrics["overall"],
        },
    }
    path = root / "FINAL_RECEIPT.json"
    _atomic_json(path, receipt)
    return receipt


_STAGE_RUNNERS = {
    "inspect": _stage_inspect,
    "candidates": _stage_candidates,
    "candidate_anchor": _stage_candidate_anchor,
    "pred": _stage_pred,
    "solve_materialize": _stage_solve_materialize,
    "pre_repair_anchor": _stage_pre_repair_anchor,
    "repair": _stage_repair,
    "final_score": _stage_final_score,
}


def _load_stage(
    path: Path,
    *,
    stage: str,
    plan_sha256: str,
    prior_stage_sha256: Mapping[str, str],
) -> dict[str, Any] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if (
        receipt.get("schema") == STAGE_SCHEMA
        and receipt.get("status") == "PASS"
        and receipt.get("stage") == stage
        and receipt.get("plan_sha256") == plan_sha256
        and receipt.get("prior_stage_sha256") == dict(prior_stage_sha256)
        and isinstance(receipt.get("result"), dict)
    ):
        return dict(receipt["result"])
    return None


def _validate_bound_file(
    record: object,
    *,
    path_field: str = "path",
    sha_field: str = "sha256",
    bytes_field: str = "bytes",
) -> bool:
    if not isinstance(record, Mapping) or not isinstance(record.get(path_field), str):
        return False
    path = Path(record[path_field])
    if path.is_symlink() or not path.is_file():
        return False
    expected_bytes = record.get(bytes_field)
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
    ):
        return False
    if _sha_file(path) != record.get(sha_field):
        return False
    return path.stat().st_size == expected_bytes


def _validate_status_json_receipt(value: object) -> bool:
    if not isinstance(value, str):
        return False
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, Mapping) and payload.get("status") == "PASS"


def _validate_stage_artifacts(
    stage: str,
    result: Mapping[str, Any],
    *,
    plan: BackpackPlan,
    run_root: Path,
) -> bool:
    """Re-verify materialized pack outputs before admitting a stage receipt."""

    if stage == "inspect":
        if not _validate_bound_file(result.get("model_manifest")):
            return False
        if not _validate_bound_file(result.get("anchor_bank")):
            return False
        teacher = result.get("teacher")
        if not isinstance(teacher, Mapping):
            return False
        if teacher.get("kind") == "model":
            manifest = result.get("model_manifest")
            if not isinstance(manifest, Mapping) or teacher.get(
                "model_manifest_sha256"
            ) != manifest.get("sha256"):
                return False
        elif teacher.get("kind") == "npy":
            if not _validate_bound_file(teacher):
                return False
        else:
            return False
        cells = result.get("cell_artifacts")
        fixed = result.get("fixed_artifacts")
        if not isinstance(cells, list) or not isinstance(fixed, list):
            return False
        if not all(_validate_bound_file(row) for row in cells) or not all(
            _validate_bound_file(row, path_field="source") for row in fixed
        ):
            return False
        reuse = result.get("receipt_reuse")
        if reuse is None:
            return True
        if not isinstance(reuse, Mapping) or not _validate_bound_file(
            reuse,
            path_field="receipt",
            sha_field="receipt_sha256",
            bytes_field="receipt_bytes",
        ):
            return False
        rows = reuse.get("receipts")
        return isinstance(rows, list) and all(_validate_bound_file(row) for row in rows)
    if stage == "candidates":
        tier_rows = result.get("candidate_tiers")
        if not isinstance(tier_rows, list):
            return False
        try:
            _manifest, cells = _load_cells(plan)
        except Exception:
            return False
        tiers_by_id = {str(tier["id"]): tier for tier in plan.tiers}
        cells_by_id = {str(cell["cell_id"]): cell for cell in cells}
        if len(tier_rows) != len(tiers_by_id):
            return False
        seen_tiers: set[str] = set()
        for tier_row in tier_rows:
            if not isinstance(tier_row, Mapping) or not isinstance(
                tier_row.get("cells"), list
            ):
                return False
            tier_id = tier_row.get("tier")
            if not isinstance(tier_id, str) or tier_id in seen_tiers:
                return False
            seen_tiers.add(tier_id)
            expected_tier = tiers_by_id.get(tier_id)
            if (
                expected_tier is None
                or tier_row.get("family") != expected_tier["family"]
            ):
                return False
            qtip_geometries = (
                _exact_qtip_geometries(expected_tier, cells)
                if expected_tier["family"] == "qtip"
                else None
            )
            cell_rows = tier_row["cells"]
            if len(cell_rows) != len(cells_by_id):
                return False
            seen_cells: set[str] = set()
            for cell_row in cell_rows:
                if not isinstance(cell_row, Mapping):
                    return False
                cell_id = cell_row.get("cell_id")
                if not isinstance(cell_id, str) or cell_id in seen_cells:
                    return False
                seen_cells.add(cell_id)
                expected_cell = cells_by_id.get(cell_id)
                if (
                    expected_cell is None
                    or cell_row.get("projection") != expected_cell["projection"]
                    or not _validate_candidate_receipt(
                        cell_row.get("receipt"),
                        tier=expected_tier,
                        cell=expected_cell,
                        geometry_by_identity=qtip_geometries,
                    )
                ):
                    return False
        return True
    if stage == "candidate_anchor":
        anchor_rows = result.get("anchors")
        if not isinstance(anchor_rows, list):
            return False
        return all(
            isinstance(row, Mapping)
            and _validate_bound_file(
                row,
                path_field="receipt",
                sha_field="receipt_sha256",
                bytes_field="receipt_bytes",
            )
            and _validate_status_json_receipt(row.get("receipt"))
            for row in anchor_rows
        )
    if stage == "pred":
        return _validate_bound_file(
            result,
            path_field="receipt",
            sha_field="receipt_sha256",
            bytes_field="receipt_bytes",
        ) and _validate_status_json_receipt(result.get("receipt"))
    if stage == "pre_repair_anchor":
        return _validate_bound_file(
            result,
            path_field="receipt",
            sha_field="receipt_sha256",
            bytes_field="receipt_bytes",
        ) and _validate_status_json_receipt(result.get("receipt"))
    if stage not in {"solve_materialize", "repair", "final_score"}:
        return True
    from .contract import verify_pack

    if stage == "solve_materialize":
        pack_value = result.get("pre_repair_pack")
        expected_instance = f"{plan.output['instance_id']}-pre-repair"
    else:
        pack_value = plan.output["pack"]
        expected_instance = f"{plan.output['instance_id']}-final"
    if not isinstance(pack_value, str):
        return False
    pack = Path(pack_value)
    if pack.is_symlink() or not pack.is_dir():
        return False
    try:
        verification = verify_pack(pack)
        _verify_backpack_accounting(
            plan,
            pack,
            include_repair=(
                stage != "solve_materialize"
                and plan.repair["method"] == REPAIR_BUNDLE_METHOD
            ),
        )
    except Exception:
        return False
    if verification.get("instance_id") != expected_instance:
        return False
    manifest_path = pack / "BANANA_PACK_MANIFEST.json"
    manifest_sha_field = {
        "solve_materialize": "pre_repair_pack_manifest_sha256",
        "repair": "pack_manifest_sha256",
        "final_score": "scored_pack_manifest_sha256",
    }[stage]
    if result.get(manifest_sha_field) != _sha_file(manifest_path):
        return False
    if stage == "final_score":
        final_receipt = run_root / "FINAL_RECEIPT.json"
        if final_receipt.is_symlink() or not final_receipt.is_file():
            return False
        try:
            return json.loads(final_receipt.read_text()) == dict(result)
        except (OSError, json.JSONDecodeError):
            return False
    return True


def _candidate_label_rows(array: np.ndarray, *, width: int) -> list[str] | None:
    if array.dtype != np.uint8 or array.ndim != 2 or array.shape[1] != width:
        return None
    labels: list[str] = []
    for row in array:
        raw = bytes(row)
        value, separator, padding = raw.partition(b"\0")
        if separator and any(padding):
            return None
        try:
            labels.append(value.decode("utf-8"))
        except UnicodeDecodeError:
            return None
    return labels


def _validate_packaged_qtip_units(
    receipt: Mapping[str, Any],
    *,
    tier: Mapping[str, Any],
    identities: Sequence[tuple[int, int, str]],
    geometries: Sequence[tuple[int, int, int]],
) -> bool:
    from .qtip_materialize import require_qtip_ring_manifest

    source_root = Path(str(tier.get("source_root", ""))).expanduser()
    if source_root.is_symlink() or not source_root.is_dir():
        return False
    try:
        require_qtip_ring_manifest(source_root, tier["bpw"])
        sealed_root = source_root.resolve()
    except (OSError, TypeError, ValueError):
        return False
    rows = receipt.get("source_units")
    if not isinstance(rows, list) or len(rows) != len(identities):
        return False
    for row, identity, geometry in zip(rows, identities, geometries, strict=True):
        if not isinstance(row, Mapping) or not isinstance(row.get("receipt"), str):
            return False
        layer, expert, projection = identity
        source_receipt = Path(row["receipt"]).expanduser()
        if source_receipt.is_symlink() or not source_receipt.is_file():
            return False
        try:
            source_receipt.resolve().relative_to(sealed_root)
            source_payload = json.loads(source_receipt.read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        if (
            not isinstance(source_payload, Mapping)
            or source_payload.get("schema") != "banana-smasher-qtip-solve-v1"
            or source_payload.get("status") != "PASS"
            or source_payload.get("layer") != layer
            or source_payload.get("expert") != expert
            or source_payload.get("projection") != projection
            or row.get("receipt_sha256") != _sha_file(source_receipt)
        ):
            return False
        config = sealed_root / f"L{layer:03d}" / f"E{expert:03d}_{projection}.json"
        if config.is_symlink() or not config.is_file():
            return False
        try:
            config_payload = json.loads(config.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        expected_geometry = {key: value for key, value in zip(("L", "K", "V"), geometry)}
        if (
            not isinstance(config_payload, Mapping)
            or config_payload.get("layer") != layer
            or config_payload.get("expert") != expert
            or config_payload.get("projection") != projection
            or config_payload.get("geometry") != expected_geometry
            or source_payload.get("config_sha256") != _sha_file(config)
        ):
            return False
        artifact_value = source_payload.get("artifact")
        if not isinstance(artifact_value, str):
            return False
        artifact = Path(artifact_value)
        if not artifact.is_absolute():
            artifact = source_receipt.parent / artifact
        if artifact.is_symlink() or not artifact.is_file():
            return False
        try:
            artifact.resolve().relative_to(sealed_root)
        except (OSError, ValueError):
            return False
        if (
            source_payload.get("artifact_sha256") != _sha_file(artifact)
            or row.get("artifact_sha256") != source_payload.get("artifact_sha256")
            or not isinstance(source_payload.get("assignment_sha256"), str)
        ):
            return False
    return True


def _validate_candidate_receipt(
    value: object,
    *,
    tier: Mapping[str, Any],
    cell: Mapping[str, Any],
    geometry_by_identity: Mapping[
        tuple[int, int, str], tuple[int, int, int]
    ]
    | None = None,
) -> bool:
    if not isinstance(value, str):
        return False
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        return False
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema") != "banana-smasher-backpack-candidate-cell-v1"
        or receipt.get("status") != "PASS"
        or receipt.get("tier") != tier["id"]
        or receipt.get("family") != tier["family"]
        or receipt.get("cell_id") != cell["cell_id"]
    ):
        return False
    arrays = receipt.get("arrays", [])
    if not isinstance(arrays, list):
        return False
    records = [receipt.get("wire"), receipt.get("decoded"), *arrays]
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            return False
        artifact = Path(record["path"])
        if artifact.is_symlink() or not artifact.is_file():
            return False
        try:
            artifact.resolve().relative_to(path.parent.resolve())
        except ValueError:
            return False
        if _sha_file(artifact) != record.get("sha256"):
            return False
        if "bytes" in record and artifact.stat().st_size != record["bytes"]:
            return False
    named = {
        str(record.get("name")): Path(str(record["path"]))
        for record in arrays
        if isinstance(record, Mapping)
        and isinstance(record.get("name"), str)
        and isinstance(record.get("path"), str)
    }
    required = {
        "codebooks",
        "scales",
        "expert_ids",
        "tensor_offsets",
        "record_tiers",
        "record_geometry",
        "record_projections",
    }
    if len(named) != len(arrays) or set(named) != required:
        return False
    try:
        codebooks = np.asarray(np.load(named["codebooks"], allow_pickle=False))
        scales = np.asarray(np.load(named["scales"], allow_pickle=False))
        expert_ids = np.asarray(np.load(named["expert_ids"], allow_pickle=False))
        offsets = np.asarray(np.load(named["tensor_offsets"], allow_pickle=False))
        tiers = np.asarray(np.load(named["record_tiers"], allow_pickle=False))
        geometry = np.asarray(np.load(named["record_geometry"], allow_pickle=False))
        projections = np.asarray(
            np.load(named["record_projections"], allow_pickle=False)
        )
        decoded = np.asarray(
            np.load(Path(str(receipt["decoded"]["path"])), allow_pickle=False)
        )
    except Exception:
        return False
    expected_experts = [int(expert) for expert in cell["expert_ids"]]
    projection = str(cell["projection"])
    if expert_ids.dtype != np.int16 or expert_ids.tolist() != expected_experts:
        return False
    record_count = expert_ids.size
    tier_labels = _candidate_label_rows(tiers, width=32)
    projection_labels = _candidate_label_rows(projections, width=8)
    wire = receipt.get("wire")
    if not isinstance(wire, Mapping) or not isinstance(wire.get("bytes"), int):
        return False
    physical_bytes = int(wire["bytes"]) + sum(
        int(value.nbytes)
        for value in (
            codebooks,
            scales,
            expert_ids,
            offsets,
            tiers,
            geometry,
            projections,
        )
    )
    if (
        receipt.get("physical_bytes") != physical_bytes
        or receipt.get("weight_count") != int(decoded.size)
        or decoded.dtype != np.float32
        or decoded.size != np.asarray(cell["weights"]).size
        or receipt.get("projection") != projection
        or tier_labels is None
        or projection_labels != [projection] * record_count
    ):
        return False
    if tier["family"] == "vector_vq":
        dimension, bits = _tier_geometry(tier)
        expected_geometry = [(dimension, bits, 1 << bits)] * record_count
        expected_tiers = [str(tier["id"])] * record_count
        expected_records = _record_rows(
            expert_ids=expected_experts,
            projection=projection,
            tier=str(tier["id"]),
            geometries=expected_geometry,
            geometry_fields=("dimension", "bits", "codebook_size"),
        )
        if (
            receipt.get("algorithm") != "nearest-vector-codeword"
            or receipt.get("dimension") != dimension
            or receipt.get("bits") != bits
            or receipt.get("codebook_size") != 1 << bits
            or receipt.get("record_geometry_fields")
            != ["dimension", "bits", "codebook_size"]
        ):
            return False
    else:
        from .qtip_rings import qtip_ring_manifest, resolve_qtip_ring

        if geometry_by_identity is None:
            return False
        ring = resolve_qtip_ring(tier["bpw"])
        identities = _cell_identities(cell)
        try:
            expected_geometry = [geometry_by_identity[identity] for identity in identities]
        except KeyError:
            return False
        expected_tiers = [ring.tier] * record_count
        expected_records = _record_rows(
            expert_ids=expected_experts,
            projection=projection,
            tier=ring.tier,
            geometries=expected_geometry,
            geometry_fields=("L", "K", "V"),
        )
        backend = str(tier["backend"])
        expected_algorithm = (
            "qtip-fixture-reference"
            if backend == "fixture_reference"
            else "qtip-packaged-v1"
        )
        if (
            receipt.get("algorithm") != expected_algorithm
            or receipt.get("backend") != backend
            or receipt.get("bpw") != float(tier["bpw"])
            or receipt.get("ring") != qtip_ring_manifest(ring)
            or receipt.get("record_geometry_fields") != ["L", "K", "V"]
        ):
            return False
        if backend == "packaged_qtip" and not _validate_packaged_qtip_units(
            receipt,
            tier=tier,
            identities=identities,
            geometries=expected_geometry,
        ):
            return False
    return (
        offsets.dtype == np.int64
        and offsets.shape == (record_count + 1, 3)
        and np.array_equal(offsets[0], np.zeros(3, dtype=np.int64))
        and bool(np.all(np.diff(offsets, axis=0) >= 0))
        and offsets[-1].tolist()
        == [int(wire["bytes"]), int(scales.nbytes), int(codebooks.nbytes)]
        and tier_labels == expected_tiers
        and geometry.dtype == np.int32
        and geometry.shape == (record_count, 3)
        and [tuple(int(value) for value in row) for row in geometry]
        == expected_geometry
        and receipt.get("records") == expected_records
    )


def _load_verified_stage(
    path: Path,
    *,
    stage: str,
    plan_sha256: str,
    prior_stage_sha256: Mapping[str, str],
    plan: BackpackPlan,
) -> dict[str, Any] | None:
    result = _load_stage(
        path,
        stage=stage,
        plan_sha256=plan_sha256,
        prior_stage_sha256=prior_stage_sha256,
    )
    if result is None or not _validate_stage_artifacts(
        stage, result, plan=plan, run_root=path.parent.parent
    ):
        return None
    return result


def _bound_reuse_receipts(
    receipts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not receipts:
        return []
    bound = []
    for index, raw in enumerate(receipts):
        row = _object(raw, f"receipts[{index}]")
        role = _nonempty(row.get("role"), f"receipts[{index}].role")
        source_input = Path(
            _nonempty(row.get("path"), f"receipts[{index}].path")
        ).expanduser()
        if source_input.is_symlink() or not source_input.is_file():
            raise BackpackPlanError(f"reused receipt must be a regular file: {source_input}")
        source = source_input.resolve()
        expected = _nonempty(row.get("sha256"), f"receipts[{index}].sha256")
        actual = _sha_file(source)
        if actual != expected:
            raise BackpackPlanError(
                f"reused receipt SHA-256 mismatch for {role}: expected {expected}, got {actual}"
            )
        try:
            payload = json.loads(source.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise BackpackPlanError(f"reused receipt is not valid JSON: {source}") from exc
        if not isinstance(payload, Mapping):
            raise BackpackPlanError(f"reused receipt must be a JSON object: {source}")
        admission = row.get("admission", "admitted")
        if admission not in {"admitted", "evidence_only"}:
            raise BackpackPlanError(
                f"receipts[{index}].admission must be admitted or evidence_only"
            )
        if "schema" in row and payload.get("schema") != row["schema"]:
            raise BackpackPlanError(
                f"reused receipt schema mismatch for {role}: "
                f"{payload.get('schema')!r} != {row['schema']!r}"
            )
        if "stage" in row:
            if row["stage"] not in STAGES:
                raise BackpackPlanError(
                    f"reused receipt stage must be one of {', '.join(STAGES)}"
                )
            if payload.get("stage") != row["stage"]:
                raise BackpackPlanError(
                    f"reused receipt stage mismatch for {role}: "
                    f"{payload.get('stage')!r} != {row['stage']!r}"
                )
        status = str(payload.get("status", payload.get("outcome", "UNKNOWN")))
        if admission == "admitted" and any(
            marker in status.upper() for marker in ("FAIL", "REJECT", "QUARANTIN")
        ):
            raise BackpackPlanError(
                f"cannot admit {role} receipt with status {status!r}; bind it evidence_only"
            )
        bound.append(
            {
                "role": role,
                "admission": admission,
                "path": str(source),
                "bytes": source.stat().st_size,
                "sha256": actual,
                "schema": payload.get("schema"),
                "source_status": status,
                **({"stage": payload.get("stage")} if payload.get("stage") in STAGES else {}),
                "payload": dict(payload),
            }
        )
    return bound


def _import_reusable_stage_chain(
    *,
    source_root: Path,
    destination_root: Path,
    stage: str,
    plan: BackpackPlan,
    plan_sha256: str,
    expected_sha256: str | None = None,
    imported: set[str] | None = None,
) -> None:
    imported = set() if imported is None else imported
    if stage in imported:
        return
    stage_index = STAGES.index(stage) + 1
    source = _stage_path(source_root, stage_index, stage)
    if source.is_symlink() or not source.is_file():
        raise BackpackPlanError(f"reused stage {stage} source is not a regular file")
    actual_sha256 = _sha_file(source)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise BackpackPlanError(
            f"reused stage {stage} receipt mismatch: {actual_sha256!r} != {expected_sha256!r}"
        )
    try:
        payload = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BackpackPlanError(f"reused stage {stage} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise BackpackPlanError(f"reused stage {stage} payload must be an object")
    if payload.get("schema") != STAGE_SCHEMA or payload.get("status") != "PASS":
        raise BackpackPlanError(f"reused stage {stage} must be a passing {STAGE_SCHEMA} receipt")
    if payload.get("stage") != stage:
        raise BackpackPlanError(f"reused stage {stage} identity mismatch")
    if payload.get("plan_sha256") != plan_sha256:
        raise BackpackPlanError(
            f"reused stage {stage} plan mismatch: "
            f"{payload.get('plan_sha256')!r} != {plan_sha256!r}"
        )
    prior = payload.get("prior_stage_sha256")
    if not isinstance(prior, Mapping):
        raise BackpackPlanError(f"reused stage {stage} lacks prior stage bindings")
    for prior_stage, prior_sha in prior.items():
        if prior_stage not in STAGES or not isinstance(prior_sha, str):
            raise BackpackPlanError(
                f"reused stage {stage} has invalid prior binding: {prior_stage!r}"
            )
        _import_reusable_stage_chain(
            source_root=source_root,
            destination_root=destination_root,
            stage=prior_stage,
            plan=plan,
            plan_sha256=plan_sha256,
            expected_sha256=prior_sha,
            imported=imported,
        )
    result = payload.get("result")
    if not isinstance(result, Mapping) or not _validate_stage_artifacts(
        stage,
        result,
        plan=plan,
        run_root=source_root,
    ):
        raise BackpackPlanError(f"reused stage {stage} artifacts are not valid")
    destination = _stage_path(destination_root, stage_index, stage)
    if destination.is_file():
        if _sha_file(destination) != actual_sha256:
            raise BackpackPlanError(f"reused stage {stage} destination mismatch")
        imported.add(stage)
        return
    _atomic_bytes(destination, source.read_bytes())
    imported.add(stage)


def _admit_reusable_stage_receipts(
    plan: BackpackPlan,
    *,
    root: Path,
    plan_sha256: str,
) -> None:
    reusable = _bound_reuse_receipts(plan.reuse_receipts)
    indexed = {
        str(row.get("stage")): row
        for row in reusable
        if row["admission"] == "admitted"
        and row.get("schema") == STAGE_SCHEMA
        and row.get("stage") in REUSABLE_STAGE_IMPORTS
    }
    imported: set[str] = set()
    for stage in STAGES:
        row = indexed.get(stage)
        if row is None:
            continue
        _import_reusable_stage_chain(
            source_root=Path(row["path"]).parent.parent,
            destination_root=root,
            stage=stage,
            plan=plan,
            plan_sha256=plan_sha256,
            expected_sha256=row["sha256"],
            imported=imported,
        )


def _bind_run(
    plan: BackpackPlan | Mapping[str, Any], run_root: str | Path
) -> tuple[BackpackPlan, Path, str]:
    parsed = plan if isinstance(plan, BackpackPlan) else BackpackPlan.from_mapping(plan)
    root_input = Path(run_root).expanduser()
    if root_input.is_symlink():
        raise BackpackPlanError(f"run root must not be a direct symlink: {root_input}")
    root = root_input.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in (
        "PLAN.json",
        "stages",
        "candidates",
        "anchors",
        "pred",
        "materialized",
        "pre-repair-pack",
        "repair",
        "REUSED_RECEIPTS.json",
        "FINAL_RECEIPT.json",
    ):
        path = root / name
        if path.is_symlink():
            raise BackpackPlanError(f"reserved run output path must not be a symlink: {path}")
    plan_payload = _canonical_bytes(parsed.as_mapping())
    plan_sha256 = _execution_plan_sha(parsed)
    plan_path = root / "PLAN.json"
    if plan_path.exists() and plan_path.read_bytes() != plan_payload:
        raise BackpackPlanError("run root is already bound to a different Backpack plan")
    _atomic_bytes(plan_path, plan_payload)
    if parsed.reuse_receipts:
        _admit_reusable_stage_receipts(parsed, root=root, plan_sha256=plan_sha256)
    return parsed, root, plan_sha256


def _execute_public_stage(
    plan: BackpackPlan | Mapping[str, Any], *, run_root: str | Path, stage: str
) -> dict[str, Any]:
    parsed, root, plan_sha256 = _bind_run(plan, run_root)
    index = STAGES.index(stage) + 1
    path = _stage_path(root, index, stage)
    prior: dict[str, Any] = {}
    prior_stage_sha256: dict[str, str] = {}
    for prior_index, prior_stage in enumerate(STAGES[: index - 1], 1):
        prior_path = _stage_path(root, prior_index, prior_stage)
        prior_result = _load_verified_stage(
            prior_path,
            stage=prior_stage,
            plan_sha256=plan_sha256,
            prior_stage_sha256=prior_stage_sha256,
            plan=parsed,
        )
        if prior_result is None:
            raise BackpackPlanError(
                f"stage {stage} requires completed stage {prior_stage}; "
                f"run {prior_stage} first"
            )
        prior[prior_stage] = prior_result
        prior_stage_sha256[prior_stage] = _sha_file(prior_path)
    existing = _load_verified_stage(
        path,
        stage=stage,
        plan_sha256=plan_sha256,
        prior_stage_sha256=prior_stage_sha256,
        plan=parsed,
    )
    if existing is not None:
        return existing
    runner = _STAGE_RUNNERS[stage]
    try:
        result = runner(parsed, root) if stage == "inspect" else runner(parsed, root, prior)
    except Exception as exc:
        _atomic_json(
            path,
            {
                "schema": STAGE_SCHEMA,
                "status": "FAIL",
                "stage": stage,
                "stage_index": index,
                "plan_sha256": plan_sha256,
                "prior_stage_sha256": prior_stage_sha256,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    result = {"status": "PASS", **result}
    _atomic_json(
        path,
        {
            "schema": STAGE_SCHEMA,
            "status": "PASS",
            "stage": stage,
            "stage_index": index,
            "plan_sha256": plan_sha256,
            "prior_stage_sha256": prior_stage_sha256,
            "result": result,
        },
    )
    return result


def inspect_backpack(
    plan: BackpackPlan | Mapping[str, Any], *, run_root: str | Path
) -> dict[str, Any]:
    """Bind model geometry, Anchor64 inputs, and whole-model byte target."""
    return _execute_public_stage(plan, run_root=run_root, stage="inspect")


def generate_backpack_candidates(
    plan: BackpackPlan | Mapping[str, Any], *, run_root: str | Path
) -> dict[str, Any]:
    """Generate every declared D4, D8, and QTIP candidate tier."""
    return _execute_public_stage(plan, run_root=run_root, stage="candidates")


def anchor_backpack_candidates(
    plan: BackpackPlan | Mapping[str, Any], *, run_root: str | Path
) -> dict[str, Any]:
    """Measure every candidate tier with the same Anchor64 instrument."""
    return _execute_public_stage(plan, run_root=run_root, stage="candidate_anchor")


def predict_backpack(
    plan: BackpackPlan | Mapping[str, Any], *, run_root: str | Path
) -> dict[str, Any]:
    """Build per-cell, per-tier six-class prediction rows."""
    return _execute_public_stage(plan, run_root=run_root, stage="pred")


def solve_backpack(
    plan: BackpackPlan | Mapping[str, Any], *, run_root: str | Path
) -> dict[str, Any]:
    """Solve exact bytes and materialize a verified pre-repair bs-pack."""
    return _execute_public_stage(plan, run_root=run_root, stage="solve_materialize")


def anchor_backpack(
    plan: BackpackPlan | Mapping[str, Any], *, run_root: str | Path
) -> dict[str, Any]:
    """Measure the selected mixed Backpack before repair."""
    return _execute_public_stage(plan, run_root=run_root, stage="pre_repair_anchor")


def repair_backpack(
    plan: BackpackPlan | Mapping[str, Any], *, run_root: str | Path
) -> dict[str, Any]:
    """Run repair/update and materialize the final verified pack."""
    return _execute_public_stage(plan, run_root=run_root, stage="repair")


def score_backpack(
    plan: BackpackPlan | Mapping[str, Any], *, run_root: str | Path
) -> dict[str, Any]:
    """Run final Anchor64 and emit the combined final receipt/table."""
    return _execute_public_stage(plan, run_root=run_root, stage="final_score")


_PUBLIC_STAGE_APIS = (
    "inspect_backpack",
    "generate_backpack_candidates",
    "anchor_backpack_candidates",
    "predict_backpack",
    "solve_backpack",
    "anchor_backpack",
    "repair_backpack",
    "score_backpack",
)


def reuse_backpack_receipts(
    receipts: Sequence[Mapping[str, Any]], *, output: str | Path
) -> dict[str, Any]:
    """Bind completed campaign receipts without replaying model execution.

    ``admission='admitted'`` rejects receipts whose status is explicitly failed,
    rejected, or quarantined. ``admission='evidence_only'`` preserves such a
    receipt as diagnostic input but never promotes it into a Backpack solve.
    """

    if not receipts:
        raise BackpackPlanError("receipt reuse requires at least one receipt")
    bound = [
        {key: value for key, value in row.items() if key != "payload"}
        for row in _bound_reuse_receipts(receipts)
    ]
    result = {
        "schema": "banana-smasher-backpack-receipt-reuse-v1",
        "status": "PASS",
        "execution": {
            "transformer_replay": False,
            "candidate_generation": False,
            "anchor_replay": False,
        },
        "receipts": bound,
        "admitted": sum(row["admission"] == "admitted" for row in bound),
        "evidence_only": sum(row["admission"] == "evidence_only" for row in bound),
    }
    unresolved_destination = Path(output).expanduser()
    if unresolved_destination.is_symlink():
        raise BackpackPlanError("receipt reuse output must not be a direct symlink")
    destination = unresolved_destination.resolve()
    _atomic_json(destination, result)
    return {
        **result,
        "receipt": str(destination),
        "receipt_sha256": _sha_file(destination),
        "receipt_bytes": destination.stat().st_size,
    }


def build_backpack(
    plan: BackpackPlan | Mapping[str, Any], *, run_root: str | Path
) -> dict[str, Any]:
    """Execute or resume the complete eight-stage Backpack construction DAG."""

    parsed, root, plan_sha256 = _bind_run(plan, run_root)
    resumed: list[str] = []
    final_result: dict[str, Any] | None = None
    prior_stage_sha256: dict[str, str] = {}
    for index, (stage, stage_api_name) in enumerate(zip(STAGES, _PUBLIC_STAGE_APIS), 1):
        path = _stage_path(root, index, stage)
        existing = _load_verified_stage(
            path,
            stage=stage,
            plan_sha256=plan_sha256,
            prior_stage_sha256=prior_stage_sha256,
            plan=parsed,
        )
        if existing is not None:
            resumed.append(stage)
        final_result = globals()[stage_api_name](parsed, run_root=root)
        prior_stage_sha256[stage] = _sha_file(path)

    assert final_result is not None
    final = dict(final_result)
    final_path = root / "FINAL_RECEIPT.json"
    final.update(
        {
            "stages": list(STAGES),
            "resumed_stages": resumed,
            "run_root": str(root),
            "final_receipt": str(final_path),
            "final_receipt_sha256": _sha_file(final_path),
        }
    )
    return final


def _completed_run_results(root: Path, through: str) -> tuple[BackpackPlan, dict[str, Any]]:
    plan_path = root / "PLAN.json"
    if root.is_symlink() or plan_path.is_symlink() or not plan_path.is_file():
        raise BackpackPlanError(f"Backpack run root is missing a regular PLAN.json: {root}")
    plan = BackpackPlan.from_mapping(json.loads(plan_path.read_text()))
    status = status_backpack(root)
    required = STAGES[: STAGES.index(through) + 1]
    states = {row["stage"]: row["status"] for row in status["stages"]}
    incomplete = [stage for stage in required if states.get(stage) != "PASS"]
    if incomplete:
        raise BackpackPlanError(
            f"lifecycle export requires completed stage {incomplete[0]}: {root}"
        )
    results = {
        stage: json.loads(_stage_path(root, index, stage).read_text())["result"]
        for index, stage in enumerate(required, 1)
    }
    return plan, results


def _pack_shape_receipt(manifest: Mapping[str, Any], root: Path) -> dict[str, Any]:
    plane_shapes = [
        {
            "name": name,
            "shape": list(row["shape"]),
            "dtype": str(row["dtype"]),
        }
        for name, row in sorted(manifest["tensor_index"].items())
    ]
    base_shapes: list[dict[str, Any]] = []
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file() and not index_path.is_symlink():
        from safetensors import safe_open

        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map", {})
        for shard in sorted(set(weight_map.values())):
            with safe_open(root / shard, framework="np") as handle:
                for name in sorted(handle.keys()):
                    base_shapes.append(
                        {
                            "name": name,
                            "shape": list(handle.get_slice(name).get_shape()),
                        }
                    )
    whole_shapes = {"expert_planes": plane_shapes, "base_tensors": base_shapes}
    wire_layout = [
        {
            "name": name,
            "shape": list(row["shape"]),
            "dtype": str(row["dtype"]),
            "data_bytes": int(row["data_bytes"]),
        }
        for name, row in sorted(manifest["tensor_index"].items())
    ]
    payload_roles = {"npy_plane", "banana_smasher_raw_plane", "base_weights_shard"}
    return {
        "expert_plane_bytes": sum(
            int(row["data_bytes"]) for row in manifest["tensor_index"].values()
        ),
        "base_weight_file_bytes": sum(
            int(row["bytes"])
            for row in manifest["files"]
            if row.get("role") == "base_weights_shard"
        ),
        "repair_state_bytes": sum(
            int(row["bytes"])
            for row in manifest["files"]
            if row.get("role") == "repair_state"
        ),
        "metadata_bytes": sum(
            int(row["bytes"])
            for row in manifest["files"]
            if row.get("role") not in payload_roles | {"repair_state"}
        ),
        "expert_wire_layout_sha256": _sha(_canonical_bytes(wire_layout)),
        "whole_model_shape_sha256": _sha(_canonical_bytes(whole_shapes)),
    }


def export_backpack_lifecycle(
    run_root: str | Path,
    *,
    lifecycle: str,
    output: str | Path,
    serving_model_root: str | Path,
    kernel_cache_root: str | Path | None = None,
    tier: str | None = None,
) -> dict[str, Any]:
    """Export uniform, selected pre-repair, or selected post-repair through bs-pack."""

    if lifecycle not in {"uniform-anchor", "pre-repair", "post-repair"}:
        raise BackpackPlanError(
            "lifecycle must be uniform-anchor, pre-repair, or post-repair"
        )
    root_input = Path(run_root).expanduser()
    if root_input.is_symlink():
        raise BackpackPlanError(f"run root must not be a direct symlink: {root_input}")
    root = root_input.resolve()
    output_input = Path(output).expanduser()
    if output_input.is_symlink():
        raise BackpackPlanError(f"lifecycle output must not be a direct symlink: {output_input}")
    destination = output_input.resolve()
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    cache_input = (
        Path(kernel_cache_root).expanduser()
        if kernel_cache_root is not None
        else Path(serving_model_root).expanduser() / "kernel-cache"
    )
    if cache_input.is_symlink() or not (
        cache_input / "BS_KERNEL_CACHE_MANIFEST.json"
    ).is_file():
        raise BackpackPlanError(
            "lifecycle export requires a regular portable kernel cache: "
            f"{cache_input}"
        )

    through = "candidates" if lifecycle == "uniform-anchor" else (
        "solve_materialize" if lifecycle == "pre-repair" else "repair"
    )
    plan, results = _completed_run_results(root, through)
    _manifest, cells = _load_cells(plan)
    candidates = results["candidates"]
    if lifecycle == "uniform-anchor":
        selected_tier = _safe_id(tier, "tier")
        if selected_tier not in {str(row["id"]) for row in plan.tiers}:
            raise BackpackPlanError(f"uniform lifecycle tier is not declared: {selected_tier}")
        assignment = [
            {"cell_id": str(cell["cell_id"]), "tier": selected_tier}
            for cell in cells
        ]
        artifact_roots = {
            str(cell["cell_id"]): candidate_artifact_root(
                candidates,
                tier=selected_tier,
                cell_id=str(cell["cell_id"]),
            )
            for cell in cells
        }
        scratch = Path(tempfile.mkdtemp(prefix=".lifecycle-uniform-", dir=root))
        source = scratch / "source"
        materialize_backpack_source(
            source,
            plan=plan,
            cells=cells,
            assignment=assignment,
            artifact_roots=artifact_roots,
        )
        repair = None
    else:
        assignment = list(results["solve_materialize"]["assignment"])
        scratch = None
        if lifecycle == "pre-repair":
            repair = None
            source = root / "materialized" / "pre-repair-source"
        else:
            repair = _configured_repair_bundle(plan)
            if repair is None and plan.repair["method"] == "fixture_residual":
                repair = _fixture_lifecycle_repair_bundle(
                    plan, root, serving_model_root
                )
            source = root / "materialized" / (
                "pre-repair-source" if repair is not None else "final-source"
            )

    assignment_sha256 = _sha(_canonical_bytes(assignment))
    instance_suffix = f"uniform-{tier}" if lifecycle == "uniform-anchor" else lifecycle
    from .contract import (
        MANIFEST_NAME,
        export_pack,
        verify_pack,
        verify_serve_compatibility,
    )

    try:
        manifest = export_pack(
            source_root=source,
            output=destination,
            model_id=plan.output["model_id"],
            instance_id=f"{plan.output['instance_id']}-{instance_suffix}",
            link_mode="copy",
            repair=repair,
            serving_model_root=serving_model_root,
            kernel_cache_root=cache_input,
            runtime_floor_bytes=0,
        )
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
    manifest["backpack_lifecycle"] = {
        "schema": "banana-smasher-backpack-lifecycle-v1",
        "lifecycle": lifecycle,
        "tier": tier,
        "assignment_sha256": assignment_sha256,
    }
    _atomic_json(destination / MANIFEST_NAME, manifest)
    verification = verify_pack(destination)
    serve_compatibility = verify_serve_compatibility(
        destination,
        destination / "kernel-cache",
        architecture="sm_120",
    )
    assignment_map = {str(row["cell_id"]): str(row["tier"]) for row in assignment}
    return {
        "schema": "banana-smasher-backpack-lifecycle-export-v1",
        "status": "PASS",
        "lifecycle": lifecycle,
        "tier": tier,
        "run_root": str(root),
        "model": str(destination),
        "assignment": assignment_map,
        "assignment_rows": assignment,
        "assignment_sha256": assignment_sha256,
        "pack_manifest_sha256": _sha_file(destination / MANIFEST_NAME),
        "verification": verification,
        "serve_compatibility": serve_compatibility,
        "repair": manifest.get("repair"),
        **_pack_shape_receipt(manifest, destination),
    }


def status_backpack(run_root: str | Path) -> dict[str, Any]:
    """Report completed stages and the first incomplete/failed DAG boundary."""

    root = Path(run_root).expanduser().resolve()
    parsed_plan: BackpackPlan | None = None
    plan_sha256: str | None = None
    plan_path = root / "PLAN.json"
    if not plan_path.is_symlink() and plan_path.is_file():
        try:
            parsed_plan = BackpackPlan.from_mapping(json.loads(plan_path.read_text()))
            plan_sha256 = _execution_plan_sha(parsed_plan)
        except (OSError, json.JSONDecodeError, BackpackPlanError):
            pass
    completed = []
    failed = None
    first_incomplete = None
    stages = []
    prior_stage_sha256: dict[str, str] = {}
    chain_valid = True
    for index, stage in enumerate(STAGES, 1):
        path = _stage_path(root, index, stage)
        state = "MISSING"
        error = None
        if path.is_file():
            try:
                receipt = json.loads(path.read_text())
                state = str(receipt.get("status", "INVALID"))
                error = receipt.get("error")
                if (
                    state == "PASS"
                    and parsed_plan is not None
                    and plan_sha256 is not None
                    and (
                        not chain_valid
                        or _load_verified_stage(
                            path,
                            stage=stage,
                            plan_sha256=plan_sha256,
                            prior_stage_sha256=prior_stage_sha256,
                            plan=parsed_plan,
                        )
                        is None
                    )
                ):
                    state = "INVALID"
            except (OSError, json.JSONDecodeError):
                state = "INVALID"
        if state == "PASS":
            prior_stage_sha256[stage] = _sha_file(path)
        else:
            chain_valid = False
        if state == "PASS" and first_incomplete is None:
            completed.append(stage)
        elif first_incomplete is None:
            first_incomplete = stage
            if state == "FAIL":
                failed = stage
        stages.append(
            {
                "index": index,
                "stage": stage,
                "status": state,
                "receipt": str(path),
                **({"error": error} if error is not None else {}),
            }
        )
    return {
        "schema": "banana-smasher-backpack-status-v1",
        "status": "PASS" if first_incomplete is None else "INCOMPLETE",
        "run_root": str(root),
        "completed_stages": len(completed),
        "first_incomplete_stage": first_incomplete,
        "failed_stage": failed,
        "stages": stages,
        "final_receipt": str(root / "FINAL_RECEIPT.json")
        if (root / "FINAL_RECEIPT.json").is_file()
        else None,
    }
