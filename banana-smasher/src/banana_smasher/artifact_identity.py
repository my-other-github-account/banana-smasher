"""Per-artifact scientific identity used by every resident rail."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .contract import PackValidationError

IDENTITY_SCHEMA = "banana-smasher-artifact-identity-v1"


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PackValidationError(f"identity.{field} must be an object")
    return value


def _sha(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PackValidationError(f"identity.{field} must be a lowercase SHA-256")
    return value


def _finite(value: object, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PackValidationError(f"identity.{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise PackValidationError(f"identity.{field} must be finite")
    return result


@dataclass(frozen=True)
class CanaryIdentity:
    reference_kld: float
    reference_top1: int
    kld_abs_tolerance: float
    top1_abs_tolerance: int


@dataclass(frozen=True)
class ArtifactIdentity:
    root: Path
    path: Path
    sha256: str
    basis_sha256: str
    official_physical_layer_sha256: str | None
    corpora: Mapping[str, str]
    checkpoints: Mapping[str, Mapping[str, Any]]
    composition_kind: str
    composition: tuple[Mapping[str, Any], ...]
    canary: CanaryIdentity
    runtime: Mapping[str, Mapping[str, Any]]
    document: Mapping[str, Any]

    @classmethod
    def load(cls, artifact_root: str | Path) -> "ArtifactIdentity":
        root = Path(artifact_root).expanduser().resolve()
        path = root / "identity.json"
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PackValidationError(f"cannot read valid identity.json at {path}: {exc}") from exc
        document = _mapping(value, "root")
        if document.get("schema") != IDENTITY_SCHEMA:
            raise PackValidationError(f"identity.schema must be {IDENTITY_SCHEMA}")
        basis = _mapping(document.get("basis"), "basis")
        corpora_row = _mapping(document.get("corpora"), "corpora")
        required_corpora = {
            "builder_eval_sha256",
            "train_score_sha256",
            "u0_lock_sha256",
            "teacher_inventory_sha256",
        }
        if set(corpora_row) != required_corpora:
            raise PackValidationError(
                f"identity.corpora must contain exactly {sorted(required_corpora)}"
            )
        corpora = {key: _sha(corpora_row[key], f"corpora.{key}") for key in sorted(corpora_row)}
        checkpoint_rows = _mapping(document.get("checkpoints"), "checkpoints")
        if not checkpoint_rows:
            raise PackValidationError("identity.checkpoints must not be empty")
        checkpoints: dict[str, Mapping[str, Any]] = {}
        for name, raw_row in checkpoint_rows.items():
            if not isinstance(name, str) or not name:
                raise PackValidationError("identity.checkpoints names must be non-empty")
            row = dict(_mapping(raw_row, f"checkpoints.{name}"))
            _sha(row.get("sha256"), f"checkpoints.{name}.sha256")
            _sha(row.get("identity_sha256"), f"checkpoints.{name}.identity_sha256")
            for field in ("parent_sha256", "lock_sha256", "trajectory_sha256"):
                if row.get(field) is not None:
                    _sha(row[field], f"checkpoints.{name}.{field}")
            checkpoints[name] = row
        composition_row = _mapping(document.get("composition"), "composition")
        kind = composition_row.get("kind")
        layers = composition_row.get("layers")
        if not isinstance(kind, str) or not kind:
            raise PackValidationError("identity.composition.kind must be non-empty")
        if not isinstance(layers, list) or not all(isinstance(row, Mapping) for row in layers):
            raise PackValidationError("identity.composition.layers must be objects")
        seen: set[int] = set()
        normalized_layers = []
        for index, row0 in enumerate(layers):
            row = dict(row0)
            layer, tiers = row.get("layer"), row.get("tiers")
            if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 or layer in seen:
                raise PackValidationError(f"identity.composition.layers[{index}].layer is invalid")
            if not isinstance(tiers, Mapping) or not tiers or any(
                not isinstance(name, str)
                or not name
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
                for name, count in tiers.items()
            ):
                raise PackValidationError(f"identity.composition.layers[{index}].tiers is invalid")
            seen.add(layer)
            normalized_layers.append(row)
        canary_row = _mapping(document.get("canary"), "canary")
        reference = _mapping(canary_row.get("reference"), "canary.reference")
        tolerance = _mapping(canary_row.get("tolerance"), "canary.tolerance")
        top1, top1_abs = reference.get("top1"), tolerance.get("top1_abs")
        if isinstance(top1, bool) or not isinstance(top1, int):
            raise PackValidationError("identity.canary.reference.top1 must be an integer")
        if isinstance(top1_abs, bool) or not isinstance(top1_abs, int) or top1_abs < 0:
            raise PackValidationError("identity.canary.tolerance.top1_abs must be nonnegative")
        runtime_row = _mapping(document.get("runtime", {}), "runtime")
        runtime: dict[str, Mapping[str, Any]] = {}
        for name, raw_binding in runtime_row.items():
            binding = dict(_mapping(raw_binding, f"runtime.{name}"))
            for field, item in binding.items():
                if field.endswith("_sha256"):
                    _sha(item, f"runtime.{name}.{field}")
            runtime[str(name)] = binding
        physical = basis.get("official_physical_layer_sha256")
        return cls(
            root=root,
            path=path,
            sha256=hashlib.sha256(raw).hexdigest(),
            basis_sha256=_sha(basis.get("model_index_sha256"), "basis.model_index_sha256"),
            official_physical_layer_sha256=(
                None if physical is None else _sha(physical, "basis.official_physical_layer_sha256")
            ),
            corpora=corpora,
            checkpoints=checkpoints,
            composition_kind=kind,
            composition=tuple(normalized_layers),
            canary=CanaryIdentity(
                reference_kld=_finite(reference.get("kld"), "canary.reference.kld"),
                reference_top1=top1,
                kld_abs_tolerance=_finite(
                    tolerance.get("kld_abs"), "canary.tolerance.kld_abs", nonnegative=True
                ),
                top1_abs_tolerance=top1_abs,
            ),
            runtime=runtime,
            document=dict(document),
        )

    def require_canary(self, *, kld: float, top1: int) -> None:
        if abs(float(kld) - self.canary.reference_kld) > self.canary.kld_abs_tolerance:
            raise PackValidationError("artifact canary KLD is outside declared tolerance")
        if abs(int(top1) - self.canary.reference_top1) > self.canary.top1_abs_tolerance:
            raise PackValidationError("artifact canary Top-1 is outside declared tolerance")


__all__ = ["ArtifactIdentity", "CanaryIdentity", "IDENTITY_SCHEMA"]
