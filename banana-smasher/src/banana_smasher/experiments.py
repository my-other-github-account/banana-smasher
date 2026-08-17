"""Typed, JSON-first experiment reproducibility contracts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 1
LOCK_SCHEMA = "banana-smasher-experiment-lock-v1"
_SHA256_LENGTH = 64


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class ArtifactRef:
    id: str
    sha256: str | None
    identity_status: str = "verified"
    note: str | None = None

    @classmethod
    def from_dict(cls, value: object, label: str) -> "ArtifactRef":
        raw = _require_object(value, label)
        artifact_id = raw.get("id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError(f"{label}.id must be a nonempty logical artifact ID")
        status = raw.get("identity_status", "verified")
        digest = raw.get("sha256")
        note = raw.get("note")
        if status == "verified":
            digest = _require_sha256(digest, f"{label}.sha256")
        elif status == "unavailable":
            if digest is not None:
                raise ValueError(
                    f"{label}.sha256 must be null when identity is unavailable"
                )
            if not isinstance(note, str) or not note:
                raise ValueError(f"{label}.note must explain unavailable identity")
        else:
            raise ValueError(f"{label}.identity_status must be verified or unavailable")
        return cls(artifact_id, digest, status, note if isinstance(note, str) else None)


@dataclass(frozen=True)
class ParentSpec:
    checkpoint: ArtifactRef
    next_update: int

    @classmethod
    def from_dict(cls, value: object) -> "ParentSpec":
        raw = _require_object(value, "scientific.parent")
        next_update = raw.get("next_update")
        if (
            isinstance(next_update, bool)
            or not isinstance(next_update, int)
            or next_update < 0
        ):
            raise ValueError(
                "scientific.parent.next_update must be a nonnegative integer"
            )
        return cls(
            checkpoint=ArtifactRef.from_dict(
                raw.get("checkpoint"), "scientific.parent.checkpoint"
            ),
            next_update=next_update,
        )


@dataclass(frozen=True)
class DataSpec:
    prompt: ArtifactRef
    corpus: ArtifactRef
    teacher: ArtifactRef

    @classmethod
    def from_dict(cls, value: object) -> "DataSpec":
        raw = _require_object(value, "scientific.data")
        return cls(
            prompt=ArtifactRef.from_dict(raw.get("prompt"), "scientific.data.prompt"),
            corpus=ArtifactRef.from_dict(raw.get("corpus"), "scientific.data.corpus"),
            teacher=ArtifactRef.from_dict(
                raw.get("teacher"), "scientific.data.teacher"
            ),
        )


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    lut_gain_lr: float
    norm_lr: float
    scheduler: str
    cosine_min_ratio: float
    cosine_updates: int
    warmup_updates: int
    first_applied_lut_gain_lr: float

    @classmethod
    def from_dict(cls, value: object) -> "OptimizerSpec":
        raw = _require_object(value, "scientific.optimizer")
        name = raw.get("name")
        scheduler = raw.get("scheduler")
        if not isinstance(name, str) or not name:
            raise ValueError("scientific.optimizer.name must be nonempty")
        if scheduler != "cosine":
            raise ValueError("scientific.optimizer.scheduler must be cosine")
        numbers: dict[str, float] = {}
        for field in (
            "lut_gain_lr",
            "norm_lr",
            "cosine_min_ratio",
            "first_applied_lut_gain_lr",
        ):
            value_at_field = raw.get(field)
            if (
                isinstance(value_at_field, bool)
                or not isinstance(value_at_field, (int, float))
                or not math.isfinite(float(value_at_field))
                or float(value_at_field) < 0
            ):
                raise ValueError(
                    f"scientific.optimizer.{field} must be finite and nonnegative"
                )
            numbers[field] = float(value_at_field)
        integers: dict[str, int] = {}
        for field, minimum in (("cosine_updates", 1), ("warmup_updates", 0)):
            value_at_field = raw.get(field)
            if (
                isinstance(value_at_field, bool)
                or not isinstance(value_at_field, int)
                or value_at_field < minimum
            ):
                raise ValueError(f"scientific.optimizer.{field} must be >= {minimum}")
            integers[field] = value_at_field
        result = cls(
            name=name,
            lut_gain_lr=numbers["lut_gain_lr"],
            norm_lr=numbers["norm_lr"],
            scheduler=scheduler,
            cosine_min_ratio=numbers["cosine_min_ratio"],
            cosine_updates=integers["cosine_updates"],
            warmup_updates=integers["warmup_updates"],
            first_applied_lut_gain_lr=numbers["first_applied_lut_gain_lr"],
        )

        expected_first = result.learning_rates_for_update(1)[0]
        if not math.isclose(
            result.first_applied_lut_gain_lr,
            expected_first,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "scientific.optimizer.first_applied_lut_gain_lr differs from "
                "base LR/warmup semantics"
            )
        return result

    def learning_rates_for_update(self, update: int) -> tuple[float, float]:
        if update < 1:
            raise ValueError("update must be >= 1")
        if self.warmup_updates and update <= self.warmup_updates:
            factor = update / self.warmup_updates
        else:
            cosine_step = max(0, update - self.warmup_updates - 1)
            cosine_step = min(cosine_step, self.cosine_updates)
            factor = (
                self.cosine_min_ratio
                + (1.0 - self.cosine_min_ratio)
                * (1.0 + math.cos(math.pi * cosine_step / self.cosine_updates))
                / 2.0
            )
        return self.lut_gain_lr * factor, self.norm_lr * factor


@dataclass(frozen=True)
class WindowSchedule:
    """Ordered training windows grouped by optimizer update."""

    mode: str
    ordered_windows: tuple[int, ...]
    windows_per_optimizer_step: int
    explicit_groups: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in {"sequential", "explicit"}:
            raise ValueError("window schedule mode must be sequential or explicit")
        if self.windows_per_optimizer_step < 1:
            raise ValueError("windows_per_optimizer_step must be positive")
        if len(set(self.ordered_windows)) != len(self.ordered_windows):
            raise ValueError("ordered training windows must be unique")
        if self.mode == "explicit":
            flattened = tuple(
                window for group in self.explicit_groups for window in group
            )
            if any(
                len(group) != self.windows_per_optimizer_step
                for group in self.explicit_groups
            ):
                raise ValueError(
                    "every explicit group must match windows_per_optimizer_step"
                )
            if flattened != self.ordered_windows:
                raise ValueError(
                    "explicit groups must exactly reproduce ordered_windows"
                )

    @classmethod
    def from_dict(cls, value: object) -> "WindowSchedule":
        raw = _require_object(value, "scientific.window_schedule")
        ordered = raw.get("ordered_windows")
        if not isinstance(ordered, list) or any(
            isinstance(window, bool) or not isinstance(window, int)
            for window in ordered
        ):
            raise ValueError(
                "scientific.window_schedule.ordered_windows must be integer IDs"
            )
        groups = raw.get("explicit_groups", [])
        if not isinstance(groups, list) or any(
            not isinstance(group, list) for group in groups
        ):
            raise ValueError(
                "scientific.window_schedule.explicit_groups must be arrays"
            )
        return cls(
            mode=str(raw.get("mode")),
            ordered_windows=tuple(ordered),
            windows_per_optimizer_step=int(raw.get("windows_per_optimizer_step", 0)),
            explicit_groups=tuple(
                tuple(int(window) for window in group) for group in groups
            ),
        )

    def windows_for_update(self, update: int) -> tuple[int, ...]:
        if update < 1:
            raise ValueError("update must be >= 1")
        if self.mode == "explicit":
            try:
                return self.explicit_groups[update - 1]
            except IndexError as exc:
                raise ValueError(f"no explicit windows for update {update}") from exc
        start = (update - 1) * self.windows_per_optimizer_step
        stop = start + self.windows_per_optimizer_step
        group = self.ordered_windows[start:stop]
        if len(group) != self.windows_per_optimizer_step:
            raise ValueError(f"no complete sequential window group for update {update}")
        return group


@dataclass(frozen=True)
class EvaluationSpec:
    suite_lock: ArtifactRef
    pre: tuple[tuple[str, Any], ...]
    scorer_contract: str

    @classmethod
    def from_dict(cls, value: object) -> "EvaluationSpec":
        raw = _require_object(value, "scientific.evaluation")
        pre = _require_object(raw.get("pre"), "scientific.evaluation.pre")
        scorer = raw.get("scorer_contract")
        if not isinstance(scorer, str) or not scorer:
            raise ValueError("scientific.evaluation.scorer_contract must be nonempty")
        return cls(
            suite_lock=ArtifactRef.from_dict(
                raw.get("suite_lock"), "scientific.evaluation.suite_lock"
            ),
            pre=tuple(sorted(pre.items())),
            scorer_contract=scorer,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_lock": asdict(self.suite_lock),
            "pre": dict(self.pre),
            "scorer_contract": self.scorer_contract,
        }


@dataclass(frozen=True)
class ScientificSpec:
    basis: ArtifactRef
    identity_source: ArtifactRef
    parent: ParentSpec
    data: DataSpec
    window_schedule: WindowSchedule
    batch_size: int
    loss_scaling: str
    optimizer: OptimizerSpec
    mutable_surfaces: tuple[str, ...]
    frozen_surfaces: tuple[str, ...]
    evaluation: EvaluationSpec

    @classmethod
    def from_dict(cls, value: object) -> "ScientificSpec":
        raw = _require_object(value, "scientific")
        batch = raw.get("batch_size")
        if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
            raise ValueError("scientific.batch_size must be positive")
        loss_scaling = raw.get("loss_scaling")
        if not isinstance(loss_scaling, str) or not loss_scaling:
            raise ValueError("scientific.loss_scaling must be nonempty")
        surfaces: dict[str, tuple[str, ...]] = {}
        for field in ("mutable_surfaces", "frozen_surfaces"):
            rows = raw.get(field)
            if not isinstance(rows, list) or any(
                not isinstance(row, str) or not row for row in rows
            ):
                raise ValueError(f"scientific.{field} must be nonempty strings")
            surfaces[field] = tuple(rows)
        return cls(
            basis=ArtifactRef.from_dict(raw.get("basis"), "scientific.basis"),
            identity_source=ArtifactRef.from_dict(
                raw.get("identity_source"), "scientific.identity_source"
            ),
            parent=ParentSpec.from_dict(raw.get("parent")),
            data=DataSpec.from_dict(raw.get("data")),
            window_schedule=WindowSchedule.from_dict(raw.get("window_schedule")),
            batch_size=batch,
            loss_scaling=loss_scaling,
            optimizer=OptimizerSpec.from_dict(raw.get("optimizer")),
            mutable_surfaces=surfaces["mutable_surfaces"],
            frozen_surfaces=surfaces["frozen_surfaces"],
            evaluation=EvaluationSpec.from_dict(raw.get("evaluation")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "basis": asdict(self.basis),
            "identity_source": asdict(self.identity_source),
            "parent": asdict(self.parent),
            "data": asdict(self.data),
            "window_schedule": asdict(self.window_schedule),
            "batch_size": self.batch_size,
            "loss_scaling": self.loss_scaling,
            "optimizer": asdict(self.optimizer),
            "mutable_surfaces": list(self.mutable_surfaces),
            "frozen_surfaces": list(self.frozen_surfaces),
            "evaluation": self.evaluation.to_dict(),
        }


@dataclass(frozen=True)
class ExecutionSpec:
    device: str
    kernel: str
    launcher: str | None = None

    @classmethod
    def from_dict(cls, value: object) -> "ExecutionSpec":
        raw = _require_object(value, "execution")
        device = raw.get("device")
        kernel = raw.get("kernel")
        launcher = raw.get("launcher")
        if not isinstance(device, str) or not device:
            raise ValueError("execution.device must be nonempty")
        if not isinstance(kernel, str) or not kernel:
            raise ValueError("execution.kernel must be nonempty")
        if launcher is not None and (not isinstance(launcher, str) or not launcher):
            raise ValueError("execution.launcher must be null or nonempty")
        return cls(device, kernel, launcher)


@dataclass(frozen=True)
class ExperimentSpec:
    schema_version: int
    name: str
    mode: str
    reproduction_of: str | None
    scientific: ScientificSpec
    execution: ExecutionSpec
    source_path: Path | None = None

    @classmethod
    def from_dict(
        cls, value: object, *, source_path: Path | None = None
    ) -> "ExperimentSpec":
        raw = _require_object(value, "experiment")
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        name = raw.get("name")
        mode = raw.get("mode")
        reproduction_of = raw.get("reproduction_of")
        if not isinstance(name, str) or not name:
            raise ValueError("experiment.name must be nonempty")
        if mode not in {"reproduce", "extend"}:
            raise ValueError("experiment.mode must be reproduce or extend")
        if mode == "reproduce" and (
            not isinstance(reproduction_of, str) or not reproduction_of
        ):
            raise ValueError("reproduce mode requires reproduction_of")
        if mode == "extend" and reproduction_of is not None:
            raise ValueError("extend mode must not claim reproduction_of")
        return cls(
            schema_version=SCHEMA_VERSION,
            name=name,
            mode=mode,
            reproduction_of=reproduction_of
            if isinstance(reproduction_of, str)
            else None,
            scientific=ScientificSpec.from_dict(raw.get("scientific")),
            execution=ExecutionSpec.from_dict(raw.get("execution")),
            source_path=source_path,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "mode": self.mode,
            "reproduction_of": self.reproduction_of,
            "scientific": self.scientific.to_dict(),
            "execution": asdict(self.execution),
        }

    @property
    def scientific_identity_sha256(self) -> str:
        return _sha256_bytes(_canonical_bytes(self.scientific.to_dict()))


def load_experiment(path: str | Path) -> ExperimentSpec:
    resolved = Path(path).expanduser().resolve()
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid experiment JSON {resolved}: {exc}") from exc
    return ExperimentSpec.from_dict(raw, source_path=resolved)


def _display_windows(windows: tuple[int, ...]) -> str:
    if not windows:
        return "[]"
    contiguous = all(right == left + 1 for left, right in zip(windows, windows[1:]))
    if contiguous:
        return f"W{windows[0]}-{windows[-1]}"
    return "[" + ",".join(f"W{window}" for window in windows) + "]"


def explain_experiment(spec: ExperimentSpec) -> str:
    science = spec.scientific
    next_group_ordinal = science.parent.next_update + 1
    next_group = science.window_schedule.windows_for_update(next_group_ordinal)
    first_group = science.window_schedule.windows_for_update(1)
    previous_group = (
        science.window_schedule.windows_for_update(science.parent.next_update)
        if science.parent.next_update
        else ()
    )
    group_sequence = f"first U1 {_display_windows(first_group)}; "
    if previous_group:
        group_sequence += (
            f"U{science.parent.next_update} {_display_windows(previous_group)}; "
        )
    group_sequence += (
        f"next {_display_windows(next_group)}; "
        f"{science.window_schedule.windows_per_optimizer_step} windows/step"
    )
    lut_lr, norm_lr = science.optimizer.learning_rates_for_update(next_group_ordinal)
    reproduction = (
        "REPRODUCTION"
        if spec.mode == "reproduce"
        else "NOT A REPRODUCTION — EXTENDED SCIENCE"
    )
    return (
        "\n".join(
            (
                f"{spec.name}: {reproduction}",
                f"parent: {science.parent.checkpoint.id} sha256={science.parent.checkpoint.sha256 or 'UNAVAILABLE'}; next_update={science.parent.next_update}",
                f"groups: {group_sequence}",
                f"LR: {science.optimizer.name} LUT/gain={lut_lr:.12g} norm={norm_lr:.12g}; warmup={science.optimizer.warmup_updates}; cosine min={science.optimizer.cosine_min_ratio} over {science.optimizer.cosine_updates}",
                f"data: prompt={science.data.prompt.id}; corpus={science.data.corpus.id}; teacher={science.data.teacher.id}; batch={science.batch_size}; scaling={science.loss_scaling}",
                f"scorer: {science.evaluation.suite_lock.id} sha256={science.evaluation.suite_lock.sha256}; {science.evaluation.scorer_contract}",
                f"scientific_identity_sha256: {spec.scientific_identity_sha256}",
                f"execution: device={spec.execution.device}; kernel={spec.execution.kernel}",
            )
        )
        + "\n"
    )


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(value[key], child))
        return result
    if isinstance(value, (list, tuple)):
        return {prefix: list(value)}
    return {prefix: value}


def diff_experiments(
    reference: ExperimentSpec, candidate: ExperimentSpec
) -> dict[str, Any]:
    scientific_reference = {"mode": reference.mode, **reference.scientific.to_dict()}
    scientific_candidate = {"mode": candidate.mode, **candidate.scientific.to_dict()}
    execution_reference = asdict(reference.execution)
    execution_candidate = asdict(candidate.execution)
    changes: list[dict[str, Any]] = []
    for classification, left, right in (
        ("SCIENTIFIC", scientific_reference, scientific_candidate),
        ("EXECUTION_ONLY", execution_reference, execution_candidate),
    ):
        left_flat = _flatten(left)
        right_flat = _flatten(right)
        for path in sorted(set(left_flat) | set(right_flat)):
            if left_flat.get(path) != right_flat.get(path):
                changes.append(
                    {
                        "classification": classification,
                        "path": path,
                        "reference": left_flat.get(path),
                        "candidate": right_flat.get(path),
                    }
                )
    scientific_drift = any(
        change["classification"] == "SCIENTIFIC" for change in changes
    )
    return {
        "classification": "SCIENTIFIC" if scientific_drift else "EXECUTION_ONLY",
        "scientific_drift": scientific_drift,
        "reference_identity": reference.scientific_identity_sha256,
        "candidate_identity": candidate.scientific_identity_sha256,
        "changes": changes,
    }


def format_experiment_diff(diff: Mapping[str, Any]) -> str:
    lines = [
        f"{diff['classification']}: reference={diff['reference_identity']} candidate={diff['candidate_identity']}"
    ]
    changes = diff.get("changes", [])
    if not changes:
        lines.append("no changes")
    else:
        priority = {
            "parent": 0,
            "window_schedule": 1,
            "optimizer": 2,
            "data": 3,
            "evaluation": 4,
        }
        ordered_changes = sorted(
            changes,
            key=lambda change: (
                priority.get(str(change["path"]).split(".", 1)[0], 5),
                str(change["path"]),
            ),
        )
        for change in ordered_changes:
            lines.append(
                f"{change['classification']} {change['path']}: {change['reference']!r} -> {change['candidate']!r}"
            )
    return "\n".join(lines) + "\n"


def _resolve_reproduction_reference(spec: ExperimentSpec) -> str:
    assert spec.reproduction_of is not None
    if len(spec.reproduction_of) == 64 and all(
        character in "0123456789abcdef" for character in spec.reproduction_of
    ):
        return spec.reproduction_of
    if spec.source_path is None:
        raise ValueError("config reproduction_of requires a source path")
    reference_path = (spec.source_path.parent / spec.reproduction_of).resolve()
    if reference_path == spec.source_path:
        return spec.scientific_identity_sha256
    return load_experiment(reference_path).scientific_identity_sha256


def runtime_contract_for(spec: ExperimentSpec) -> dict[str, Any]:
    science = spec.scientific
    update = science.parent.next_update
    group_ordinal = update + 1
    lut_lr, norm_lr = science.optimizer.learning_rates_for_update(group_ordinal)
    return {
        "parent": asdict(science.parent.checkpoint),
        "update": update,
        "windows": list(science.window_schedule.windows_for_update(group_ordinal)),
        "optimizer": {
            "name": science.optimizer.name,
            "lut_gain_lr": lut_lr,
            "norm_lr": norm_lr,
            "warmup_updates": science.optimizer.warmup_updates,
        },
        "data": asdict(science.data),
        "scorer": {
            "suite_lock": asdict(science.evaluation.suite_lock),
            "contract": science.evaluation.scorer_contract,
        },
    }


def build_experiment_lock(spec: ExperimentSpec) -> dict[str, Any]:
    if spec.mode == "reproduce":
        expected = _resolve_reproduction_reference(spec)
        if expected != spec.scientific_identity_sha256:
            raise ValueError(
                "reproduction scientific drift: "
                f"expected={expected} actual={spec.scientific_identity_sha256}"
            )
    science = spec.scientific
    if spec.source_path is None:
        source_sha = _sha256_bytes(_canonical_bytes(spec.to_dict()))
        source_config = None
    else:
        source_sha = _sha256_bytes(spec.source_path.read_bytes())
        source_config = spec.source_path.name
    return {
        "schema": LOCK_SCHEMA,
        "status": "REPRODUCTION" if spec.mode == "reproduce" else "NOT A REPRODUCTION",
        "mode": spec.mode,
        "source_config": source_config,
        "source_config_sha256": source_sha,
        "scientific_identity_sha256": spec.scientific_identity_sha256,
        "normalized_spec": spec.to_dict(),
        "derived": {
            "first_group": list(science.window_schedule.windows_for_update(1)),
            "next_update": science.parent.next_update,
            "next_group": list(
                science.window_schedule.windows_for_update(
                    science.parent.next_update + 1
                )
            ),
        },
        "runtime_contract": runtime_contract_for(spec),
    }


def write_experiment_lock(spec: ExperimentSpec, output: str | Path) -> dict[str, Any]:
    lock = build_experiment_lock(spec)
    Path(output).write_text(
        json.dumps(lock, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return lock


def document_experiment(spec: ExperimentSpec) -> str:
    science = spec.scientific
    lines = [
        f"# {spec.name}",
        "",
        f"> {'REPRODUCTION' if spec.mode == 'reproduce' else 'NOT A REPRODUCTION — EXTENDED SCIENCE'}",
        "",
        "```text",
        explain_experiment(spec).rstrip(),
        "```",
        "",
        "## Scientific surfaces",
        "",
        f"- Mutable: {', '.join(science.mutable_surfaces)}",
        f"- Frozen: {', '.join(science.frozen_surfaces)}",
        f"- Evaluation authority: `{science.evaluation.suite_lock.id}` (`{science.evaluation.suite_lock.sha256}`)",
        "",
        "The evaluation protocol is referenced by suite-lock identity rather than copied here.",
    ]
    return "\n".join(lines) + "\n"


def _load_json_object(path: str | Path, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def runtime_mismatches(
    lock: Mapping[str, Any], observed: Mapping[str, Any]
) -> list[str]:
    if lock.get("schema") != LOCK_SCHEMA:
        raise ValueError(f"experiment lock schema must be {LOCK_SCHEMA}")
    expected = _require_object(lock.get("runtime_contract"), "lock.runtime_contract")
    left = _flatten(expected)
    right = _flatten(observed)
    mismatches = []
    for path in sorted(set(left) | set(right)):
        if left.get(path) != right.get(path):
            mismatches.append(
                f"{path}: expected={left.get(path)!r} observed={right.get(path)!r}"
            )
    return mismatches


def validate_runtime_contract(
    lock: Mapping[str, Any] | str | Path,
    observed: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    lock_object = (
        _load_json_object(lock, "experiment lock")
        if isinstance(lock, (str, Path))
        else dict(lock)
    )
    observed_object = (
        _load_json_object(observed, "observed runtime")
        if isinstance(observed, (str, Path))
        else dict(observed)
    )
    mismatches = runtime_mismatches(lock_object, observed_object)
    if mismatches:
        raise ValueError("runtime contract mismatch: " + "; ".join(mismatches))
    return {
        "status": "PASS",
        "scientific_identity_sha256": lock_object["scientific_identity_sha256"],
        "validated_fields": len(_flatten(lock_object["runtime_contract"])),
    }
