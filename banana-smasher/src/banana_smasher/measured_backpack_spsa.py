"""Measured end-to-end SPSA optimizer for mixed-tier Backpack assignments.

Routing measurements are used only to form shared-logit groups.  No local error,
per-cell KLD prediction, or proxy quality coefficient enters the projection.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .qtip25_native_v4 import native_v4_geometry

CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")
SHIPPING_BYTES = 102_000_000_000
FIXED_NONEXPERT_BYTES = 9_032_112_614
EXPERT_ENVELOPE_BYTES = 92_967_887_386


@dataclass(frozen=True)
class TierDeclaration:
    id: str
    family: str
    source_key: str
    B: int | None = None
    code_bpw: float | None = None
    provider: str | None = None
    bpw: float | None = None
    phase_count: int | None = None
    alternation: bool | None = None
    member_averaging: bool | None = None
    feedback_mode: str | None = None
    scale_semantics: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TierDeclaration":
        tier_id = value.get("id")
        family = value.get("family")
        if not isinstance(tier_id, str) or not tier_id:
            raise ValueError("tier declaration id must be a non-empty string")
        if not isinstance(family, str) or not family:
            raise ValueError(f"tier {tier_id!r} family must be a non-empty string")
        source_key = value.get("source_key", tier_id)
        if not isinstance(source_key, str) or not source_key:
            raise ValueError(f"tier {tier_id!r} source_key must be a non-empty string")
        raw_B = value.get("B")
        raw_bpw = value.get("code_bpw")
        if family == "qtip_native_v4":
            declared_bpw = value.get("bpw", raw_bpw)
            geometry = native_v4_geometry(declared_bpw)
            if isinstance(raw_B, bool) or not isinstance(raw_B, int) or raw_B != geometry.B:
                raise ValueError(f"native-V4 tier {tier_id!r} requires positive integer B")
            expected_bpw = geometry.B / geometry.V
            if raw_bpw is not None and float(raw_bpw) != expected_bpw:
                raise ValueError(f"native-V4 tier {tier_id!r} code_bpw must equal B/4")
            expected_provider = f"qtip-native-v4@{expected_bpw:.2f}"
            if value.get("provider", expected_provider) != expected_provider:
                raise ValueError(f"native-V4 tier {tier_id!r} provider does not match its rate")
            if value.get("phase_count", 1) != 1:
                raise ValueError(f"native-V4 tier {tier_id!r} must have phase_count=1")
            if value.get("alternation", False) is not False:
                raise ValueError(f"native-V4 tier {tier_id!r} cannot alternate")
            if value.get("member_averaging", False) is not False:
                raise ValueError(f"native-V4 tier {tier_id!r} cannot average members")
            feedback_mode = value.get("feedback_mode", "off")
            if feedback_mode not in {"off", "reverse_16"}:
                raise ValueError(f"native-V4 tier {tier_id!r} feedback mode is invalid")
            scale_semantics = value.get("scale_semantics", "absolute_unit")
            if scale_semantics not in {"absolute_unit", "relative_search"}:
                raise ValueError(f"native-V4 tier {tier_id!r} scale semantics are invalid")
            if feedback_mode == "off" and scale_semantics != "absolute_unit":
                raise ValueError(
                    f"native-V4 tier {tier_id!r} relative search requires reverse_16 feedback"
                )
            return cls(
                tier_id,
                family,
                source_key,
                geometry.B,
                expected_bpw,
                expected_provider,
                expected_bpw,
                1,
                False,
                False,
                feedback_mode,
                scale_semantics,
            )
        if any(
            value.get(field) is not None
            for field in (
                "B",
                "code_bpw",
                "provider",
                "bpw",
                "phase_count",
                "alternation",
                "member_averaging",
            )
        ):
            raise ValueError(f"non-native-V4 tier {tier_id!r} cannot declare B/code_bpw")
        return cls(tier_id, family, source_key)

    def as_mapping(self) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "id": self.id,
            "family": self.family,
            "source_key": self.source_key,
        }
        if self.B is not None:
            mapping.update(
                {
                    "provider": self.provider,
                    "bpw": self.bpw,
                    "B": self.B,
                    "code_bpw": self.code_bpw,
                    "phase_count": self.phase_count,
                    "alternation": self.alternation,
                    "member_averaging": self.member_averaging,
                    "feedback_mode": self.feedback_mode,
                    "scale_semantics": self.scale_semantics,
                }
            )
        return mapping

    def code_bytes(self, weights: int) -> int:
        if self.B is None:
            raise ValueError(f"tier {self.id!r} has no homogeneous native-V4 code rate")
        if isinstance(weights, bool) or not isinstance(weights, int) or weights < 0:
            raise ValueError("native-V4 weight count must be a nonnegative integer")
        numerator = weights * self.B
        if numerator % 32:
            raise ValueError("native-V4 code bytes do not close weights*B/32 exactly")
        return numerator // 32


@dataclass(frozen=True)
class TierMenu:
    declarations: tuple[TierDeclaration, ...]

    @classmethod
    def from_declarations(
        cls, values: Sequence[Mapping[str, Any] | TierDeclaration]
    ) -> "TierMenu":
        declarations = tuple(
            value if isinstance(value, TierDeclaration) else TierDeclaration.from_mapping(value)
            for value in values
        )
        if not declarations:
            raise ValueError("tier menu must not be empty")
        ids = [value.id for value in declarations]
        if len(ids) != len(set(ids)):
            raise ValueError("tier menu ids must be unique")
        source_keys = [value.source_key for value in declarations]
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("tier menu source keys must be unique")
        return cls(declarations)

    @property
    def tier_ids(self) -> tuple[str, ...]:
        return tuple(value.id for value in self.declarations)

    @property
    def sha256(self) -> str:
        return _sha256(_canonical([value.as_mapping() for value in self.declarations]))

    def as_mappings(self) -> list[dict[str, Any]]:
        return [value.as_mapping() for value in self.declarations]


DEFAULT_QTIP_V5_MENU = TierMenu.from_declarations(
    (
        {"id": "native_mxfp4", "family": "native_mxfp4"},
        *(
            {
                "id": f"native_v4_b{bits}",
                "family": "qtip_native_v4",
                "provider": f"qtip-native-v4@{bits / 4:.2f}",
                "bpw": bits / 4,
                "B": bits,
                "code_bpw": bits / 4,
                "phase_count": 1,
                "alternation": False,
                "member_averaging": False,
                "feedback_mode": "off",
                "scale_semantics": "absolute_unit",
            }
            for bits in range(4, 17)
        ),
        {"id": "d4_k2048", "family": "d4"},
        {"id": "d4_k4096", "family": "d4"},
    )
)


@dataclass(frozen=True)
class WireOption:
    cell_id: str
    tier: str
    layer: int
    expert: int
    projection: str
    physical_bytes: int
    activation_ids: tuple[str, ...]
    physical_producer: Mapping[str, Any] | None


@dataclass(frozen=True)
class RoutingCell:
    cell_id: str
    layer: int
    expert: int
    projection: str
    usage_by_class: Mapping[str, float]
    dominant_class: str
    total_usage: float
    entropy: float


Evaluator = Callable[[Path, Mapping[str, Any], Path], Mapping[str, Any]]


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    raw = _canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_cell(cell_id: str) -> tuple[int, int, str]:
    try:
        layer_text, expert_text, projection = cell_id.split(":")
        layer = int(layer_text.removeprefix("L"))
        expert = int(expert_text.removeprefix("E"))
    except Exception as exc:
        raise ValueError(f"invalid Backpack cell id: {cell_id!r}") from exc
    if layer < 0 or expert < 0 or projection not in {"down", "fused13"}:
        raise ValueError(f"invalid Backpack cell id: {cell_id!r}")
    return layer, expert, projection


def load_full_wire_menu(
    ledger_path: str | Path,
    *,
    expected_basis_sha256: str,
    tier_menu: TierMenu = DEFAULT_QTIP_V5_MENU,
) -> tuple[dict[str, dict[str, WireOption]], dict[str, int], dict[str, Any]]:
    """Load physical wire facts once while deliberately discarding quality proxies."""

    path = Path(ledger_path).expanduser().resolve()
    tier_ids = tier_menu.tier_ids
    declarations = {value.id: value for value in tier_menu.declarations}
    digest = hashlib.sha256()
    by_cell: dict[str, dict[str, WireOption]] = {}
    activation_bytes: dict[str, int] = {}
    identity: dict[str, Any] | None = None
    rows = 0
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"wire row {line_number} is not an object")
            if value.get("basis_sha256") != expected_basis_sha256:
                raise ValueError(f"wire row {line_number} basis mismatch")
            cell_id = str(value.get("cell_id", ""))
            tier = str(value.get("tier", ""))
            layer, expert, projection = _parse_cell(cell_id)
            if tier not in tier_ids:
                raise ValueError(f"wire row {line_number} has unexpected tier {tier!r}")
            declaration = declarations[tier]
            if value.get("family", declaration.family) != declaration.family:
                raise ValueError(f"wire row {line_number} tier family mismatch")
            if value.get("source_key", declaration.source_key) != declaration.source_key:
                raise ValueError(f"wire row {line_number} tier source mismatch")
            if any(
                value.get(field) != expected
                for field, expected in (
                    ("layer", layer),
                    ("expert", expert),
                    ("projection", projection),
                )
            ):
                raise ValueError(f"wire row {line_number} cell geometry mismatch")
            physical_bytes = value.get("physical_bytes")
            if (
                isinstance(physical_bytes, bool)
                or not isinstance(physical_bytes, int)
                or physical_bytes <= 0
            ):
                raise ValueError(f"wire row {line_number} physical_bytes is invalid")
            raw_artifacts = value.get("activation_artifacts", [])
            if not isinstance(raw_artifacts, list):
                raise ValueError(f"wire row {line_number} activation_artifacts is invalid")
            ids: list[str] = []
            for artifact in raw_artifacts:
                if not isinstance(artifact, dict):
                    raise ValueError(f"wire row {line_number} activation artifact is invalid")
                artifact_id = str(artifact.get("id", artifact.get("artifact_id", "")))
                byte_count = artifact.get("bytes")
                if (
                    not artifact_id
                    or isinstance(byte_count, bool)
                    or not isinstance(byte_count, int)
                    or byte_count < 0
                ):
                    raise ValueError(f"wire row {line_number} activation artifact is invalid")
                if artifact_id in activation_bytes and activation_bytes[artifact_id] != byte_count:
                    raise ValueError(f"activation byte conflict for {artifact_id}")
                activation_bytes[artifact_id] = byte_count
                ids.append(artifact_id)
            declared_ids = sorted(str(item) for item in value.get("activation_ids", []))
            if declared_ids != sorted(ids):
                raise ValueError(f"wire row {line_number} activation identity mismatch")
            cell_options = by_cell.setdefault(cell_id, {})
            if tier in cell_options:
                raise ValueError(f"duplicate wire option {(cell_id, tier)!r}")
            # prediction_by_class and every other quality-looking field are omitted.
            cell_options[tier] = WireOption(
                cell_id=cell_id,
                tier=tier,
                layer=layer,
                expert=expert,
                projection=projection,
                physical_bytes=physical_bytes,
                activation_ids=tuple(sorted(ids)),
                physical_producer=(
                    dict(value["physical_producer"])
                    if isinstance(value.get("physical_producer"), dict)
                    else None
                ),
            )
            row_identity = {
                key: value.get(key)
                for key in ("model_id", "model_revision", "basis_sha256")
            }
            if identity is None:
                identity = row_identity
            elif row_identity != identity:
                raise ValueError(f"wire row {line_number} model identity drift")
            rows += 1
    tier_set = set(tier_ids)
    if not by_cell or any(set(options) != tier_set for options in by_cell.values()):
        raise ValueError("full-wire menu is not a complete cell/tier matrix")
    evidence = {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
        "rows": rows,
        "cells": len(by_cell),
        "ordered_tier_ids": list(tier_ids),
        "tier_menu": tier_menu.as_mappings(),
        "tier_menu_sha256": tier_menu.sha256,
        "quality_coefficients_loaded": False,
        "identity": identity,
    }
    return by_cell, activation_bytes, evidence


def load_routing_usage(
    path: str | Path,
    *,
    expected_basis_sha256: str,
    expected_cells: set[str],
) -> tuple[dict[str, RoutingCell], dict[str, Any]]:
    """Load FF0731 class usage used only as grouping metadata."""

    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    document = json.loads(raw)
    if (
        not isinstance(document, dict)
        or document.get("schema") != "banana-smasher-ff0731-routing-usage-v1"
        or document.get("status") != "PASS"
        or document.get("basis_sha256") != expected_basis_sha256
    ):
        raise ValueError("routing-usage identity/schema mismatch")
    rows = document.get("cells")
    if not isinstance(rows, list):
        raise ValueError("routing-usage cells must be an array")
    result: dict[str, RoutingCell] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("routing-usage row is not an object")
        cell_id = str(row.get("cell_id", ""))
        if cell_id in result:
            raise ValueError(f"duplicate routing cell {cell_id}")
        layer, expert, projection = _parse_cell(cell_id)
        raw_usage = row.get("usage_by_class")
        if not isinstance(raw_usage, dict) or set(raw_usage) != set(CLASSES):
            raise ValueError(f"routing usage classes mismatch for {cell_id}")
        usage = {name: float(raw_usage[name]) for name in CLASSES}
        if any(not math.isfinite(value) or value < 0 for value in usage.values()):
            raise ValueError(f"routing usage value is invalid for {cell_id}")
        total = sum(usage.values())
        dominant = max(CLASSES, key=lambda name: (usage[name], -CLASSES.index(name)))
        probabilities = [value / total for value in usage.values() if total > 0 and value > 0]
        entropy = -sum(value * math.log(value) for value in probabilities)
        result[cell_id] = RoutingCell(
            cell_id=cell_id,
            layer=layer,
            expert=expert,
            projection=projection,
            usage_by_class=usage,
            dominant_class=dominant,
            total_usage=total,
            entropy=entropy,
        )
    if set(result) != expected_cells:
        missing = sorted(expected_cells - set(result))
        extra = sorted(set(result) - expected_cells)
        raise ValueError(f"routing-usage coverage mismatch: missing={missing[:3]} extra={extra[:3]}")
    return result, {
        "path": str(source),
        "sha256": _sha256(raw),
        "bytes": len(raw),
        "cells": len(result),
        "use": "grouping-metadata-only",
    }


def build_hierarchical_groups(
    routing: Mapping[str, RoutingCell], *, quantile_bins: int = 2
) -> dict[str, str]:
    if quantile_bins < 1:
        raise ValueError("quantile_bins must be positive")
    strata: dict[tuple[int, str, str], list[RoutingCell]] = defaultdict(list)
    for row in routing.values():
        strata[(row.layer, row.projection, row.dominant_class)].append(row)
    groups: dict[str, str] = {}
    for (layer, projection, dominant), rows in sorted(strata.items()):
        ordered = sorted(rows, key=lambda row: (row.total_usage, row.expert, row.cell_id))
        for rank, row in enumerate(ordered):
            quantile = min(quantile_bins - 1, rank * quantile_bins // len(ordered))
            groups[row.cell_id] = f"L{layer:03d}:{projection}:{dominant}:q{quantile}"
    return groups


def refine_influential_groups(
    group_by_cell: Mapping[str, str],
    routing: Mapping[str, RoutingCell],
    logits: Mapping[str, Sequence[float]],
    *,
    fraction: float = 0.25,
) -> tuple[dict[str, str], dict[str, list[float]]]:
    if not 0 < fraction <= 1:
        raise ValueError("refinement fraction must be in (0, 1]")
    members: dict[str, list[str]] = defaultdict(list)
    for cell_id, group in group_by_cell.items():
        members[group].append(cell_id)
    eligible = [group for group, cells in members.items() if len(cells) >= 4]
    count = max(1, math.ceil(len(eligible) * fraction)) if eligible else 0
    selected = set(
        sorted(
            eligible,
            key=lambda group: (-float(np.linalg.norm(logits[group])), group),
        )[:count]
    )
    refined: dict[str, str] = {}
    refined_logits: dict[str, list[float]] = {}
    for group, cells in sorted(members.items()):
        if group not in selected:
            refined_logits[group] = [float(value) for value in logits[group]]
            for cell_id in cells:
                refined[cell_id] = group
            continue
        ordered = sorted(cells, key=lambda cell_id: (routing[cell_id].entropy, cell_id))
        midpoint = len(ordered) // 2
        for split, split_cells in enumerate((ordered[:midpoint], ordered[midpoint:])):
            child = f"{group}/r{split}"
            refined_logits[child] = [float(value) for value in logits[group]]
            for cell_id in split_cells:
                refined[cell_id] = child
    return refined, refined_logits


def _assignment_sha(rows: Sequence[Mapping[str, Any]]) -> str:
    assignment = {str(row["cell_id"]): str(row["tier"]) for row in rows}
    return _sha256(_canonical(assignment))


def project_group_logits(
    menu: Mapping[str, Mapping[str, WireOption]],
    activation_bytes: Mapping[str, int],
    group_by_cell: Mapping[str, str],
    logits: Mapping[str, Sequence[float]],
    *,
    shipping_bytes: int = SHIPPING_BYTES,
    fixed_nonexpert_bytes: int = FIXED_NONEXPERT_BYTES,
    repair_bytes: int = 0,
    max_tier_fraction_per_layer_projection: float = 0.85,
    max_tier_fraction_per_dominant_class: float = 0.90,
    routing: Mapping[str, RoutingCell] | None = None,
    time_limit_seconds: float = 60.0,
    tier_menu: TierMenu = DEFAULT_QTIP_V5_MENU,
) -> dict[str, Any]:
    """Project shared logits to one whole-model-cap-safe physical assignment."""

    if repair_bytes != 0:
        raise ValueError("measured SPSA search is PRE_REPAIR and requires repair_bytes=0")
    if shipping_bytes <= fixed_nonexpert_bytes:
        raise ValueError("whole-model target must exceed fixed nonexpert bytes")
    tier_ids = tier_menu.tier_ids
    declarations = {value.id: value for value in tier_menu.declarations}
    if set(menu) != set(group_by_cell):
        raise ValueError("group mapping does not cover the wire menu")
    if any(set(options) != set(tier_ids) for options in menu.values()):
        raise ValueError("wire options do not match the declared tier menu")
    groups: dict[str, list[str]] = defaultdict(list)
    for cell_id, group in group_by_cell.items():
        groups[group].append(cell_id)
    if set(groups) != set(logits):
        raise ValueError("logit groups do not match the cell grouping")
    group_names = sorted(groups)
    option_keys = [(group, tier) for group in group_names for tier in tier_ids]
    option_index = {key: index for index, key in enumerate(option_keys)}
    option_bytes: dict[tuple[str, str], int] = {}
    option_activations: dict[tuple[str, str], set[str]] = {}
    for key in option_keys:
        group, tier = key
        option_bytes[key] = sum(menu[cell_id][tier].physical_bytes for cell_id in groups[group])
        option_activations[key] = {
            artifact_id
            for cell_id in groups[group]
            for artifact_id in menu[cell_id][tier].activation_ids
        }
    used_activation_ids = sorted(
        {artifact_id for values in option_activations.values() for artifact_id in values}
    )
    missing_activations = set(used_activation_ids) - set(activation_bytes)
    if missing_activations:
        raise ValueError(f"missing activation byte declarations: {sorted(missing_activations)[:3]}")
    activation_index = {
        artifact_id: len(option_keys) + offset
        for offset, artifact_id in enumerate(used_activation_ids)
    }
    variable_count = len(option_keys) + len(used_activation_ids)
    objective = np.zeros(variable_count, dtype=np.float64)
    for group in group_names:
        values = logits[group]
        if len(values) != len(tier_ids):
            raise ValueError(
                f"group {group} does not have {len(tier_ids)} declared tier logits"
            )
        for tier_index, tier in enumerate(tier_ids):
            key = (group, tier)
            # Byte utilization only breaks near-ties; learned measured logits dominate.
            utilization_tie_break = 1e-4 * option_bytes[key] / (shipping_bytes - fixed_nonexpert_bytes)
            objective[option_index[key]] = -float(values[tier_index]) - utilization_tie_break

    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(coefficients: Mapping[int, float], lb: float, ub: float) -> None:
        row_index = len(lower)
        for column, value in coefficients.items():
            if value:
                matrix_rows.append(row_index)
                matrix_columns.append(column)
                matrix_values.append(float(value))
        lower.append(float(lb))
        upper.append(float(ub))

    for group in group_names:
        add_constraint(
            {option_index[(group, tier)]: 1.0 for tier in tier_ids}, 1.0, 1.0
        )
    budget = {
        option_index[key]: float(byte_count) for key, byte_count in option_bytes.items()
    }
    budget.update(
        {
            activation_index[artifact_id]: float(activation_bytes[artifact_id])
            for artifact_id in used_activation_ids
        }
    )
    expert_cap = shipping_bytes - fixed_nonexpert_bytes
    add_constraint(budget, -np.inf, float(expert_cap))
    for key, ids in option_activations.items():
        x = option_index[key]
        for artifact_id in ids:
            add_constraint({x: 1.0, activation_index[artifact_id]: -1.0}, -np.inf, 0.0)
    for artifact_id in used_activation_ids:
        users = [option_index[key] for key, ids in option_activations.items() if artifact_id in ids]
        coefficients = {activation_index[artifact_id]: 1.0}
        coefficients.update({user: -1.0 for user in users})
        add_constraint(coefficients, -np.inf, 0.0)

    layer_projection_cells: dict[tuple[int, str], list[str]] = defaultdict(list)
    for cell_id, options in menu.items():
        sample = next(iter(options.values()))
        layer_projection_cells[(sample.layer, sample.projection)].append(cell_id)
    for cells in layer_projection_cells.values():
        cell_set = set(cells)
        limit = math.floor(len(cells) * max_tier_fraction_per_layer_projection)
        for tier in tier_ids:
            coefficients = {
                option_index[(group, tier)]: float(len(cell_set.intersection(group_cells)))
                for group, group_cells in groups.items()
                if cell_set.intersection(group_cells)
            }
            add_constraint(coefficients, -np.inf, float(limit))
    if routing is not None:
        class_cells = {
            name: {cell_id for cell_id, row in routing.items() if row.dominant_class == name}
            for name in CLASSES
        }
        for cells in class_cells.values():
            if not cells:
                continue
            limit = math.floor(len(cells) * max_tier_fraction_per_dominant_class)
            for tier in tier_ids:
                coefficients = {
                    option_index[(group, tier)]: float(len(cells.intersection(group_cells)))
                    for group, group_cells in groups.items()
                    if cells.intersection(group_cells)
                }
                add_constraint(coefficients, -np.inf, float(limit))

    matrix = coo_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(len(lower), variable_count),
    ).tocsr()
    solved = milp(
        objective,
        integrality=np.ones(variable_count, dtype=np.int8),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": float(time_limit_seconds)},
    )
    if solved.x is None:
        raise RuntimeError(f"exact-budget group projection failed: {solved.message}")
    chosen_by_group: dict[str, str] = {}
    for group in group_names:
        selected = [
            tier for tier in tier_ids if solved.x[option_index[(group, tier)]] >= 0.5
        ]
        if len(selected) != 1:
            raise RuntimeError(f"projection returned invalid group choice for {group}: {selected}")
        chosen_by_group[group] = selected[0]
    rows: list[dict[str, Any]] = []
    activated: set[str] = set()
    tier_counts: Counter[str] = Counter()
    payload_bytes = 0
    for cell_id in sorted(menu, key=_parse_cell):
        group = group_by_cell[cell_id]
        tier = chosen_by_group[group]
        option = menu[cell_id][tier]
        payload_bytes += option.physical_bytes
        tier_counts[tier] += 1
        activated.update(option.activation_ids)
        rows.append(
            {
                "cell_id": cell_id,
                "layer": option.layer,
                "expert": option.expert,
                "projection": option.projection,
                "selection_group": group,
                "tier": tier,
                "source_key": declarations[tier].source_key,
                "physical_bytes": option.physical_bytes,
                "activation_artifact_ids": list(option.activation_ids),
                **(
                    {"physical_producer": dict(option.physical_producer)}
                    if option.physical_producer is not None
                    else {}
                ),
            }
        )
    shared_bytes = sum(activation_bytes[artifact_id] for artifact_id in activated)
    expert_bytes = payload_bytes + shared_bytes
    whole_bytes = fixed_nonexpert_bytes + expert_bytes
    if whole_bytes > shipping_bytes:
        raise RuntimeError("projected assignment exceeds the whole-model target")
    assignment_sha256 = _assignment_sha(rows)
    return {
        "schema": "banana-smasher-measured-spsa-assignment-v1",
        "status": "PASS_EXACT_BUDGET_PROJECTION",
        "phase": "PRE_REPAIR",
        "assignment_sha256": assignment_sha256,
        "tier_counts": {tier: tier_counts.get(tier, 0) for tier in tier_ids},
        "ordered_tier_ids": list(tier_ids),
        "tier_menu_sha256": tier_menu.sha256,
        "whole_model_accounting": {
            "shipping_bytes_cap": shipping_bytes,
            "expert_envelope_bytes": expert_cap,
            "selected_cell_payload_bytes": payload_bytes,
            "selected_activation_bytes": shared_bytes,
            "expert_physical_wire_bytes": expert_bytes,
            "fixed_nonexpert_bytes": fixed_nonexpert_bytes,
            "repair_bytes": repair_bytes,
            "whole_shipping_bytes": whole_bytes,
            "shipping_slack_bytes": shipping_bytes - whole_bytes,
        },
        "activated_artifacts": [
            {"id": artifact_id, "bytes": activation_bytes[artifact_id]}
            for artifact_id in sorted(activated)
        ],
        "assignments": rows,
        "projection": {
            "solver_status": int(solved.status),
            "solver_message": str(solved.message),
            "groups": len(group_names),
            "variables": variable_count,
            "quality_signal": "measured-group-logits-only",
            "routing_use": "grouping-and-concentration-only",
            "max_tier_fraction_per_layer_projection": max_tier_fraction_per_layer_projection,
            "max_tier_fraction_per_dominant_class": max_tier_fraction_per_dominant_class,
        },
    }


def measured_objective(
    score: Mapping[str, Any],
    *,
    worst_class_weight: float,
    top1_floor: float,
) -> float:
    if score.get("split") != "TRAIN" or score.get("kld_reference") != "own-base":
        raise ValueError("measurement must be own-base TRAIN KLD")
    class_kld = score.get("class_kld")
    if not isinstance(class_kld, Mapping) or set(class_kld) != set(CLASSES):
        raise ValueError("measurement must contain all six class KLD values")
    values = [float(class_kld[name]) for name in CLASSES]
    mean_kld = float(score.get("mean_kld"))
    top1 = float(score.get("top1_agreement"))
    if any(not math.isfinite(value) or value < 0 for value in [*values, mean_kld]):
        raise ValueError("measurement KLD is invalid")
    if not math.isfinite(top1) or not 0 <= top1 <= 1:
        raise ValueError("measurement Top-1 is invalid")
    if top1 < float(top1_floor):
        return math.inf
    return mean_kld + float(worst_class_weight) * max(values)


def _validate_measurement(
    value: Mapping[str, Any],
    *,
    assignment_sha256: str,
    train_slice: Mapping[str, Any],
) -> dict[str, Any]:
    if value.get("status") != "PASS":
        raise ValueError("SPSA evaluator did not return terminal PASS")
    if value.get("assignment_sha256") != assignment_sha256:
        raise ValueError("SPSA measurement assignment mismatch")
    if value.get("slice_id") != train_slice.get("slice_id"):
        raise ValueError("SPSA measurement TRAIN slice mismatch")
    if value.get("window_ids") != train_slice.get("window_ids"):
        raise ValueError("SPSA measurement window order mismatch")
    if value.get("windows") != 8:
        raise ValueError("SPSA TRAIN measurement must contain exactly eight windows")
    if value.get("split") != "TRAIN" or value.get("kld_reference") != "own-base":
        raise ValueError("SPSA measurement must be own-base TRAIN KLD")
    class_kld = value.get("class_kld")
    if not isinstance(class_kld, Mapping) or set(class_kld) != set(CLASSES):
        raise ValueError("SPSA measurement lacks the six-class KLD breakdown")
    for field in ("mean_kld", "top1_agreement"):
        if isinstance(value.get(field), bool) or not isinstance(value.get(field), (int, float)):
            raise ValueError(f"SPSA measurement {field} is invalid")
    return dict(value)


def command_evaluator(command: Sequence[str]) -> Evaluator:
    """Build an evaluator from an argv template with assignment/slice/output fields."""

    template = [str(item) for item in command]
    if not template:
        raise ValueError("evaluator command must not be empty")

    def evaluate(
        assignment_path: Path, train_slice: Mapping[str, Any], output_path: Path
    ) -> Mapping[str, Any]:
        slice_path = output_path.with_suffix(".slice.json")
        _atomic_json(slice_path, train_slice)
        values = {
            "assignment": str(assignment_path),
            "slice": str(slice_path),
            "output": str(output_path),
        }
        argv = [item.format(**values) for item in template]
        completed = subprocess.run(argv, check=False, text=True, capture_output=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"SPSA evaluator failed ({completed.returncode}): "
                f"{shlex.join(argv)}\n{completed.stderr[-4000:]}"
            )
        return json.loads(output_path.read_text())

    return evaluate


def run_measured_spsa(
    menu: Mapping[str, Mapping[str, WireOption]],
    activation_bytes: Mapping[str, int],
    routing: Mapping[str, RoutingCell],
    train_slices: Sequence[Mapping[str, Any]],
    evaluator: Evaluator,
    output_root: str | Path,
    *,
    coarse_iterations: int = 16,
    refine_iterations: int = 8,
    perturbation: float = 0.12,
    learning_rate: float = 0.08,
    worst_class_weight: float = 0.25,
    top1_floor: float = 0.90,
    seed: int = 731,
    tier_menu: TierMenu = DEFAULT_QTIP_V5_MENU,
) -> dict[str, Any]:
    """Run resumable antithetic SPSA over measured eight-window model forwards."""

    tier_ids = tier_menu.tier_ids
    if not train_slices:
        raise ValueError("at least one balanced TRAIN slice is required")
    normalized_slices: list[dict[str, Any]] = []
    for value in train_slices:
        row = dict(value)
        if (
            not isinstance(row.get("slice_id"), str)
            or not isinstance(row.get("window_ids"), list)
            or len(row["window_ids"]) != 8
            or row.get("holdout_used", False)
        ):
            raise ValueError("every TRAIN slice must identify eight ordered non-HOLDOUT windows")
        normalized_slices.append(row)
    root = Path(output_root).expanduser().resolve()
    state_path = root / "STATE.json"
    root.mkdir(parents=True, exist_ok=True)
    total_iterations = coarse_iterations + refine_iterations
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("schema") != "banana-smasher-measured-spsa-state-v2":
            raise ValueError("SPSA resume state schema mismatch")
    else:
        group_by_cell = build_hierarchical_groups(routing)
        state = {
            "schema": "banana-smasher-measured-spsa-state-v2",
            "status": "RUNNING",
            "ordered_tier_ids": list(tier_ids),
            "tier_menu": tier_menu.as_mappings(),
            "tier_menu_sha256": tier_menu.sha256,
            "next_iteration": 0,
            "refined": False,
            "group_by_cell": group_by_cell,
            "logits": {
                group: [0.0] * len(tier_ids)
                for group in sorted(set(group_by_cell.values()))
            },
            "curve": [],
            "best": None,
            "config": {
                "coarse_iterations": coarse_iterations,
                "refine_iterations": refine_iterations,
                "perturbation": perturbation,
                "learning_rate": learning_rate,
                "worst_class_weight": worst_class_weight,
                "top1_floor": top1_floor,
                "seed": seed,
            },
        }
        _atomic_json(state_path, state)
    if (
        state.get("ordered_tier_ids") != list(tier_ids)
        or state.get("tier_menu") != tier_menu.as_mappings()
        or state.get("tier_menu_sha256") != tier_menu.sha256
    ):
        raise ValueError("SPSA resume tier menu mismatch")
    if state.get("config") != {
        "coarse_iterations": coarse_iterations,
        "refine_iterations": refine_iterations,
        "perturbation": perturbation,
        "learning_rate": learning_rate,
        "worst_class_weight": worst_class_weight,
        "top1_floor": top1_floor,
        "seed": seed,
    }:
        raise ValueError("SPSA resume configuration mismatch")

    for iteration in range(int(state["next_iteration"]), total_iterations):
        if iteration == coarse_iterations and not state["refined"]:
            refined_groups, refined_logits = refine_influential_groups(
                state["group_by_cell"], routing, state["logits"]
            )
            state["group_by_cell"] = refined_groups
            state["logits"] = refined_logits
            state["refined"] = True
            _atomic_json(state_path, state)
        group_names = sorted(state["logits"])
        generator = np.random.default_rng(seed + iteration)
        delta = generator.choice((-1.0, 1.0), size=(len(group_names), len(tier_ids)))
        train_slice = normalized_slices[iteration % len(normalized_slices)]
        pair: dict[str, tuple[dict[str, Any], dict[str, Any], float]] = {}
        for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
            perturbed = {
                group: (
                    np.asarray(state["logits"][group], dtype=np.float64)
                    + sign * perturbation * delta[group_index]
                ).tolist()
                for group_index, group in enumerate(group_names)
            }
            assignment = project_group_logits(
                menu,
                activation_bytes,
                state["group_by_cell"],
                perturbed,
                routing=routing,
                tier_menu=tier_menu,
            )
            assignment.update(
                {
                    "search_iteration": iteration,
                    "antithetic_arm": sign_name,
                    "train_slice_id": train_slice["slice_id"],
                }
            )
            pair_root = root / "iterations" / f"{iteration:03d}"
            assignment_path = pair_root / f"{sign_name}.assignment.json"
            measurement_path = pair_root / f"{sign_name}.measurement.json"
            _atomic_json(assignment_path, assignment)
            if measurement_path.exists():
                measurement = json.loads(measurement_path.read_text())
            else:
                measurement = dict(evaluator(assignment_path, train_slice, measurement_path))
                _atomic_json(measurement_path, measurement)
            measurement = _validate_measurement(
                measurement,
                assignment_sha256=assignment["assignment_sha256"],
                train_slice=train_slice,
            )
            objective = measured_objective(
                measurement,
                worst_class_weight=worst_class_weight,
                top1_floor=top1_floor,
            )
            pair[sign_name] = (assignment, measurement, objective)
            best = state.get("best")
            if best is None or objective < float(best["objective"]):
                state["best"] = {
                    "objective": objective,
                    "iteration": iteration,
                    "arm": sign_name,
                    "assignment_path": str(assignment_path),
                    "measurement_path": str(measurement_path),
                    "assignment_sha256": assignment["assignment_sha256"],
                }
        difference = pair["plus"][2] - pair["minus"][2]
        scale = difference / (2.0 * perturbation)
        for group_index, group in enumerate(group_names):
            updated = (
                np.asarray(state["logits"][group], dtype=np.float64)
                - learning_rate * scale * delta[group_index]
            )
            updated -= updated.mean()
            state["logits"][group] = updated.tolist()
        state["curve"].append(
            {
                "iteration": iteration,
                "phase": "coarse" if iteration < coarse_iterations else "refined",
                "slice_id": train_slice["slice_id"],
                "groups": len(group_names),
                "plus": {
                    "assignment_sha256": pair["plus"][0]["assignment_sha256"],
                    "objective": pair["plus"][2],
                    "mean_kld": pair["plus"][1]["mean_kld"],
                    "class_kld": pair["plus"][1]["class_kld"],
                    "top1_agreement": pair["plus"][1]["top1_agreement"],
                },
                "minus": {
                    "assignment_sha256": pair["minus"][0]["assignment_sha256"],
                    "objective": pair["minus"][2],
                    "mean_kld": pair["minus"][1]["mean_kld"],
                    "class_kld": pair["minus"][1]["class_kld"],
                    "top1_agreement": pair["minus"][1]["top1_agreement"],
                },
                "objective_difference": difference,
            }
        )
        state["next_iteration"] = iteration + 1
        _atomic_json(state_path, state)

    best = state.get("best")
    if not isinstance(best, dict):
        raise RuntimeError("SPSA search completed without a measured assignment")
    frozen = json.loads(Path(best["assignment_path"]).read_text())
    frozen["status"] = "PASS_FROZEN_PRE_REPAIR"
    frozen["frozen_from"] = dict(best)
    frozen_path = root / "FROZEN_ASSIGNMENT.json"
    _atomic_json(frozen_path, frozen)
    result = {
        "schema": "banana-smasher-measured-spsa-search-v1",
        "status": "PASS_FROZEN_PRE_REPAIR",
        "iterations": total_iterations,
        "evaluations": total_iterations * 2,
        "curve": state["curve"],
        "best": best,
        "frozen_assignment": {
            "path": str(frozen_path),
            "sha256": _sha256(frozen_path.read_bytes()),
            "assignment_sha256": frozen["assignment_sha256"],
        },
        "holdout_used": False,
        "repair_applied": False,
        "quality_signal": "measured-six-class-end-to-end-kld-and-top1-only",
    }
    _atomic_json(root / "SEARCH_TERMINAL.json", result)
    state["status"] = "PASS_FROZEN_PRE_REPAIR"
    _atomic_json(state_path, state)
    return result
