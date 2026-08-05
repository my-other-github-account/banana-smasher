from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any


CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")
CURRENT_CAMPAIGN_TIER_POLICY = ("qtip2.5",)
_CLASS_WEIGHT_PRESETS = {
    "parity-all-ones": {
        "agentic": 1.0,
        "chat": 1.0,
        "code": 1.0,
        "multilingual": 1.0,
        "prose": 1.0,
        "reasoning": 1.0,
    },
    "legacy-preview": {
        "agentic": 1.0,
        "chat": 1.0,
        "code": 1.5,
        "multilingual": 2.0,
        "prose": 1.5,
        "reasoning": 1.0,
    },
}
_AUTHORITY_FIELDS = (
    "six_class_predictions_sha256",
    "routing_importance_sha256",
    "projection_correction_sha256",
    "physical_bytes_sha256",
)
_HEX = frozenset("0123456789abcdef")


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _finite(value: object, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or (nonnegative and parsed < 0.0):
        qualifier = "non-negative finite" if nonnegative else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return parsed


def _six_class_values(
    value: object, label: str, *, nonnegative: bool = False
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(CLASSES):
        raise ValueError(f"{label} must exactly cover the six canonical classes")
    return {
        name: _finite(value[name], f"{label}.{name}", nonnegative=nonnegative)
        for name in CLASSES
    }


def _sha_field(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or set(value) > _HEX
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def class_weight_preset(name: str = "parity-all-ones") -> dict[str, float]:
    """Return an explicit six-class preset; parity is all ones."""

    try:
        return dict(_CLASS_WEIGHT_PRESETS[name])
    except KeyError as exc:
        raise ValueError(
            f"unknown class weight preset {name!r}; expected {sorted(_CLASS_WEIGHT_PRESETS)}"
        ) from exc


def _normalized_weights(weights: Mapping[str, object]) -> dict[str, float]:
    parsed = _six_class_values(weights, "class_weights", nonnegative=True)
    total = math.fsum(parsed.values())
    if total <= 0.0:
        raise ValueError("class_weights must have positive total mass")
    return {name: parsed[name] / total for name in CLASSES}


def resolve_tier_menu(
    menu: Sequence[str],
    *,
    include_tiers: Sequence[str] | None = None,
    exclude_tiers: Sequence[str] = (),
) -> list[str]:
    """Resolve a generic tier menu under explicit include/exclude policy.

    The current campaign defaults to QTIP2.5, while callers can name any tier-A/
    tier-B experiment without changing this implementation.
    """

    if not menu or any(not isinstance(tier, str) or not tier for tier in menu):
        raise ValueError("tier menu must contain non-empty strings")
    if len(set(menu)) != len(menu):
        raise ValueError("tier menu must not contain duplicates")
    included = list(CURRENT_CAMPAIGN_TIER_POLICY if include_tiers is None else include_tiers)
    excluded = list(exclude_tiers)
    if any(not isinstance(tier, str) or not tier for tier in [*included, *excluded]):
        raise ValueError("included and excluded tiers must be non-empty strings")
    unknown_included = sorted(set(included) - set(menu))
    unknown_excluded = sorted(set(excluded) - set(menu))
    if unknown_included:
        raise ValueError(f"unknown included tiers: {unknown_included}")
    if unknown_excluded:
        raise ValueError(f"unknown excluded tiers: {unknown_excluded}")
    selected = [tier for tier in menu if tier in set(included) and tier not in set(excluded)]
    if not selected:
        raise ValueError("tier policy selected no options")
    return selected


def _projection_corrections(value: object) -> dict[str, float]:
    if isinstance(value, Mapping):
        return _six_class_values(value, "projection_correction")
    correction = _finite(value, "projection_correction")
    return {name: correction for name in CLASSES}


def prepare_preview_u12_options(
    rows: Sequence[Mapping[str, Any]],
    *,
    basis_sha256: str,
    include_tiers: Sequence[str] | None = None,
    exclude_tiers: Sequence[str] = (),
    class_weight_preset_name: str = "parity-all-ones",
    class_weights: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Build authenticated Preview-U12 option-level six-class costs.

    For class ``c``, the F521 parity cost is
    ``max(0, (prediction[c] + projection_correction[c])
              * routing_importance[c] * projection_weight)``.
    """

    basis = _sha_field(basis_sha256, "basis_sha256")
    if not rows:
        raise ValueError("Preview-U12 options require non-empty rows")
    menu: list[str] = []
    for row in rows:
        tier = row.get("tier")
        if isinstance(tier, str) and tier and tier not in menu:
            menu.append(tier)
    selected_tiers = resolve_tier_menu(
        menu, include_tiers=include_tiers, exclude_tiers=exclude_tiers
    )
    selected = set(selected_tiers)

    if class_weights is None:
        raw_weights = class_weight_preset(class_weight_preset_name)
        preset_name = class_weight_preset_name
    else:
        raw_weights = _six_class_values(
            class_weights, "class_weights", nonnegative=True
        )
        preset_name = "custom"
    normalized_weights = _normalized_weights(raw_weights)

    options: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        tier = row.get("tier")
        if tier not in selected:
            continue
        if row.get("basis_sha256") != basis:
            raise ValueError(f"option row {index} basis mismatch")
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError(f"option row {index} cell_id must be non-empty")
        key = (cell_id, str(tier))
        if key in seen:
            raise ValueError(f"duplicate option for cell/tier {key!r}")
        seen.add(key)
        physical_bytes = row.get("physical_bytes")
        if (
            isinstance(physical_bytes, bool)
            or not isinstance(physical_bytes, int)
            or physical_bytes < 0
        ):
            raise ValueError(f"option {key!r} physical_bytes must be non-negative integer")
        predictions = _six_class_values(
            row.get("six_class_predictions"),
            f"option {key!r} six_class_predictions",
            nonnegative=True,
        )
        routing = _six_class_values(
            row.get("routing_importance_by_class"),
            f"option {key!r} routing_importance_by_class",
            nonnegative=True,
        )
        projection_weight = _finite(
            row.get("projection_weight"),
            f"option {key!r} projection_weight",
            nonnegative=True,
        )
        corrections = _projection_corrections(row.get("projection_correction"))
        authority = row.get("authority", row.get("dimension_authority"))
        if not isinstance(authority, Mapping):
            raise ValueError(f"option {key!r} requires dimension authority")
        authenticated = {
            field: _sha_field(authority.get(field), f"option {key!r} authority.{field}")
            for field in _AUTHORITY_FIELDS
        }
        inputs = {
            "basis_sha256": basis,
            "cell_id": cell_id,
            "tier": tier,
            "physical_bytes": physical_bytes,
            "six_class_predictions": predictions,
            "routing_importance_by_class": routing,
            "projection_weight": projection_weight,
            "projection_correction_by_class": corrections,
            "dimension_authority": authenticated,
        }
        costs = {
            name: max(
                0.0,
                float(
                    (Decimal(str(predictions[name])) + Decimal(str(corrections[name])))
                    * Decimal(str(routing[name]))
                    * Decimal(str(projection_weight))
                ),
            )
            for name in CLASSES
        }
        options.append(
            {
                "cell_id": cell_id,
                "tier": tier,
                "physical_bytes": physical_bytes,
                "six_class_costs": costs,
                "cost_authority": {
                    "status": "AUTHENTICATED",
                    "formula": "max(0,(prediction+projection_correction)*routing_importance*projection_weight)",
                    "inputs_sha256": _sha256(inputs),
                    "six_class_costs_sha256": _sha256(costs),
                    "dimension_authority": authenticated,
                },
            }
        )
    if not options:
        raise ValueError("selected tier policy has no option rows")
    return {
        "schema": "banana-smasher-preview-u12-options-v1",
        "status": "PASS_AUTHENTICATED_OPTION_COSTS",
        "basis_sha256": basis,
        "classes": list(CLASSES),
        "selected_tiers": selected_tiers,
        "class_weight_preset": preset_name,
        "raw_class_weights": raw_weights,
        "normalized_class_weights": normalized_weights,
        "options": options,
    }


def pareto_prune_six_class_options(
    options: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Prune only byte-and-six-class dominated options within each cell."""

    parsed: list[dict[str, Any]] = []
    for index, option in enumerate(options):
        cell_id = option.get("cell_id")
        tier = option.get("tier")
        physical_bytes = option.get("physical_bytes")
        if not isinstance(cell_id, str) or not cell_id or not isinstance(tier, str) or not tier:
            raise ValueError(f"Pareto option {index} requires cell_id and tier")
        if (
            isinstance(physical_bytes, bool)
            or not isinstance(physical_bytes, int)
            or physical_bytes < 0
        ):
            raise ValueError(f"Pareto option {index} physical_bytes must be non-negative")
        costs = _six_class_values(
            option.get("six_class_costs"),
            f"Pareto option {index} six_class_costs",
        )
        parsed.append({**option, "six_class_costs": costs})

    kept: list[dict[str, Any]] = []
    pruned: list[dict[str, str]] = []
    for candidate in parsed:
        dominator: dict[str, Any] | None = None
        for other in parsed:
            if other is candidate or other["cell_id"] != candidate["cell_id"]:
                continue
            no_worse = other["physical_bytes"] <= candidate["physical_bytes"] and all(
                other["six_class_costs"][name]
                <= candidate["six_class_costs"][name]
                for name in CLASSES
            )
            strictly_better = other["physical_bytes"] < candidate["physical_bytes"] or any(
                other["six_class_costs"][name]
                < candidate["six_class_costs"][name]
                for name in CLASSES
            )
            if no_worse and strictly_better:
                dominator = other
                break
        if dominator is None:
            kept.append(candidate)
        else:
            pruned.append(
                {
                    "cell_id": str(candidate["cell_id"]),
                    "tier": str(candidate["tier"]),
                    "dominated_by": str(dominator["tier"]),
                }
            )
    return {
        "schema": "banana-smasher-six-class-pareto-prune-v1",
        "status": "PASS",
        "options": kept,
        "pruned": pruned,
    }


def solve_preview_u12_options(
    prepared: Mapping[str, Any],
    *,
    envelope_bytes: int,
    class_caps: Mapping[str, object] | None = None,
    class_kld_bounds: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Solve authenticated options under explicit per-class KLD bounds.

    Lower KLD is better, so a minimum-quality requirement is expressed as a
    ``max_kld`` ceiling. ``class_caps`` remains a compatibility alias for
    callers of the first Preview-U12 API revision.
    """

    if (
        prepared.get("schema") != "banana-smasher-preview-u12-options-v1"
        or prepared.get("status") != "PASS_AUTHENTICATED_OPTION_COSTS"
        or prepared.get("classes") != list(CLASSES)
    ):
        raise ValueError("prepared Preview-U12 options schema/status/classes mismatch")
    options = prepared.get("options")
    tiers = prepared.get("selected_tiers")
    if not isinstance(options, list) or not options:
        raise ValueError("prepared Preview-U12 options must be non-empty")
    if (
        not isinstance(tiers, list)
        or not tiers
        or any(not isinstance(tier, str) or not tier for tier in tiers)
    ):
        raise ValueError("prepared Preview-U12 selected_tiers are invalid")

    cells: list[str] = []
    bytes_by_option: dict[tuple[str, str], int] = {}
    class_costs_by_option: dict[tuple[str, str], dict[str, float]] = {}
    for index, option in enumerate(options):
        if not isinstance(option, Mapping):
            raise ValueError(f"prepared Preview-U12 option {index} must be an object")
        cell_id = option.get("cell_id")
        tier = option.get("tier")
        authority = option.get("cost_authority")
        if (
            not isinstance(cell_id, str)
            or not cell_id
            or tier not in tiers
            or not isinstance(authority, Mapping)
            or authority.get("status") != "AUTHENTICATED"
        ):
            raise ValueError(f"prepared Preview-U12 option {index} identity/authority mismatch")
        if cell_id not in cells:
            cells.append(cell_id)
        key = (cell_id, str(tier))
        if key in bytes_by_option:
            raise ValueError(f"duplicate prepared Preview-U12 option {key!r}")
        physical_bytes = option.get("physical_bytes")
        if (
            isinstance(physical_bytes, bool)
            or not isinstance(physical_bytes, int)
            or physical_bytes < 0
        ):
            raise ValueError(f"prepared Preview-U12 option {key!r} bytes are invalid")
        bytes_by_option[key] = physical_bytes
        class_costs_by_option[key] = _six_class_values(
            option.get("six_class_costs"),
            f"prepared Preview-U12 option {key!r} six_class_costs",
            nonnegative=True,
        )

    expected = {(cell, tier) for cell in cells for tier in tiers}
    if set(bytes_by_option) != expected:
        raise ValueError("prepared Preview-U12 options must cover every selected cell/tier")
    if class_kld_bounds is not None and class_caps is not None:
        raise ValueError("provide class_kld_bounds or class_caps, not both")
    if class_kld_bounds is None:
        if class_caps is None:
            raise ValueError("class_kld_bounds must provide a max_kld ceiling for each class")
        caps = _six_class_values(class_caps, "class_caps", nonnegative=True)
        bounds = {
            name: {"min_kld": 0.0, "max_kld": caps[name]} for name in CLASSES
        }
    else:
        if set(class_kld_bounds) != set(CLASSES):
            raise ValueError("class_kld_bounds must exactly cover the six canonical classes")
        bounds = {}
        for name in CLASSES:
            value = class_kld_bounds[name]
            if not isinstance(value, Mapping) or set(value) - {"min_kld", "max_kld"}:
                raise ValueError(
                    f"class_kld_bounds.{name} must contain max_kld and optional min_kld"
                )
            if "max_kld" not in value:
                raise ValueError(f"class_kld_bounds.{name}.max_kld is required")
            minimum = _finite(
                value.get("min_kld", 0.0),
                f"class_kld_bounds.{name}.min_kld",
                nonnegative=True,
            )
            maximum = _finite(
                value["max_kld"],
                f"class_kld_bounds.{name}.max_kld",
                nonnegative=True,
            )
            if minimum > maximum:
                raise ValueError(f"class_kld_bounds.{name} min_kld exceeds max_kld")
            bounds[name] = {"min_kld": minimum, "max_kld": maximum}
        caps = {name: bounds[name]["max_kld"] for name in CLASSES}
    raw_weights = _six_class_values(
        prepared.get("raw_class_weights"), "raw_class_weights", nonnegative=True
    )
    preset = prepared.get("class_weight_preset")

    from .knapsack import solve_class_balanced_options

    try:
        solved = solve_class_balanced_options(
            cells=cells,
            tiers=list(tiers),
            bytes_by_option=bytes_by_option,
            class_costs_by_option=class_costs_by_option,
            envelope_bytes=envelope_bytes,
            class_caps=caps,
            class_weights=None if preset == "parity-all-ones" else raw_weights,
            class_floors={name: bounds[name]["min_kld"] for name in CLASSES},
        )
    except RuntimeError as exc:
        raise ValueError(f"infeasible class_kld_bounds: {exc}") from exc
    predicted = solved["prediction_by_class"]
    if any(
        predicted[name] < bounds[name]["min_kld"] - 1e-10
        or predicted[name] > bounds[name]["max_kld"] + 1e-10
        for name in CLASSES
    ):
        raise RuntimeError("Preview-U12 returned prediction violates class_kld_bounds")
    return {
        **solved,
        "schema": "banana-smasher-preview-u12-solve-v1",
        "basis_sha256": prepared.get("basis_sha256"),
        "class_weight_preset": preset,
        "raw_class_weights": raw_weights,
        "class_kld_bounds": bounds,
        "bounds_verification": {
            "status": "PASS",
            "semantics": "lower_kld_is_better; minimum quality is a max_kld ceiling",
        },
    }
