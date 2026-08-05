from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from banana_smasher.backpack_preview import (
    CLASSES,
    class_weight_preset,
    pareto_prune_six_class_options,
    prepare_preview_u12_options,
    resolve_tier_menu,
    solve_preview_u12_options,
)


FIXTURE = Path(__file__).parent / "fixtures" / "f521_preview_u12.json"
FIXTURE_SHA256 = "d089dc414d7f1c0ecabdad5fa7dd77d7dff50b41c2aab7d706aade894c12dc8e"


def test_sealed_f521_preview_u12_fixture_parity() -> None:
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    fixture = json.loads(FIXTURE.read_text())

    prepared = prepare_preview_u12_options(
        fixture["rows"],
        basis_sha256=fixture["basis_sha256"],
        include_tiers=["qtip2.5", "tier-b"],
    )

    by_key = {
        f"{row['cell_id']}:{row['tier']}": row["six_class_costs"]
        for row in prepared["options"]
    }
    assert by_key == fixture["expected"]["costs"]
    assert prepared["normalized_class_weights"] == fixture["expected"][
        "class_weights"
    ]
    assert prepared["class_weight_preset"] == "parity-all-ones"
    assert all(row["cost_authority"]["status"] == "AUTHENTICATED" for row in prepared["options"])
    solved = solve_preview_u12_options(
        prepared,
        envelope_bytes=fixture["expected"]["solve"]["assigned_bytes"],
        class_kld_bounds=fixture["class_kld_bounds"],
    )
    assert solved["assigned_bytes"] == fixture["expected"]["solve"]["assigned_bytes"]
    assert [
        {"cell_id": row["cell_id"], "tier": row["tier"]}
        for row in solved["assignments"]
    ] == fixture["expected"]["solve"]["assignments"]
    assert solved["prediction_by_class"] == pytest.approx(
        fixture["expected"]["solve"]["prediction_by_class"]
    )
    assert solved["bounds_verification"]["status"] == "PASS"


def test_legacy_weighted_objective_remains_optional() -> None:
    assert class_weight_preset("parity-all-ones") == {name: 1.0 for name in CLASSES}
    assert class_weight_preset("legacy-preview") == {
        "agentic": 1.0,
        "chat": 1.0,
        "code": 1.5,
        "multilingual": 2.0,
        "prose": 1.5,
        "reasoning": 1.0,
    }
    fixture = json.loads(FIXTURE.read_text())
    weighted = prepare_preview_u12_options(
        fixture["rows"],
        basis_sha256=fixture["basis_sha256"],
        include_tiers=["qtip2.5", "tier-b"],
        class_weight_preset_name="legacy-preview",
    )
    assert weighted["normalized_class_weights"] == {
        "agentic": 0.125,
        "chat": 0.125,
        "code": 0.1875,
        "multilingual": 0.25,
        "prose": 0.1875,
        "reasoning": 0.125,
    }
    solved = solve_preview_u12_options(
        weighted,
        envelope_bytes=3,
        class_kld_bounds=fixture["class_kld_bounds"],
    )
    assert solved["objective"]["name"] == "weighted_mean_per_class_predicted_damage"
    assert solved["objective"]["normalized_class_weights"] == weighted[
        "normalized_class_weights"
    ]


def test_pareto_prune_is_safe_across_all_six_classes() -> None:
    base = {name: 1.0 for name in CLASSES}
    options = [
        {"cell_id": "c0", "tier": "a", "physical_bytes": 1, "six_class_costs": base},
        {
            "cell_id": "c0",
            "tier": "b",
            "physical_bytes": 2,
            "six_class_costs": {name: 2.0 for name in CLASSES},
        },
        {
            "cell_id": "c0",
            "tier": "tradeoff",
            "physical_bytes": 2,
            "six_class_costs": {**base, "reasoning": 0.5},
        },
        {"cell_id": "c1", "tier": "b", "physical_bytes": 9, "six_class_costs": base},
    ]

    result = pareto_prune_six_class_options(options)

    assert [(row["cell_id"], row["tier"]) for row in result["options"]] == [
        ("c0", "a"),
        ("c0", "tradeoff"),
        ("c1", "b"),
    ]
    assert result["pruned"] == [{"cell_id": "c0", "tier": "b", "dominated_by": "a"}]


def test_tier_menu_defaults_to_current_qtip25_policy_but_overrides_are_generic() -> None:
    menu = ["qtip2", "qtip2.5", "tier-a", "tier-b"]

    assert resolve_tier_menu(menu) == ["qtip2.5"]
    assert resolve_tier_menu(menu, include_tiers=["tier-a", "tier-b"]) == [
        "tier-a",
        "tier-b",
    ]
    assert resolve_tier_menu(
        menu,
        include_tiers=["tier-a", "tier-b"],
        exclude_tiers=["tier-a"],
    ) == ["tier-b"]
    with pytest.raises(ValueError, match="unknown included tiers"):
        resolve_tier_menu(menu, include_tiers=["not-in-menu"])


def test_preview_solver_uses_uniform_six_class_objective_and_hard_caps() -> None:
    fixture = json.loads(FIXTURE.read_text())
    prepared = prepare_preview_u12_options(
        fixture["rows"],
        basis_sha256=fixture["basis_sha256"],
        include_tiers=["qtip2.5", "tier-b"],
    )
    caps = {name: 100.0 for name in CLASSES}
    caps["chat"] = 1.0

    solved = solve_preview_u12_options(
        prepared,
        envelope_bytes=3,
        class_caps=caps,
    )

    assert [(row["cell_id"], row["tier"]) for row in solved["assignments"]] == [
        ("c0", "qtip2.5"),
        ("c1", "qtip2.5"),
    ]
    assert solved["prediction_by_class"]["chat"] == pytest.approx(0.65)
    assert solved["objective"]["name"] == "uniform_mean_per_class_predicted_damage"
    assert solved["objective"]["normalized_class_weights"] == {
        name: pytest.approx(1.0 / 6.0) for name in CLASSES
    }
    expected = sum(solved["prediction_by_class"].values()) / 6.0
    assert solved["objective"]["value"] == pytest.approx(expected)


def test_preview_solver_exposes_explicit_per_class_max_kld_bounds() -> None:
    fixture = json.loads(FIXTURE.read_text())
    prepared = prepare_preview_u12_options(
        fixture["rows"],
        basis_sha256=fixture["basis_sha256"],
        include_tiers=["qtip2.5", "tier-b"],
    )
    bounds = {name: {"max_kld": 100.0} for name in CLASSES}
    bounds["chat"]["max_kld"] = 1.0

    solved = solve_preview_u12_options(
        prepared,
        envelope_bytes=3,
        class_kld_bounds=bounds,
    )

    assert solved["class_kld_bounds"] == {
        name: {
            "min_kld": 0.0,
            "max_kld": 1.0 if name == "chat" else 100.0,
        }
        for name in CLASSES
    }
    assert solved["bounds_verification"] == {
        "status": "PASS",
        "semantics": "lower_kld_is_better; minimum quality is a max_kld ceiling",
    }
    assert solved["prediction_by_class"]["chat"] <= 1.0


def test_preview_solver_accepts_explicit_min_kld_floor() -> None:
    fixture = json.loads(FIXTURE.read_text())
    prepared = prepare_preview_u12_options(
        fixture["rows"],
        basis_sha256=fixture["basis_sha256"],
        include_tiers=["qtip2.5", "tier-b"],
    )
    bounds = {name: {"max_kld": 100.0} for name in CLASSES}
    bounds["agentic"]["min_kld"] = 1.0

    solved = solve_preview_u12_options(
        prepared,
        envelope_bytes=3,
        class_kld_bounds=bounds,
    )

    assert solved["prediction_by_class"]["agentic"] >= 1.0
    assert solved["class_kld_bounds"]["agentic"] == {
        "min_kld": 1.0,
        "max_kld": 100.0,
    }


def test_preview_solver_rejects_infeasible_per_class_kld_bound() -> None:
    fixture = json.loads(FIXTURE.read_text())
    prepared = prepare_preview_u12_options(
        fixture["rows"],
        basis_sha256=fixture["basis_sha256"],
        include_tiers=["qtip2.5", "tier-b"],
    )
    bounds = {name: {"max_kld": 100.0} for name in CLASSES}
    bounds["reasoning"]["max_kld"] = 0.0

    with pytest.raises(ValueError, match="infeasible class_kld_bounds"):
        solve_preview_u12_options(
            prepared,
            envelope_bytes=3,
            class_kld_bounds=bounds,
        )
