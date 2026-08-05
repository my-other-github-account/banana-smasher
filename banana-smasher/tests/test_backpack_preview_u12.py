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
FIXTURE_SHA256 = "2ebf895ce77e8b99ae7390892dc6cd31c587f360e2032ee2eed97682b91adfc4"


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
        class_caps=fixture["class_caps"],
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
