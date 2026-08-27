from types import SimpleNamespace

import numpy as np
import pytest
import scipy.optimize

from banana_smasher.knapsack import KnapsackValidationError, solve_class_balanced_options


def test_solver_rejects_scalar_best_assignment_that_breaks_code_cap():
    result = solve_class_balanced_options(
        cells=["L000.E000.down", "L000.E001.down"],
        tiers=["qtip2", "d4"],
        bytes_by_option={
            ("L000.E000.down", "qtip2"): 1,
            ("L000.E000.down", "d4"): 2,
            ("L000.E001.down", "qtip2"): 1,
            ("L000.E001.down", "d4"): 2,
        },
        class_costs_by_option={
            ("L000.E000.down", "qtip2"): {"chat": 0.05, "code": 0.70},
            ("L000.E000.down", "d4"): {"chat": 0.30, "code": 0.10},
            ("L000.E001.down", "qtip2"): {"chat": 0.05, "code": 0.70},
            ("L000.E001.down", "d4"): {"chat": 0.30, "code": 0.10},
        },
        envelope_bytes=3,
        class_caps={"chat": 1.0, "code": 0.8},
    )

    assert result["assigned_bytes"] == 3
    assert result["prediction_by_class"]["code"] <= 0.8
    assert sorted(row["tier"] for row in result["assignments"]) == ["d4", "qtip2"]
    assert result["objective"]["name"] == "uniform_mean_per_class_predicted_damage"


def test_equal_options_use_explicit_manifest_order_tie_breaker():
    arguments = {
        "cells": ["L000.E000.down", "L000.E001.down"],
        "tiers": ["qtip2", "d4"],
        "bytes_by_option": {
            (cell, tier): 1
            for cell in ("L000.E000.down", "L000.E001.down")
            for tier in ("qtip2", "d4")
        },
        "class_costs_by_option": {
            (cell, tier): {"chat": 0.25, "code": 0.5}
            for cell in ("L000.E000.down", "L000.E001.down")
            for tier in ("qtip2", "d4")
        },
        "envelope_bytes": 2,
        "class_caps": {"chat": 1.0, "code": 1.0},
    }

    repeated = [solve_class_balanced_options(**arguments) for _ in range(3)]

    assert repeated[0] == repeated[1] == repeated[2]
    assert [row["tier"] for row in repeated[0]["assignments"]] == [
        "qtip2",
        "qtip2",
    ]
    assert repeated[0]["solver"]["equal_option_tie_breaker"] == (
        "first_manifest_tier"
    )


def test_exact_envelope_selects_exact_option_instead_of_cheaper_underfill():
    result = solve_class_balanced_options(
        cells=["cell"],
        tiers=["small", "exact"],
        bytes_by_option={("cell", "small"): 1, ("cell", "exact"): 2},
        class_costs_by_option={
            ("cell", "small"): {"chat": 0.0},
            ("cell", "exact"): {"chat": 1.0},
        },
        envelope_bytes=2,
        class_caps={"chat": 2.0},
        exact_envelope=True,
    )

    assert result["assigned_bytes"] == 2
    assert result["slack_bytes"] == 0
    assert result["assignments"][0]["tier"] == "exact"


def test_class_balanced_solver_forces_fallback_when_q3_is_unavailable():
    cells = ["L000.E000.down", "L001.E000.down"]
    tiers = ["qtip2", "qtip3"]
    options = {(cell, tier) for cell in cells for tier in tiers}

    result = solve_class_balanced_options(
        cells=cells,
        tiers=tiers,
        bytes_by_option={key: 1 if key[1] == "qtip2" else 2 for key in options},
        class_costs_by_option={
            key: {"quality": 1.0 if key[1] == "qtip2" else 0.0} for key in options
        },
        envelope_bytes=3,
        class_caps={"quality": 10.0},
        available_options={
            ("L000.E000.down", "qtip2"),
            ("L000.E000.down", "qtip3"),
            ("L001.E000.down", "qtip2"),
        },
    )

    assert {row["cell_id"]: row["tier"] for row in result["assignments"]} == {
        "L000.E000.down": "qtip3",
        "L001.E000.down": "qtip2",
    }


def test_class_balanced_solver_rejects_cell_without_available_tier():
    cells = ["L000.E000.down"]
    tiers = ["qtip2", "qtip3"]
    options = {(cell, tier) for cell in cells for tier in tiers}

    with pytest.raises(KnapsackValidationError, match="no available tier"):
        solve_class_balanced_options(
            cells=cells,
            tiers=tiers,
            bytes_by_option={key: 1 for key in options},
            class_costs_by_option={key: {"quality": 1.0} for key in options},
            envelope_bytes=1,
            class_caps={"quality": 10.0},
            available_options=set(),
        )
def test_status_zero_integral_solution_accepts_representational_nonzero_gap(monkeypatch):
    monkeypatch.setattr(
        scipy.optimize,
        "milp",
        lambda **_kwargs: SimpleNamespace(
            success=True,
            x=np.array([1.0]),
            status=0,
            message="optimal within solver numerics",
            mip_gap=1e-12,
        ),
    )

    result = solve_class_balanced_options(
        cells=["cell"],
        tiers=["tier"],
        bytes_by_option={("cell", "tier"): 1},
        class_costs_by_option={("cell", "tier"): {"chat": 0.25}},
        envelope_bytes=1,
        class_caps={"chat": 1.0},
    )

    assert result["solver"]["status"] == 0
    assert result["solver"]["mip_gap"] == 1e-12
    assert result["assignments"] == [
        {
            "cell_id": "cell",
            "tier": "tier",
            "bytes": 1,
            "prediction_by_class": {"chat": 0.25},
        }
    ]


def test_solver_enforces_layer_class_concentration_caps():
    result = solve_class_balanced_options(
        cells=["L000:E000:down", "L001:E000:down"],
        tiers=["cheap", "safe"],
        bytes_by_option={
            ("L000:E000:down", "cheap"): 1,
            ("L000:E000:down", "safe"): 2,
            ("L001:E000:down", "cheap"): 1,
            ("L001:E000:down", "safe"): 2,
        },
        class_costs_by_option={
            ("L000:E000:down", "cheap"): {"code": 0.9},
            ("L000:E000:down", "safe"): {"code": 0.1},
            ("L001:E000:down", "cheap"): {"code": 0.9},
            ("L001:E000:down", "safe"): {"code": 0.1},
        },
        envelope_bytes=3,
        class_caps={"code": 1.8},
        concentration_groups_by_cell={
            "L000:E000:down": "L000",
            "L001:E000:down": "L001",
        },
        concentration_caps={
            "L000": {"code": 0.2},
            "L001": {"code": 0.9},
        },
    )

    assert [row["tier"] for row in result["assignments"]] == ["safe", "cheap"]
    assert result["concentration_totals"] == {
        "L000": {"code": 0.1},
        "L001": {"code": 0.9},
    }
