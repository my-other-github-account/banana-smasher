from banana_smasher.knapsack import solve_class_balanced_options


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
