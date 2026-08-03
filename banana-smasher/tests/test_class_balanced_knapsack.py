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
