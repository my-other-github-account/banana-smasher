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


def test_solver_preserves_verified_feasible_incumbent_at_time_limit(monkeypatch):
    from types import SimpleNamespace

    import numpy as np
    import scipy.optimize

    observed_options = {}

    def limited_milp(*args, **kwargs):
        observed_options.update(kwargs["options"])
        return SimpleNamespace(
            success=False,
            status=1,
            message="Time limit reached",
            x=np.array([1.0, 0.0, 0.0, 1.0]),
            mip_gap=0.01,
        )

    monkeypatch.setattr(scipy.optimize, "milp", limited_milp)
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
        time_limit_seconds=0.01,
        require_optimal=False,
    )

    assert observed_options["time_limit"] == 0.01
    assert result["status"] == "PASS_FEASIBLE_PREDICTION_ONLY"
    assert result["assigned_bytes"] == 3
    assert result["solver"]["status"] == 1
    assert result["solver"]["mip_gap"] == 0.01
    assert result["solver"]["optimality_proven"] is False
