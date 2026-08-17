from __future__ import annotations

from pathlib import Path

import pytest

from banana_smasher.cli import main
from banana_smasher.experiments import (
    ExperimentSpec,
    WindowSchedule,
    build_experiment_lock,
    diff_experiments,
    document_experiment,
    load_experiment,
    validate_runtime_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "experiments"


def _spec_dict() -> dict:
    sha_a = "a" * 64
    sha_b = "b" * 64
    sha_c = "c" * 64
    return {
        "schema_version": 1,
        "name": "fixture",
        "mode": "extend",
        "reproduction_of": None,
        "scientific": {
            "basis": {"id": "basis", "sha256": sha_a},
            "identity_source": {"id": "source", "sha256": sha_b},
            "parent": {
                "checkpoint": {"id": "UPDATE_005", "sha256": sha_c},
                "next_update": 5,
            },
            "data": {
                "prompt": {"id": "prompt", "sha256": sha_a},
                "corpus": {"id": "corpus", "sha256": sha_b},
                "teacher": {"id": "teacher", "sha256": sha_c},
            },
            "window_schedule": {
                "mode": "sequential",
                "ordered_windows": list(range(20, 84)),
                "windows_per_optimizer_step": 4,
            },
            "batch_size": 4,
            "loss_scaling": "mean; no extra 4/64 factor",
            "optimizer": {
                "name": "Adam",
                "lut_gain_lr": 0.01,
                "norm_lr": 0.0001,
                "scheduler": "cosine",
                "cosine_min_ratio": 0.1,
                "cosine_updates": 64,
                "warmup_updates": 0,
                "first_applied_lut_gain_lr": 0.01,
            },
            "mutable_surfaces": ["43 layer LUTs"],
            "frozen_surfaces": ["codes"],
            "evaluation": {
                "suite_lock": {
                    "id": "evaluations/configs/balanced64-v1.json",
                    "sha256": "d5610f11c23b75f81e196e74407cb7e642a4f4a2e12f55925e13e5a7fe43ffb9",
                },
                "pre": {
                    "checkpoint_sha256": sha_c,
                    "update": 5,
                    "kld": 0.226162314683653,
                    "top1_matches": 56700,
                    "positions": 65536,
                },
                "scorer_contract": "BALANCED64_V1 suite lock",
            },
        },
        "execution": {"device": "cuda:0", "kernel": "official-k2"},
    }


def test_sequential_green_groups_cover_u1_through_next_group() -> None:
    schedule = WindowSchedule(
        mode="sequential",
        ordered_windows=tuple(range(20, 84)),
        windows_per_optimizer_step=4,
    )

    assert [schedule.windows_for_update(update) for update in range(1, 6)] == [
        (20, 21, 22, 23),
        (24, 25, 26, 27),
        (28, 29, 30, 31),
        (32, 33, 34, 35),
        (36, 37, 38, 39),
    ]
    assert schedule.windows_for_update(6) == (40, 41, 42, 43)


def test_execution_changes_do_not_change_scientific_identity() -> None:
    reference = ExperimentSpec.from_dict(_spec_dict())
    changed = _spec_dict()
    changed["execution"] = {"device": "cuda:7", "kernel": "experimental-kernel"}
    candidate = ExperimentSpec.from_dict(changed)

    assert reference.scientific_identity_sha256 == candidate.scientific_identity_sha256


def test_green_no_warmup_cosine_and_checked_in_groups() -> None:
    green = load_experiment(EXPERIMENTS / "green-run1978-u5.json")
    optimizer = green.scientific.optimizer

    assert optimizer.learning_rates_for_update(1) == pytest.approx((0.01, 0.0001))
    expected_u2_factor = (
        0.1 + 0.9 * (1 + __import__("math").cos(__import__("math").pi / 64)) / 2
    )
    assert optimizer.learning_rates_for_update(2) == pytest.approx(
        (0.01 * expected_u2_factor, 0.0001 * expected_u2_factor)
    )
    assert green.scientific.window_schedule.windows_for_update(6) == (40, 41, 42, 43)


def test_all64_diagnostic_diff_names_scientific_drift() -> None:
    green = load_experiment(EXPERIMENTS / "green-run1978-u5.json")
    diagnostic = load_experiment(EXPERIMENTS / "all64-warmup-diagnostic.json")

    result = diff_experiments(green, diagnostic)
    paths = {change["path"] for change in result["changes"]}

    assert result["classification"] == "SCIENTIFIC"
    assert {
        "mode",
        "parent.checkpoint.id",
        "window_schedule.windows_per_optimizer_step",
        "optimizer.warmup_updates",
        "optimizer.first_applied_lut_gain_lr",
    } <= paths


def test_green_document_and_lock_are_derived_from_config() -> None:
    green = load_experiment(EXPERIMENTS / "green-run1978-u5.json")

    assert (
        document_experiment(green) == (EXPERIMENTS / "green-run1978-u5.md").read_text()
    )
    lock = build_experiment_lock(green)
    assert lock["status"] == "REPRODUCTION"
    assert lock["derived"]["first_group"] == [20, 21, 22, 23]
    assert lock["derived"]["next_update"] == 5
    assert lock["derived"]["next_group"] == [40, 41, 42, 43]
    assert lock["runtime_contract"]["update"] == 5


def test_runtime_mismatch_stops_before_joint_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    green = load_experiment(EXPERIMENTS / "green-run1978-u5.json")
    lock = build_experiment_lock(green)
    observed = dict(lock["runtime_contract"])
    observed["update"] = 999
    lock_path = tmp_path / "lock.json"
    observed_path = tmp_path / "observed.json"
    lock_path.write_text(__import__("json").dumps(lock))
    observed_path.write_text(__import__("json").dumps(observed))
    called = False

    def forbidden_loader(**_: object) -> dict:
        nonlocal called
        called = True
        raise AssertionError("trainer/model loader must not run")

    monkeypatch.setattr(
        "banana_smasher.qtip_v7_joint_workflow.train_joint", forbidden_loader
    )
    rc = main(
        [
            "qtip-v7-joint-repair",
            "train",
            "--freeze",
            str(tmp_path / "freeze.json"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--target-update",
            "6",
            "--trainer",
            str(tmp_path / "trainer.py"),
            "--experiment-lock",
            str(lock_path),
            "--experiment-observed",
            str(observed_path),
        ]
    )

    assert rc != 0
    assert called is False
    assert "runtime contract mismatch" in capsys.readouterr().err


def test_validate_runtime_contract_accepts_exact_observation() -> None:
    lock = build_experiment_lock(load_experiment(EXPERIMENTS / "green-run1978-u5.json"))
    result = validate_runtime_contract(lock, lock["runtime_contract"])
    assert result["status"] == "PASS"


def test_reproduce_mode_rejects_scientific_override() -> None:
    raw = __import__("json").loads((EXPERIMENTS / "green-run1978-u5.json").read_text())
    raw["scientific"]["batch_size"] = 8
    changed = ExperimentSpec.from_dict(
        raw, source_path=EXPERIMENTS / "green-run1978-u5.json"
    )

    with pytest.raises(ValueError, match="reproduction scientific drift"):
        build_experiment_lock(changed)


def test_explicit_window_groups_are_exact() -> None:
    schedule = WindowSchedule(
        mode="explicit",
        ordered_windows=(9, 2, 7, 5),
        windows_per_optimizer_step=2,
        explicit_groups=((9, 2), (7, 5)),
    )
    assert schedule.windows_for_update(2) == (7, 5)
