from __future__ import annotations

import copy
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

from repair_api.api import ResidentRepairAPI
from repair_api.balanced64 import ArtifactError
from repair_api.modern_green_resident import (
    ModernGreenResidentEngine,
    PUBLISHED_PRE_RECIPE_ID,
    PUBLISHED_PRE_SHA256,
)


class _FakeDist:
    @staticmethod
    def all_gather_object(rows, local):
        rows[0] = copy.deepcopy(local)
        rows[1] = copy.deepcopy(local)


def test_sign_inverted_diagnostic_perturbs_negative_direction_and_restores_zero_persistence(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = False
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda _step: 1.0)
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)

    def pipeline_pass(self, _windows, *, loss_divisor):
        parameter.grad = torch.tensor([1.0])
        return 1.5, {"forward_seconds": 0.1, "backward_seconds": 0.2}

    observed = {}

    def validate(self, windows, teacher_root):
        observed["perturbed_value"] = float(parameter.detach()[0])
        return {
            "kld_mean": 0.12,
            "windows": list(windows),
            "runtime_counters": {
                "timed_model_payload_reads": 0,
                "timed_score_file_reads": 0,
            },
        }

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(validate, engine)
    optimizer_before = copy.deepcopy(engine.optimizer.state_dict())
    scheduler_before = copy.deepcopy(engine.scheduler.state_dict())

    result = engine.diagnostic_perturb_and_validate([28], tmp_path, direction=-1)

    assert observed["perturbed_value"] > 2.0
    assert parameter.detach().tolist() == [2.0]
    assert engine.optimizer.state_dict() == optimizer_before
    assert engine.scheduler.state_dict() == scheduler_before
    assert engine.global_step == 40
    assert result["perturbation_direction"] == -1
    assert result["checkpoint_persisted"] is False
    assert result["lineage_claimed"] is False
    assert result["restoration"]["exact"] is True
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("train_windows", ([28, 28], [28, 56], [56, 28], [56, 56]))
def test_objective_aligned_diagnostic_consumes_controlled_windows_and_restores(
    tmp_path: Path, monkeypatch, train_windows,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [28, 56]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}], momentum=0.9
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda _step: 1.0)
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)
    observed = {"train": [], "validation": []}

    def pipeline_pass(self, windows, *, loss_divisor):
        observed["train"].append(list(windows))
        parameter.grad = torch.tensor([1.0])
        return 1.5, {"forward_seconds": 0.1, "backward_seconds": 0.2}

    def validate(self, windows, teacher_root):
        observed["validation"].append(list(windows))
        assert teacher_root == tmp_path
        assert float(parameter.detach()[0]) < 2.0
        return {
            "kld_mean": 0.12,
            "windows": list(windows),
            "runtime_counters": {"timed_model_payload_reads": 0, "timed_score_file_reads": 0},
        }

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(validate, engine)
    parameter_before = parameter.detach().clone()
    optimizer_before = copy.deepcopy(engine.optimizer.state_dict())
    scheduler_before = copy.deepcopy(engine.scheduler.state_dict())

    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=train_windows
    )

    assert observed == {"train": [train_windows], "validation": [[28]]}
    assert torch.equal(parameter.detach(), parameter_before)
    assert engine.optimizer.state_dict() == optimizer_before
    assert engine.scheduler.state_dict() == scheduler_before
    assert engine.global_step == 40
    assert result["train_windows"] == train_windows
    assert result["validation"]["windows"] == [28]
    assert result["checkpoint_persisted"] is False
    assert result["lineage_claimed"] is False
    assert result["restoration"]["exact"] is True
    assert list(tmp_path.iterdir()) == []


def test_equal_norm_objective_composition_normalizes_constituents_then_averages(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda _step: 1.0)
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)
    observed = {"train": []}

    def pipeline_pass(self, windows, *, loss_divisor):
        observed["train"].append(list(windows))
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([0.0, 4.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    def validate(self, windows, teacher_root):
        observed["composed_gradient"] = parameter.grad.detach().clone()
        return {"kld_mean": 0.12, "windows": list(windows)}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(validate, engine)

    result = engine.diagnostic_perturb_and_validate(
        [28],
        tmp_path,
        direction=1,
        train_windows=[56, 28],
        objective_composition="equal_norm",
    )

    expected_local = torch.tensor([0.5 / (2.0**0.5), 0.5 / (2.0**0.5)])
    assert observed["train"] == [[56], [28]]
    assert torch.allclose(observed["composed_gradient"], expected_local)
    assert result["objective_composition"] == "equal_norm"
    assert result["step"]["constituent_gradient_norms"] == pytest.approx([18.0**0.5, 32.0**0.5])
    assert result["train_windows"] == [56, 28]


def test_pcgrad_equal_norm_projects_conflicting_constituents_symmetrically(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda _step: 1.0)
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)
    observed = {"train": []}

    def pipeline_pass(self, windows, *, loss_divisor):
        observed["train"].append(list(windows))
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([-2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    def validate(self, windows, teacher_root):
        observed["composed_gradient"] = parameter.grad.detach().clone()
        return {"kld_mean": 0.12, "windows": list(windows)}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(validate, engine)

    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition="pcgrad_equal_norm",
    )

    expected = torch.tensor([0.1767767, 0.4267767])
    assert observed["train"] == [[56], [28]]
    assert torch.allclose(observed["composed_gradient"], expected)
    assert result["objective_composition"] == "pcgrad_equal_norm"
    assert result["step"]["constituent_gradient_dot"] == pytest.approx(-(2.0**-0.5))
    assert result["step"]["conflict_projected"] is True


def test_pcgrad_equal_norm_refuses_nonconflicting_constituents(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda _step: 1.0)
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)
    parameter_before = parameter.detach().clone()

    def pipeline_pass(self, windows, *, loss_divisor):
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([0.0, 4.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(lambda self, windows, teacher_root: {}, engine)

    with pytest.raises(ArtifactError, match="negative constituent gradient dot product"):
        engine.diagnostic_perturb_and_validate(
            [28], tmp_path, direction=1, train_windows=[56, 28],
            objective_composition="pcgrad_equal_norm",
        )
    assert torch.equal(parameter.detach(), parameter_before)
    assert engine.global_step == 40


def test_symmetric_always_project_equal_norm_projects_aligned_constituents(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda _step: 1.0)
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)
    observed = {"train": []}

    def pipeline_pass(self, windows, *, loss_divisor):
        observed["train"].append(list(windows))
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    def validate(self, windows, teacher_root):
        observed["composed_gradient"] = parameter.grad.detach().clone()
        return {"kld_mean": 0.12, "windows": list(windows)}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(validate, engine)
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition="symmetric_always_project_equal_norm",
    )

    assert observed["train"] == [[56], [28]]
    # _FakeDist models two identical ranks, so each local residual carries the
    # expected 1/sqrt(2) share of the globally normalized composition.
    assert torch.allclose(
        observed["composed_gradient"], torch.tensor([0.1767767, 0.0732233])
    )
    assert result["step"]["constituent_gradient_dot"] == pytest.approx(2.0**-0.5)
    assert result["step"]["conflict_projected"] is True


def test_symmetric_always_project_residual_equal_norm_renormalizes_residuals(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda _step: 1.0)
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)
    observed = {"train": []}

    def pipeline_pass(self, windows, *, loss_divisor):
        observed["train"].append(list(windows))
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    def validate(self, windows, teacher_root):
        observed["composed_gradient"] = parameter.grad.detach().clone()
        return {"kld_mean": 0.12, "windows": list(windows)}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(validate, engine)
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition="symmetric_always_project_residual_equal_norm",
    )

    assert observed["train"] == [[56], [28]]
    assert torch.allclose(
        observed["composed_gradient"], torch.tensor([0.25, 0.1035534])
    )
    assert result["step"]["constituent_gradient_dot"] == pytest.approx(2.0**-0.5)
    assert result["step"]["projected_residual_norms"] == pytest.approx(
        [2.0**-0.5, 2.0**-0.5]
    )
    assert result["step"]["conflict_projected"] is True


def test_ordered_second_project_residual_equal_norm_preserves_w56(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda _step: 1.0)
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)
    observed = {"train": []}

    def pipeline_pass(self, windows, *, loss_divisor):
        observed["train"].append(list(windows))
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    def validate(self, windows, teacher_root):
        observed["composed_gradient"] = parameter.grad.detach().clone()
        return {"kld_mean": 0.12, "windows": list(windows)}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(validate, engine)
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition="ordered_second_project_residual_equal_norm",
    )

    assert observed["train"] == [[56], [28]]
    assert torch.allclose(
        observed["composed_gradient"], torch.tensor([0.3535534, 0.3535534])
    )
    assert result["step"]["constituent_gradient_dot"] == pytest.approx(2.0**-0.5)
    assert result["step"]["projected_residual_norms"] == pytest.approx([2.0**-0.5])
    assert result["step"]["conflict_projected"] is True


def test_ordered_first_project_residual_equal_norm_preserves_w28(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda _step: 1.0)
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)
    observed = {"train": []}

    def pipeline_pass(self, windows, *, loss_divisor):
        observed["train"].append(list(windows))
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    def validate(self, windows, teacher_root):
        observed["composed_gradient"] = parameter.grad.detach().clone()
        return {"kld_mean": 0.12, "windows": list(windows)}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(validate, engine)
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition="ordered_first_project_residual_equal_norm",
    )

    assert observed["train"] == [[56], [28]]
    assert torch.allclose(
        observed["composed_gradient"], torch.tensor([0.5, 0.0]), atol=2.0e-8
    )
    assert result["step"]["constituent_gradient_dot"] == pytest.approx(2.0**-0.5)
    assert result["step"]["projected_residual_norms"] == pytest.approx([2.0**-0.5])
    assert result["step"]["conflict_projected"] is True


def test_ordered_first_project_equal_norm_preserves_natural_w56_residual(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda _step: 1.0)
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)
    observed = {"train": []}

    def pipeline_pass(self, windows, *, loss_divisor):
        observed["train"].append(list(windows))
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    def validate(self, windows, teacher_root):
        observed["composed_gradient"] = parameter.grad.detach().clone()
        return {"kld_mean": 0.12, "windows": list(windows)}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(validate, engine)
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition="ordered_first_project_equal_norm",
    )

    assert observed["train"] == [[56], [28]]
    assert torch.allclose(
        observed["composed_gradient"],
        torch.tensor([0.4267767, 0.0732233]),
        atol=2.0e-8,
    )
    assert result["step"]["constituent_gradient_dot"] == pytest.approx(2.0**-0.5)
    assert result["step"]["projected_residual_norms"] == pytest.approx([2.0**-0.5])
    assert result["step"]["conflict_projected"] is True


def test_ordered_second_project_equal_norm_preserves_natural_w28_residual(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda _step: 1.0)
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)
    observed = {"train": []}

    def pipeline_pass(self, windows, *, loss_divisor):
        observed["train"].append(list(windows))
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    def validate(self, windows, teacher_root):
        observed["composed_gradient"] = parameter.grad.detach().clone()
        return {"kld_mean": 0.12, "windows": list(windows)}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(validate, engine)
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition="ordered_second_project_equal_norm",
    )

    assert observed["train"] == [[56], [28]]
    assert torch.allclose(
        observed["composed_gradient"],
        torch.tensor([0.35355338, 0.25]),
        atol=2.0e-8,
    )
    assert result["step"]["constituent_gradient_dot"] == pytest.approx(2.0**-0.5)
    assert result["step"]["projected_residual_norms"] == pytest.approx([2.0**-0.5])
    assert result["step"]["conflict_projected"] is True


def test_ordered_second_project_original_mean_norm_restores_unprojected_mean_norm(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda _step: 1.0)
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)
    observed = {"train": []}

    def pipeline_pass(self, windows, *, loss_divisor):
        observed["train"].append(list(windows))
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    def validate(self, windows, teacher_root):
        observed["composed_gradient"] = parameter.grad.detach().clone()
        return {"kld_mean": 0.12, "windows": list(windows)}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(validate, engine)
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition="ordered_second_project_original_mean_norm",
    )

    assert observed["train"] == [[56], [28]]
    assert torch.allclose(
        observed["composed_gradient"],
        torch.tensor([0.5334021, 0.37717224]),
        atol=3.0e-8,
    )
    assert result["step"]["constituent_gradient_dot"] == pytest.approx(2.0**-0.5)
    assert result["step"]["projected_residual_norms"] == pytest.approx([2.0**-0.5])
    assert result["step"]["unprojected_composed_gradient_norm"] == pytest.approx(0.9238795325)
    assert result["step"]["projected_composed_gradient_norm"] == pytest.approx(0.6123724357)
    assert result["step"]["composed_gradient_rescale"] == pytest.approx(1.508688959)
    assert result["step"]["conflict_projected"] is True


def test_ordered_second_project_residual_equal_norm_original_mean_norm_combines_both_gates(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda _step: 1.0)
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)
    observed = {"train": []}

    def pipeline_pass(self, windows, *, loss_divisor):
        observed["train"].append(list(windows))
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    def validate(self, windows, teacher_root):
        observed["composed_gradient"] = parameter.grad.detach().clone()
        return {"kld_mean": 0.12, "windows": list(windows)}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(validate, engine)
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition=(
            "ordered_second_project_residual_equal_norm_original_mean_norm"
        ),
    )

    assert observed["train"] == [[56], [28]]
    assert torch.allclose(
        observed["composed_gradient"],
        torch.tensor([0.46193975, 0.46193975]),
        atol=3.0e-8,
    )
    assert result["step"]["constituent_gradient_dot"] == pytest.approx(2.0**-0.5)
    assert result["step"]["projected_residual_norms"] == pytest.approx([2.0**-0.5])
    assert result["step"]["projected_residual_rescale"] == pytest.approx(2.0**0.5)
    assert result["step"]["unprojected_composed_gradient_norm"] == pytest.approx(0.9238795325)
    assert result["step"]["projected_composed_gradient_norm"] == pytest.approx(2.0**-0.5)
    assert result["step"]["composed_gradient_rescale"] == pytest.approx(1.306562965)
    assert result["step"]["conflict_projected"] is True


def test_ordered_second_project_residual_equal_norm_residual_only_original_mean_norm(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda _step: 1.0)
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)
    observed = {"train": []}

    def pipeline_pass(self, windows, *, loss_divisor):
        observed["train"].append(list(windows))
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    def validate(self, windows, teacher_root):
        observed["composed_gradient"] = parameter.grad.detach().clone()
        return {"kld_mean": 0.12, "windows": list(windows)}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(validate, engine)
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition=(
            "ordered_second_project_residual_equal_norm_residual_only_original_mean_norm"
        ),
    )

    assert observed["train"] == [[56], [28]]
    assert torch.allclose(
        observed["composed_gradient"],
        torch.tensor([0.35355338, 0.54934102]),
        atol=3.0e-8,
    )
    assert result["step"]["constituent_gradient_dot"] == pytest.approx(2.0**-0.5)
    assert result["step"]["projected_residual_norms"] == pytest.approx([2.0**-0.5])
    assert result["step"]["projected_residual_rescale"] == pytest.approx(2.0**0.5)
    assert result["step"]["unprojected_composed_gradient_norm"] == pytest.approx(0.9238795325)
    assert result["step"]["projected_composed_gradient_norm"] == pytest.approx(2.0**-0.5)
    assert result["step"]["composed_gradient_rescale"] is None
    assert result["step"]["residual_only_original_mean_rescale"] == pytest.approx(1.553773974)
    assert result["step"]["conflict_projected"] is True


def test_ordered_second_project_residual_equal_norm_first_only_original_mean_norm(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda _step: 1.0)
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)
    observed = {"train": []}

    def pipeline_pass(self, windows, *, loss_divisor):
        observed["train"].append(list(windows))
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    def validate(self, windows, teacher_root):
        observed["composed_gradient"] = parameter.grad.detach().clone()
        return {"kld_mean": 0.12, "windows": list(windows)}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(validate, engine)
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition=(
            "ordered_second_project_residual_equal_norm_first_only_original_mean_norm"
        ),
    )

    assert observed["train"] == [[56], [28]]
    assert torch.allclose(
        observed["composed_gradient"],
        torch.tensor([0.54934102, 0.35355338]),
        atol=3.0e-8,
    )
    assert result["step"]["constituent_gradient_dot"] == pytest.approx(2.0**-0.5)
    assert result["step"]["projected_residual_norms"] == pytest.approx([2.0**-0.5])
    assert result["step"]["projected_residual_rescale"] == pytest.approx(2.0**0.5)
    assert result["step"]["unprojected_composed_gradient_norm"] == pytest.approx(0.9238795325)
    assert result["step"]["projected_composed_gradient_norm"] == pytest.approx(2.0**-0.5)
    assert result["step"]["composed_gradient_rescale"] is None
    assert result["step"]["first_only_original_mean_rescale"] == pytest.approx(1.553773974)
    assert result["step"]["residual_only_original_mean_rescale"] is None
    assert result["step"]["conflict_projected"] is True


@pytest.mark.parametrize(
    (
        "objective_composition",
        "expected_gradient",
        "expected_residual_rescale",
        "expected_projected_norm",
        "expected_reciprocal_rescale",
        "expected_reciprocal_inverse",
    ),
    [
        (
            "ordered_second_project_residual_equal_norm_reciprocal_original_mean_norm",
            [0.62155630, 0.20110809],
            2.0**0.5,
            2.0**-0.5,
            1.7580266923,
            0.5688195773,
        ),
        (
            "ordered_second_project_residual_equal_norm_reciprocal_residual_original_mean_norm",
            [0.20110809, 0.62155630],
            2.0**0.5,
            2.0**-0.5,
            1.7580266923,
            0.5688195773,
        ),
        (
            "ordered_second_project_residual_reciprocal_original_mean_norm",
            [0.13844349, 0.63844347],
            None,
            (3.0 / 8.0) ** 0.5,
            2.5537739740,
            0.3915773323,
        ),
        (
            "ordered_second_project_residual_reciprocal_first_original_mean_norm",
            [0.63844347, 0.13844349],
            None,
            (3.0 / 8.0) ** 0.5,
            1.8057908946,
            0.5537739740,
        ),
        (
            "ordered_first_project_residual_reciprocal_second_original_mean_norm",
            [0.54934206, 0.35355339],
            None,
            (3.0 / 8.0) ** 0.5,
            1.8057908946,
            0.5537739740,
        ),
        (
            "ordered_first_project_residual_reciprocal_first_original_mean_norm",
            [0.54934206, -0.35355339],
            None,
            (3.0 / 8.0) ** 0.5,
            2.5537739740,
            0.3915773323,
        ),
    ],
)
def test_ordered_second_project_residual_equal_norm_reciprocal_original_mean_norm(
    tmp_path: Path,
    monkeypatch,
    objective_composition: str,
    expected_gradient: list[float],
    expected_residual_rescale: float | None,
    expected_projected_norm: float,
    expected_reciprocal_rescale: float,
    expected_reciprocal_inverse: float,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda _step: 1.0)
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)
    observed = {"train": []}

    def pipeline_pass(self, windows, *, loss_divisor):
        observed["train"].append(list(windows))
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    def validate(self, windows, teacher_root):
        observed["composed_gradient"] = parameter.grad.detach().clone()
        return {"kld_mean": 0.12, "windows": list(windows)}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(validate, engine)
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition=objective_composition,
    )

    assert observed["train"] == [[56], [28]]
    assert torch.allclose(
        observed["composed_gradient"],
        torch.tensor(expected_gradient),
        atol=3.0e-8,
    )
    step = result["step"]
    assert step["constituent_gradient_dot"] == pytest.approx(2.0**-0.5)
    assert step["projected_residual_norms"] == pytest.approx([2.0**-0.5])
    if expected_residual_rescale is None:
        assert step["projected_residual_rescale"] is None
    else:
        assert step["projected_residual_rescale"] == pytest.approx(
            expected_residual_rescale
        )
    assert step["unprojected_composed_gradient_norm"] == pytest.approx(0.9238795325)
    assert step["projected_composed_gradient_norm"] == pytest.approx(
        expected_projected_norm
    )
    assert step["reciprocal_original_mean_rescale"] == pytest.approx(
        expected_reciprocal_rescale
    )
    assert step["reciprocal_original_mean_inverse_rescale"] == pytest.approx(
        expected_reciprocal_inverse
    )
    assert step["reciprocal_original_mean_root_count"] == 1
    assert step["reciprocal_original_mean_norm_closed"] is True
    assert step["composed_gradient_rescale"] is None
    assert step["first_only_original_mean_rescale"] is None
    assert step["residual_only_original_mean_rescale"] is None
    assert step["conflict_projected"] is True


def test_cli_accepts_natural_residual_reciprocal_original_mean_objective() -> None:
    from repair_api.cli import build_parser

    args = build_parser().parse_args([
        "diagnostic-perturb-validate",
        "--artifact-root", "/tmp/artifact",
        "--start-checkpoint", "SCHEDULE_E186B108124B_UPDATE_040",
        "--config", "/tmp/config.json",
        "--objective-composition",
        "ordered_second_project_residual_reciprocal_original_mean_norm",
        "--receipt", "/tmp/receipt.json",
    ])
    assert args.objective_composition == (
        "ordered_second_project_residual_reciprocal_original_mean_norm"
    )


def test_cli_accepts_natural_residual_reciprocal_first_original_mean_objective() -> None:
    from repair_api.cli import build_parser

    args = build_parser().parse_args([
        "diagnostic-perturb-validate",
        "--artifact-root", "/tmp/artifact",
        "--start-checkpoint", "SCHEDULE_E186B108124B_UPDATE_040",
        "--config", "/tmp/config.json",
        "--objective-composition",
        "ordered_second_project_residual_reciprocal_first_original_mean_norm",
        "--receipt", "/tmp/receipt.json",
    ])
    assert args.objective_composition == (
        "ordered_second_project_residual_reciprocal_first_original_mean_norm"
    )


def test_cli_accepts_ordered_first_natural_residual_reciprocal_second_objective() -> None:
    from repair_api.cli import build_parser

    args = build_parser().parse_args([
        "diagnostic-perturb-validate",
        "--artifact-root", "/tmp/artifact",
        "--start-checkpoint", "SCHEDULE_E186B108124B_UPDATE_040",
        "--config", "/tmp/config.json",
        "--objective-composition",
        "ordered_first_project_residual_reciprocal_second_original_mean_norm",
        "--receipt", "/tmp/receipt.json",
    ])
    assert args.objective_composition == (
        "ordered_first_project_residual_reciprocal_second_original_mean_norm"
    )


def test_cli_accepts_ordered_first_natural_residual_reciprocal_first_objective() -> None:
    from repair_api.cli import build_parser

    args = build_parser().parse_args([
        "diagnostic-perturb-validate",
        "--artifact-root", "/tmp/artifact",
        "--start-checkpoint", "SCHEDULE_E186B108124B_UPDATE_040",
        "--config", "/tmp/config.json",
        "--objective-composition",
        "ordered_first_project_residual_reciprocal_first_original_mean_norm",
        "--receipt", "/tmp/receipt.json",
    ])
    assert args.objective_composition == (
        "ordered_first_project_residual_reciprocal_first_original_mean_norm"
    )


def test_symmetric_natural_residual_reciprocal_original_mean_norm(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda _step: 1.0)
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)
    observed = {"train": []}

    def pipeline_pass(self, windows, *, loss_divisor):
        observed["train"].append(list(windows))
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    def validate(self, windows, teacher_root):
        observed["composed_gradient"] = parameter.grad.detach().clone()
        return {"kld_mean": 0.12, "windows": list(windows)}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(validate, engine)
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition=(
            "symmetric_always_project_residual_reciprocal_original_mean_norm"
        ),
    )

    assert observed["train"] == [[56], [28]]
    assert torch.allclose(
        observed["composed_gradient"],
        torch.tensor([0.50371992, -0.41598431]),
        atol=3.0e-8,
    )
    step = result["step"]
    assert step["constituent_gradient_dot"] == pytest.approx(2.0**-0.5)
    assert step["projected_residual_norms"] == pytest.approx(
        [2.0**-0.5, 2.0**-0.5]
    )
    assert step["projected_residual_rescale"] is None
    assert step["unprojected_composed_gradient_norm"] == pytest.approx(0.9238795325)
    assert step["projected_composed_gradient_norm"] == pytest.approx(0.2705980501)
    assert step["reciprocal_original_mean_rescale"] == pytest.approx(2.8494701423)
    assert step["reciprocal_original_mean_inverse_rescale"] == pytest.approx(0.3509424384)
    assert step["reciprocal_original_mean_root_count"] == 1
    assert step["reciprocal_original_mean_norm_closed"] is True
    assert step["composed_gradient_rescale"] is None
    assert step["conflict_projected"] is True


def test_symmetric_natural_residual_reciprocal_second_original_mean_norm(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda _step: 1.0)
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)
    observed = {"train": []}

    def pipeline_pass(self, windows, *, loss_divisor):
        observed["train"].append(list(windows))
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    def validate(self, windows, teacher_root):
        observed["composed_gradient"] = parameter.grad.detach().clone()
        return {"kld_mean": 0.12, "windows": list(windows)}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(validate, engine)
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition=(
            "symmetric_always_project_residual_reciprocal_second_original_mean_norm"
        ),
    )

    assert observed["train"] == [[56], [28]]
    assert torch.allclose(
        observed["composed_gradient"],
        torch.tensor([0.06203844, 0.65032911]),
        atol=3.0e-8,
    )
    step = result["step"]
    assert step["constituent_gradient_dot"] == pytest.approx(2.0**-0.5)
    assert step["projected_residual_norms"] == pytest.approx(
        [2.0**-0.5, 2.0**-0.5]
    )
    assert step["projected_residual_rescale"] is None
    assert step["unprojected_composed_gradient_norm"] == pytest.approx(0.9238795325)
    assert step["projected_composed_gradient_norm"] == pytest.approx(0.2705980501)
    assert step["reciprocal_original_mean_rescale"] == pytest.approx(2.8494701423)
    assert step["reciprocal_original_mean_inverse_rescale"] == pytest.approx(0.3509424384)
    assert step["reciprocal_original_mean_root_count"] == 1
    assert step["reciprocal_original_mean_norm_closed"] is True
    assert step["reciprocal_original_mean_rescale_assignment"] == "second"
    assert step["composed_gradient_rescale"] is None
    assert step["conflict_projected"] is True


def test_symmetric_natural_residual_common_original_mean_norm(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda _step: 1.0)
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)
    observed = {"train": []}

    def pipeline_pass(self, windows, *, loss_divisor):
        observed["train"].append(list(windows))
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    def validate(self, windows, teacher_root):
        observed["composed_gradient"] = parameter.grad.detach().clone()
        return {"kld_mean": 0.12, "windows": list(windows)}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(validate, engine)
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition="symmetric_always_project_residual_original_mean_norm",
    )

    assert observed["train"] == [[56], [28]]
    assert torch.allclose(
        observed["composed_gradient"],
        torch.tensor([0.60355341, 0.25000000]),
        atol=3.0e-8,
    )
    step = result["step"]
    assert step["constituent_gradient_dot"] == pytest.approx(2.0**-0.5)
    assert step["projected_residual_norms"] == pytest.approx(
        [2.0**-0.5, 2.0**-0.5]
    )
    assert step["projected_residual_rescale"] is None
    assert step["unprojected_composed_gradient_norm"] == pytest.approx(0.9238795325)
    assert step["projected_composed_gradient_norm"] == pytest.approx(0.2705980501)
    assert step["symmetric_original_mean_rescale"] == pytest.approx(3.4142135624)
    assert step["symmetric_original_mean_applied_scales"] == pytest.approx(
        [3.4142135624, 3.4142135624]
    )
    assert step["symmetric_original_mean_norm_closed"] is True
    assert step["composed_gradient_rescale"] is None
    assert step["reciprocal_original_mean_rescale"] is None
    assert step["conflict_projected"] is True


def test_cli_accepts_symmetric_natural_residual_common_original_mean_objective() -> None:
    from repair_api.cli import build_parser

    args = build_parser().parse_args([
        "diagnostic-perturb-validate",
        "--artifact-root", "/tmp/artifact",
        "--start-checkpoint", "SCHEDULE_E186B108124B_UPDATE_040",
        "--config", "/tmp/config.json",
        "--objective-composition",
        "symmetric_always_project_residual_original_mean_norm",
        "--receipt", "/tmp/receipt.json",
    ])
    assert args.objective_composition == (
        "symmetric_always_project_residual_original_mean_norm"
    )


def test_symmetric_natural_residual_second_only_original_mean_norm(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda _step: 1.0)
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)
    observed = {}

    def pipeline_pass(self, windows, *, loss_divisor):
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    def validate(self, windows, teacher_root):
        observed["composed_gradient"] = parameter.grad.detach().clone()
        return {"kld_mean": 0.12, "windows": list(windows)}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(validate, engine)
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition=(
            "symmetric_always_project_residual_second_only_original_mean_norm"
        ),
    )

    assert torch.allclose(
        observed["composed_gradient"],
        torch.tensor([0.17677670, 0.62890911]),
        atol=3.0e-8,
    )
    step = result["step"]
    assert step["projected_residual_norms"] == pytest.approx(
        [2.0**-0.5, 2.0**-0.5]
    )
    assert step["projected_residual_rescale"] is None
    assert step["unprojected_composed_gradient_norm"] == pytest.approx(0.9238795325)
    assert step["projected_composed_gradient_norm"] == pytest.approx(0.2705980501)
    assert step["symmetric_original_mean_rescale"] == pytest.approx(3.2227433060)
    assert step["symmetric_original_mean_applied_scales"] == pytest.approx(
        [1.0, 3.2227433060]
    )
    assert step["symmetric_original_mean_root_count"] == 1
    assert step["symmetric_original_mean_norm_closed"] is True
    assert step["conflict_projected"] is True


def test_symmetric_natural_residual_first_only_original_mean_norm(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(
        engine.optimizer, lambda _step: 1.0
    )
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)
    observed = {}

    def pipeline_pass(self, windows, *, loss_divisor):
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    def validate(self, windows, teacher_root):
        observed["composed_gradient"] = parameter.grad.detach().clone()
        return {"kld_mean": 0.12, "windows": list(windows)}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(validate, engine)
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition=(
            "symmetric_always_project_residual_first_only_original_mean_norm"
        ),
    )

    step = result["step"]
    assert step["projected_residual_norms"] == pytest.approx(
        [2.0**-0.5, 2.0**-0.5]
    )
    assert step["projected_residual_rescale"] is None
    assert step["unprojected_composed_gradient_norm"] == pytest.approx(0.9238795325)
    assert step["projected_composed_gradient_norm"] == pytest.approx(0.2705980501)
    assert step["symmetric_original_mean_rescale"] == pytest.approx(3.2227433060)
    assert step["symmetric_original_mean_applied_scales"] == pytest.approx(
        [3.2227433060, 1.0]
    )
    assert step["symmetric_original_mean_root_count"] == 1
    assert step["symmetric_original_mean_norm_closed"] is True
    assert step["conflict_projected"] is True
    assert not torch.allclose(
        observed["composed_gradient"],
        torch.tensor([0.17677670, 0.62890911]),
        atol=3.0e-8,
    )


def test_symmetric_natural_residual_first_only_original_mean_projection(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(
        engine.optimizer, lambda _step: 1.0
    )
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)
    observed = {}

    def pipeline_pass(self, windows, *, loss_divisor):
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    def validate(self, windows, teacher_root):
        observed["composed_gradient"] = parameter.grad.detach().clone()
        return {"kld_mean": 0.12, "windows": list(windows)}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(validate, engine)
    objective = "symmetric_always_project_residual_first_only_original_mean_projection"
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition=objective,
    )

    assert torch.allclose(
        observed["composed_gradient"],
        torch.tensor([1.03033009, -0.78033009]),
        atol=3.0e-8,
    )
    step = result["step"]
    assert step["projected_residual_norms"] == pytest.approx(
        [2.0**-0.5, 2.0**-0.5]
    )
    assert step["projected_residual_rescale"] is None
    assert step["symmetric_original_mean_rescale"] == pytest.approx(5.8284271247)
    assert step["symmetric_original_mean_applied_scales"] == pytest.approx(
        [5.8284271247, 1.0]
    )
    assert step["symmetric_original_mean_root_count"] == 1
    assert step["symmetric_original_mean_projection_denominator"] == pytest.approx(0.25)
    assert step["symmetric_original_mean_projection_target"] == pytest.approx(
        0.8535533906
    )
    assert step["symmetric_original_mean_projection_achieved"] == pytest.approx(
        0.8535533906
    )
    assert step["symmetric_original_mean_projection_closed"] is True
    assert step["conflict_projected"] is True


def test_cli_accepts_symmetric_natural_residual_first_only_projection_objective() -> None:
    from repair_api.cli import build_parser

    objective = "symmetric_always_project_residual_first_only_original_mean_projection"
    args = build_parser().parse_args([
        "diagnostic-perturb-validate",
        "--artifact-root", "/tmp/artifact",
        "--start-checkpoint", "SCHEDULE_E186B108124B_UPDATE_040",
        "--config", "/tmp/config.json",
        "--objective-composition", objective,
        "--receipt", "/tmp/receipt.json",
    ])
    assert args.objective_composition == objective


def test_symmetric_natural_residual_second_only_original_mean_projection(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(
        engine.optimizer, lambda _step: 1.0
    )
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)
    observed = {}

    def pipeline_pass(self, windows, *, loss_divisor):
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    def validate(self, windows, teacher_root):
        observed["composed_gradient"] = parameter.grad.detach().clone()
        return {"kld_mean": 0.12, "windows": list(windows)}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(validate, engine)
    objective = "symmetric_always_project_residual_second_only_original_mean_projection"
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition=objective,
    )

    assert torch.allclose(
        observed["composed_gradient"],
        torch.tensor([0.17677670, 1.28033009]),
        atol=3.0e-8,
    )
    step = result["step"]
    assert step["projected_residual_norms"] == pytest.approx(
        [2.0**-0.5, 2.0**-0.5]
    )
    assert step["projected_residual_rescale"] is None
    assert step["symmetric_original_mean_rescale"] == pytest.approx(5.8284271247)
    assert step["symmetric_original_mean_applied_scales"] == pytest.approx(
        [1.0, 5.8284271247]
    )
    assert step["symmetric_original_mean_root_count"] == 1
    assert step["symmetric_original_mean_projection_denominator"] == pytest.approx(0.25)
    assert step["symmetric_original_mean_projection_target"] == pytest.approx(
        0.8535533906
    )
    assert step["symmetric_original_mean_projection_achieved"] == pytest.approx(
        0.8535533906
    )
    assert step["symmetric_original_mean_projection_closed"] is True
    assert step["conflict_projected"] is True


def test_symmetric_natural_residual_common_original_mean_projection(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(
        engine.optimizer, lambda _step: 1.0
    )
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)
    observed = {}

    def pipeline_pass(self, windows, *, loss_divisor):
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    def validate(self, windows, teacher_root):
        observed["composed_gradient"] = parameter.grad.detach().clone()
        return {"kld_mean": 0.12, "windows": list(windows)}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(validate, engine)
    objective = "symmetric_always_project_residual_common_original_mean_projection"
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition=objective,
    )

    assert torch.allclose(
        observed["composed_gradient"],
        torch.tensor([0.60355339, 0.25]),
        atol=3.0e-8,
    )
    step = result["step"]
    assert step["projected_residual_norms"] == pytest.approx(
        [2.0**-0.5, 2.0**-0.5]
    )
    assert step["projected_residual_rescale"] is None
    assert step["symmetric_original_mean_rescale"] == pytest.approx(3.4142135624)
    assert step["symmetric_original_mean_applied_scales"] == pytest.approx(
        [3.4142135624, 3.4142135624]
    )
    assert step["symmetric_original_mean_root_count"] == 1
    assert step["symmetric_original_mean_projection_denominator"] == pytest.approx(0.5)
    assert step["symmetric_original_mean_projection_target"] == pytest.approx(
        0.8535533906
    )
    assert step["symmetric_original_mean_projection_achieved"] == pytest.approx(
        0.8535533906
    )
    assert step["symmetric_original_mean_projection_closed"] is True
    assert step["conflict_projected"] is True


def test_cli_accepts_symmetric_natural_residual_common_projection_objective() -> None:
    from repair_api.cli import build_parser

    objective = "symmetric_always_project_residual_common_original_mean_projection"
    args = build_parser().parse_args([
        "diagnostic-perturb-validate",
        "--artifact-root", "/tmp/artifact",
        "--start-checkpoint", "SCHEDULE_E186B108124B_UPDATE_040",
        "--config", "/tmp/config.json",
        "--objective-composition", objective,
        "--receipt", "/tmp/receipt.json",
    ])
    assert args.objective_composition == objective


def test_symmetric_natural_residual_reciprocal_original_mean_projection(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(
        engine.optimizer, lambda _step: 1.0
    )
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)

    def pipeline_pass(self, windows, *, loss_divisor):
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(
        lambda self, windows, teacher_root: {
            "kld_mean": 0.12,
            "windows": list(windows),
        },
        engine,
    )
    objective = "symmetric_always_project_residual_reciprocal_original_mean_projection"
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition=objective,
    )

    step = result["step"]
    scale = step["symmetric_original_mean_rescale"]
    assert scale == pytest.approx(6.6786973270)
    assert step["symmetric_original_mean_applied_scales"] == pytest.approx(
        [scale, 1.0 / scale]
    )
    assert step["symmetric_original_mean_root_count"] == 1
    assert step["symmetric_original_mean_reciprocal_product_closed"] is True
    assert step["symmetric_original_mean_projection_coefficients"] == pytest.approx(
        [0.25, 0.25]
    )
    assert step["symmetric_original_mean_projection_target"] == pytest.approx(
        0.8535533906
    )
    assert step["symmetric_original_mean_projection_achieved"] == pytest.approx(
        0.8535533906
    )
    assert step["symmetric_original_mean_projection_closed"] is True


def test_cli_accepts_symmetric_natural_residual_reciprocal_projection_objective() -> None:
    from repair_api.cli import build_parser

    objective = "symmetric_always_project_residual_reciprocal_original_mean_projection"
    args = build_parser().parse_args([
        "diagnostic-perturb-validate",
        "--artifact-root", "/tmp/artifact",
        "--start-checkpoint", "SCHEDULE_E186B108124B_UPDATE_040",
        "--config", "/tmp/config.json",
        "--objective-composition", objective,
        "--receipt", "/tmp/receipt.json",
    ])
    assert args.objective_composition == objective


def test_symmetric_natural_residual_reciprocal_second_original_mean_projection(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(
        engine.optimizer, lambda _step: 1.0
    )
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)

    def pipeline_pass(self, windows, *, loss_divisor):
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(
        lambda self, windows, teacher_root: {
            "kld_mean": 0.12,
            "windows": list(windows),
        },
        engine,
    )
    objective = (
        "symmetric_always_project_residual_reciprocal_second_original_mean_projection"
    )
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition=objective,
    )

    step = result["step"]
    scale = step["symmetric_original_mean_rescale"]
    assert scale == pytest.approx(6.6786973270)
    assert step["symmetric_original_mean_applied_scales"] == pytest.approx(
        [1.0 / scale, scale]
    )
    assert step["symmetric_original_mean_root_count"] == 1
    assert step["symmetric_original_mean_reciprocal_product_closed"] is True
    assert step["symmetric_original_mean_projection_coefficients"] == pytest.approx(
        [0.25, 0.25]
    )
    assert step["symmetric_original_mean_projection_target"] == pytest.approx(
        0.8535533906
    )
    assert step["symmetric_original_mean_projection_achieved"] == pytest.approx(
        0.8535533906
    )
    assert step["symmetric_original_mean_projection_closed"] is True


def test_reciprocal_second_first_constituent_projection_admits_unique_subunit_root(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(
        engine.optimizer, lambda _step: 1.0
    )
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)

    def pipeline_pass(self, windows, *, loss_divisor):
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(
        lambda self, windows, teacher_root: {
            "kld_mean": 0.12,
            "windows": list(windows),
        },
        engine,
    )
    objective = (
        "symmetric_always_project_residual_reciprocal_second_"
        "first_constituent_projection"
    )
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition=objective,
    )

    step = result["step"]
    expected_root = 1.0 - 2.0**-0.5
    assert step["symmetric_original_mean_rescale"] == pytest.approx(expected_root)
    assert 0.0 < step["symmetric_original_mean_rescale"] < 1.0
    assert step["symmetric_original_mean_rescale"] == pytest.approx(
        1.0 - step["constituent_gradient_dot"], rel=1.0e-12, abs=1.0e-12
    )
    assert step["symmetric_original_mean_root_count"] == 1
    assert step["symmetric_original_mean_applied_scales"] == pytest.approx(
        [1.0 / expected_root, expected_root]
    )
    assert step["symmetric_original_mean_reciprocal_product_closed"] is True
    assert step["symmetric_original_mean_projection_target"] == pytest.approx(
        (1.0 + 2.0**-0.5) / 2.0
    )
    assert step["symmetric_original_mean_projection_achieved"] == pytest.approx(
        step["symmetric_original_mean_projection_target"]
    )
    assert step["symmetric_original_mean_projection_closed"] is True


def test_reciprocal_second_second_constituent_projection_admits_unique_superunit_root(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(
        engine.optimizer, lambda _step: 1.0
    )
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)

    def pipeline_pass(self, windows, *, loss_divisor):
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(
        lambda self, windows, teacher_root: {
            "kld_mean": 0.12,
            "windows": list(windows),
        },
        engine,
    )
    objective = (
        "symmetric_always_project_residual_reciprocal_second_"
        "second_constituent_projection"
    )
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition=objective,
    )

    step = result["step"]
    expected_root = 1.0 / (1.0 - 2.0**-0.5)
    assert step["symmetric_original_mean_rescale"] == pytest.approx(expected_root)
    assert step["symmetric_original_mean_rescale"] > 1.0
    assert step["symmetric_original_mean_rescale"] == pytest.approx(
        1.0 / (1.0 - step["constituent_gradient_dot"]),
        rel=1.0e-12,
        abs=1.0e-12,
    )
    assert step["symmetric_original_mean_root_count"] == 1
    assert step["symmetric_original_mean_applied_scales"] == pytest.approx(
        [1.0 / expected_root, expected_root]
    )
    assert step["symmetric_original_mean_reciprocal_product_closed"] is True
    assert step["symmetric_original_mean_projection_target"] == pytest.approx(
        (1.0 + 2.0**-0.5) / 2.0
    )
    assert step["symmetric_original_mean_projection_achieved"] == pytest.approx(
        step["symmetric_original_mean_projection_target"]
    )
    assert step["symmetric_original_mean_projection_closed"] is True


def test_reciprocal_first_second_constituent_projection_admits_unique_subunit_root(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(
        engine.optimizer, lambda _step: 1.0
    )
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)

    def pipeline_pass(self, windows, *, loss_divisor):
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(
        lambda self, windows, teacher_root: {
            "kld_mean": 0.12,
            "windows": list(windows),
        },
        engine,
    )
    objective = (
        "symmetric_always_project_residual_reciprocal_first_"
        "second_constituent_projection"
    )
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition=objective,
    )

    step = result["step"]
    expected_root = 1.0 - 2.0**-0.5
    assert step["symmetric_original_mean_rescale"] == pytest.approx(expected_root)
    assert 0.0 < step["symmetric_original_mean_rescale"] < 1.0
    assert step["symmetric_original_mean_rescale"] == pytest.approx(
        1.0 - step["constituent_gradient_dot"], rel=1.0e-12, abs=1.0e-12
    )
    assert step["symmetric_original_mean_root_count"] == 1
    assert step["symmetric_original_mean_applied_scales"] == pytest.approx(
        [expected_root, 1.0 / expected_root]
    )
    assert step["symmetric_original_mean_reciprocal_product_closed"] is True
    assert step["symmetric_original_mean_projection_achieved"] == pytest.approx(
        step["symmetric_original_mean_projection_target"]
    )
    assert step["symmetric_original_mean_projection_closed"] is True


def test_reciprocal_first_first_constituent_projection_admits_unique_superunit_root(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(
        engine.optimizer, lambda _step: 1.0
    )
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)

    def pipeline_pass(self, windows, *, loss_divisor):
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(
        lambda self, windows, teacher_root: {
            "kld_mean": 0.12,
            "windows": list(windows),
        },
        engine,
    )
    objective = (
        "symmetric_always_project_residual_reciprocal_first_"
        "first_constituent_projection"
    )
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition=objective,
    )

    step = result["step"]
    expected_root = 1.0 / (1.0 - 2.0**-0.5)
    assert step["symmetric_original_mean_rescale"] == pytest.approx(expected_root)
    assert step["symmetric_original_mean_rescale"] > 1.0
    assert step["symmetric_original_mean_rescale"] == pytest.approx(
        1.0 / (1.0 - step["constituent_gradient_dot"]),
        rel=1.0e-12,
        abs=1.0e-12,
    )
    assert step["symmetric_original_mean_root_count"] == 1
    assert step["symmetric_original_mean_applied_scales"] == pytest.approx(
        [expected_root, 1.0 / expected_root]
    )
    assert step["symmetric_original_mean_reciprocal_product_closed"] is True
    assert step["symmetric_original_mean_projection_achieved"] == pytest.approx(
        step["symmetric_original_mean_projection_target"]
    )
    assert step["symmetric_original_mean_projection_closed"] is True


def test_reciprocal_first_first_constituent_projected_mean_target_admits_identity_root(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(
        engine.optimizer, lambda _step: 1.0
    )
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)

    def pipeline_pass(self, windows, *, loss_divisor):
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(
        lambda self, windows, teacher_root: {
            "kld_mean": 0.12,
            "windows": list(windows),
        },
        engine,
    )
    objective = (
        "symmetric_always_project_residual_reciprocal_first_first_"
        "constituent_projected_mean_target"
    )
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition=objective,
    )

    step = result["step"]
    assert step["symmetric_original_mean_rescale"] == 1.0
    assert step["symmetric_original_mean_root_count"] == 1
    assert step["symmetric_original_mean_applied_scales"] == [1.0, 1.0]
    assert step["symmetric_original_mean_reciprocal_product_closed"] is True
    assert step["symmetric_original_mean_projection_target"] == pytest.approx(
        step["symmetric_original_mean_projection_achieved"]
    )
    assert step["symmetric_original_mean_projection_closed"] is True


def test_reciprocal_first_second_constituent_projected_mean_target_admits_identity_root(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(
        engine.optimizer, lambda _step: 1.0
    )
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)

    def pipeline_pass(self, windows, *, loss_divisor):
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(
        lambda self, windows, teacher_root: {
            "kld_mean": 0.12,
            "windows": list(windows),
        },
        engine,
    )
    objective = (
        "symmetric_always_project_residual_reciprocal_first_second_"
        "constituent_projected_mean_target"
    )
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition=objective,
    )

    step = result["step"]
    assert step["symmetric_original_mean_rescale"] == 1.0
    assert step["symmetric_original_mean_root_count"] == 1
    assert step["symmetric_original_mean_applied_scales"] == [1.0, 1.0]
    assert step["symmetric_original_mean_reciprocal_product_closed"] is True
    assert step["symmetric_original_mean_projection_target"] == pytest.approx(
        step["symmetric_original_mean_projection_achieved"]
    )
    assert step["symmetric_original_mean_projection_closed"] is True


def test_reciprocal_second_first_constituent_projected_mean_target_admits_identity_root(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(
        engine.optimizer, lambda _step: 1.0
    )
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)

    def pipeline_pass(self, windows, *, loss_divisor):
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(
        lambda self, windows, teacher_root: {
            "kld_mean": 0.12,
            "windows": list(windows),
        },
        engine,
    )
    objective = (
        "symmetric_always_project_residual_reciprocal_second_first_"
        "constituent_projected_mean_target"
    )
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition=objective,
    )

    step = result["step"]
    assert step["symmetric_original_mean_rescale"] == 1.0
    assert step["symmetric_original_mean_root_count"] == 1
    assert step["symmetric_original_mean_applied_scales"] == [1.0, 1.0]
    assert step["symmetric_original_mean_reciprocal_product_closed"] is True
    assert step["symmetric_original_mean_projection_target"] == pytest.approx(
        step["symmetric_original_mean_projection_achieved"]
    )
    assert step["symmetric_original_mean_projection_closed"] is True


def test_reciprocal_second_projected_mean_axis_projected_mean_target_admits_identity_root(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(
        engine.optimizer, lambda _step: 1.0
    )
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)

    def pipeline_pass(self, windows, *, loss_divisor):
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(
        lambda self, windows, teacher_root: {
            "kld_mean": 0.12,
            "windows": list(windows),
        },
        engine,
    )
    objective = (
        "symmetric_always_project_residual_reciprocal_second_"
        "projected_mean_axis_projected_mean_target"
    )
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition=objective,
    )

    step = result["step"]
    assert step["symmetric_original_mean_rescale"] == 1.0
    assert step["symmetric_original_mean_root_count"] == 1
    assert step["symmetric_original_mean_applied_scales"] == [1.0, 1.0]
    assert step["symmetric_original_mean_reciprocal_product_closed"] is True
    assert step["symmetric_original_mean_projection_axis_norm"] == pytest.approx(1.0)
    assert step["symmetric_original_mean_projection_axis_closed"] is True
    assert step["symmetric_original_mean_projection_target"] == pytest.approx(
        step["symmetric_original_mean_projection_achieved"]
    )
    assert step["symmetric_original_mean_projection_closed"] is True


def test_reciprocal_second_renormalized_projected_mean_axis_admits_identity_root(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(
        engine.optimizer, lambda _step: 1.0
    )
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)

    def pipeline_pass(self, windows, *, loss_divisor):
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(
        lambda self, windows, teacher_root: {
            "kld_mean": 0.12,
            "windows": list(windows),
        },
        engine,
    )
    objective = (
        "symmetric_always_project_residual_reciprocal_second_"
        "projected_mean_axis_renormalized_projected_mean_target"
    )
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition=objective,
    )

    step = result["step"]
    assert step["symmetric_original_mean_rescale"] == 1.0
    assert step["symmetric_original_mean_root_count"] == 1
    assert step["symmetric_original_mean_applied_scales"] == [1.0, 1.0]
    assert step["symmetric_original_mean_reciprocal_product_closed"] is True
    assert step["symmetric_original_mean_projection_axis_second_pass_renormalized"] is True
    assert step["symmetric_original_mean_projection_axis_norm"] == pytest.approx(1.0)
    assert step["symmetric_original_mean_projection_axis_closed"] is True
    assert step["symmetric_original_mean_projection_target"] == pytest.approx(
        step["symmetric_original_mean_projection_achieved"]
    )
    assert step["symmetric_original_mean_projection_closed"] is True


def test_reciprocal_second_second_constituent_projected_mean_target_admits_identity_root(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(
        engine.optimizer, lambda _step: 1.0
    )
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)

    def pipeline_pass(self, windows, *, loss_divisor):
        parameter.grad = {
            56: torch.tensor([3.0, 0.0]),
            28: torch.tensor([2.0, 2.0]),
        }[windows[0]].clone()
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(
        lambda self, windows, teacher_root: {
            "kld_mean": 0.12,
            "windows": list(windows),
        },
        engine,
    )
    objective = (
        "symmetric_always_project_residual_reciprocal_second_second_"
        "constituent_projected_mean_target"
    )
    result = engine.diagnostic_perturb_and_validate(
        [28], tmp_path, direction=1, train_windows=[56, 28],
        objective_composition=objective,
    )

    step = result["step"]
    assert step["symmetric_original_mean_rescale"] == 1.0
    assert step["symmetric_original_mean_root_count"] == 1
    assert step["symmetric_original_mean_applied_scales"] == [1.0, 1.0]
    assert step["symmetric_original_mean_reciprocal_product_closed"] is True
    assert step["symmetric_original_mean_projection_target"] == pytest.approx(
        step["symmetric_original_mean_projection_achieved"]
    )
    assert step["symmetric_original_mean_projection_closed"] is True


def test_cli_accepts_symmetric_natural_residual_reciprocal_second_projection_objective() -> None:
    from repair_api.cli import build_parser

    objective = (
        "symmetric_always_project_residual_reciprocal_second_original_mean_projection"
    )
    args = build_parser().parse_args([
        "diagnostic-perturb-validate",
        "--artifact-root", "/tmp/artifact",
        "--start-checkpoint", "SCHEDULE_E186B108124B_UPDATE_040",
        "--config", "/tmp/config.json",
        "--objective-composition", objective,
        "--receipt", "/tmp/receipt.json",
    ])
    assert args.objective_composition == objective


def test_cli_accepts_reciprocal_second_first_constituent_projection_objective() -> None:
    from repair_api.cli import build_parser

    objective = (
        "symmetric_always_project_residual_reciprocal_second_"
        "first_constituent_projection"
    )
    args = build_parser().parse_args([
        "diagnostic-perturb-validate",
        "--artifact-root", "/tmp/artifact",
        "--start-checkpoint", "SCHEDULE_E186B108124B_UPDATE_040",
        "--config", "/tmp/config.json",
        "--objective-composition", objective,
        "--receipt", "/tmp/receipt.json",
    ])
    assert args.objective_composition == objective


def test_cli_accepts_reciprocal_second_second_constituent_projection_objective() -> None:
    from repair_api.cli import build_parser

    objective = (
        "symmetric_always_project_residual_reciprocal_second_"
        "second_constituent_projection"
    )
    args = build_parser().parse_args([
        "diagnostic-perturb-validate",
        "--artifact-root", "/tmp/artifact",
        "--start-checkpoint", "SCHEDULE_E186B108124B_UPDATE_040",
        "--config", "/tmp/config.json",
        "--objective-composition", objective,
        "--receipt", "/tmp/receipt.json",
    ])
    assert args.objective_composition == objective


def test_cli_accepts_reciprocal_first_second_constituent_projection_objective() -> None:
    from repair_api.cli import build_parser

    objective = (
        "symmetric_always_project_residual_reciprocal_first_"
        "second_constituent_projection"
    )
    args = build_parser().parse_args([
        "diagnostic-perturb-validate",
        "--artifact-root", "/tmp/artifact",
        "--start-checkpoint", "SCHEDULE_E186B108124B_UPDATE_040",
        "--config", "/tmp/config.json",
        "--objective-composition", objective,
        "--receipt", "/tmp/receipt.json",
    ])
    assert args.objective_composition == objective


def test_cli_accepts_reciprocal_first_first_constituent_projection_objective() -> None:
    from repair_api.cli import build_parser

    objective = (
        "symmetric_always_project_residual_reciprocal_first_"
        "first_constituent_projection"
    )
    args = build_parser().parse_args([
        "diagnostic-perturb-validate",
        "--artifact-root", "/tmp/artifact",
        "--start-checkpoint", "SCHEDULE_E186B108124B_UPDATE_040",
        "--config", "/tmp/config.json",
        "--objective-composition", objective,
        "--receipt", "/tmp/receipt.json",
    ])
    assert args.objective_composition == objective


def test_cli_accepts_reciprocal_first_first_constituent_projected_mean_target() -> None:
    from repair_api.cli import build_parser

    objective = (
        "symmetric_always_project_residual_reciprocal_first_first_"
        "constituent_projected_mean_target"
    )
    args = build_parser().parse_args([
        "diagnostic-perturb-validate",
        "--artifact-root", "/tmp/artifact",
        "--start-checkpoint", "SCHEDULE_E186B108124B_UPDATE_040",
        "--config", "/tmp/config.json",
        "--objective-composition", objective,
        "--receipt", "/tmp/receipt.json",
    ])
    assert args.objective_composition == objective


def test_cli_accepts_reciprocal_first_second_constituent_projected_mean_target() -> None:
    from repair_api.cli import build_parser

    objective = (
        "symmetric_always_project_residual_reciprocal_first_second_"
        "constituent_projected_mean_target"
    )
    args = build_parser().parse_args([
        "diagnostic-perturb-validate",
        "--artifact-root", "/tmp/artifact",
        "--start-checkpoint", "SCHEDULE_E186B108124B_UPDATE_040",
        "--config", "/tmp/config.json",
        "--objective-composition", objective,
        "--receipt", "/tmp/receipt.json",
    ])
    assert args.objective_composition == objective


def test_cli_accepts_reciprocal_second_first_constituent_projected_mean_target() -> None:
    from repair_api.cli import build_parser

    objective = (
        "symmetric_always_project_residual_reciprocal_second_first_"
        "constituent_projected_mean_target"
    )
    args = build_parser().parse_args([
        "diagnostic-perturb-validate",
        "--artifact-root", "/tmp/artifact",
        "--start-checkpoint", "SCHEDULE_E186B108124B_UPDATE_040",
        "--config", "/tmp/config.json",
        "--objective-composition", objective,
        "--receipt", "/tmp/receipt.json",
    ])
    assert args.objective_composition == objective


def test_cli_accepts_reciprocal_second_projected_mean_axis_projected_mean_target() -> None:
    from repair_api.cli import build_parser

    objective = (
        "symmetric_always_project_residual_reciprocal_second_"
        "projected_mean_axis_projected_mean_target"
    )
    args = build_parser().parse_args([
        "diagnostic-perturb-validate",
        "--artifact-root", "/tmp/artifact",
        "--start-checkpoint", "SCHEDULE_E186B108124B_UPDATE_040",
        "--config", "/tmp/config.json",
        "--objective-composition", objective,
        "--receipt", "/tmp/receipt.json",
    ])
    assert args.objective_composition == objective


def test_cli_accepts_reciprocal_second_renormalized_projected_mean_axis_target() -> None:
    from repair_api.cli import build_parser

    objective = (
        "symmetric_always_project_residual_reciprocal_second_"
        "projected_mean_axis_renormalized_projected_mean_target"
    )
    args = build_parser().parse_args([
        "diagnostic-perturb-validate",
        "--artifact-root", "/tmp/artifact",
        "--start-checkpoint", "SCHEDULE_E186B108124B_UPDATE_040",
        "--config", "/tmp/config.json",
        "--objective-composition", objective,
        "--receipt", "/tmp/receipt.json",
    ])
    assert args.objective_composition == objective


def test_cli_accepts_reciprocal_second_second_constituent_projected_mean_target() -> None:
    from repair_api.cli import build_parser

    objective = (
        "symmetric_always_project_residual_reciprocal_second_second_"
        "constituent_projected_mean_target"
    )
    args = build_parser().parse_args([
        "diagnostic-perturb-validate",
        "--artifact-root", "/tmp/artifact",
        "--start-checkpoint", "SCHEDULE_E186B108124B_UPDATE_040",
        "--config", "/tmp/config.json",
        "--objective-composition", objective,
        "--receipt", "/tmp/receipt.json",
    ])
    assert args.objective_composition == objective


def test_cli_accepts_symmetric_natural_residual_second_only_projection_objective() -> None:
    from repair_api.cli import build_parser

    objective = "symmetric_always_project_residual_second_only_original_mean_projection"
    args = build_parser().parse_args([
        "diagnostic-perturb-validate",
        "--artifact-root", "/tmp/artifact",
        "--start-checkpoint", "SCHEDULE_E186B108124B_UPDATE_040",
        "--config", "/tmp/config.json",
        "--objective-composition", objective,
        "--receipt", "/tmp/receipt.json",
    ])
    assert args.objective_composition == objective


def test_cli_accepts_symmetric_natural_residual_first_only_objective() -> None:
    from repair_api.cli import build_parser

    objective = "symmetric_always_project_residual_first_only_original_mean_norm"
    args = build_parser().parse_args([
        "diagnostic-perturb-validate",
        "--artifact-root", "/tmp/artifact",
        "--start-checkpoint", "SCHEDULE_E186B108124B_UPDATE_040",
        "--config", "/tmp/config.json",
        "--objective-composition", objective,
        "--receipt", "/tmp/receipt.json",
    ])
    assert args.objective_composition == objective


def test_cli_accepts_symmetric_natural_residual_second_only_objective() -> None:
    from repair_api.cli import build_parser

    objective = "symmetric_always_project_residual_second_only_original_mean_norm"
    args = build_parser().parse_args([
        "diagnostic-perturb-validate",
        "--artifact-root", "/tmp/artifact",
        "--start-checkpoint", "SCHEDULE_E186B108124B_UPDATE_040",
        "--config", "/tmp/config.json",
        "--objective-composition", objective,
        "--receipt", "/tmp/receipt.json",
    ])
    assert args.objective_composition == objective


def test_cli_accepts_symmetric_natural_residual_reciprocal_objective() -> None:
    from repair_api.cli import build_parser

    args = build_parser().parse_args([
        "diagnostic-perturb-validate",
        "--artifact-root", "/tmp/artifact",
        "--start-checkpoint", "SCHEDULE_E186B108124B_UPDATE_040",
        "--config", "/tmp/config.json",
        "--objective-composition",
        "symmetric_always_project_residual_reciprocal_original_mean_norm",
        "--receipt", "/tmp/receipt.json",
    ])
    assert args.objective_composition == (
        "symmetric_always_project_residual_reciprocal_original_mean_norm"
    )


def test_cli_accepts_symmetric_natural_residual_reciprocal_second_objective() -> None:
    from repair_api.cli import build_parser

    args = build_parser().parse_args([
        "diagnostic-perturb-validate",
        "--artifact-root", "/tmp/artifact",
        "--start-checkpoint", "SCHEDULE_E186B108124B_UPDATE_040",
        "--config", "/tmp/config.json",
        "--objective-composition",
        "symmetric_always_project_residual_reciprocal_second_original_mean_norm",
        "--receipt", "/tmp/receipt.json",
    ])
    assert args.objective_composition == (
        "symmetric_always_project_residual_reciprocal_second_original_mean_norm"
    )


def test_symmetric_always_project_equal_norm_refuses_zero_residual(
    tmp_path: Path, monkeypatch,
) -> None:
    import repair_api.modern_green_resident as engine_module

    monkeypatch.setattr(engine_module, "_cuda_sync", lambda _torch: None)
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    engine = object.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.dist = _FakeDist()
    engine.rank = 0
    engine.global_step = 40
    engine.published_pre_recipe = True
    engine.published_pre_controlled_windows = True
    engine.controlled_windows = {40: [56, 28]}
    engine.controlled_arm = False
    engine.controlled_arm_id = None
    engine.pipeline_microbatch = 2
    engine.config = {
        "recipe_id": PUBLISHED_PRE_RECIPE_ID,
        "published_pre_checkpoint_sha256": PUBLISHED_PRE_SHA256,
        "checkpoint_sha256": "c" * 64,
        "lr_scale": 0.375,
        "controlled_windows_per_update": 2,
    }
    engine.student = SimpleNamespace(device="cpu")
    engine.optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 1.0, "group_name": "luts"}]
    )
    engine.scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda _step: 1.0)
    engine._local_params = MethodType(lambda self: [("probe", parameter)], engine)
    parameter_before = parameter.detach().clone()

    def pipeline_pass(self, windows, *, loss_divisor):
        parameter.grad = torch.tensor([float(windows[0]), 0.0])
        return float(windows[0]), {"forward_seconds": 0.1, "backward_seconds": 0.2}

    engine._pipeline_pass = MethodType(pipeline_pass, engine)
    engine.validate = MethodType(lambda self, windows, teacher_root: {}, engine)
    with pytest.raises(ArtifactError, match="nonzero finite projected residuals"):
        engine.diagnostic_perturb_and_validate(
            [28], tmp_path, direction=1, train_windows=[56, 28],
            objective_composition="symmetric_always_project_equal_norm",
        )
    assert torch.equal(parameter.detach(), parameter_before)
    assert engine.global_step == 40


def test_public_diagnostic_forwards_only_controlled_objective_compositions() -> None:
    api = object.__new__(ResidentRepairAPI)
    api.artifact = SimpleNamespace(
        checkpoint_key=lambda _value: "UPDATE_040",
        manifest={"checkpoints": {"UPDATE_040": {"update": 40}}},
    )
    forwarded = []

    def continue_two_spark_real(self, start, milestones, *, config, receipt_path):
        forwarded.append(
            {
                "start": start,
                "milestones": list(milestones),
                "config": dict(config),
                "receipt_path": receipt_path,
            }
        )
        return {"status": "PASS"}

    api.continue_two_spark_real = MethodType(continue_two_spark_real, api)

    for train_windows in ([56, 56], [56, 28], [28, 56], [28, 28]):
        assert api.diagnostic_perturb_and_validate(
            40,
            config={"seed": 1701},
            train_windows=train_windows,
            windows=[28],
            direction=1,
            receipt_path="diagnostic.json",
        ) == {"status": "PASS"}

    assert [row["config"]["diagnostic_train_windows"] for row in forwarded] == [
        [56, 56],
        [56, 28],
        [28, 56],
        [28, 28],
    ]
    assert all(row["config"]["validation_windows"] == [28] for row in forwarded)
    assert api.diagnostic_perturb_and_validate(
        40,
        config={"seed": 1701},
        train_windows=[56, 28],
        windows=[28],
        direction=1,
        objective_composition="equal_norm",
        receipt_path="diagnostic.json",
    ) == {"status": "PASS"}
    assert forwarded[-1]["config"]["diagnostic_objective_composition"] == "equal_norm"
    assert api.diagnostic_perturb_and_validate(
        40,
        config={"seed": 1701},
        train_windows=[56, 28],
        windows=[28],
        direction=1,
        objective_composition="pcgrad_equal_norm",
        receipt_path="diagnostic.json",
    ) == {"status": "PASS"}
    assert forwarded[-1]["config"]["diagnostic_objective_composition"] == "pcgrad_equal_norm"
    assert api.diagnostic_perturb_and_validate(
        40,
        config={"seed": 1701},
        train_windows=[56, 28],
        windows=[28],
        direction=1,
        objective_composition="symmetric_always_project_equal_norm",
        receipt_path="diagnostic.json",
    ) == {"status": "PASS"}
    assert forwarded[-1]["config"]["diagnostic_objective_composition"] == (
        "symmetric_always_project_equal_norm"
    )
    assert api.diagnostic_perturb_and_validate(
        40,
        config={"seed": 1701},
        train_windows=[56, 28],
        windows=[28],
        direction=1,
        objective_composition="symmetric_always_project_residual_equal_norm",
        receipt_path="diagnostic.json",
    ) == {"status": "PASS"}
    assert forwarded[-1]["config"]["diagnostic_objective_composition"] == (
        "symmetric_always_project_residual_equal_norm"
    )
    assert api.diagnostic_perturb_and_validate(
        40,
        config={"seed": 1701},
        train_windows=[56, 28],
        windows=[28],
        direction=1,
        objective_composition="symmetric_always_project_residual_original_mean_norm",
        receipt_path="diagnostic.json",
    ) == {"status": "PASS"}
    assert forwarded[-1]["config"]["diagnostic_objective_composition"] == (
        "symmetric_always_project_residual_original_mean_norm"
    )
    objective = "symmetric_always_project_residual_second_only_original_mean_norm"
    assert api.diagnostic_perturb_and_validate(
        40,
        config={"seed": 1701},
        train_windows=[56, 28],
        windows=[28],
        direction=1,
        objective_composition=objective,
        receipt_path="diagnostic.json",
    ) == {"status": "PASS"}
    assert forwarded[-1]["config"]["diagnostic_objective_composition"] == objective
    objective = "symmetric_always_project_residual_first_only_original_mean_norm"
    assert api.diagnostic_perturb_and_validate(
        40,
        config={"seed": 1701},
        train_windows=[56, 28],
        windows=[28],
        direction=1,
        objective_composition=objective,
        receipt_path="diagnostic.json",
    ) == {"status": "PASS"}
    assert forwarded[-1]["config"]["diagnostic_objective_composition"] == objective
    objective = "symmetric_always_project_residual_first_only_original_mean_projection"
    assert api.diagnostic_perturb_and_validate(
        40,
        config={"seed": 1701},
        train_windows=[56, 28],
        windows=[28],
        direction=1,
        objective_composition=objective,
        receipt_path="diagnostic.json",
    ) == {"status": "PASS"}
    assert forwarded[-1]["config"]["diagnostic_objective_composition"] == objective
    assert api.diagnostic_perturb_and_validate(
        40,
        config={"seed": 1701},
        train_windows=[56, 28],
        windows=[28],
        direction=1,
        objective_composition="ordered_second_project_residual_equal_norm",
        receipt_path="diagnostic.json",
    ) == {"status": "PASS"}
    assert forwarded[-1]["config"]["diagnostic_objective_composition"] == (
        "ordered_second_project_residual_equal_norm"
    )
    assert api.diagnostic_perturb_and_validate(
        40,
        config={"seed": 1701},
        train_windows=[56, 28],
        windows=[28],
        direction=1,
        objective_composition="ordered_first_project_residual_equal_norm",
        receipt_path="diagnostic.json",
    ) == {"status": "PASS"}
    assert forwarded[-1]["config"]["diagnostic_objective_composition"] == (
        "ordered_first_project_residual_equal_norm"
    )
    assert api.diagnostic_perturb_and_validate(
        40,
        config={"seed": 1701},
        train_windows=[56, 28],
        windows=[28],
        direction=1,
        objective_composition="ordered_first_project_equal_norm",
        receipt_path="diagnostic.json",
    ) == {"status": "PASS"}
    assert forwarded[-1]["config"]["diagnostic_objective_composition"] == (
        "ordered_first_project_equal_norm"
    )
    with pytest.raises(ArtifactError, match="objective composition"):
        api.diagnostic_perturb_and_validate(
            40,
            config={"seed": 1701},
            train_windows=[56, 28],
            windows=[28],
            direction=1,
            objective_composition="arbitrary",
            receipt_path="diagnostic.json",
        )
    with pytest.raises(ArtifactError, match="controlled train_windows"):
        api.diagnostic_perturb_and_validate(
            40,
            config={"seed": 1701},
            train_windows=[28, 42],
            windows=[28],
            direction=1,
            receipt_path="diagnostic.json",
        )
