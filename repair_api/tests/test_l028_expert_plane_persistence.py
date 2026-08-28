from __future__ import annotations

import pytest
import torch

from repair_api.balanced64 import ArtifactError
from repair_api.modern_green_resident import (
    EXPERT_PLANE_SURFACE,
    ModernGreenResidentEngine,
    _activate_expert_plane_surface,
    _fp64_state_adam,
    _merge_expanded_optimizer_state,
    _validated_expert_plane_expansion,
)


def _contract() -> dict[str, object]:
    return {
        "surface": EXPERT_PLANE_SURFACE,
        "layer": 28,
        "components": ["SU", "SV"],
        "projections": ["w1", "w2", "w3"],
        "static_w28": True,
        "immutable_wire": True,
        "roster_sha256": "3f2c6f97cb65f6e2e862881b8096993797c4d86b373c27632a287163215fc1ec",
        "learning_rate": 1.0e-4,
    }


def test_exact_l028_contract_promotes_pre_and_reloads_checkpoint_state() -> None:
    class Expert:
        def __init__(self) -> None:
            self.parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
            self.loaded = None

        def promote_l028_su_sv(self):
            return [("model.layers.28.mlp.experts.E000.w1.SU", self.parameter)]

        def load_expert_plane_state(self, state):
            self.loaded = state

    contract = _validated_expert_plane_expansion({"expert_plane_expansion": _contract()})
    assert contract is not None
    expert = Expert()
    student = type("Student", (), {"experts": {28: expert}})()

    pre_rows = _activate_expert_plane_surface(
        student, {}, contract, checkpoint_cursor=0
    )
    assert pre_rows == [("model.layers.28.mlp.experts.E000.w1.SU", expert.parameter)]
    assert expert.loaded is None

    saved = {pre_rows[0][0]: torch.tensor([2.0])}
    resumed_rows = _activate_expert_plane_surface(
        student, {EXPERT_PLANE_SURFACE: saved}, contract, checkpoint_cursor=1
    )
    assert resumed_rows == pre_rows
    assert expert.loaded is saved


def test_l028_contract_rejects_missing_resume_surface_and_wscale() -> None:
    contract = _contract()
    contract["components"] = ["SU", "SV", "Wscale"]
    with pytest.raises(ArtifactError, match="exactly SU/SV"):
        _validated_expert_plane_expansion({"expert_plane_expansion": contract})

    wrong_roster = {
        "expert_plane_expansion": {
            **_contract(),
            "roster_sha256": "a" * 64,
        }
    }
    with pytest.raises(ArtifactError, match="roster identity drift"):
        _validated_expert_plane_expansion(wrong_roster)

    with pytest.raises(ArtifactError, match="cannot combine with tailfix"):
        _validated_expert_plane_expansion(
            {"expert_plane_expansion": _contract(), "tailfix_wholesale": True}
        )

    class Expert:
        def promote_l028_su_sv(self):
            return []

    with pytest.raises(ArtifactError, match="missing L028 SU/SV"):
        _activate_expert_plane_surface(
            type("Student", (), {"experts": {28: Expert()}})(),
            {},
            _contract(),
            checkpoint_cursor=1,
        )


def test_expanded_optimizer_merge_preserves_four_surface_name_identity() -> None:
    surfaces = ("luts", "norms", "outputs", EXPERT_PLANE_SURFACE)
    ordered = {
        "luts": {"lut": torch.tensor([1.0])},
        "norms": {"norm": torch.tensor([2.0])},
        "outputs": {"output": torch.tensor([3.0])},
        EXPERT_PLANE_SURFACE: {"plane": torch.tensor([4.0])},
    }
    rows = [
        {
            "param_names": {surface: list(ordered[surface]) for surface in surfaces},
            "optimizer": {
                "state": {
                    index: {"step": torch.tensor(1.0), "exp_avg": torch.tensor([index + 1.0])}
                    for index in range(4)
                },
                "param_groups": [
                    {"params": [index], "lr": 10.0 ** (-index - 1), "group_name": surface}
                    for index, surface in enumerate(surfaces)
                ],
            },
        }
    ]

    merged = _merge_expanded_optimizer_state(rows, ordered, surfaces, set())

    assert [group["group_name"] for group in merged["param_groups"]] == list(surfaces)
    assert [group["params"] for group in merged["param_groups"]] == [[0], [1], [2], [3]]
    assert set(merged["state"]) == {0, 1, 2, 3}


def test_resident_reload_restores_fourth_expert_plane_optimizer_group() -> None:
    surfaces = ("luts", "norms", "outputs", EXPERT_PLANE_SURFACE)
    source_parameters = [torch.nn.Parameter(torch.tensor([float(index + 1)])) for index in range(4)]
    source = _fp64_state_adam(
        torch,
        [
            {"params": [parameter], "lr": 1.0e-3, "group_name": surface}
            for surface, parameter in zip(surfaces, source_parameters)
        ],
    )
    for parameter in source_parameters:
        parameter.grad = torch.ones_like(parameter)
    source.step()
    source_state = source.state_dict()

    target_parameters = [torch.nn.Parameter(torch.tensor([0.0])) for _ in surfaces]
    engine = ModernGreenResidentEngine.__new__(ModernGreenResidentEngine)
    engine.torch = torch
    engine.luts = [("lut", target_parameters[0])]
    engine.norms = [("norm", target_parameters[1])]
    engine.outputs = [("output", target_parameters[2])]
    engine.expert_planes = [("plane", target_parameters[3])]
    engine.expert_plane_contract = _contract()
    engine.state = {
        surface: {name: torch.tensor([0.0])}
        for surface, name in zip(surfaces, ("lut", "norm", "output", "plane"))
    }
    engine.payload = {
        "optimizer": source_state,
        "scheduler": {"last_epoch": 1},
    }
    engine.optimizer = _fp64_state_adam(
        torch,
        [
            {"params": [parameter], "lr": 1.0e-3, "group_name": surface}
            for surface, parameter in zip(surfaces, target_parameters)
        ],
    )

    class Scheduler:
        def __init__(self) -> None:
            self.loaded = None

        def load_state_dict(self, state) -> None:
            self.loaded = state

    engine.scheduler = Scheduler()
    engine.published_pre_recipe = False
    engine.global_step = 1
    engine.controlled_arm = False

    engine._load_optimizer_scheduler_state()

    assert len(engine.optimizer.param_groups) == 4
    plane_state = engine.optimizer.state[target_parameters[3]]
    assert int(plane_state["step"].item()) == 1
    assert torch.equal(plane_state["exp_avg"], source_state["state"][3]["exp_avg"])
    assert engine.scheduler.loaded == {"last_epoch": 1}


def test_local_parameter_surface_is_expert_only_for_l028_probe() -> None:
    engine = ModernGreenResidentEngine.__new__(ModernGreenResidentEngine)
    engine.luts = [("lut", torch.nn.Parameter(torch.tensor([1.0])))]
    engine.norms = [("norm", torch.nn.Parameter(torch.tensor([2.0])))]
    engine.outputs = [("output", torch.nn.Parameter(torch.tensor([3.0])))]
    engine.expert_planes = [("plane", torch.nn.Parameter(torch.tensor([4.0])))]
    engine.expert_plane_contract = _contract()

    rows = engine._local_params()

    assert [name for name, _parameter in rows] == ["plane"]
