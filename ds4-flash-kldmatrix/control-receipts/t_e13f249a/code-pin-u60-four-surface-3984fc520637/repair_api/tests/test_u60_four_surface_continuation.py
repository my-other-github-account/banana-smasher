from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from repair_api.balanced64 import ArtifactError
from repair_api.modern_green_resident import (
    AUTHENTICATED_U60_CHECKPOINT_SHA256,
    EXPERT_PLANE_SURFACE,
    _bind_l28_expert_plane_state,
    _canonical_l28_expert_plane_spec,
    _fp64_state_adam,
    _merge_sharded_optimizer_state,
    _validate_trainable_state_schema,
)


def _u60_planes() -> dict[str, torch.Tensor]:
    return {
        name: torch.arange(shape[0], dtype=torch.float32)
        for name, shape in _canonical_l28_expert_plane_spec()
    }


def _payload(state, *, step: int, sha: str | None = None):
    identity = {"next_update": step}
    if sha is not None:
        identity["checkpoint_sha256"] = sha
    return {"state": state, "next_update": step, "identity": identity}


def test_three_surface_u0_remains_admitted() -> None:
    state = {"luts": {}, "norms": {}, "outputs": {}}
    assert _validate_trainable_state_schema(torch, _payload(state, step=0)) is False


def test_four_surface_is_only_authenticated_u60_and_canonical_order() -> None:
    planes = _u60_planes()
    state = {"luts": {}, "norms": {}, "outputs": {}, EXPERT_PLANE_SURFACE: planes}
    payload = _payload(state, step=60, sha=AUTHENTICATED_U60_CHECKPOINT_SHA256)
    assert _validate_trainable_state_schema(torch, payload) is True

    reordered = dict(reversed(list(planes.items())))
    with pytest.raises(ArtifactError, match="canonical order"):
        _validate_trainable_state_schema(
            torch,
            _payload(
                {"luts": {}, "norms": {}, "outputs": {}, EXPERT_PLANE_SURFACE: reordered},
                step=60,
                sha=AUTHENTICATED_U60_CHECKPOINT_SHA256,
            ),
        )

    with pytest.raises(ArtifactError, match="authenticated UPDATE_060"):
        _validate_trainable_state_schema(
            torch, _payload(state, step=60, sha="0" * 64)
        )


def test_four_surface_rejects_wrong_dtype_and_geometry() -> None:
    planes = _u60_planes()
    first = next(iter(planes))
    planes[first] = planes[first].to(torch.float16)
    state = {"luts": {}, "norms": {}, "outputs": {}, EXPERT_PLANE_SURFACE: planes}
    with pytest.raises(ArtifactError, match="float32"):
        _validate_trainable_state_schema(
            torch,
            _payload(state, step=60, sha=AUTHENTICATED_U60_CHECKPOINT_SHA256),
        )


def test_l28_runtime_binding_reads_checkpoint_candidates_and_is_trainable() -> None:
    class Provider(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for projection, k, m in (("w1", 4096, 2048), ("w2", 2048, 4096), ("w3", 4096, 2048)):
                self.register_buffer(f"su_{projection}", torch.zeros((256, k), dtype=torch.float16), persistent=False)
                self.register_buffer(f"sv_{projection}", torch.zeros((256, m), dtype=torch.float16), persistent=False)

        def forward(self):
            return self.su_w1[0, 0] + self.sv_w3[-1, -1]

    provider = Provider()
    layers = [SimpleNamespace(mlp=SimpleNamespace(experts=Provider())) for _ in range(29)]
    layers[28] = SimpleNamespace(mlp=SimpleNamespace(experts=provider))
    student = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(layers=layers)), device=torch.device("cpu"))
    planes = _u60_planes()
    rows, proof = _bind_l28_expert_plane_state(torch, student, planes)

    assert len(rows) == 1536
    assert [name for name, _ in rows] == list(planes)
    assert all(parameter.dtype == torch.float32 and parameter.requires_grad for _, parameter in rows)
    assert proof["candidate_name"] == "model.layers.28.mlp.experts.E000.w1.SU"
    assert proof["candidate_read"] is True

    observed = provider()
    expected = planes[proof["candidate_name"]][0].to(torch.float16)
    assert torch.equal(provider.su_w1[0], planes[proof["candidate_name"]].to(torch.float16))
    assert observed.dtype == torch.float16
    assert torch.equal(provider.su_w1[0, 0], expected)
    observed.backward()
    assert rows[0][1].grad is not None


def test_fourth_group_optimizer_scheduler_round_trip_and_nonzero_delta() -> None:
    names = [name for name, _shape in _canonical_l28_expert_plane_spec()]
    params = [torch.nn.Parameter(torch.tensor([float(i % 7)], dtype=torch.float32)) for i in range(len(names))]
    optimizer = torch.optim.Adam([
        {"params": [], "lr": 0.0, "group_name": "luts"},
        {"params": [], "lr": 0.0, "group_name": "norms"},
        {"params": [], "lr": 0.0, "group_name": "outputs"},
        {"params": params, "lr": 7.5e-5, "group_name": EXPERT_PLANE_SURFACE},
    ])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda _s: 1.0] * 4)
    before = params[0].detach().clone()
    params[0].grad = torch.ones_like(params[0])
    optimizer.step()
    scheduler.step()
    assert not torch.equal(before, params[0])

    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()
    restored_params = [torch.nn.Parameter(torch.tensor([float(i % 7)], dtype=torch.float32)) for i in range(len(names))]
    restored_optimizer = torch.optim.Adam([
        {"params": [], "lr": 0.0, "group_name": "luts"},
        {"params": [], "lr": 0.0, "group_name": "norms"},
        {"params": [], "lr": 0.0, "group_name": "outputs"},
        {"params": restored_params, "lr": 7.5e-5, "group_name": EXPERT_PLANE_SURFACE},
    ])
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer, lr_lambda=[lambda _s: 1.0] * 4
    )
    restored_optimizer.load_state_dict(optimizer_state)
    restored_scheduler.load_state_dict(scheduler_state)
    assert len(restored_optimizer.param_groups) == 4
    assert len(restored_optimizer.param_groups[3]["params"]) == 1536
    assert len(restored_scheduler.base_lrs) == 4
    assert restored_scheduler.state_dict()["last_epoch"] == scheduler_state["last_epoch"]


def test_four_group_sharded_optimizer_merge_preserves_checkpoint_order() -> None:
    surfaces = ("luts", "norms", "outputs", EXPERT_PLANE_SURFACE)
    ordered = {
        "luts": {"lut": torch.tensor([1.0])},
        "norms": {"norm": torch.tensor([2.0])},
        "outputs": {"output": torch.tensor([3.0])},
        EXPERT_PLANE_SURFACE: {"plane0": torch.tensor([4.0]), "plane1": torch.tensor([5.0])},
    }
    templates = [
        {"lr": 0.0, "group_name": "luts"},
        {"lr": 0.0, "group_name": "norms"},
        {"lr": 0.0, "group_name": "outputs"},
        {"lr": 7.5e-5, "group_name": EXPERT_PLANE_SURFACE},
    ]
    rows = [
        {
            "param_names": {
                "luts": ["lut"], "norms": [], "outputs": ["output"],
                EXPERT_PLANE_SURFACE: [],
            },
            "optimizer": {
                "state": {0: {"step": 60}, 1: {"step": 60}},
                "param_groups": [
                    {**templates[0], "params": [0]},
                    {**templates[1], "params": []},
                    {**templates[2], "params": [1]},
                    {**templates[3], "params": []},
                ],
            },
        },
        {
            "param_names": {
                "luts": [], "norms": ["norm"], "outputs": [],
                EXPERT_PLANE_SURFACE: ["plane0", "plane1"],
            },
            "optimizer": {
                "state": {0: {"step": 60}, 1: {"step": 60}, 2: {"step": 60}},
                "param_groups": [
                    {**templates[0], "params": []},
                    {**templates[1], "params": [0]},
                    {**templates[2], "params": []},
                    {**templates[3], "params": [1, 2]},
                ],
            },
        },
    ]
    merged = _merge_sharded_optimizer_state(rows, ordered, surfaces)
    assert [group["group_name"] for group in merged["param_groups"]] == list(surfaces)
    assert [group["params"] for group in merged["param_groups"]] == [[0], [1], [2], [3, 4]]
    assert set(merged["state"]) == {0, 1, 2, 3, 4}


def test_fp64_adam_restores_integer_step_from_authenticated_u60_style_state() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = _fp64_state_adam(torch, [{"params": [parameter], "lr": 1.0e-4}])
    optimizer.load_state_dict({
        "state": {0: {
            "step": 60,
            "exp_avg": torch.zeros(1, dtype=torch.float64),
            "exp_avg_sq": torch.ones(1, dtype=torch.float64),
        }},
        "param_groups": [{
            "params": [0], "lr": 1.0e-4, "betas": (0.9, 0.999),
            "eps": 1.0e-8,
        }],
    })
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert int(optimizer.state[parameter]["step"].item()) == 61
