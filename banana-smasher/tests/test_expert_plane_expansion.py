from __future__ import annotations

import hashlib
import importlib.util
import sys

import pytest
import torch

from banana_smasher.resident_balanced64 import ArtifactError
from banana_smasher.resident_continuation import (
    EXPERT_PLANE_SURFACE,
    _activate_expert_plane_surface,
    _classify_expert_plane_update,
    _validated_expert_plane_expansion,
)
from banana_smasher.resident_proven_api import _persisted_surface_reload_evidence

_RUNNER_DIR = (
    __import__("pathlib").Path(__file__).resolve().parents[2]
    / "runtime"
    / "v7"
    / "runner"
)
sys.path.insert(0, str(_RUNNER_DIR))
try:
    _spec = importlib.util.spec_from_file_location(
        "banana_smasher_test_physical_experts", _RUNNER_DIR / "fast_v7_expert_base.py"
    )
    assert _spec is not None and _spec.loader is not None
    _physical = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_physical)
finally:
    sys.path.remove(str(_RUNNER_DIR))
PROJECTION_SHAPES = _physical.PROJECTION_SHAPES
FullyResidentGroupedV7Experts = _physical.FullyResidentGroupedV7Experts


def _l028_module() -> FullyResidentGroupedV7Experts:
    module = FullyResidentGroupedV7Experts.__new__(FullyResidentGroupedV7Experts)
    torch.nn.Module.__init__(module)
    module.L = 28
    for projection, (m, k) in PROJECTION_SHAPES.items():
        su = torch.arange(256 * k, dtype=torch.int64).reshape(256, k).remainder(997).to(torch.float16)
        sv = torch.arange(256 * m, dtype=torch.int64).reshape(256, m).remainder(991).to(torch.float16)
        module.register_buffer(f"su_{projection}", su, persistent=False)
        module.register_buffer(f"sv_{projection}", sv, persistent=False)
    return module


def test_l028_su_sv_promotion_preserves_wire_views_and_exact_roster() -> None:
    module = _l028_module()
    immutable_views = {
        (projection, surface): getattr(module, f"{surface.lower()}_{projection}").clone()
        for projection in ("w1", "w2", "w3")
        for surface in ("SU", "SV")
    }

    rows = module.promote_l028_su_sv()

    assert len(rows) == 1536
    assert sum(parameter.numel() for _name, parameter in rows) == 4_718_592
    assert rows[0][0] == "model.layers.28.mlp.experts.E000.w1.SU"
    assert rows[-1][0] == "model.layers.28.mlp.experts.E255.w3.SV"
    assert len({name for name, _parameter in rows}) == 1536
    assert all(parameter.dtype == torch.float32 and parameter.requires_grad for _name, parameter in rows)
    for projection in ("w1", "w2", "w3"):
        for surface in ("SU", "SV"):
            observed = module.expert_plane_wire_view(projection, surface)
            expected = immutable_views[(projection, surface)]
            assert observed.dtype == torch.float16
            assert observed.equal(expected)
            assert hashlib.sha256(observed.detach().numpy().tobytes()).digest() == hashlib.sha256(expected.numpy().tobytes()).digest()

    ordinary_objective = sum(
        module.expert_plane_wire_view(projection, surface).float().sum()
        for projection in ("w1", "w2", "w3")
        for surface in ("SU", "SV")
    )
    ordinary_objective.backward()
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad) == parameter.numel()
        for _name, parameter in rows
    )

    state = module.expert_plane_state()
    changed_name = "model.layers.28.mlp.experts.E007.w2.SV"
    state[changed_name] = state[changed_name] + 1.0
    module.load_expert_plane_state(state)
    restored = dict(module.expert_plane_parameters())[changed_name]
    assert restored.detach().cpu().equal(state[changed_name])


def test_e116_zero_route_remains_trainable_and_optimizer_consumable() -> None:
    names = [
        "model.layers.28.mlp.experts.E116.w1.SU",
        "model.layers.28.mlp.experts.E116.w1.SV",
        "model.layers.28.mlp.experts.E116.w2.SU",
    ]
    rows = [
        (name, torch.nn.Parameter(torch.ones(2, dtype=torch.float32)))
        for name in names
    ]
    before = [parameter.detach().clone() for _name, parameter in rows]
    for _name, parameter in rows:
        parameter.grad = torch.zeros_like(parameter)

    coverage = _classify_expert_plane_update(torch, rows, before)

    assert coverage["missing_trainable"] == []
    assert coverage["missing_gradient"] == []
    assert coverage["missing_delta"] == []
    assert coverage["gradient_present"] == "3/3"
    assert coverage["gradient_nonzero"] == "0/3"
    assert coverage["delta_nonzero"] == "0/3"
    assert coverage["gradient"] == "3/3"
    assert coverage["delta"] == "3/3"

    optimizer = torch.optim.Adam([parameter for _name, parameter in rows], lr=1.0e-4)
    optimizer.step()
    assert len(optimizer.state) == 3


def test_expert_plane_expansion_contract_is_exact_and_refuses_wscale() -> None:
    roster_sha = "a" * 64
    config = {
        "expert_plane_expansion": {
            "surface": EXPERT_PLANE_SURFACE,
            "layer": 28,
            "components": ["SU", "SV"],
            "projections": ["w1", "w2", "w3"],
            "static_w28": True,
            "immutable_wire": True,
            "roster_sha256": roster_sha,
            "learning_rate": 1.0e-4,
        }
    }
    assert _validated_expert_plane_expansion(config)["roster_sha256"] == roster_sha

    bad = {"expert_plane_expansion": {**config["expert_plane_expansion"], "components": ["SU", "SV", "Wscale"]}}
    with pytest.raises(ArtifactError, match="exactly SU/SV"):
        _validated_expert_plane_expansion(bad)


def test_public_surface_activation_seeds_pre_and_loads_resume_state() -> None:
    class Expert:
        def __init__(self) -> None:
            self.parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
            self.loaded = None

        def promote_l028_su_sv(self):
            return [("model.layers.28.mlp.experts.E000.w1.SU", self.parameter)]

        def load_expert_plane_state(self, state):
            self.loaded = state

    expert = Expert()
    student = type("Student", (), {"experts": {28: expert}})()
    contract = {"layer": 28}

    rows = _activate_expert_plane_surface(student, {}, contract, checkpoint_cursor=0)
    assert rows == [("model.layers.28.mlp.experts.E000.w1.SU", expert.parameter)]
    assert expert.loaded is None

    saved = {rows[0][0]: torch.tensor([2.0])}
    resumed = _activate_expert_plane_surface(
        student, {EXPERT_PLANE_SURFACE: saved}, contract, checkpoint_cursor=1
    )
    assert resumed == rows
    assert expert.loaded is saved

    with pytest.raises(ArtifactError, match="missing L028 SU/SV"):
        _activate_expert_plane_surface(student, {}, contract, checkpoint_cursor=1)


def test_persisted_surface_reload_evidence_requires_exact_expert_state() -> None:
    state = {
        "luts": {"layer": torch.tensor([1.0])},
        "norms": {"norm": torch.tensor([1.0])},
        "outputs": {"output": torch.tensor([1.0])},
        EXPERT_PLANE_SURFACE: {
            f"expert-{index:04d}": torch.tensor([float(index)])
            for index in range(1536)
        },
    }

    evidence = _persisted_surface_reload_evidence(
        {"state": state},
        trainable_surfaces=["luts", "rmsnorms", "output_gains", EXPERT_PLANE_SURFACE],
    )

    assert evidence == {
        "resident_state_persisted": True,
        "checkpoint_reload_verified": True,
        "persisted_trainable_surfaces": [
            "luts",
            "norms",
            "outputs",
            EXPERT_PLANE_SURFACE,
        ],
        "persisted_expert_plane_tensors": 1536,
    }
    with pytest.raises(ArtifactError, match="reload omitted configured"):
        _persisted_surface_reload_evidence(
            {
                "state": {
                    key: value
                    for key, value in state.items()
                    if key != EXPERT_PLANE_SURFACE
                }
            },
            trainable_surfaces=[EXPERT_PLANE_SURFACE],
        )
