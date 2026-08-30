from types import SimpleNamespace

import pytest
import torch

from repair_api.api import _validate_trainable_scale_candidate_contract
from repair_api.balanced64 import ArtifactError
from repair_api.modern_green_resident import (
    BASE_LRS,
    _configure_trainable_quantization_scales,
    _project_quantization_scale_trust_region,
    _resident_optimizer_param_groups,
)


class FakeExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("packed_w1", torch.ones(1, dtype=torch.int16), persistent=False)
        self.register_buffer("su_w1", torch.tensor([[1.0, 2.0]], dtype=torch.float16), persistent=False)
        self.register_buffer("sv_w1", torch.tensor([[3.0]], dtype=torch.float16), persistent=False)
        self.register_buffer("su_w2", torch.tensor([[4.0]], dtype=torch.float16), persistent=False)
        self.register_buffer("sv_w2", torch.tensor([[5.0, 6.0]], dtype=torch.float16), persistent=False)
        self.register_buffer("su_w3", torch.tensor([[7.0, 8.0]], dtype=torch.float16), persistent=False)
        self.register_buffer("sv_w3", torch.tensor([[9.0]], dtype=torch.float16), persistent=False)


def _student():
    return SimpleNamespace(experts={3: FakeExpert()})


def test_scale_surface_is_opt_in_and_default_path_is_unchanged():
    student = _student()
    rows, manifest = _configure_trainable_quantization_scales({}, student, saved=None)

    assert rows == []
    assert manifest == {"mode": "frozen", "trainable": 0}
    assert "su_w1" in student.experts[3]._buffers
    assert not student.experts[3].su_w1.requires_grad


def test_scale_surface_promotes_all_su_sv_buffers_to_fp32_parameters_without_value_drift():
    student = _student()
    packed_before = student.experts[3].packed_w1.clone()
    expected = student.experts[3].su_w1.float().clone()

    rows, manifest = _configure_trainable_quantization_scales(
        {"trainable_quantization_scales": True}, student, saved=None
    )

    assert [name for name, _ in rows] == [
        "layers.3.scales.su_w1", "layers.3.scales.sv_w1",
        "layers.3.scales.su_w2", "layers.3.scales.sv_w2",
        "layers.3.scales.su_w3", "layers.3.scales.sv_w3",
    ]
    assert manifest == {
        "mode": "trainable", "trainable": 6, "layers": [3],
        "relative_trust_region": None,
    }
    assert "su_w1" in student.experts[3]._parameters
    assert student.experts[3].su_w1.dtype == torch.float32
    assert student.experts[3].su_w1.requires_grad
    assert torch.equal(student.experts[3].su_w1.detach(), expected)
    assert torch.equal(student.experts[3].packed_w1, packed_before)
    assert not student.experts[3].packed_w1.requires_grad


def test_scale_surface_loads_checkpoint_values_and_refuses_incomplete_state():
    student = _student()
    names = [
        f"layers.3.scales.{axis}_{projection}"
        for projection in ("w1", "w2", "w3")
        for axis in ("su", "sv")
    ]
    saved = {
        name: torch.full_like(getattr(student.experts[3], name.rsplit(".", 1)[1]).float(), index + 10.0)
        for index, name in enumerate(names)
    }

    rows, _manifest = _configure_trainable_quantization_scales(
        {"trainable_quantization_scales": True}, student, saved=saved
    )
    assert all(torch.equal(parameter.detach(), saved[name]) for name, parameter in rows)

    with pytest.raises(ArtifactError, match="checkpoint missing trainable quantization scale"):
        _configure_trainable_quantization_scales(
            {"trainable_quantization_scales": True}, _student(), saved={}
        )


def test_scale_optimizer_group_reuses_lut_schedule_and_default_group_count_stays_three():
    parameter = torch.nn.Parameter(torch.ones(1))
    rows = {
        "luts": [("lut", parameter)],
        "norms": [],
        "outputs": [],
        "scales": [("scale", torch.nn.Parameter(torch.ones(1)))],
    }

    default_groups = _resident_optimizer_param_groups({}, rows, BASE_LRS)
    scale_groups = _resident_optimizer_param_groups(
        {"trainable_quantization_scales": True}, rows, BASE_LRS
    )

    assert [group["group_name"] for group in default_groups] == ["luts", "norms", "outputs"]
    assert [group["group_name"] for group in scale_groups] == ["luts", "norms", "outputs", "scales"]
    assert scale_groups[-1]["lr"] == BASE_LRS["luts"]


def test_scale_relative_trust_region_projects_positive_negative_and_zero_controls():
    student = _student()
    rows, manifest = _configure_trainable_quantization_scales(
        {
            "trainable_quantization_scales": True,
            "quantization_scale_relative_trust_region": 0.05,
        },
        student,
        saved=None,
    )
    assert manifest["relative_trust_region"] == 0.05
    name, parameter = rows[0]
    with torch.no_grad():
        parameter.copy_(torch.tensor([[4.0, -4.0]]))
    report = _project_quantization_scale_trust_region(
        {"quantization_scale_relative_trust_region": 0.05}, [(name, parameter)]
    )
    assert torch.equal(parameter, torch.tensor([[1.05, 1.9]]))
    assert report["enabled"] is True
    assert report["clipped_elements"] == 2
    assert report["maximum_relative_delta"] == pytest.approx(0.05)

    zero = torch.nn.Parameter(torch.tensor([0.0]))
    zero._banana_scale_control = torch.tensor([0.0])
    with torch.no_grad():
        zero.fill_(1.0)
    _project_quantization_scale_trust_region(
        {"quantization_scale_relative_trust_region": 0.05}, [("zero", zero)]
    )
    assert zero.item() == 0.0


def test_rank_partitioned_scale_optimizer_state_merges_by_parameter_name():
    from repair_api.modern_green_resident import ModernGreenResidentEngine

    rows = [
        {
            "param_names": {"luts": ["l0"], "norms": [], "outputs": [], "scales": ["s0"]},
            "optimizer": {
                "state": {0: {"step": 1}, 1: {"step": 1}},
                "param_groups": [
                    {"params": [0], "group_name": "luts"},
                    {"params": [], "group_name": "norms"},
                    {"params": [], "group_name": "outputs"},
                    {"params": [1], "group_name": "scales"},
                ],
            },
        },
        {
            "param_names": {"luts": ["l1"], "norms": [], "outputs": [], "scales": ["s1"]},
            "optimizer": {
                "state": {0: {"step": 1}, 1: {"step": 1}},
                "param_groups": [
                    {"params": [0], "group_name": "luts"},
                    {"params": [], "group_name": "norms"},
                    {"params": [], "group_name": "outputs"},
                    {"params": [1], "group_name": "scales"},
                ],
            },
        },
    ]
    merged = ModernGreenResidentEngine._merge_named_optimizer_state(
        ModernGreenResidentEngine.__new__(ModernGreenResidentEngine),
        rows,
        {"luts": {"l0": 0, "l1": 0}, "norms": {}, "outputs": {}, "scales": {"s0": 0, "s1": 0}},
        ("luts", "norms", "outputs", "scales"),
    )

    assert [group["params"] for group in merged["param_groups"]] == [[0, 1], [], [], [2, 3]]
    assert set(merged["state"]) == {0, 1, 2, 3}


def test_public_scale_candidate_contract_pins_u20_to_u24_and_uniform_mean():
    config = {
        "trainable_quantization_scales": True,
        "token_kld_reduction": "mean",
        "scientific_identity": (
            "Candidate C: U20-to-U24; sole variable is grouped-K2 quantization scales frozen-to-trainable"
        ),
    }
    assert _validate_trainable_scale_candidate_contract(
        start_update=20,
        start_sha="2502bd03cc2c9deac966a24f8e8712633b1b0e0cb192d5eee71d10e91e77cccd",
        requested=(21, 22, 23, 24),
        config=config,
    ) is True

    for changed in (
        {"token_kld_reduction": "cvar_tail"},
        {"scientific_identity": "different"},
    ):
        with pytest.raises(ArtifactError):
            _validate_trainable_scale_candidate_contract(
                start_update=20,
                start_sha="2502bd03cc2c9deac966a24f8e8712633b1b0e0cb192d5eee71d10e91e77cccd",
                requested=(21, 22, 23, 24),
                config={**config, **changed},
            )


def test_public_candidate_d_contract_pins_mechanism_and_relative_bound():
    config = {
        "trainable_quantization_scales": True,
        "quantization_scale_relative_trust_region": 0.05,
        "mechanism_receipt_sha256": (
            "2a706eece007225b1a37d9977102659e5bdedd736d04585b577128e0c5918d36"
        ),
        "token_kld_reduction": "mean",
        "scientific_identity": (
            "Candidate D: U20-to-U24; sole variable versus Candidate C is a 5% U20-relative scale trust region"
        ),
    }
    assert _validate_trainable_scale_candidate_contract(
        start_update=20,
        start_sha="2502bd03cc2c9deac966a24f8e8712633b1b0e0cb192d5eee71d10e91e77cccd",
        requested=(21, 22, 23, 24),
        config=config,
    ) is True
    with pytest.raises(ArtifactError):
        _validate_trainable_scale_candidate_contract(
            start_update=20,
            start_sha="2502bd03cc2c9deac966a24f8e8712633b1b0e0cb192d5eee71d10e91e77cccd",
            requested=(21, 22, 23, 24),
            config={**config, "quantization_scale_relative_trust_region": 0.10},
        )
