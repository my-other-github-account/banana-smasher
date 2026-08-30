from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from repair_api import ArtifactError, ResidentRepairAPI
from repair_api.sealed_pre_forward import bind_sealed_pre_resident_config


PROVIDER_SHA256 = "942c3074d89f8872f8c52df78941c908d9fce87edae7c21671d339f3e891d3cb"


class Immutable942cProjectionProvider:
    """Behavioral fixture for the immutable provider's projection call order."""

    def __init__(self) -> None:
        self.packed_w1 = torch.tensor([1])
        self.packed_w2 = torch.tensor([2])
        self.packed_w3 = torch.tensor([3])
        self.su_w1 = torch.tensor([11])
        self.su_w2 = torch.tensor([12])
        self.su_w3 = torch.tensor([13])
        self.sv_w1 = torch.tensor([21])
        self.sv_w2 = torch.tensor([22])
        self.sv_w3 = torch.tensor([23])
        self.plane_source = type("Plane", (), {"wire_lut": lambda _self: torch.tensor([31])})()

    def _project(
        self,
        projection: str,
        x: torch.Tensor,
        assignments: torch.Tensor,
        packed: torch.Tensor,
        lut_master: torch.Tensor,
        su: torch.Tensor,
        sv: torch.Tensor,
        *_args,
    ) -> torch.Tensor:
        del assignments, packed, lut_master, su, sv
        if projection == "w1":
            return torch.full_like(x, -1.0)
        if projection == "w3":
            return torch.full_like(x, 0.5)
        assert projection == "w2"
        return x

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        assignments = top_k_index.reshape(-1).to(torch.int64)
        routed = hidden_states.repeat_interleave(top_k_index.shape[1], dim=0)
        lut = self.plane_source.wire_lut()
        gate = self._project(
            "w1", routed, assignments, self.packed_w1, lut, self.su_w1, self.sv_w1
        )
        up = self._project(
            "w3", routed, assignments, self.packed_w3, lut, self.su_w3, self.sv_w3
        )
        product = torch.nn.functional.silu(gate) * up
        down = self._project(
            "w2", product, assignments, self.packed_w2, lut, self.su_w2, self.sv_w2
        )
        return (down * top_k_weights.reshape(-1, 1)).to(hidden_states.dtype)


def test_production_config_builder_activates_existing_projection_binder() -> None:
    """Production config must carry the exact gate that RUN7002 proved effective."""
    from repair_api.modern_green_resident import (
        SEALED_GATE_UP_RUNTIME_MARKER,
        _bind_installed_projection_runtime,
        _uses_exact_sealed_reconstruction,
    )

    config: dict[str, object] = {
        "resident_validation_expert_implementation": "accepted_static_w28"
    }
    bind_sealed_pre_resident_config(config)

    # RUN7003 proved that retaining accepted_static_w28 leaves the production
    # product at the composed-provider hashes with projection_calls=0.  The
    # canonical sealed-PRE builder must select the exact full-weight resident
    # boundary before installing the already-implemented projection wrapper.
    assert config["resident_validation_expert_implementation"] == (
        "sealed_bf16_full_weight"
    )
    assert config["resident_gate_up_projection"] == "combined_4096_bf16_f_linear_v1"
    assert config["resident_gate_up_provider_sha256"] == PROVIDER_SHA256
    assert _uses_exact_sealed_reconstruction(config) is True

    installed = type(
        "InstalledProvider",
        (),
        {"_sealed_gate_up_runtime_marker": SEALED_GATE_UP_RUNTIME_MARKER},
    )
    trainer = SimpleNamespace(FullyResidentGroupedV7Experts=object)
    receipt = _bind_installed_projection_runtime(trainer, installed, config)

    assert receipt["status"] == "BOUND_TO_ORDINARY_TRAINER_GLOBAL"
    assert receipt["provider_expert_sha256"] == PROVIDER_SHA256
    assert trainer.FullyResidentGroupedV7Experts is installed


def test_public_projection_wrapper_binds_combined_gate_up_and_product_exactly() -> None:
    config = {"basis_sha256": "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"}
    bound = ResidentRepairAPI.bind_combined_gate_up_projection(
        config, provider_expert_sha256=PROVIDER_SHA256
    )

    assert config == {"basis_sha256": bound["basis_sha256"]}
    assert bound["resident_gate_up_projection"] == "combined_4096_bf16_f_linear_v1"
    assert bound["resident_gate_up_provider_sha256"] == PROVIDER_SHA256
    with pytest.raises(ArtifactError, match="requires provider 942c3074"):
        ResidentRepairAPI.bind_combined_gate_up_projection(
            config, provider_expert_sha256="0" * 64
        )

    from repair_api.modern_green_resident import _bind_sealed_gate_up_projection

    calls: list[tuple[torch.Tensor, ...]] = []

    def combined_projection(*args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(args)
        x = args[0]
        return torch.ones_like(x), torch.full_like(x, 2.0)

    hidden = torch.zeros((1, 1), dtype=torch.bfloat16)
    indices = torch.tensor([[0]], dtype=torch.int64)
    weights = torch.ones((1, 1), dtype=torch.bfloat16)
    baseline = Immutable942cProjectionProvider().forward(hidden, indices, weights)
    wrapped_type = _bind_sealed_gate_up_projection(
        Immutable942cProjectionProvider,
        bound,
        combined_projection=combined_projection,
        native_down_projection=lambda x, *_args: x,
    )
    observed = wrapped_type().forward(hidden, indices, weights)
    expected = (torch.nn.functional.silu(torch.ones_like(hidden)) * 2).to(torch.bfloat16)

    assert not torch.equal(baseline, expected)
    assert torch.equal(observed, expected)
    assert len(calls) == 1
    assert calls[0][2] is not calls[0][3]
    assert calls[0][2].item() == 1
    assert calls[0][3].item() == 3
    assert wrapped_type.__name__ == Immutable942cProjectionProvider.__name__


def test_product_runner_does_not_shadow_the_construction_time_projection_wrapper() -> None:
    source = (
        __import__("pathlib").Path(__file__).parents[1] / "resident_full64_accept.py"
    ).read_text()
    binding = source.index("config = api.bind_combined_gate_up_projection(")
    construction = source.index("engine = ModernGreenResidentEngine(")

    assert binding < construction
    assert "projection_binding = _install_r20_full_weight_projection(engine)" not in source
    assert "projection_binding = engine.sealed_gate_up_runtime_witness(" in source[construction:]
    assert "require_activation=False" in source[construction:]


def test_combined_projection_uses_one_expert_local_bf16_linear() -> None:
    from repair_api.modern_green_resident import (
        _sealed_builder_combined_gate_up_projection,
    )

    torch.manual_seed(942)
    hidden = torch.randn(4, 3, dtype=torch.bfloat16)
    assignments = torch.tensor([1, 0, 1, 0], dtype=torch.int64)
    packed_w1 = torch.tensor([0, 1])
    packed_w3 = torch.tensor([2, 3])
    weights = [torch.randn(3, 2, dtype=torch.bfloat16) for _ in range(4)]
    su = torch.zeros((2, 3))
    sv = torch.zeros((2, 2))

    def full_weight_builder(packed, _lut, _su, _sv):
        return weights[int(packed.item())]

    gate, up = _sealed_builder_combined_gate_up_projection(
        hidden,
        assignments,
        packed_w1,
        packed_w3,
        torch.zeros(1),
        su,
        sv,
        su,
        sv,
        full_weight_builder=full_weight_builder,
    )
    expected = torch.empty((4, 4), dtype=torch.bfloat16)
    for expert in (0, 1):
        mask = assignments == expert
        combined_weight = torch.cat(
            (weights[expert].T.contiguous(), weights[expert + 2].T.contiguous()),
            dim=0,
        )
        expected[mask] = torch.nn.functional.linear(
            hidden[mask].contiguous(), combined_weight
        )
    expected_gate, expected_up = expected.chunk(2, dim=-1)

    assert torch.equal(gate, expected_gate)
    assert torch.equal(up, expected_up)


def test_sealed_w2_source_codes_reproduce_builder_mode_w2_bytes_and_linear() -> None:
    from repair_api.modern_green_resident import (
        _decode_sealed_w2_source_weight,
        _pack_sealed_w2_source_codes,
        _sealed_w2_source_projection,
    )

    # Two original FP4 bytes contain nibbles 0,5,8,13.  Builder mode=w2 maps
    # them to +1,+4,-1,-4 before applying the e8m0 block scale.
    packed = torch.tensor([[0x50, 0xD8] * 8], dtype=torch.uint8)
    codes = _pack_sealed_w2_source_codes(packed)
    scales = torch.full((1, 1), 127, dtype=torch.uint8)
    weight = _decode_sealed_w2_source_weight(codes, scales)
    expected_weight = torch.tensor(
        [[1.0, 4.0, -1.0, -4.0] * 8], dtype=torch.bfloat16
    )

    assert torch.equal(weight, expected_weight)
    hidden = torch.arange(32, dtype=torch.bfloat16).reshape(1, 32)
    observed = _sealed_w2_source_projection(
        hidden, torch.tensor([0]), codes.unsqueeze(0), scales.unsqueeze(0)
    )
    assert torch.equal(observed, torch.nn.functional.linear(hidden, expected_weight))


def test_public_projection_wrapper_calls_native_bf16_w2_for_every_provider_instance() -> None:
    native_calls: list[tuple[torch.Tensor, ...]] = []

    def native_down_projection(*args: torch.Tensor) -> torch.Tensor:
        native_calls.append(args)
        return torch.full_like(args[0], 7.0)

    config = ResidentRepairAPI.bind_combined_gate_up_projection(
        {}, provider_expert_sha256=PROVIDER_SHA256
    )
    from repair_api.modern_green_resident import _bind_sealed_gate_up_projection

    wrapped_type = _bind_sealed_gate_up_projection(
        Immutable942cProjectionProvider,
        config,
        combined_projection=lambda *args: (
            torch.ones_like(args[0]), torch.ones_like(args[0])
        ),
        native_down_projection=native_down_projection,
    )
    assert wrapped_type._sealed_native_bf16_w2_scope == "provider_class_all_instances_v1"

    dispatch_implementations = set()
    for layer in (0, 21, 42):
        provider = wrapped_type()
        provider.L = layer
        provider._sealed_aligned_positions = None
        dispatch_implementations.add(
            provider._sealed_native_bf16_down_projection.__func__
        )
        activated = torch.tensor([[0.006927490234375]], dtype=torch.bfloat16)
        observed = provider._project(
            "w2", activated, torch.tensor([0]), provider.packed_w2,
            provider.plane_source.wire_lut(), provider.su_w2, provider.sv_w2,
        )
        assert observed.dtype == torch.bfloat16
        assert torch.equal(observed, torch.full_like(activated, 7.0))

    assert len(native_calls) == 3
    assert dispatch_implementations == {
        wrapped_type._sealed_native_bf16_down_projection
    }


def test_routed_return_matches_grouped_mm_reshape_sum_not_expert_index_add() -> None:
    from repair_api.modern_green_resident import _sealed_builder_accumulate_routes

    hidden = torch.zeros((1, 1), dtype=torch.bfloat16)
    routed_output = torch.tensor(
        [[0.302734375], [0.578125], [-0.8125], [-0.94140625],
         [0.671875], [-0.134765625]],
        dtype=torch.bfloat16,
    )
    top_k_index = torch.tensor([[5, 4, 3, 2, 1, 0]], dtype=torch.int64)
    top_k_weights = torch.ones((1, 6), dtype=torch.float32)

    observed = _sealed_builder_accumulate_routes(
        hidden, routed_output, top_k_index, top_k_weights
    )
    weighted = (routed_output * top_k_weights.reshape(-1, 1)).to(torch.bfloat16)
    grouped_mm = weighted.view(1, 6, 1).sum(dim=1).to(torch.bfloat16)
    expert_ordered = torch.zeros_like(hidden)
    for expert in torch.unique(top_k_index, sorted=True):
        expert_ordered.index_add_(
            0, torch.tensor([0]), weighted[top_k_index.reshape(-1) == expert]
        )

    assert not torch.equal(grouped_mm, expert_ordered)
    assert torch.equal(observed, grouped_mm)
