from __future__ import annotations

import pytest
import torch

from repair_api import ArtifactError, ResidentRepairAPI


PROVIDER_SHA256 = "942c3074d89f8872f8c52df78941c908d9fce87edae7c21671d339f3e891d3cb"
ATTEMPT106BP_TERMINAL_SHA256 = (
    "a5964b1276475629e0e2ab1a22ec0fe82fd81ef0302f31b06873c31c2b358faa"
)
ATTEMPT106BP_ACTIVATION_SHA256 = (
    "61528b2cca0fafb243624f7d2d641d4c51c973448926f3826e3740dd857b5cf4"
)
ATTEMPT106BP_BUILDER_W2_SHA256 = (
    "f4ac5d41507ddbf581d95fbe80087e9abb105bab7f95cbf2f7f2c6ba1a6ca4db"
)
ATTEMPT106BP_PUBLIC_W2_SHA256 = (
    "b254218fcd731ec0d049c55eb76d1d7a2b44664ca45f15c3fbb442738735038a"
)
ATTEMPT106BQ_TERMINAL_SHA256 = (
    "8212b751d5f08a2f80c5d83c3a362779ee43adb54ac45fa42438ef7666a0c6ab"
)
ATTEMPT106BQ_PUBLIC_CAST_W2_SHA256 = (
    "e5e853db9d49d2ded1f640cc1ea3a63e15748f353924b08ed5f8d99e563788dc"
)


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


def test_public_projection_wrapper_casts_grouped_w2_to_hidden_dtype_before_weighting() -> None:
    """Attempt106bp: preserve the builder's BF16 W2 handoff, not FP32 weighting."""
    assert ATTEMPT106BP_TERMINAL_SHA256 == (
        "a5964b1276475629e0e2ab1a22ec0fe82fd81ef0302f31b06873c31c2b358faa"
    )
    assert ATTEMPT106BP_ACTIVATION_SHA256 == (
        "61528b2cca0fafb243624f7d2d641d4c51c973448926f3826e3740dd857b5cf4"
    )
    assert ATTEMPT106BP_BUILDER_W2_SHA256 != ATTEMPT106BP_PUBLIC_W2_SHA256

    class Float32W2Provider(Immutable942cProjectionProvider):
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
            value = super()._project(
                projection, x, assignments, packed, lut_master, su, sv, *_args
            )
            return value.float() if projection == "w2" else value

    config = ResidentRepairAPI.bind_combined_gate_up_projection(
        {}, provider_expert_sha256=PROVIDER_SHA256
    )

    def combined_projection(*args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = args[0]
        return torch.ones_like(x), torch.full_like(x, 2.0)

    wrapped_type = _bind_sealed_gate_up_projection_for_test(
        Float32W2Provider,
        config,
        combined_projection,
        native_down_projection=lambda x, *_args: x,
    )
    provider = wrapped_type()
    provider._sealed_aligned_positions = None
    activated = torch.tensor([[0.006927490234375]], dtype=torch.bfloat16)
    assignments = torch.tensor([0], dtype=torch.int64)
    observed = provider._project(
        "w2",
        activated,
        assignments,
        provider.packed_w2,
        provider.plane_source.wire_lut(),
        provider.su_w2,
        provider.sv_w2,
    )

    assert observed.dtype == activated.dtype
    assert torch.equal(observed, activated)


def test_public_projection_wrapper_calls_native_bf16_w2_before_weighting() -> None:
    """Attempt106bq: call the proven builder F.linear, not FP32-then-cast."""
    assert ATTEMPT106BQ_TERMINAL_SHA256 == (
        "8212b751d5f08a2f80c5d83c3a362779ee43adb54ac45fa42438ef7666a0c6ab"
    )
    assert ATTEMPT106BP_BUILDER_W2_SHA256 != ATTEMPT106BQ_PUBLIC_CAST_W2_SHA256

    native_calls: list[tuple[torch.Tensor, ...]] = []

    def native_down_projection(*args: torch.Tensor) -> torch.Tensor:
        native_calls.append(args)
        activated = args[0]
        return torch.full_like(activated, 7.0)

    config = ResidentRepairAPI.bind_combined_gate_up_projection(
        {}, provider_expert_sha256=PROVIDER_SHA256
    )
    wrapped_type = _bind_sealed_gate_up_projection_for_test(
        Immutable942cProjectionProvider,
        config,
        lambda *args: (torch.ones_like(args[0]), torch.ones_like(args[0])),
        native_down_projection=native_down_projection,
    )
    provider = wrapped_type()
    provider._sealed_aligned_positions = None
    activated = torch.tensor([[0.006927490234375]], dtype=torch.bfloat16)
    assignments = torch.tensor([0], dtype=torch.int64)
    observed = provider._project(
        "w2",
        activated,
        assignments,
        provider.packed_w2,
        provider.plane_source.wire_lut(),
        provider.su_w2,
        provider.sv_w2,
    )

    assert torch.equal(observed, torch.full_like(activated, 7.0))
    assert len(native_calls) == 1
    assert native_calls[0][0] is activated
    assert native_calls[0][1] is assignments
    assert native_calls[0][2] is provider.packed_w2


def test_historical_provider_adapter_preserves_swiglu_limit() -> None:
    """The constructor handoff must retain the accepted model clamp operand."""
    from repair_api.modern_green_resident import _bind_historical_swiglu_limit

    class HistoricalProvider:
        def __init__(self, marker=None):
            self.marker = marker

    adapted = _bind_historical_swiglu_limit(HistoricalProvider)
    provider = adapted(marker="ok", swiglu_limit=0.75)

    assert provider.marker == "ok"
    assert provider.limit == 0.75


def test_historical_provider_adapter_uses_sealed_model_limit() -> None:
    """A legacy trainer may omit the constructor-only limit argument."""
    from repair_api.modern_green_resident import _bind_historical_swiglu_limit

    class HistoricalProvider:
        def __init__(self, marker=None):
            self.marker = marker

    adapted = _bind_historical_swiglu_limit(
        HistoricalProvider, sealed_limit=10.0
    )
    provider = adapted(marker="ok")

    assert provider.marker == "ok"
    assert provider.limit == 10.0


def test_provider_global_projection_clamps_swiglu_operands_before_activation() -> None:
    """Ordinary provider instances must consume the accepted clamped operands."""
    config = ResidentRepairAPI.bind_combined_gate_up_projection(
        {}, provider_expert_sha256=PROVIDER_SHA256
    )

    def combined_projection(*args):
        x = args[0]
        return torch.full_like(x, 2.0), torch.full_like(x, -3.0)

    wrapped_type = _bind_sealed_gate_up_projection_for_test(
        Immutable942cProjectionProvider,
        config,
        combined_projection,
        native_down_projection=lambda x, *_args: x,
    )
    for layer in (0, 1, 42):
        provider = wrapped_type()
        provider.L = layer
        provider.limit = 0.5
        provider._sealed_aligned_positions = None
        hidden = torch.ones((1, 1), dtype=torch.bfloat16)
        assignments = torch.zeros(1, dtype=torch.int64)
        lut = provider.plane_source.wire_lut()
        gate = provider._project(
            "w1", hidden, assignments, provider.packed_w1, lut,
            provider.su_w1, provider.sv_w1,
        )
        up = provider._project(
            "w3", hidden, assignments, provider.packed_w3, lut,
            provider.su_w3, provider.sv_w3,
        )
        assert torch.equal(gate, torch.full_like(gate, 0.5))
        assert torch.equal(up, torch.full_like(up, -0.5))


def test_native_bf16_w2_scope_is_provider_global_and_dtype_guarded() -> None:
    """The repaired W2 path applies identically to every layer instance."""
    config = ResidentRepairAPI.bind_combined_gate_up_projection(
        {}, provider_expert_sha256=PROVIDER_SHA256
    )

    def native_down_projection(x: torch.Tensor, *_args: torch.Tensor) -> torch.Tensor:
        return x.clone()

    wrapped_type = _bind_sealed_gate_up_projection_for_test(
        Immutable942cProjectionProvider,
        config,
        lambda *args: (torch.ones_like(args[0]), torch.ones_like(args[0])),
        native_down_projection=native_down_projection,
    )
    assert wrapped_type._sealed_native_bf16_w2_scope == "provider_class_all_instances_v1"

    for layer in (0, 1, 42):
        provider = wrapped_type()
        provider.L = layer
        provider._sealed_aligned_positions = None
        activated = torch.tensor([[0.006927490234375]], dtype=torch.bfloat16)
        observed = provider._project(
            "w2",
            activated,
            torch.tensor([0], dtype=torch.int64),
            provider.packed_w2,
            provider.plane_source.wire_lut(),
            provider.su_w2,
            provider.sv_w2,
        )
        assert observed.dtype == torch.bfloat16
        assert torch.equal(observed, activated)


def _bind_sealed_gate_up_projection_for_test(
    provider, config, combined_projection, *, native_down_projection=None
):
    from repair_api.modern_green_resident import _bind_sealed_gate_up_projection

    return _bind_sealed_gate_up_projection(
        provider,
        config,
        combined_projection=combined_projection,
        native_down_projection=native_down_projection,
    )


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
