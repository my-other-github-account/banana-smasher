from __future__ import annotations

import pytest
import torch

from repair_api import ArtifactError, ResidentRepairAPI


PROVIDER_SHA256 = "c3cf5410d180aeaa321b59d399d2ef8160ce200cd0c85457f51975bb17589c29"
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
    with pytest.raises(ArtifactError, match="requires provider ca554e44"):
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
        native_down_projection=lambda x, *_args: x.float(),
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


def test_public_projection_wrapper_exposes_native_bf16_w2_as_fp32_before_weighting() -> None:
    """Expose BF16-rounded W2 through grouped_mm's FP32 provider buffer."""
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
        native_down_projection=lambda x, *_args: x.float(),
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

    assert observed.dtype == torch.float32
    assert torch.equal(observed, activated.float())


def test_native_down_projection_uses_expert_local_bf16_f_linear(monkeypatch) -> None:
    from repair_api.modern_green_resident import _sealed_builder_native_down_projection

    torch.manual_seed(106)
    hidden = torch.randn(4, 3, dtype=torch.bfloat16)
    assignments = torch.tensor([1, 0, 1, 0], dtype=torch.int64)
    packed_w2 = torch.tensor([0, 1])
    weights = [torch.randn(3, 2, dtype=torch.bfloat16) for _ in range(2)]
    su_w2 = torch.zeros((2, 3))
    sv_w2 = torch.zeros((2, 2))
    original_linear = torch.nn.functional.linear
    linear_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def recording_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        linear_calls.append((x, weight))
        return original_linear(x, weight)

    monkeypatch.setattr(torch.nn.functional, "linear", recording_linear)
    observed = _sealed_builder_native_down_projection(
        hidden, assignments, packed_w2, torch.zeros(1), su_w2, sv_w2,
        full_weight_builder=lambda packed, *_args: weights[int(packed.item())],
    )

    expected = torch.empty((4, 2), dtype=torch.bfloat16)
    for expert in (0, 1):
        mask = assignments == expert
        expected[mask] = original_linear(hidden[mask].contiguous(), weights[expert].T.contiguous())
    assert torch.equal(observed, expected)
    assert observed.dtype == torch.bfloat16
    assert len(linear_calls) == 2
    assert all(x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16 for x, weight in linear_calls)
    assert all(x.is_contiguous() and weight.is_contiguous() for x, weight in linear_calls)


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
        return torch.full_like(activated, 7.0).float()

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


def test_historical_provider_adapter_forwards_required_swiglu_limit() -> None:
    """The compatibility wrapper must not strip a provider's required operand."""
    from repair_api.modern_green_resident import _bind_historical_swiglu_limit

    class CurrentProvider:
        def __init__(self, *, swiglu_limit):
            self.received_limit = swiglu_limit

    adapted = _bind_historical_swiglu_limit(CurrentProvider, sealed_limit=10.0)
    provider = adapted()

    assert provider.received_limit == 10.0
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
        native_down_projection=lambda x, *_args: x.float(),
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
        return x.float()

    wrapped_type = _bind_sealed_gate_up_projection_for_test(
        Immutable942cProjectionProvider,
        config,
        lambda *args: (torch.ones_like(args[0]), torch.ones_like(args[0])),
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
            "w2",
            activated,
            torch.tensor([0], dtype=torch.int64),
            provider.packed_w2,
            provider.plane_source.wire_lut(),
            provider.su_w2,
            provider.sv_w2,
        )
        assert observed.dtype == torch.float32
        assert torch.equal(observed, activated.float())
    assert dispatch_implementations == {
        wrapped_type._sealed_native_bf16_down_projection
    }


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
