from __future__ import annotations

import torch

from banana_smasher import resident_mixed_experts


def test_authenticated_mixed_wire_linear_geometry() -> None:
    """The sealed producer's row orientation matches torch F.linear directly."""

    routed = torch.empty((168, 4096), device="meta")
    fused13 = torch.empty((4096, 4096), device="meta")
    down = torch.empty((4096, 2048), device="meta")

    gate, up = torch.nn.functional.linear(routed, fused13).chunk(2, dim=-1)
    assert gate.shape == up.shape == (168, 2048)
    output = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, down)
    assert output.shape == (168, 4096)


def test_mixed_expert_matches_canonical_swiglu_clamps(monkeypatch) -> None:
    """The resident bridge must preserve DeepseekV4Experts._apply_gate."""

    gate_up = torch.tensor(
        [[[20.0, 0.0], [0.0, 20.0], [30.0, 0.0], [0.0, -30.0]]]
    )
    down = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    class Runtime:
        def _load_vq3u_experts(self, layer):
            assert layer == 0
            return gate_up, down

    class PlaneSource:
        layer = 0
        master = object()

    monkeypatch.setattr(resident_mixed_experts, "_RUNTIME", Runtime())
    module = resident_mixed_experts.FullyResidentGroupedV7Experts(
        0, plane_source=PlaneSource(), swiglu_limit=10.0
    )
    hidden = torch.tensor([[1.0, 1.0]])
    index = torch.tensor([[0]])
    weights = torch.tensor([[1.0]])

    projected = torch.nn.functional.linear(hidden, gate_up[0])
    gate, up = projected.chunk(2, dim=-1)
    actual = module._apply_gate(projected)
    expected = (
        torch.nn.functional.silu(gate.clamp(max=10.0))
        * up.clamp(min=-10.0, max=10.0)
    )
    assert torch.equal(actual, expected)

    unclamped = torch.nn.functional.silu(gate) * up
    assert not torch.equal(actual, unclamped)


def test_mixed_expert_reuses_canonical_grouped_mm_dispatch(monkeypatch) -> None:
    """Resident tensors must traverse the canonical grouped-mm arithmetic rail."""

    experts, hidden_size, intermediate, tokens, top_k = 4, 8, 4, 6, 3
    gate_up = torch.randn(experts, 2 * intermediate, hidden_size)
    down = torch.randn(experts, hidden_size, intermediate)
    hidden = torch.randn(tokens, hidden_size)
    index = torch.randint(0, experts, (tokens, top_k))
    weights = torch.rand(tokens, top_k)

    class Runtime:
        def _load_vq3u_experts(self, layer):
            assert layer == 0
            return gate_up, down

    class PlaneSource:
        layer = 0
        master = object()

    monkeypatch.setattr(resident_mixed_experts, "_RUNTIME", Runtime())
    module = resident_mixed_experts.FullyResidentGroupedV7Experts(
        0, plane_source=PlaneSource(), swiglu_limit=10.0
    )
    expected = torch.randn_like(hidden)
    calls = []

    def grouped(module_arg, hidden_arg, index_arg, weights_arg):
        calls.append((module_arg, hidden_arg, index_arg, weights_arg))
        assert module_arg.num_experts == experts
        assert module_arg.has_gate is True
        assert module_arg.has_bias is False
        assert module_arg.is_transposed is False
        assert module_arg.gate_up_proj is gate_up
        assert module_arg.down_proj is down
        return expected

    monkeypatch.setattr(resident_mixed_experts, "_canonical_grouped_mm_forward", grouped)
    actual = module(hidden, index, weights)

    assert actual is expected
    assert calls == [(module, hidden, index, weights)]
    assert not hasattr(module, "gate_up_proj")
    assert not hasattr(module, "down_proj")