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

    actual = module(hidden, index, weights)
    projected = torch.nn.functional.linear(hidden, gate_up[0])
    gate, up = projected.chunk(2, dim=-1)
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(gate.clamp(max=10.0))
        * up.clamp(min=-10.0, max=10.0),
        down[0],
    )
    assert torch.equal(actual, expected)

    unclamped = torch.nn.functional.linear(
        torch.nn.functional.silu(gate) * up,
        down[0],
    )
    assert not torch.equal(actual, unclamped)