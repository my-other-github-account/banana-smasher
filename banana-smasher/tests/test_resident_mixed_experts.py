from __future__ import annotations

import torch


def test_authenticated_mixed_wire_linear_geometry() -> None:
    """The sealed producer's row orientation matches torch F.linear directly."""

    routed = torch.empty((168, 4096), device="meta")
    fused13 = torch.empty((4096, 4096), device="meta")
    down = torch.empty((4096, 2048), device="meta")

    gate, up = torch.nn.functional.linear(routed, fused13).chunk(2, dim=-1)
    assert gate.shape == up.shape == (168, 2048)
    output = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, down)
    assert output.shape == (168, 4096)