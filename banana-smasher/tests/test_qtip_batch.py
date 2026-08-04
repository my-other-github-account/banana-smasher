from __future__ import annotations

import types

import torch

from banana_smasher.qtip_batch import block_ldl_batch, ldlq_batch


def test_current_k2_cross_unit_cpu_shapes_and_unit_axis_match() -> None:
    generator = torch.Generator().manual_seed(20260804)
    factors = torch.randn((2, 16, 16), generator=generator)
    hessians = factors @ factors.transpose(1, 2) + torch.eye(16) * 0.25
    batched_lower = block_ldl_batch(hessians, 16)
    serial_lower = torch.stack(
        [block_ldl_batch(hessian.unsqueeze(0), 16)[0] for hessian in hessians]
    )
    assert batched_lower.shape == (2, 16, 16)
    assert torch.allclose(batched_lower, serial_lower, atol=2e-5, rtol=2e-5)

    class IdentityCodebook:
        idx_dtype = torch.int32

        def quantize(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            states = torch.zeros((values.shape[0], 128), dtype=torch.int32)
            return values.clone(), states

    weights = torch.randn((2, 16, 128), generator=generator)
    lower = torch.zeros((2, 128, 128))
    args = types.SimpleNamespace(td_x=16, td_y=16, V=2)
    quantized, states = ldlq_batch(
        weights, lower, IdentityCodebook(), args, buf_cols=128
    )
    serial = [
        ldlq_batch(
            weights[unit : unit + 1],
            lower[unit : unit + 1],
            IdentityCodebook(),
            args,
            buf_cols=128,
        )
        for unit in range(2)
    ]

    assert quantized.shape == weights.shape
    assert states.shape == (2, 16, 64)
    assert torch.equal(quantized, torch.cat([row[0] for row in serial]))
    assert torch.equal(states, torch.cat([row[1] for row in serial]))
