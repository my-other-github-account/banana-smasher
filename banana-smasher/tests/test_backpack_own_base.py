from __future__ import annotations

import pytest
import torch

from banana_smasher.backpack_own_base import measure_local_projection_losses


def test_measures_class_conditioned_fused_and_down_function_loss():
    inputs = torch.tensor([[1.0, 2.0], [2.0, -1.0]])
    hessian_weights = torch.tensor([1.0, 4.0])
    class_ids = torch.tensor([0, 1])
    native_fused = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.0, 0.5],
        ]
    )
    native_down = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    tier_fused = native_fused.clone()
    tier_fused[2, 0] = 1.0
    tier_down = native_down.clone()
    tier_down[0, 1] = 0.5

    actual = measure_local_projection_losses(
        inputs=inputs,
        hessian_weights=hessian_weights,
        class_ids=class_ids,
        class_names=("code", "reasoning"),
        native_fused=native_fused,
        native_down=native_down,
        tier_fused=tier_fused,
        tier_down=tier_down,
        swiglu_limit=100.0,
    )

    gate, up = torch.nn.functional.linear(inputs, native_fused).chunk(2, dim=-1)
    native_intermediate = torch.nn.functional.silu(gate) * up
    native_output = torch.nn.functional.linear(native_intermediate, native_down)
    tier_gate, tier_up = torch.nn.functional.linear(inputs, tier_fused).chunk(2, dim=-1)
    fused_output = torch.nn.functional.linear(
        torch.nn.functional.silu(tier_gate) * tier_up,
        native_down,
    )
    down_output = torch.nn.functional.linear(native_intermediate, tier_down)
    expected_fused = 0.5 * hessian_weights * (fused_output - native_output).square().sum(-1)
    expected_down = 0.5 * hessian_weights * (down_output - native_output).square().sum(-1)

    assert actual["fused13"] == pytest.approx(
        {
            "code": expected_fused[0].item(),
            "reasoning": expected_fused[1].item(),
        }
    )
    assert actual["down"] == pytest.approx(
        {
            "code": expected_down[0].item(),
            "reasoning": expected_down[1].item(),
        }
    )


def test_native_operator_measures_exact_zero_against_itself():
    inputs = torch.tensor([[1.0, -1.0]])
    fused = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.0, 0.5],
        ]
    )
    down = torch.eye(2)

    actual = measure_local_projection_losses(
        inputs=inputs,
        hessian_weights=torch.tensor([0.75]),
        class_ids=torch.tensor([0]),
        class_names=("chat",),
        native_fused=fused,
        native_down=down,
        tier_fused=fused,
        tier_down=down,
        swiglu_limit=100.0,
    )

    assert actual == {
        "fused13": {"chat": 0.0},
        "down": {"chat": 0.0},
    }
