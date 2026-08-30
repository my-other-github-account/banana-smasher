from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _runner_module():
    path = Path(__file__).parents[1] / "vendor/site/banana_smasher/update_backends/joint_v7_repair.py"
    spec = importlib.util.spec_from_file_location("joint_v7_repair_official_gain_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_output_gain_matches_official_builder_bf16_operand_order():
    runner = _runner_module()
    linear = torch.nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        linear.weight.copy_(
            torch.tensor(
                [[0.333, -0.219, 0.117], [-0.443, 0.281, 0.097]],
                dtype=torch.bfloat16,
            )
        )
    log_gain = torch.nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
    linear.register_parameter("_banana_smasher_output_log_gain", log_gain)
    wire_before = linear.weight.detach().clone()
    multiplier_bf16 = torch.exp(log_gain.detach()).to(torch.bfloat16)
    expected = wire_before * multiplier_bf16
    high_precision_then_cast = (
        wire_before.to(torch.float64) * torch.exp(log_gain.detach().to(torch.float64))
    ).to(torch.bfloat16)
    assert not torch.equal(expected, high_precision_then_cast)

    receipt = runner.materialize_official_builder_output_gains(
        torch,
        [("model.layers.0.self_attn.o_b_proj.output_log_gain", linear, log_gain)],
    )

    assert torch.equal(linear.weight, expected)
    assert log_gain.item() == 0.0
    assert receipt[0]["multiplier_bf16"] == float(multiplier_bf16)
    assert receipt[0]["wire_dtype"] == "torch.bfloat16"


def test_official_output_gain_refuses_non_bf16_weight():
    runner = _runner_module()
    linear = torch.nn.Linear(2, 2, bias=False, dtype=torch.float32)
    gain = torch.nn.Parameter(torch.tensor(0.01))
    linear.register_parameter("_banana_smasher_output_log_gain", gain)
    with pytest.raises(RuntimeError, match="requires a BF16 weight matrix"):
        runner.materialize_official_builder_output_gains(
            torch,
            [("bad.output_log_gain", linear, gain)],
        )
