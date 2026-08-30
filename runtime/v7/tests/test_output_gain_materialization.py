from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _runner_module():
    path = (
        Path(__file__).parents[1]
        / "vendor/site/banana_smasher/update_backends/joint_v7_repair.py"
    )
    spec = importlib.util.spec_from_file_location("joint_v7_repair_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_output_gain_is_folded_before_projection_like_serving_export():
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
    expected_wire = (
        wire_before.to(torch.float64) * torch.exp(log_gain.detach().to(torch.float64))
    ).to(torch.bfloat16)

    receipt = runner.materialize_output_log_gains(
        torch,
        [("model.layers.0.self_attn.o_b_proj.output_log_gain", linear, log_gain)],
    )

    assert torch.equal(linear.weight, expected_wire)
    assert log_gain.item() == 0.0
    assert receipt[0]["name"].endswith(".output_log_gain")
    assert receipt[0]["wire_dtype"] == "torch.bfloat16"


def test_output_gain_materialization_refuses_missing_weight():
    runner = _runner_module()
    module = torch.nn.Identity()
    log_gain = torch.nn.Parameter(torch.tensor(0.01))
    module.register_parameter("_banana_smasher_output_log_gain", log_gain)
    with pytest.raises(RuntimeError, match="requires a weight matrix"):
        runner.materialize_output_log_gains(torch, [("bad.output_log_gain", module, log_gain)])