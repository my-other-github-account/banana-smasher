from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
import torch

from repair_api.official_k2_resident_score import OFFICIAL_PHYSICAL_LAYER_SHA256


ROOT = Path(__file__).resolve().parents[2]
CURRENT = ROOT / "control-receipts/t_d2057251/parity_outputs/current_metric.json"
IDENTITY_REFERENCE = ROOT / "control-receipts/t_d2057251/parity_outputs/eager_reference.json"
SEALED_FIXTURE = ROOT / "control-receipts/t_df19ce3f/q8192_win28.pt"
EXPERT_SOURCE = ROOT / "repair_api/assets/fast_v7_expert_base.py"
REQUIRED_TAPS = (
    "ids", "embeddings", *(f"L{layer:03d}" for layer in range(43)),
    "hc_head", "norm", "logits", "q_lp_at_ref", "q_argmax",
)
ZERO_READ_COUNTERS = (
    "timed_model_payload_reads", "fallback_calls", "reconstruction_calls",
    "reference_fwht_calls", "cpu_relay_bytes", "layer_streaming_calls",
)
BF16_RTOL = 1.0e-2
BF16_ATOL = 1.0e-2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_expert_module():
    shim = types.ModuleType("fast_k2_grouped")
    setattr(shim, "grouped_k2_stats", lambda: {})
    setattr(shim, "grouped_packed_projection", lambda *args, **kwargs: None)
    sys.modules["fast_k2_grouped"] = shim
    spec = importlib.util.spec_from_file_location("u0_w28_expert_source", EXPERT_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_u0_window28_q_lp_regression_binds_fixed_bf16_expert_seam() -> None:
    current = json.loads(CURRENT.read_text())
    identity_reference = json.loads(IDENTITY_REFERENCE.read_text())
    sealed_q_lp = torch.load(
        SEALED_FIXTURE, map_location="cpu", mmap=True, weights_only=True
    )["q_lp_at_ref"]

    assert (current["checkpoint"], current["window"], current["mode"]) == (
        "UPDATE_000", 28, "current",
    )
    assert set(current["taps"]) == set(REQUIRED_TAPS)
    # The independently useful upstream identity taps are unchanged for the
    # byte-identical request. The shared-code reference is not used as the
    # q_lp answer key.
    for tap in ("ids", "embeddings"):
        assert current["taps"][tap] == identity_reference["taps"][tap]
    assert all(current["runtime_counters"][name] == 0 for name in ZERO_READ_COUNTERS)

    current_sample = torch.tensor(
        current["taps"]["q_lp_at_ref"]["sample"], dtype=torch.float32
    )
    sealed_sample = sealed_q_lp.reshape(-1)[: current_sample.numel()].float()
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            current_sample, sealed_sample, rtol=BF16_RTOL, atol=BF16_ATOL
        )

    # The first divergent q_lp is downstream of routed-expert accumulation.
    # Pin the production API to the canonical asset and require its public
    # DeepseekV4 BF16 projection/clamp boundaries in focused seam tests.
    assert OFFICIAL_PHYSICAL_LAYER_SHA256 == _sha256(EXPERT_SOURCE)

    module = _load_expert_module()
    expert = module.FullyResidentGroupedV7Experts.__new__(
        module.FullyResidentGroupedV7Experts
    )
    torch.nn.Module.__init__(expert)
    expert.trace_enabled = False
    expert.act = torch.nn.functional.silu
    expert.limit = 10.0
    expert.cpu_relay_bytes = 0
    expert.reconstruction_calls = 0

    class PlaneSource:
        master = torch.zeros(1)

        @staticmethod
        def wire_lut():
            return torch.zeros(1)

    expert.__dict__["plane_source"] = PlaneSource()
    for projection in ("w1", "w2", "w3"):
        setattr(expert, f"packed_{projection}", torch.empty(0))
        setattr(expert, f"su_{projection}", torch.empty(0))
        setattr(expert, f"sv_{projection}", torch.empty(0))
    expert._project = lambda _projection, x, *_args: torch.ones(
        (x.shape[0], 1), dtype=torch.bfloat16
    )

    observed = expert(
        torch.ones((1, 1), dtype=torch.bfloat16),
        torch.tensor([[2, 0, 1]], dtype=torch.int64),
        torch.tensor([[0.1, 0.1, 0.5001]], dtype=torch.float32),
    )
    expected = torch.tensor([[0.703125]], dtype=torch.bfloat16)
    torch.testing.assert_close(observed, expected, rtol=BF16_RTOL, atol=BF16_ATOL)
