from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types

import torch

from repair_api.balanced64 import ArtifactError
from repair_api.official_k2_resident_score import (
    OFFICIAL_PHYSICAL_LAYER_SHA256,
    _configured_attention_implementation,
    _configured_expert_source_sha256,
)


DEFAULT_SOURCE = Path(__file__).parents[1] / "assets" / "fast_v7_expert_base.py"
SOURCE = Path(os.environ.get("FAST_V7_SOURCE", DEFAULT_SOURCE))


def _load_module():
    shim = types.ModuleType("fast_k2_grouped")
    shim.grouped_k2_stats = lambda: {}
    shim.grouped_packed_projection = lambda *args, **kwargs: None
    sys.modules["fast_k2_grouped"] = shim
    spec = importlib.util.spec_from_file_location("fast_v7_expert_base_under_test", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resident_grouped_expert_matches_sealed_bf16_projection_and_swiglu_seam():
    module = _load_module()
    expert = module.FullyResidentGroupedV7Experts.__new__(module.FullyResidentGroupedV7Experts)
    torch.nn.Module.__init__(expert)
    expert.trace_enabled = False
    expert.act = torch.nn.functional.silu
    expert.limit = 10.0
    expert.cpu_relay_bytes = 0
    expert.reconstruction_calls = 0

    class PlaneSource:
        master = torch.zeros(1024)

        @staticmethod
        def wire_lut():
            return torch.zeros(1024)

    expert.__dict__["plane_source"] = PlaneSource()
    for projection in ("w1", "w2", "w3"):
        setattr(expert, f"packed_{projection}", torch.empty(0))
        setattr(expert, f"su_{projection}", torch.empty(0))
        setattr(expert, f"sv_{projection}", torch.empty(0))

    seen = {}

    def project(projection, x, *_args):
        seen[projection] = x.dtype
        if projection == "w1":
            return torch.full((x.shape[0], 1), 20.123, dtype=torch.float32)
        if projection == "w3":
            return torch.full((x.shape[0], 1), 12.123, dtype=torch.float32)
        if projection == "w2":
            return x.float() + 0.123
        raise AssertionError(projection)

    expert._project = project
    observed = expert(
        torch.ones((1, 1), dtype=torch.bfloat16),
        torch.zeros((1, 1), dtype=torch.int64),
        torch.ones((1, 1)),
    )
    gate = torch.tensor(20.123, dtype=torch.float32).to(torch.bfloat16).clamp(max=10.0)
    up = torch.tensor(12.123, dtype=torch.float32).to(torch.bfloat16).clamp(min=-10.0, max=10.0)
    activated = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    expected = (activated.float() + 0.123).to(torch.bfloat16)
    assert torch.equal(observed.squeeze(), expected)
    assert seen == {"w1": torch.bfloat16, "w3": torch.bfloat16, "w2": torch.bfloat16}


def test_published_pre_sealed_expert_uses_unclamped_run1698_swiglu(monkeypatch):
    module = _load_module()
    expert = module.FullyResidentGroupedV7Experts.__new__(module.FullyResidentGroupedV7Experts)
    torch.nn.Module.__init__(expert)
    expert.trace_enabled = False
    expert.act = torch.nn.functional.silu
    expert.limit = 10.0
    expert.cpu_relay_bytes = 0
    expert.reconstruction_calls = 0
    monkeypatch.setenv("FAST_K2_SEALED_NO_SWIGLU_CLAMP", "1")

    class PlaneSource:
        master = torch.zeros(1024)

        @staticmethod
        def wire_lut():
            return torch.zeros(1024)

    expert.__dict__["plane_source"] = PlaneSource()
    for projection in ("w1", "w2", "w3"):
        setattr(expert, f"packed_{projection}", torch.empty(0))
        setattr(expert, f"su_{projection}", torch.empty(0))
        setattr(expert, f"sv_{projection}", torch.empty(0))

    def project(projection, x, *_args):
        if projection == "w1":
            return torch.full((x.shape[0], 1), 20.0)
        if projection == "w3":
            return torch.full((x.shape[0], 1), 12.0)
        return x.float()

    expert._project = project
    observed = expert(
        torch.ones((1, 1), dtype=torch.bfloat16),
        torch.zeros((1, 1), dtype=torch.int64),
        torch.ones((1, 1)),
    )
    expected = (torch.nn.functional.silu(torch.tensor(20.0)) * 12.0).to(torch.bfloat16)
    assert torch.equal(observed.squeeze(), expected)


def test_resident_grouped_expert_matches_sealed_bf16_route_accumulation():
    module = _load_module()
    expert = module.FullyResidentGroupedV7Experts.__new__(module.FullyResidentGroupedV7Experts)
    torch.nn.Module.__init__(expert)
    expert.trace_enabled = False
    expert.act = torch.nn.functional.silu
    expert.limit = 10.0
    expert.cpu_relay_bytes = 0
    expert.reconstruction_calls = 0

    class PlaneSource:
        master = torch.zeros(1024)

        @staticmethod
        def wire_lut():
            return torch.zeros(1024)

    expert.__dict__["plane_source"] = PlaneSource()
    for projection in ("w1", "w2", "w3"):
        setattr(expert, f"packed_{projection}", torch.empty(0))
        setattr(expert, f"su_{projection}", torch.empty(0))
        setattr(expert, f"sv_{projection}", torch.empty(0))

    def project(projection, x, *_args):
        if projection in ("w1", "w3"):
            return torch.ones((x.shape[0], 1), dtype=torch.bfloat16)
        return torch.ones((x.shape[0], 1), dtype=torch.bfloat16)

    expert._project = project
    hidden = torch.ones((1, 1), dtype=torch.bfloat16)
    weights = torch.tensor([[0.3333, 0.5001]], dtype=torch.float32)
    observed = expert(hidden, torch.tensor([[0, 1]]), weights)

    first = (torch.tensor(1.0, dtype=torch.bfloat16).float() * weights[0, 0]).to(torch.bfloat16)
    second = (torch.tensor(1.0, dtype=torch.bfloat16).float() * weights[0, 1]).to(torch.bfloat16)
    expected = (first + second).to(torch.bfloat16)
    assert expected.item() == 0.8359375
    assert torch.equal(observed.squeeze(), expected)


def test_resident_grouped_expert_matches_sealed_expert_major_bf16_sum_order():
    module = _load_module()
    expert = module.FullyResidentGroupedV7Experts.__new__(module.FullyResidentGroupedV7Experts)
    torch.nn.Module.__init__(expert)
    expert.trace_enabled = False
    expert.act = torch.nn.functional.silu
    expert.limit = 10.0
    expert.cpu_relay_bytes = 0
    expert.reconstruction_calls = 0

    class PlaneSource:
        master = torch.zeros(1024)

        @staticmethod
        def wire_lut():
            return torch.zeros(1024)

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
    expected = torch.tensor(0.0, dtype=torch.bfloat16)
    # Sealed DeepseekV4Experts visits hit experts in ascending expert index.
    for value in (0.1, 0.5001, 0.1):
        expected = (
            expected + torch.tensor(value, dtype=torch.float32).to(torch.bfloat16)
        ).to(torch.bfloat16)
    assert expected.item() == 0.703125
    assert torch.equal(observed.squeeze(), expected)


def test_resident_expert_source_sha_is_artifact_bound_and_fail_closed():
    replacement = "53ce8076bb67d0b1e8a15c63fb67ce911651b260d9d687950c54ac04f56f9c0a"
    assert _configured_expert_source_sha256({}) == OFFICIAL_PHYSICAL_LAYER_SHA256
    assert _configured_expert_source_sha256({"official_expert_source_sha256": replacement}) == replacement
    for malformed in ("", "f" * 63, "g" * 64):
        try:
            _configured_expert_source_sha256({"official_expert_source_sha256": malformed})
        except ArtifactError:
            pass
        else:
            raise AssertionError(f"malformed expert source SHA was accepted: {malformed!r}")


def test_resident_attention_default_matches_sealed_eager_builder():
    assert _configured_attention_implementation({}) == "eager"
    assert _configured_attention_implementation({"attention_implementation": "sdpa"}) == "sdpa"
    assert _configured_attention_implementation({
        "attention_implementation": "sdpa",
        "attention_implementation_override": "eager",
    }) == "eager"
    try:
        _configured_attention_implementation({"attention_implementation": "flash_attention_2"})
    except ArtifactError:
        pass
    else:
        raise AssertionError("unadmitted attention implementation was accepted")
