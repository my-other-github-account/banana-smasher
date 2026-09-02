from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import sys
import types

import torch


ROOT = Path(__file__).parents[2]
GROUPED_SOURCE = ROOT / "repair_api" / "assets" / "fast_k2_grouped.py"


def _normalized_hadamard_128(device, dtype):
    matrix = torch.tensor([[1.0]])
    while matrix.shape[0] < 128:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0
        )
    return (matrix / (128**0.5)).to(device=device, dtype=dtype)


def _load_grouped_module():
    codec = types.ModuleType("banana_smasher.q2_codec")
    setattr(codec, "tensor_core_permutation", lambda: list(range(256)))
    qtip = types.ModuleType("banana_smasher.qtip_k2")
    setattr(qtip, "normalized_hadamard_128", _normalized_hadamard_128)
    package = types.ModuleType("banana_smasher")
    package.__path__ = []
    with_modules = {
        "banana_smasher": package,
        "banana_smasher.q2_codec": codec,
        "banana_smasher.qtip_k2": qtip,
    }
    sys.modules.update(with_modules)
    spec = importlib.util.spec_from_file_location(
        "fast_k2_grouped_bf16_boundary_test", GROUPED_SOURCE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_decode_vectorizes_exact_circular_state_recurrence():
    grouped = _load_grouped_module()
    packed = torch.arange(64, dtype=torch.int16).reshape(2, 1, 1, 32)
    lut = torch.linspace(-1.0, 1.0, 1024, dtype=torch.float16)

    codes = grouped._unpack_codes(packed)
    state = torch.zeros(codes.shape[:-1], dtype=torch.int64)
    for position in range(248, 256):
        state = ((state << 2) | codes[..., position]) & 0xFFFF
    states = torch.empty_like(codes, dtype=torch.int64)
    for position in range(256):
        state = ((state << 2) | codes[..., position]) & 0xFFFF
        states[..., position] = state
    products = (states * grouped._MUL1) & 0xFFFFFFFF
    parents = (
        (products & 0xFF)
        + ((products >> 8) & 0xFF)
        + ((products >> 16) & 0xFF)
        + ((products >> 24) & 0xFF)
    )
    decoded = lut[parents].float()[..., grouped._inverse_permutation(torch.device("cpu"))]
    expected = decoded.reshape(2, 1, 1, 16, 16).permute(0, 1, 3, 2, 4).reshape(2, 16, 16)

    assert torch.equal(grouped.direct_decode_matrix(packed, lut), expected)
    source = GROUPED_SOURCE.read_text()
    assert "for position in range(256)" not in source
    assert ".unfold(-1, 8, 1)" in source


def test_sealed_bf16_weight_slab_matches_decode_inverse_transform_boundary():
    grouped = _load_grouped_module()
    packed = torch.zeros((2, 8, 8, 32), dtype=torch.int16)
    lut = torch.zeros(1024, dtype=torch.float32)
    lut[0] = 1.0039
    su = torch.stack(
        (torch.linspace(0.75, 1.25, 128), torch.linspace(1.25, 0.75, 128))
    ).to(torch.float16)
    sv = torch.stack(
        (torch.linspace(1.25, 0.75, 128), torch.linspace(0.75, 1.25, 128))
    ).to(torch.float16)

    observed = grouped.sealed_bf16_weight_slab(packed, lut, su, sv, 0)
    decoded = grouped.direct_decode_matrix(packed, lut)
    hadamard = _normalized_hadamard_128(torch.device("cpu"), torch.float32)
    # Match qtip_k2.inverse_transform exactly: the input-channel scale is
    # applied after the left H128 transform and before the right H128
    # transform.  It cannot be moved across the right transform.
    expected = torch.matmul(hadamard, decoded)
    expected = expected * su.float().unsqueeze(2)
    expected = torch.matmul(expected, hadamard)
    expected = (expected * sv.float().unsqueeze(1)).to(torch.bfloat16)

    assert observed.shape == (2, 128, 128)
    assert observed.dtype == torch.bfloat16
    assert torch.equal(observed, expected)


def test_sealed_full_weight_matches_official_full_matrix_boundary():
    grouped = _load_grouped_module()
    torch.manual_seed(23)
    packed = torch.randint(-(1 << 15), 1 << 15, (8, 16, 32), dtype=torch.int16)
    lut = torch.randn(1024, dtype=torch.float32)
    su = torch.randn(128, dtype=torch.float16)
    sv = torch.randn(256, dtype=torch.float16)

    observed = grouped.sealed_bf16_full_weight(packed, lut, su, sv)
    decoded = grouped.direct_decode_matrix(packed, lut)
    hadamard = _normalized_hadamard_128(torch.device("cpu"), torch.float32)
    expected = torch.matmul(hadamard, decoded.reshape(1, 128, 256)).reshape(128, 256)
    expected = expected * su.float().reshape(128, 1)
    expected = torch.matmul(expected.reshape(128, 2, 128), hadamard).reshape(128, 256)
    expected = (expected * sv.float().reshape(1, 256)).to(torch.bfloat16)

    assert observed.shape == (128, 256)
    assert torch.equal(observed, expected)


def test_grouped_projection_exposes_opt_in_sealed_bf16_physical_path():
    source = GROUPED_SOURCE.read_text()
    assert "FAST_K2_SEALED_FULL_WEIGHT_BF16" in source
    assert "sealed_bf16_weight_slab" in source
    assert "sealed_bf16_full_weight" in source
    assert "torch.matmul" in source
    assert '"sealed_bf16_slab_calls"' in source


def test_sealed_projection_and_expert_boundaries_remain_fp32_until_final_accumulation():
    grouped_source = GROUPED_SOURCE.read_text()
    expert_source = (ROOT / "repair_api" / "assets" / "static_w28_fast_v7_expert_base.py").read_text()

    assert "return sorted_output[inverse_order].float()" in grouped_source
    assert 'FAST_K2_SEALED_PROJECTION_BF16' not in expert_source
    assert "value = value.to(torch.bfloat16).float()" not in expert_source
    assert "gate = gate.clamp(max=self.limit)" not in expert_source
    assert "up = up.clamp(min=-self.limit, max=self.limit)" not in expert_source
    assert "activated = self.act(gate) * up" in expert_source
    assert "routed_output = (" in expert_source
    assert "routed_output * route_weight" in expert_source
    assert "expert_order = torch.argsort(top_k_index, dim=1, stable=True)" in expert_source
    assert "final = (final + ordered_output[:, route_slot]).to(hidden_states.dtype)" in expert_source


def test_static_w28_runtime_hashes_bind_the_provider_files_consumed_by_the_gate():
    from repair_api import modern_green_resident

    assets = ROOT / "repair_api" / "assets"
    wrapper = assets / "static_w28_fast_k2_grouped.py"
    expert = assets / "static_w28_fast_v7_expert_base.py"

    assert hashlib.sha256(wrapper.read_bytes()).hexdigest() == (
        modern_green_resident.STATIC_W28_GROUPED_WRAPPER_SHA256
    )
    assert hashlib.sha256(expert.read_bytes()).hexdigest() == (
        modern_green_resident.STATIC_W28_GROUPED_EXPERT_SHA256
    )


def test_static_resident_expert_preserves_attempt25_ordinary_provider_boundaries():
    """The zero-reconstruction provider must retain accepted pre-logit semantics."""
    static_source = (
        ROOT / "repair_api" / "assets" / "static_w28_fast_v7_expert_base.py"
    ).read_text()
    accepted_boundaries = (
        "activated = self.act(gate) * up",
        "expert_order = torch.argsort(top_k_index, dim=1, stable=True)",
        "final = (final + ordered_output[:, route_slot]).to(hidden_states.dtype)",
    )
    for boundary in accepted_boundaries:
        assert boundary in static_source
    assert "gate = gate.clamp(max=self.limit)" not in static_source
    assert "up = up.clamp(min=-self.limit, max=self.limit)" not in static_source
    assert "self.reconstruction_calls += 0" in static_source
    assert "sealed_bf16_full_weight" not in static_source
