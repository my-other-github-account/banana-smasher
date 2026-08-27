from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import sys
import types

import torch


ROOT = Path(__file__).parents[2]
GROUPED_SOURCE = ROOT / "repair_api" / "assets" / "fast_k2_grouped.py"
STATIC_GROUPED_SOURCE = ROOT / "repair_api" / "assets" / "static_w28_fast_k2_grouped.py"


def _normalized_hadamard_128(device, dtype):
    matrix = torch.tensor([[1.0]])
    while matrix.shape[0] < 128:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0
        )
    return (matrix / (128**0.5)).to(device=device, dtype=dtype)


def _load_grouped_module(source: Path = GROUPED_SOURCE):
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
        f"fast_k2_grouped_bf16_boundary_test_{source.stem}", source
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
    assert observed.stride() == (1, 128)
    assert not observed.is_contiguous()


def test_grouped_projection_exposes_opt_in_sealed_bf16_physical_path():
    source = GROUPED_SOURCE.read_text()
    assert "FAST_K2_SEALED_FULL_WEIGHT_BF16" in source
    assert "sealed_bf16_weight_slab" in source
    assert "sealed_bf16_full_weight" in source
    assert "torch.matmul" in source
    assert '"sealed_bf16_slab_calls"' in source
    expert_source = (ROOT / "repair_api" / "assets" / "fast_v7_expert_base.py").read_text()
    assert "grouped_sealed_gate_up_projection(" in expert_source
    assert "gate_up_weight = torch.cat((gate_weight, up_weight), dim=0)" in source
    assert "torch.nn.functional.linear(" in source
    assert "expert_x = x[assignments == expert_idx]" in source
    assert "sorted_x =" not in source


def test_grouped_gate_up_matches_one_contiguous_builder_linear_per_expert():
    grouped = _load_grouped_module()
    torch.manual_seed(204)
    packed_gate = torch.randint(-(1 << 15), 1 << 15, (2, 8, 16, 32), dtype=torch.int16)
    packed_up = torch.randint(-(1 << 15), 1 << 15, (2, 8, 16, 32), dtype=torch.int16)
    lut = torch.randn(1024, dtype=torch.float32)
    su_gate = torch.randn(2, 128, dtype=torch.float16)
    sv_gate = torch.randn(2, 256, dtype=torch.float16)
    su_up = torch.randn(2, 128, dtype=torch.float16)
    sv_up = torch.randn(2, 256, dtype=torch.float16)
    assignments = torch.tensor([1, 0, 1, 0], dtype=torch.int64)
    hidden = torch.randn(4, 128, dtype=torch.bfloat16)

    gate, up = grouped.grouped_sealed_gate_up_projection(
        hidden, assignments, packed_gate, packed_up, lut,
        su_gate, sv_gate, su_up, sv_up,
    )
    expected = torch.empty((4, 512), dtype=torch.bfloat16)
    for expert in (0, 1):
        mask = assignments == expert
        gate_weight = grouped.sealed_bf16_full_weight(
            packed_gate[expert], lut, su_gate[expert], sv_gate[expert]
        ).transpose(0, 1).contiguous()
        up_weight = grouped.sealed_bf16_full_weight(
            packed_up[expert], lut, su_up[expert], sv_up[expert]
        ).transpose(0, 1).contiguous()
        expected[mask] = torch.nn.functional.linear(
            hidden[mask], torch.cat((gate_weight, up_weight), dim=0)
        )
    expected_gate, expected_up = expected.chunk(2, dim=-1)
    assert torch.equal(gate, expected_gate)
    assert torch.equal(up, expected_up)


def test_r20_runner_full_weight_projection_matches_sealed_gate_tensor_exactly():
    """A7's first BF16 gate divergence is repaired at the GEMM boundary only."""
    import repair_api.resident_full64_accept as runner

    grouped = _load_grouped_module()
    torch.manual_seed(942)
    packed = torch.randint(-(1 << 15), 1 << 15, (2, 8, 16, 32), dtype=torch.int16)
    lut = torch.randn(1024, dtype=torch.float32)
    su = torch.randn(2, 128, dtype=torch.float16)
    sv = torch.randn(2, 256, dtype=torch.float16)
    assignments = torch.tensor([1, 0, 1, 0], dtype=torch.int64)
    hidden = torch.randn(4, 128, dtype=torch.bfloat16)

    observed = runner._sealed_full_weight_projection(
        hidden, assignments, packed, lut, su, sv,
        full_weight_builder=grouped.sealed_bf16_full_weight,
    )
    expected = torch.empty((4, 256), dtype=torch.bfloat16)
    for expert in (0, 1):
        mask = assignments == expert
        weight = grouped.sealed_bf16_full_weight(
            packed[expert], lut, su[expert], sv[expert]
        )
        expected[mask] = torch.nn.functional.linear(hidden[mask], weight.T)

    assert observed.dtype == torch.float32
    assert torch.equal(observed.to(torch.bfloat16), expected)
    runner_source = (ROOT / "repair_api" / "resident_full64_accept.py").read_text()
    assert "_install_r20_full_weight_projection(engine)" in runner_source


def test_static_w28_plane_row_matches_layer_streamed_decode_boundary():
    """The resident consumer must preserve the producer's exact inverse transform."""
    import hashlib
    import repair_api.modern_green_resident as resident

    grouped = _load_grouped_module(STATIC_GROUPED_SOURCE)
    torch.manual_seed(28)
    packed = torch.randint(
        -(1 << 15), 1 << 15, (1, 8, 8, 32), dtype=torch.int16
    )
    lut = torch.linspace(-1.0, 1.0, 1024, dtype=torch.float32)
    su = torch.linspace(0.75, 1.25, 128, dtype=torch.float16).reshape(1, 128)
    sv = torch.linspace(1.25, 0.75, 128, dtype=torch.float16).reshape(1, 128)
    plane_row = torch.linspace(-0.5, 0.5, 128, dtype=torch.float32).reshape(1, 128)
    assignments = torch.zeros(1, dtype=torch.int64)

    observed = grouped.grouped_packed_projection_reference(
        plane_row, assignments, packed, lut, su, sv
    )
    # Exact builder_B2_PUBLISHED_PRE/official_local_planesource boundary:
    # inverse_transform applies H128 first, then su, then decoded K2, then H128/sv.
    transformed = grouped.block_hadamard_128(plane_row.float()) * su.float()
    decoded = grouped.direct_decode_matrix(packed[0], lut.to(torch.float16))
    expected = grouped.block_hadamard_128(transformed @ decoded) * sv.float()

    assert torch.equal(observed, expected)
    source = STATIC_GROUPED_SOURCE.read_text()
    assert hashlib.sha256(STATIC_GROUPED_SOURCE.read_bytes()).hexdigest() == (
        resident.STATIC_W28_GROUPED_WRAPPER_SHA256
    )
    assert "block_hadamard_128(x.float()) * su[assignments].float()" in source
    assert "block_hadamard_128(sorted_x.float())" in source
    assert "sorted_x.float() * su[sorted_assignments].float()" not in source


def test_static_w28_one_input_matches_full_streamed_bf16_tensor_exactly():
    """Ported static boundary must match the eager physical weight before CUDA."""
    import hashlib
    import repair_api.modern_green_resident as resident

    eager = _load_grouped_module(GROUPED_SOURCE)
    static = _load_grouped_module(STATIC_GROUPED_SOURCE)
    torch.manual_seed(32)
    packed = torch.randint(-(1 << 15), 1 << 15, (8, 16, 32), dtype=torch.int16)
    lut = torch.randn(1024, dtype=torch.float32)
    su = torch.randn(128, dtype=torch.float16)
    sv = torch.randn(256, dtype=torch.float16)
    one_input = torch.linspace(-0.5, 0.5, 4 * 128).reshape(4, 128).to(torch.bfloat16)

    eager_weight = eager.sealed_bf16_full_weight(packed, lut, su, sv)
    static_weight = static.sealed_bf16_full_weight(packed, lut, su, sv)
    eager_output = torch.matmul(one_input, eager_weight)
    static_output = torch.matmul(one_input, static_weight)

    assert eager_output.shape == static_output.shape == (4, 256)
    assert eager_output.stride() == static_output.stride()
    assert torch.equal(static_weight, eager_weight)
    assert torch.equal(static_output, eager_output)
    source = STATIC_GROUPED_SOURCE.read_text()
    assert hashlib.sha256(STATIC_GROUPED_SOURCE.read_bytes()).hexdigest() == (
        resident.STATIC_W28_GROUPED_WRAPPER_SHA256
    )
    assert 'FAST_K2_SEALED_FULL_WEIGHT_BF16' in source
    assert "return sorted_output[inverse_order].float()" in source
    runner = (ROOT / "repair_api" / "resident_full64_accept.py").read_text()
    assert 'os.environ["FAST_K2_SEALED_FULL_WEIGHT_BF16"] = "1"' in runner
    assert 'os.environ["FAST_K2_SEALED_FULL_WEIGHT_BF16"] = "0"' not in runner


def test_static_route_metadata_reuses_sorted_input_and_work_rows_geometry():
    grouped = _load_grouped_module(STATIC_GROUPED_SOURCE)
    assignments = torch.tensor([3, 1, 3, 0, 1, 3], dtype=torch.int64)
    inputs = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    metadata = grouped.grouped_route_metadata(assignments, 4, input_tensor=inputs)
    legacy_order = torch.argsort(assignments, stable=True)

    assert torch.equal(metadata["order"], legacy_order)
    assert torch.equal(metadata["inverse_order"], torch.argsort(legacy_order))
    assert torch.equal(metadata["sorted_assignments"], assignments[legacy_order])
    assert torch.equal(metadata["sorted_input"], inputs[legacy_order])
    assert metadata["assignments_data_ptr"] == assignments.data_ptr()
    assert grouped.grouped_k2_stats()["route_metadata_builds"] == 1
    assert "counts + WORK_ROWS - 1, WORK_ROWS" in STATIC_GROUPED_SOURCE.read_text()

    expert_source = (
        ROOT / "repair_api" / "assets" / "static_w28_fast_v7_expert_base.py"
    ).read_text()
    assert expert_source.count("route_metadata=route_metadata") == 2
    assert expert_source.count("route_metadata,") >= 3
    assert "expert_index, EXPERTS, input_tensor=routed_hidden" in expert_source


def test_static_expert_refuses_any_forward_wider_than_the_exact_mb2_pair():
    expert_source = (
        ROOT / "repair_api" / "assets" / "static_w28_fast_v7_expert_base.py"
    ).read_text()
    assert "hidden_states.shape[0] > sealed_group_tokens" in expert_source
    assert "admits only one exact sealed mb2 pair per forward" in expert_source
    assert "FAST_K2_EXPERT_STREAM_CONCURRENCY" not in expert_source


def test_sealed_projection_and_expert_boundaries_remain_fp32_until_final_accumulation():
    grouped_source = GROUPED_SOURCE.read_text()
    expert_source = (ROOT / "repair_api" / "assets" / "static_w28_fast_v7_expert_base.py").read_text()

    assert "return sorted_output[inverse_order].float()" in grouped_source
    assert 'FAST_K2_SEALED_PROJECTION_BF16' in expert_source
    assert "value = value.to(torch.bfloat16).float()" in expert_source


def test_static_expert_tensor_parallelism_slices_every_projection_before_output_reduce():
    expert_source = (
        ROOT / "repair_api" / "assets" / "static_w28_fast_v7_expert_base.py"
    ).read_text()
    assert "tensor_parallel_rank" in expert_source
    assert "tensor_parallel_world_size" in expert_source
    assert "def configure_tensor_parallel(" in expert_source
    assert "self.packed_w1[:, :, block_lo:block_hi, :].contiguous()" in expert_source
    assert "self.packed_w3[:, :, block_lo:block_hi, :].contiguous()" in expert_source
    assert "self.packed_w2[:, block_lo:block_hi, :, :].contiguous()" in expert_source
    assert "self.su_w2[:, intermediate_lo:intermediate_hi].contiguous()" in expert_source
    assert "torch.distributed.all_reduce(local_routed_output, group=self.tensor_parallel_group)" in expert_source
    assert "local_route_mask" not in expert_source
    assert "gate = gate.clamp(max=self.limit)" not in expert_source
    assert "up = up.clamp(min=-self.limit, max=self.limit)" not in expert_source
    assert "activated = self.act(gate) * up" in expert_source
    assert "local_routed_output = (local_routed_output * route_weight).to(hidden_states.dtype)" in expert_source


def test_resident_routed_return_matches_authentic_provider_bf16_weighting():
    expert_source = (ROOT / "repair_api" / "assets" / "fast_v7_expert_base.py").read_text()
    assert "route_weight = top_k_weights.reshape(-1, 1)" in expert_source
    assert "route_weight = top_k_weights.reshape(-1, 1).float()" not in expert_source
    assert (
        "routed_output = routed_output * route_weight.to(dtype=routed_output.dtype)"
        in expert_source
    )
    assert "routed_output.view(" in expert_source
    assert "hidden_states.shape[0], top_k_index.shape[1], hidden_states.shape[-1]" in expert_source
    assert ").sum(dim=1)" in expert_source
    assert "final.index_add_(0, token_index, routed_output)" not in expert_source
    assert "return final.to(hidden_states.dtype)" in expert_source
    assert "top_k_index.transpose(0, 1)" not in expert_source

    down = torch.tensor([0.006927490234375], dtype=torch.bfloat16)
    route_weight = torch.tensor([0.3333], dtype=torch.float32)
    authentic = down * route_weight.to(dtype=down.dtype)
    widened = (down * route_weight).to(dtype=down.dtype)
    assert authentic.item() == 0.0023193359375
    assert widened.item() == 0.0023040771484375
    assert not torch.equal(authentic, widened)


def test_static_tp_preserves_sealed_expert_ordered_bf16_accumulation():
    expert_source = (
        ROOT / "repair_api" / "assets" / "static_w28_fast_v7_expert_base.py"
    ).read_text()
    assert "for expert_idx in torch.unique(expert_index, sorted=True):" in expert_source
    assert "final.index_add_(" in expert_source
    assert "expert_order = torch.argsort(top_k_index" not in expert_source
    assert "final = (final + ordered_output[:, route_slot]).to(hidden_states.dtype)" not in expert_source


def test_static_complete_expert_boundary_preserves_full_width_bf16_semantics():
    expert_source = (
        ROOT / "repair_api" / "assets" / "static_w28_fast_v7_expert_base.py"
    ).read_text()
    runner_source = (ROOT / "repair_api" / "resident_full64_accept.py").read_text()

    assert 'FAST_K2_SEALED_COMPLETE_EXPERT_BF16' in expert_source
    assert 'value = value.to(torch.bfloat16)' in expert_source
    assert 'if self.sealed_complete_expert_bf16:' in expert_source
    assert 'self.tensor_parallel_world_size = 1' in expert_source
    assert 'return' in expert_source
    assert 'os.environ["FAST_K2_SEALED_COMPLETE_EXPERT_BF16"] = "1"' in runner_source
    from repair_api import modern_green_resident
    expert_path = ROOT / "repair_api" / "assets" / "static_w28_fast_v7_expert_base.py"
    assert hashlib.sha256(expert_path.read_bytes()).hexdigest() == (
        modern_green_resident.STATIC_W28_GROUPED_EXPERT_SHA256
    )


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
        "for expert_idx in torch.unique(expert_index, sorted=True):",
        "final.index_add_(",
    )
    for boundary in accepted_boundaries:
        assert boundary in static_source
    assert "gate = gate.clamp(max=self.limit)" not in static_source
    assert "up = up.clamp(min=-self.limit, max=self.limit)" not in static_source
    assert "self.reconstruction_calls += 0" in static_source
    assert "sealed_bf16_full_weight" not in static_source


def test_exact_accepted_provider_is_kept_duplicated_without_tp_configuration() -> None:
    from repair_api import modern_green_resident

    config = {
        "expert_parallel_all_layers": True,
        "recipe_id": modern_green_resident.PUBLISHED_PRE_RECIPE_ID,
        "static_w28_gate": {},
        "sealed_pre_source_binding": {
            "builder_sha256": "d66890669faa578339a8f3fa6a4c23617fbe925c0d0ac6e38fd9481ad0cd7026",
            "planesource_sha256": "167603b5662437a2f9fc4b3ead1561d777a7a831a898133993b9e1c0c26c9f87",
        },
        "resident_validation_expert_implementation": "sealed_bf16_full_weight",
    }

    assert modern_green_resident._configure_resident_tensor_parallel(
        config, {0: object()}, rank=0
    ) == "exact-accepted-0eeb-duplicated-all43-no-tp"
