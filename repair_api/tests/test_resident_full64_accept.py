from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, cast

import torch

from repair_api.resident_full64_accept import (
    ADOPTED_PROVIDER_EXPERT_SHA256,
    ADOPTED_PROVIDER_WRAPPER_SHA256,
    BASIS,
    CHECKPOINT,
    CURRENT_PROVIDER_EXPERT_SHA256,
    CURRENT_PROVIDER_WRAPPER_SHA256,
    _resolve_config_path,
    atomic,
    validate_full64_batches,
)
from repair_api.modern_green_resident import ModernGreenResidentEngine


def test_resident_receipt_atomic_creates_missing_parent() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "receipts" / "receipt.json"
        digest = atomic(path, {"status": "PASS"})
        assert path.exists()
        assert len(digest) == 64


def test_full64_batches_resume_parent_prefix_then_compute_only_suffix() -> None:
    windows = (28, 56, 68, 71)
    zero = {name: 0 for name in (
        "checkpoint_reloads", "fallback_calls", "reconstruction_calls",
        "timed_model_payload_reads", "timed_score_file_reads",
    )}
    parent_binding = {
        "provider_wrapper_sha256": ADOPTED_PROVIDER_WRAPPER_SHA256,
        "provider_expert_sha256": ADOPTED_PROVIDER_EXPERT_SHA256,
        "geometry": "sealed-mb2",
    }
    child_binding = {
        "provider_wrapper_sha256": CURRENT_PROVIDER_WRAPPER_SHA256,
        "provider_expert_sha256": CURRENT_PROVIDER_EXPERT_SHA256,
        "geometry": "sealed-mb2",
    }

    def measurement(batch: tuple[int, ...], binding: dict[str, str]) -> dict[str, Any]:
        return {
            "windows": list(batch),
            "per_window": [
                {"window": window, "positions": 1, "kld_sum_binary64": float(window), "top1": 1}
                for window in batch
            ],
            "runtime_counters": zero,
            "phase_profiles_by_rank": [[], []],
            "validation_corpus_sha256": "corpus",
            "validation_teacher_sha256_by_window": {str(window): f"teacher-{window}" for window in batch},
            "sealed_builder_binding": binding,
        }

    with tempfile.TemporaryDirectory() as directory:
        receipts = Path(directory)
        prefix = {
            "schema": "banana-smasher-resident-full64-batch-v1", "status": "PASS",
            "task_id": "t_8b1b3a3f", "rank": 0,
            "canonical_code_commit": "ae27abc53f3ca69f6efa9a64c1f6e1d4f0193d1e",
            "basis_sha256": BASIS, "checkpoint_sha256": CHECKPOINT,
            "batch_index": 0, "windows": [28, 56],
            "measurement": measurement((28, 56), parent_binding),
        }
        atomic(receipts / "FULL64_BATCH.rank0.00.json", prefix)

        class API:
            calls: list[tuple[int, ...]] = []
            def validate(self, engine: Any, batch: tuple[int, ...], teacher_root: Path) -> dict[str, Any]:
                self.calls.append(batch)
                return measurement(batch, child_binding)

        class Engine:
            released: list[tuple[int, ...]] = []
            def release_validation_inputs(self, batch: tuple[int, ...], teacher_root: Path) -> bool:
                self.released.append(batch)
                return True

        api, engine = API(), Engine()
        result = validate_full64_batches(
            api, engine, windows, Path(directory) / "teacher", receipts,
            rank=0, canonical_code_commit="f" * 40,
            adopted_prefix_code_commit="ae27abc53f3ca69f6efa9a64c1f6e1d4f0193d1e",
            batch_size=2,
        )
        assert result["resumed_batch_count"] == 1
        assert result["computed_batch_count"] == 1
        assert api.calls == [(68, 71)]
        assert engine.released == [(68, 71)]
        assert result["windows"] == list(windows)


def test_resident_full64_accept_uses_explicit_authenticated_attempt_config(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        selected = root / "CONFIG.t_5e0f4049.attempt16.rank1.json"
        selected.write_text("{}\n")
        stale = root / "CONFIG.t_5e0f4049.rank1.json"
        stale.write_text("{}\n")
        monkeypatch.setenv("BANANA_SMASHER_CONFIG_PATH", str(selected))
        assert _resolve_config_path(root, task="t_5e0f4049", rank=1) == selected
        outside = root.parent / "foreign-config.json"
        monkeypatch.setenv("BANANA_SMASHER_CONFIG_PATH", str(outside))
        try:
            _resolve_config_path(root, task="t_5e0f4049", rank=1)
        except RuntimeError as error:
            assert str(error) == "CONFIG_PATH_OUTSIDE_PHYSICAL_ROOT"
        else:
            raise AssertionError("foreign config path was accepted")


def test_resident_full64_accept_is_single_load_one_static_admission_then_production() -> None:
    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    assert source.count('checkpoint_path("PRE")') == 1
    assert source.count("torch.load(checkpoint_path") == 1
    assert source.count("ModernGreenResidentEngine(") == 1
    assert source.count("api.validate(engine") == 2
    assert source.count('config["sealed_builder_window_microbatch"] = 2') >= 1
    assert 'config["sealed_builder_window_microbatch"] = 4' not in source
    assert 'config["sealed_builder_window_microbatch"] = 8' not in source
    assert 'config["score_window_batch_size"] = 2' in source
    assert 'config["score_pair_stream_concurrency"] = 1' in source
    assert source.index('config["score_pair_stream_concurrency"] = 1') < source.index("ModernGreenResidentEngine(")
    assert source.index('config["score_window_batch_size"] = 2') < source.index("ModernGreenResidentEngine(")
    assert 'config["score_pipeline_overlap"] = True' in source
    assert 'config["attention_query_chunk_size"] = 512' in source
    assert 'config.pop("attention_query_chunk_size", None)' not in source
    eager = source.index('config["resident_validation_attention_implementation"] = "eager"')
    admission = source.index("api.validate(engine, (28,)")
    packed = source.index(
        'config["resident_validation_expert_implementation"] = "packed_cuda_bf16_boundary"',
        admission,
    )
    production = source.index("validate_full64_batches(", admission)
    assert eager < admission < packed < production
    assert 'adopted_prefix_code_commit="ae27abc53f3ca69f6efa9a64c1f6e1d4f0193d1e"' in source
    assert 'full = api.validate(engine, windows' not in source
    assert 'PACKED_W28_CANARY' not in source
    assert 'projected_full64_wall >= 300.0' not in source
    assert "Concurrency is therefore only across intact" in source
    assert 'W28_KLD = 0.13712959240533734' in source
    assert "W28_TOP1 = 877" in source
    assert 'post_load_wall >= 300.0' in source
    assert 'banana-smasher-resident-full64-rate-low-v2' in source


def test_production_planesource_provider_is_bound_before_engine_construction() -> None:
    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    planesource = source.index(
        'config["resident_validation_expert_implementation"] = "sealed_bf16_full_weight"'
    )
    engine = source.index("ModernGreenResidentEngine(")
    assert planesource < engine
    # Imported W28 acceptance is receipt-only, so the single resident engine
    # must be constructed with the production provider it will actually run.
    assert source.count(
        'config["resident_validation_expert_implementation"] = "sealed_bf16_full_weight"'
    ) == 1


def test_production_restores_sealed_planesource_weight_boundary_after_adoption() -> None:
    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    boundary = source.index('os.environ["FAST_K2_SEALED_FULL_WEIGHT_BF16"] = "1"')
    engine = source.index("ModernGreenResidentEngine(")
    production = source.index('config["score_window_batch_size"] = 2')
    assert boundary < engine < production
    assert source.count('os.environ["FAST_K2_SEALED_FULL_WEIGHT_BF16"] = "1"') == 1


def test_accepted_separate_projection_decoder_receives_fp16_parent_lut() -> None:
    root = Path(__file__).parents[1]
    provider = (root / "assets" / "static_w28_modern_green_clean_u0.py").read_text()
    engine = (root / "modern_green_resident.py").read_text()
    wire_lut = provider[provider.index("    def wire_lut(self):") : provider.index("\n\n\nclass ResidentOfficialExperts")]
    # Keep the immutable producer bytes/hash: its master LUT is rounded through
    # fp16 then exposed as fp32 for training. The accepted decoder itself has a
    # strict fp16[1024] contract, so the production adapter restores that dtype
    # at its call boundary without changing LUT values.
    assert ".to(dtype=self.torch.float16).to(dtype=self.torch.float32)" in wire_lut
    adapter = engine[engine.index("class SealedPlaneSourceExperts") :]
    adapter = adapter[: adapter.index("self.trainer.FullyResidentGroupedV7Experts")]
    assert "accepted_wire_lut = plane_source.wire_lut" in adapter
    assert "return accepted_wire_lut().to(dtype=torch_module.float16)" in adapter
    assert "plane_source.wire_lut = decoder_wire_lut" in adapter


def test_sealed_planesource_restores_combined_native_swiglu_constructor() -> None:
    root = Path(__file__).parents[1]
    provider = (root / "assets" / "static_w28_modern_green_clean_u0.py").read_text()
    engine = (root / "modern_green_resident.py").read_text()
    assert 'TRAINER_SHA256 = "a662482336a61cd258861a1ce6f13007dcf82497104499c97f3839f443583687"' in engine
    resident = provider[provider.index("class ResidentOfficialExperts") :]
    resident = resident[: resident.index("class ResidentDenseL034")]
    # The sealed producer performs one fused gate/up GEMM and then applies the
    # model's native SwiGLU clamp before SiLU.  The public adapter must preserve
    # that constructor value instead of silently defaulting the limit to zero.
    signature = resident[resident.index("def __init__(") : resident.index("from torch import nn")]
    assert "swiglu_limit: float" in signature
    assert "gate = gate.clamp(max=mod.swiglu_limit)" in resident
    assert "up = up.clamp(min=-mod.swiglu_limit, max=mod.swiglu_limit)" in resident
    assert 'model_config.get("swiglu_limit", float("nan"))' in engine
    adapter = engine[engine.index("class SealedPlaneSourceExperts") :]
    adapter = adapter[: adapter.index("self.trainer.FullyResidentGroupedV7Experts")]
    assert "swiglu_limit: float = sealed_swiglu_limit" in adapter
    assert "swiglu_limit=swiglu_limit" in adapter
    assert 'if expert_implementation == "sealed_bf16_full_weight" and hasattr(self, "trainer"):' in engine


def test_sealed_planesource_combines_existing_w1_w3_payloads_at_w13_boundary() -> None:
    provider = (Path(__file__).parents[1] / "assets" / "static_w28_modern_green_clean_u0.py").read_text()
    resident = provider[provider.index("class ResidentOfficialExperts") :]
    resident = resident[: resident.index("class ResidentDenseL034")]
    assert 'if projection == "w13":' in resident
    assert 'mod.payloads[(expert, "w1")]' in resident
    assert 'mod.payloads[(expert, "w3")]' in resident
    assert 'lut = mod.plane_source.wire_lut().to(dtype=torch.float16)' in resident
    assert "torch.cat((gate_weight, up_weight), dim=0)" in resident
    assert 'project(mod, hidden.unsqueeze(0), expert, "w13")' in resident


def test_resident_full64_accept_binds_the_dispatched_task_from_environment():
    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    assert 'TASK = os.environ.get("BANANA_SMASHER_TASK_ID", "t_d4dac464")' in source
    assert 'W28_ADOPTION_TASK = "t_8b1b3a3f"' in source
    assert 'row.get("task_id") != W28_ADOPTION_TASK' in source
    assert 'FULL64_REQUIRES_ACCEPTED_PROVIDER' in source
    assert 'len(rows) != 64' in source
    assert 'checkpoint_reloads' in source
    assert 'per_window_diff' in source
    assert 'reference_terminal_sha256' in source


def test_sdpa_repair_retains_attention_sink_denominator() -> None:
    source = (Path(__file__).parents[1] / "modern_green_resident.py").read_text()
    assert "scaled_dot_product_attention" in source
    assert "sink_key[..., width]" in source
    assert "sink_value = value_aug.new_zeros" in source
    assert "output[..., :width]" in source
    assert "official_k2_sink_corrected_sdpa" in source


def test_packed_boundary_tap_is_diagnostic_and_first_projection_only() -> None:
    root = Path(__file__).parents[1]
    runner = (root / "resident_full64_accept.py").read_text()
    expert = (root / "assets" / "static_w28_fast_v7_expert_base.py").read_text()
    assert "PACKED_BOUNDARY_TAP_ONLY" in runner
    assert 'config["resident_validation_expert_implementation"] = "packed_cuda_bf16_boundary"' in runner
    assert 'self.L == 0 and projection == "w1"' in expert
    assert "PASS_FIRST_DIVERGENCE_L000_W1_PROJECTION_BF16_ROUND" in expert


def test_whole_chain_bisect_compares_product_and_reference_cache_handoffs():
    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    assert "WHOLE_CHAIN_BISECT_ONLY" in source
    assert "def _whole_chain_bisect" in source
    assert "product_taps, _product_hidden = _capture_product_layer_taps(" in source
    assert 'reference_taps = _capture_reference_layer_taps(engine, hidden.clone(), ids)' in source
    assert 'active_cache = DynamicCache(config=engine.student.config)' in source
    assert 'entry.keys = entry.keys.new_empty((0,))' in source
    assert 'entry.values = entry.values.new_empty((0,))' in source
    assert '"first_divergent_layer": first_divergent' in source
    assert "with torch.no_grad():" in source


def test_readout_binding_ab_localizes_post_l042_rank_handoff():
    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    assert "READOUT_BINDING_AB_ONLY" in source
    assert "def _readout_binding_ab" in source
    assert 'stage_order = ("L042", "hc_head", "norm", "logits", "q_lp_at_ref", "q_argmax")' in source
    assert '"product_source": "repair_api/modern_green_resident.py:2130-2178 (rank1 readout)"' in source
    assert '"diagnostic_source": "repair_api/official_k2_resident_score.py:2782-2831 (sealed parity readout)"' in source
    assert '"attempt37b_receipts": ["5cc1909b86095d5434bcbc1b764381776a5a8c2459e7dec29528fd91b0ba8855", "1992ab3c66af23c54a7f475d86ebe3d9e5dce1373b0134bfd361d6994ba68840"]' in source


def test_sink_token_sdpa_matches_eager_sink_equation() -> None:
    torch.manual_seed(7)
    query = torch.randn(1, 2, 4, 8)
    key = torch.randn(1, 2, 4, 8)
    value = torch.randn(1, 2, 4, 8)
    mask = torch.full((1, 1, 4, 4), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    module = SimpleNamespace(sinks=torch.tensor([0.25, -0.5]), num_key_value_groups=1)
    scaling = 8 ** -0.5
    observed, _ = ModernGreenResidentEngine._sink_corrected_sdpa_forward(
        module, query, key, value, mask, scaling,
    )
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling + mask
    sinks = module.sinks.reshape(1, 2, 1, 1).expand(1, 2, 4, 1)
    probabilities = torch.softmax(torch.cat((scores, sinks), dim=-1), dim=-1)[..., :-1]
    expected = torch.matmul(probabilities, value).transpose(1, 2).contiguous()
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)


def test_chunked_eager_attention_is_bitwise_and_bounds_query_workspace() -> None:
    torch.manual_seed(11)
    engine = cast(Any, ModernGreenResidentEngine.__new__(ModernGreenResidentEngine))
    engine.torch = torch
    engine._attention_workspace = None
    module = SimpleNamespace(num_key_value_groups=2, sinks=torch.tensor([0.25, -0.5]))
    query = torch.randn((4, 2, 9, 5), dtype=torch.bfloat16)
    key = torch.randn((4, 1, 11, 5), dtype=torch.bfloat16)
    value = torch.randn((4, 1, 11, 5), dtype=torch.bfloat16)
    mask = torch.zeros((4, 1, 9, 11), dtype=torch.bfloat16)
    repeated_key = key[:, :, None].expand(4, 1, 2, 11, 5).reshape(4, 2, 11, 5)
    repeated_value = value[:, :, None].expand(4, 1, 2, 11, 5).reshape(4, 2, 11, 5)
    weights = torch.matmul(query, repeated_key.transpose(2, 3)) * (5**-0.5) + mask
    sinks = module.sinks.reshape(1, -1, 1, 1).expand(4, -1, 9, -1)
    logits = torch.cat([weights, sinks], dim=-1)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    probabilities = torch.nn.functional.softmax(logits, dim=-1, dtype=logits.dtype)[..., :-1]
    expected = torch.matmul(probabilities.to(repeated_value.dtype), repeated_value).transpose(1, 2).contiguous()
    observed_chunks = []
    observed, returned_weights = engine._chunked_eager_attention_forward(
        module, query, key, value, mask, scaling=5**-0.5,
        query_chunk_size=3, _chunk_observer=observed_chunks.append,
        _resident_workspace_factory=engine._attention_workspace_for,
    )
    assert torch.equal(observed, expected)
    assert returned_weights is None
    assert observed_chunks == [3, 3, 3]
