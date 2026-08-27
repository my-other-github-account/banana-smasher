from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import torch

from repair_api.resident_full64_accept import (
    ADOPTED_PROVIDER_EXPERT_SHA256,
    ADOPTED_PROVIDER_WRAPPER_SHA256,
    BASIS,
    CHECKPOINT,
    CURRENT_PROVIDER_EXPERT_SHA256,
    CURRENT_PROVIDER_WRAPPER_SHA256,
    CUDA_MEMORY_FRACTION,
    _apply_cuda_memory_fraction,
    _resolve_config_path,
    atomic,
    validate_scheduled_pair_group,
)
from repair_api.modern_green_resident import (
    ModernGreenResidentEngine,
    _builder_frame_readout_logits,
)


def test_resident_receipt_atomic_creates_missing_parent() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "receipts" / "receipt.json"
        digest = atomic(path, {"status": "PASS"})
        assert path.exists()
        assert len(digest) == 64


def test_builder_frame_readout_projects_full_real_length_before_score_slice() -> None:
    calls: list[tuple[int, ...]] = []

    class Model:
        @staticmethod
        def lm_head(value: torch.Tensor) -> torch.Tensor:
            calls.append(tuple(value.shape))
            return torch.arange(value.shape[0] * 7, dtype=torch.bfloat16).reshape(
                value.shape[0], 7
            )

    final = torch.zeros((1, 2048, 4), dtype=torch.bfloat16)
    logits = _builder_frame_readout_logits(
        Model(), final, batch_index=0, real_length=1537, score_positions=1024
    )

    assert calls == [(1537, 4)]
    assert tuple(logits.shape) == (1024, 7)
    assert logits.dtype == torch.float32


def test_cuda_memory_fraction_is_applied_and_read_back_before_nccl() -> None:
    calls: list[tuple[str, object]] = []

    class FakeCuda:
        def set_per_process_memory_fraction(self, fraction: float) -> None:
            calls.append(("set", fraction))

        def get_per_process_memory_fraction(self) -> float:
            calls.append(("get", None))
            return 0.4499999999969387

    observed = _apply_cuda_memory_fraction(cast(Any, FakeCuda()))
    assert abs(observed - CUDA_MEMORY_FRACTION) < 1e-9
    assert CUDA_MEMORY_FRACTION == 0.45
    assert calls == [("set", 0.45), ("get", None)]

    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    cap = source.index("_apply_cuda_memory_fraction(torch.cuda)")
    cap_receipt = source.index('f"CUDA_MEMORY_CAP.{TASK}.rank{rank}.json"', cap)
    nccl = source.index("torch.distributed.init_process_group", cap_receipt)
    engine = source.index("ModernGreenResidentEngine(", nccl)
    assert cap < cap_receipt < nccl < engine


def test_tensor_ab_uses_r20_cgroup_memory_contract_without_cuda_fraction_cap() -> None:
    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    tensor_gate = source.index("if sealed_runtime_tensor_ab_only:")
    cap_call = source.index("_apply_cuda_memory_fraction(torch.cuda)")
    assert tensor_gate < cap_call
    assert 'cuda_memory_cap_status = "BYPASSED_R20_TENSOR_AB_CGROUP_ONLY"' in source


def test_scheduled_pair_group_calls_the_admitted_forward_once_and_seals_each_pair() -> None:
    windows = (28, 56, 68, 71, 76, 99, 107, 122)
    zero = {name: 0 for name in (
        "checkpoint_reloads", "fallback_calls", "reconstruction_calls",
        "timed_model_payload_reads", "timed_score_file_reads",
    )}
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
        class API:
            calls: list[tuple[int, ...]] = []
            def validate(self, engine: Any, batch: tuple[int, ...], teacher_root: Path) -> dict[str, Any]:
                self.calls.append(batch)
                return measurement(batch, child_binding)

        api, engine = API(), object()
        result = validate_scheduled_pair_group(
            api, engine, windows, Path(directory) / "teacher", receipts,
            rank=0, canonical_code_commit="f" * 40, attempt="schedule-ab",
            first_pair_index=3,
        )
        assert api.calls == [windows]
        assert result["windows"] == list(windows)
        assert [row["pair_index"] for row in result["scheduled_pair_receipts"]] == [3, 4, 5, 6]
        assert [row["windows"] for row in result["scheduled_pair_receipts"]] == [
            [28, 56], [68, 71], [76, 99], [107, 122],
        ]
        assert all(Path(row["path"]).is_file() for row in result["scheduled_pair_receipts"])


def test_production_rebinds_to_sealed_single_window_pre_semantics() -> None:
    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    engine_source = (Path(__file__).parents[1] / "modern_green_resident.py").read_text()
    gate = source.index("# W28 is now sealed against the immutable accepted producer")
    production = source.index("full_started = time.perf_counter()", gate)
    body = source[gate:]
    assert "W28_KLD = 0.1364830042977786" in source
    assert "W28_TOP1 = 880" in source
    assert 'int(config.get("score_window_batch_size", 0)) != 1' in source
    assert '"score_window_batch_size": 1' in source
    assert '"sealed_builder_window_microbatch": 1' in source
    assert 'config["sealed_builder_window_microbatch"] = 1' in source
    assert 'config["sealed_builder_window_microbatch"] = 2' not in source
    assert 'config["score_window_batch_size"] = 1' in body
    assert "engine.score_pipeline_microbatch = 1" in body
    assert 'config["score_pair_stream_concurrency"] = 1' in body
    assert 'config["score_pipeline_overlap"] = True' in body
    assert "published PRE validation requires sealed single-window microbatch" in engine_source
    assert "_physical_canary_batch_windows(ordered" not in engine_source
    assert "scheduled PRE pair group requires exact sealed mb=2" not in engine_source
    assert "validate_scheduled_pair_group(" in source[production:]
    assert "validate_full64_admission_pairs(" not in source[production:]
    assert "first_pair_index=0" in source[production:]


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


def test_resident_full64_accept_pair_scheduling_ab_preserves_the_imported_eager_math() -> None:
    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    engine_source = (Path(__file__).parents[1] / "modern_green_resident.py").read_text()
    assert source.count('checkpoint_path("PRE")') == 1
    assert source.count("torch.load(checkpoint_path") == 1
    assert source.count("ModernGreenResidentEngine(") == 1
    eager = source.index('config["resident_validation_attention_implementation"] = "eager"')
    engine = source.index("ModernGreenResidentEngine(")
    admission = source.index("_score_admission_windows(", engine)
    scheduling = source.index('config["score_pair_stream_concurrency"] = 1', admission)
    canary = source.index("validate_scheduled_pair_group(", scheduling)
    assert eager < engine < admission < scheduling < canary
    assert 'config.pop("resident_validation_stock_hf_attention", None)' in source
    assert 'config.pop("resident_validation_stock_hf_sdpa_math_backend", None)' in source
    scheduling_block = source[scheduling - 500:scheduling + 300]
    assert 'config["score_window_batch_size"] = 1' in scheduling_block
    assert 'config["attention_query_chunk_size"] = 64' in scheduling_block
    assert 'config["indexer_scorer_query_chunk_size"] = 128' in scheduling_block
    assert "engine.score_pipeline_microbatch = 1" in scheduling_block
    assert 'config["score_pipeline_overlap"] = False' in source[scheduling:scheduling + 300]
    assert 'config["score_pair_group_single_stream"] = True' in source[scheduling:scheduling + 300]
    assert "canary_windows = windows[2:10]" in source
    assert 'first_pair_index=1' in source
    assert 'row["kld_delta"] == 0.0' in source
    assert 'row["top1_delta"] == 0' in source
    assert 'full_groups = len(windows) // len(canary_windows)' in source
    assert 'full_groups * max(rank0_stage_ms, rank1_stage_ms)' in source
    assert 'projected_full64_wall < 300.0' in source
    assert 'banana-smasher-pair-scheduling-ab-v1' in source
    assert 'PAIR_SCHEDULING_AB_ONLY' in source
    assert 'pair_group_single_stream = bool(' in engine_source
    assert 'and not pair_group_single_stream' in engine_source


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


def test_public_full64_binds_exact_sealed_pre_sources_to_resident_zero_reload_api() -> None:
    root = Path(__file__).parents[1]
    runner = (root / "resident_full64_accept.py").read_text()
    resident = (root / "modern_green_resident.py").read_text()
    binding = (root / "sealed_pre_forward.py").read_text()
    wrapper = (root / "assets" / "static_w28_fast_k2_grouped.py").read_text()
    assert "run_sealed_pre_forward(" not in runner
    resident_binding = runner.index("bind_sealed_pre_resident_config(config)")
    engine = runner.index("ModernGreenResidentEngine(")
    assert resident_binding < engine
    assert "def bind_sealed_pre_resident_config(" in binding
    assert 'config["resident_validation_expert_implementation"] = "sealed_bf16_full_weight"' in binding
    assert 'config["sealed_pre_source_binding"] = source_binding()' in binding
    assert 'stream_sync = getattr(wrapper_module, "bind_backward_stream_sync", None)' in resident
    assert "if callable(stream_sync):" in resident
    assert "W28_KLD = 0.1364830042977786" in runner
    assert "W28_TOP1 = 880" in runner
    assert 'BUILDER_SHA256 = "11ead706db562197e76cdc320d5d13044bb254a411b6412326667f524ddf29ed"' in binding
    assert 'PLANESOURCE_SHA256 = "167603b5662437a2f9fc4b3ead1561d777a7a831a898133993b9e1c0c26c9f87"' in binding
    assert 'SEALED_MODEL_ROOT = Path("/home/dnola/models/hf/DeepSeek-V4-Flash-0731")' in binding
    assert 'source_args = ["--local-dir", str(SEALED_MODEL_ROOT)]' in binding
    assert '"--remote", "dnola@192.168.200.4:/home/dnola/models/hf/DeepSeek-V4-Flash-0731"' in binding
    assert '"--shard-buf", str(root / "shard_buf"), "--keep-shards", "3"' in binding
    from repair_api.sealed_pre_forward import source_binding

    observed = source_binding(root)
    assert observed["status"] == "PASS"
    assert observed["known_value_fixture"] == {
        "window": 28,
        "kld_mean": 0.1364830042977786,
        "top1": 880,
    }


def test_sdpa_repair_retains_attention_sink_denominator() -> None:
    source = (Path(__file__).parents[1] / "modern_green_resident.py").read_text()
    assert "scaled_dot_product_attention" in source
    assert "torch.logsumexp(logits, dim=-1)" in source
    assert "torch.sigmoid(lse - sinks)" in source
    assert "output * keep_probability" in source
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
    assert "reference_taps = _capture_reference_layer_taps(engine, ids)" in source
    assert 'active_cache = DynamicCache(config=engine.student.config)' in source
    assert "local_indices = range(engine.first, engine.last + 1)" in source
    assert "engine._batch_p2p_recv(hidden, src=0)" in source
    assert "engine._batch_p2p_send(hidden.detach().contiguous(), dst=1)" in source
    assert "tuple(f\"L{index:03d}\" for index in local_indices)" in source
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


def test_sealed_runtime_tensor_ab_executes_exact_builder_control() -> None:
    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    engine = (Path(__file__).parents[1] / "modern_green_resident.py").read_text()
    assert "SEALED_RUNTIME_TENSOR_AB_ONLY" in source
    assert "def _sealed_runtime_tensor_ab(" in source
    assert "_prepare_exact_modules(" in source
    assert "builder.build_layer_sd(0, engine.student.wm, engine.student.get_tensor, \"planes\", planes)" in source
    assert 'boundary_order = ("input_tensor", "attention_tensor", "post_expert_hidden", "lm_head_logits")' in source
    assert '"first_divergent_boundary": first_divergent' in source
    assert '"control_source": "repair_api/assets/builder_B2_PUBLISHED_PRE.py:593-643 + repair_api/assets/official_local_planesource.py:592-624"' in source
    assert "class SealedBuilderProjectionBoundaryExpert" in engine
    assert "return value.to(dtype=torch.bfloat16)" in engine
    assert "def _run_separate_gate_up_geometry(" in source
    assert "control_known_hash = \"11cc07869ffcf71c39699e5631fa352cdb3aba52a003b04b659ceb5cfa4c0662\"" in source
    assert "variant_known_hash = \"6c4dc981b8ece4f5e1fd81573a289a48c2ae742efd20546f8d83f594c4e12f1d\"" in source
    assert '"sole_variable": "concatenated_gate_up_gemm_vs_separate_w1_w3_gemms"' in source
    expert = (Path(__file__).parents[1] / "assets" / "fast_v7_expert_base.py").read_text()
    assert "top_k_index.transpose(0, 1).reshape(-1)" in expert
    assert "top_k_weights.transpose(0, 1).reshape(-1, 1)" in expert


def test_sealed_runtime_expert_trace_covers_named_internal_boundaries() -> None:
    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    assert "SEALED_RUNTIME_EXPERT_TRACE_AB_ONLY" in source
    assert "def _sealed_runtime_expert_trace_ab(" in source
    assert "def _run_one_layer_with_expert_trace(" in source
    assert '"route_key", "route_weight", "gate", "up", "activated", "w2_down"' in source
    assert '"weighted_routed_output", "full_weighted_routed_output"' in source
    assert 'local["first_weighted_divergent_expert"]' in source
    assert '"first_weighted_divergent_expert": gathered[0].get(' in source
    assert '"active_expert_census": gathered[0].get("active_expert_census")' in source
    assert '"first_unequal_active_expert": gathered[0].get(' in source
    assert '"trace-normalized expert-major/slot-major/token-major route keys"' in source
    expert = (Path(__file__).parents[1] / "assets" / "fast_v7_expert_base.py").read_text()
    assert 'authentic_trace["route_key"] = torch.stack(' in expert
    assert "(expert_index, slot_index, token_index), dim=1" in expert
    assert 'authentic_trace["route_weight"] = top_k_weights.reshape(-1).detach()' in expert
    assert 'authentic_trace["w2_down"] = routed_output.detach()' in expert
    assert 'authentic_trace["weighted_routed_output"] = routed_output.detach()' in expert
    assert "AUTHENTIC_EXPERT_TRACE_INACTIVE" not in expert
    assert source.count('"EXPERT_TRACE_TARGET_204_INACTIVE"') == 2
    assert '"post_expert_hidden"' in source
    assert '"first_divergent_boundary": first_divergent' in source
    assert "builder.build_layer_sd(" in source
    assert "replay = replay_resident if resident else replay_control" in source
    assert 'module.__class__.forward.__globals__[' in source
    assert '"grouped_sealed_gate_up_projection"' in source
    assert 'keep("per_slot_accumulation", final[trace_tokens])' in source
    assert 'experts._authentic_expert_trace = authentic_provider_trace' in source
    assert 'interface = dispatcher.__globals__["ALL_EXPERTS_FUNCTIONS"]' in source
    assert 'implementation_globals[linear_key] = observed_selected_linear' in source
    assert 'linear_key = "_grouped_linear" if implementation_name == "grouped_mm"' in source
    assert 'torch.Tensor.__mul__ = observed_mul' in source
    assert 'AUTHENTIC_PROVIDER_TRACE_COVERAGE_RED' in source
    assert 'trace.update(authentic_trace)' in source
    assert '"routed_accumulator", "shared_expert_output", "moe_combined_output"' in source
    assert 'shared_expert_output = layer.mlp.shared_experts(replay_inputs["hidden_states"])' in source
    assert '"routed": routed.detach()' in source
    assert '"shared": shared.detach()' in source
    assert '"combine": value.detach()' in source
    assert '"authentic_routed_output": authentic_routed_output.cpu()' in source
    assert '"authentic_shared_expert_output": authentic_shared_output.cpu()' in source
    assert '"authentic_return_combine": authentic_return_combine.detach().cpu()' in source
    local_order = source[source.index("local[\"boundaries\"] = {") - 800 :]
    local_order = local_order[: local_order.index("local[\"boundaries\"] = {")]
    assert '"authentic_routed_output", "authentic_shared_expert_output"' in local_order
    assert '"authentic_return_combine"' in local_order
    final_order = source[source.index("boundaries = gathered[0]") :]
    final_order = final_order[: final_order.index("receipt = {")]
    assert '"authentic_routed_output", "authentic_shared_expert_output"' in final_order
    assert '"authentic_return_combine", "authentic_moe_output"' in final_order
    assert 'ffn_hc.forward = types.MethodType(transparent_ffn_hc_forward, ffn_hc)' in source
    assert 'mlp.forward = types.MethodType(transparent_mlp_forward, mlp)' in source
    assert '"authentic_moe_output": authentic_mlp_trace["output"].cpu()' in source
    assert '"scaled_moe_branch": scaled_moe_branch.detach().cpu()' in source
    assert '"residual_projection": residual_projection.detach().cpu()' in source
    assert '"reconstructed_post_expert_hidden": (' in source


def test_restored_provider_authentic_trace_normalizes_without_replay() -> None:
    from repair_api.resident_full64_accept import _normalize_provider_authentic_trace

    hidden = torch.zeros((2, 3), dtype=torch.bfloat16)
    indices = torch.tensor([[7, 204], [204, 9]], dtype=torch.int64)
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]], dtype=torch.float32)
    route_key = torch.tensor(
        [[7, 0, 0], [204, 1, 0], [204, 0, 1], [9, 1, 1]],
        dtype=torch.int64,
    )
    w2_down = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
    weighted = w2_down * weights.reshape(-1, 1).to(torch.bfloat16)
    final = weighted.reshape(2, 2, 3).sum(dim=1).to(torch.bfloat16)
    provider_trace = {
        "route_key": route_key,
        "route_weight": weights.reshape(-1),
        "gate": torch.zeros((4, 2), dtype=torch.bfloat16),
        "up": torch.zeros((4, 2), dtype=torch.bfloat16),
        "activated": torch.zeros((4, 2), dtype=torch.bfloat16),
        "w2_down": w2_down,
        "weighted_routed_output": weighted,
    }

    capture, assembly = _normalize_provider_authentic_trace(
        provider_trace, hidden, indices, weights, final,
    )

    assert torch.equal(capture["authentic_expert_indices"], indices.reshape(-1))
    assert torch.equal(capture["authentic_token_indices"], route_key[:, 2])
    assert torch.equal(capture["authentic_weighted_buffer"], weighted)
    assert capture["authentic_ordered_weighted_buffer"].shape == (2, 2, 3)
    assert assembly["final_return"] is final
    assert assembly["routed_output"] is w2_down


def test_static_provider_w2_capture_is_output_transparent_and_restores_project() -> None:
    from repair_api.resident_full64_accept import _capture_static_provider_w2

    class Provider:
        def _project(self, projection, value, assignments):
            del projection, assignments
            return value

    class Capture:
        observed = None

        def capture_w2(self, value, assignments):
            self.observed = (value, assignments)

    provider = Provider()
    original = provider._project
    capture = Capture()
    value = torch.ones((2, 3), dtype=torch.bfloat16)
    assignments = torch.tensor([7, 204], dtype=torch.int64)
    with _capture_static_provider_w2(provider, capture):
        returned = provider._project("w2", value, assignments)
        assert returned is value
        assert capture.observed == (value, assignments)
    assert provider._project == original


def test_static_provider_capture_feeds_all_five_consumer_keys_end_to_end() -> None:
    from unittest.mock import patch
    from repair_api.resident_full64_accept import (
        _AuthenticRouteCaptureMode, _capture_static_provider_w2,
    )

    class Provider:
        def _project(self, projection, value, assignments):
            del projection, assignments
            return value

    provider = Provider()
    hidden = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    indices = torch.tensor([[7, 204], [204, 9]], dtype=torch.int64)
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]], dtype=torch.float32)
    token_index = torch.arange(2).unsqueeze(1).expand_as(indices).reshape(-1)
    expert_index = indices.reshape(-1)
    routed = hidden[token_index].contiguous()
    capture = _AuthenticRouteCaptureMode()
    capture.bind_route_inputs(indices, weights)
    original_gather = torch.gather

    def observed_gather(*args, **kwargs):
        result = original_gather(*args, **kwargs)
        capture.capture_ordered(result)
        return result

    with _capture_static_provider_w2(provider, capture), patch.object(
        torch, "gather", observed_gather
    ), capture:
        w2 = provider._project("w2", routed, expert_index)
        weighted = (w2 * weights.reshape(-1, 1).float()).to(hidden.dtype)
        routed_output = weighted.reshape(2, 2, 3)
        expert_order = torch.argsort(indices, dim=1, stable=True)
        ordered = torch.gather(
            routed_output, 1, expert_order.unsqueeze(-1).expand_as(routed_output)
        )
        ordered.sum(dim=1)

    required = {
        "authentic_w2_output": capture.w2_output,
        "authentic_route_weights": capture.route_weights,
        "authentic_expert_indices": capture.expert_indices,
        "authentic_weighted_buffer": capture.weighted_buffer,
        "authentic_ordered_weighted_buffer": capture.ordered_weighted_buffer,
    }
    assert all(value is not None for value in required.values()), required
    assert required["authentic_w2_output"] is w2
    assert torch.equal(required["authentic_expert_indices"], expert_index)
    assert torch.equal(required["authentic_ordered_weighted_buffer"], ordered)


def test_static_provider_replay_arguments_follow_actual_required_signature() -> None:
    from unittest.mock import patch
    from repair_api.resident_full64_accept import _static_provider_replay_arguments

    class Provider:
        def _project(
            self, projection, value, assignments, packed, lut, su, sv,
            route_rows_per_sample, route_metadata,
        ):
            del self, projection, assignments, packed, lut, su, sv
            return value, route_rows_per_sample, route_metadata

    routed = torch.zeros((4, 3), dtype=torch.bfloat16)
    experts = torch.tensor([7, 204, 204, 9], dtype=torch.int64)
    metadata = {"source": "static-provider"}
    with patch.dict(
        Provider._project.__globals__,
        {"EXPERTS": 256, "grouped_route_metadata": lambda *args, **kwargs: metadata},
    ):
        tail = _static_provider_replay_arguments(Provider(), routed, experts, 4)
    assert tail == (4, metadata)


def test_r20_diagnostic_decoder_loads_the_pinned_shipped_asset() -> None:
    from repair_api.resident_full64_accept import _load_r20_grouped_decoder

    decoder = _load_r20_grouped_decoder()
    assert callable(decoder.sealed_bf16_full_weight)


def test_pre_gemm_witness_binds_bytes_layout_and_invocation_geometry() -> None:
    from repair_api.resident_full64_accept import (
        _compare_pre_gemm_witnesses,
        _pre_gemm_tensor_witness,
    )

    hidden = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    weight = torch.arange(20, dtype=torch.bfloat16).reshape(5, 4)
    hidden_witness = _pre_gemm_tensor_witness(hidden, role="active_hidden_rows")
    weight_witness = _pre_gemm_tensor_witness(weight, role="decoded_gate_weight")

    assert hidden_witness["dtype"] == "torch.bfloat16"
    assert hidden_witness["shape"] == [3, 4]
    assert hidden_witness["stride"] == [4, 1]
    assert hidden_witness["storage_offset"] == 0
    assert hidden_witness["layout"] == "torch.strided"
    assert hidden_witness["role"] == "active_hidden_rows"
    comparison = _compare_pre_gemm_witnesses(
        {"hidden": hidden_witness, "gate_weight": weight_witness},
        {"hidden": dict(hidden_witness), "gate_weight": dict(weight_witness)},
        control_invocation={"operator": "torch.nn.functional.linear", "m": 3, "n": 5, "k": 4},
        variant_invocation={"operator": "torch.nn.functional.linear", "m": 3, "n": 5, "k": 4},
    )
    assert comparison["status"] == "PRE_GEMM_INPUT_WEIGHT_GEOMETRY_PARITY"
    assert comparison["first_unequal_boundary"] is None

    variant = {"hidden": dict(hidden_witness), "gate_weight": dict(weight_witness)}
    variant["gate_weight"]["sha256"] = "0" * 64
    unequal = _compare_pre_gemm_witnesses(
        {"hidden": hidden_witness, "gate_weight": weight_witness},
        variant,
        control_invocation={"operator": "torch.nn.functional.linear", "m": 3, "n": 5, "k": 4},
        variant_invocation={"operator": "torch.nn.functional.linear", "m": 3, "n": 5, "k": 4},
    )
    assert unequal["first_unequal_boundary"] == "gate_weight"

    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    assert 'pre_gemm_capture["control"]' in source
    assert 'pre_gemm_capture["variant"]' in source
    assert '"expert_204_decoded_gate_weight"' in source
    assert '"weight_transpose_semantics": "input @ weight.T"' in source
    assert '"pre_gemm_comparison": pre_gemm' in source
    assert '"MULTIPLE_GEMM_EXECUTION_MECHANISMS_PLAUSIBLE_STOP"' in source


def test_a30o_authentic_return_witness_localizes_route_order() -> None:
    from repair_api.resident_full64_accept import _routed_return_assembly_witness

    hidden = torch.zeros((2, 1), dtype=torch.bfloat16)
    top_k_index = torch.tensor([[9, 3], [3, 9]], dtype=torch.int64)
    top_k_weights = torch.ones((2, 2), dtype=torch.float32)
    routed_token_major = torch.tensor(
        [[1.0], [10.0], [100.0], [1000.0]], dtype=torch.bfloat16
    )
    observed = _routed_return_assembly_witness(
        hidden, routed_token_major, top_k_index, top_k_weights,
    )
    slot_major = routed_token_major.reshape(2, 2, 1).transpose(0, 1).reshape(4, 1)
    misaligned = _routed_return_assembly_witness(
        hidden, slot_major, top_k_index, top_k_weights,
    )

    assert observed["assembly_contract"] == "token-major-flat_then_ascending-expert-index-add"
    assert observed["destination_zero"]["nonzero"] == 0
    assert observed["token_index"]["values"] == [0, 0, 1, 1]
    assert observed["expert_index"]["values"] == [9, 3, 3, 9]
    assert observed["weighted_routes"]["sha256"] != misaligned["weighted_routes"]["sha256"]
    assert observed["final_return"]["sha256"] != misaligned["final_return"]["sha256"]

    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    assert "_sealed_builder_accumulate_routes" in source
    assert 'assembly_capture["variant"]' in source
    assert '"routed_return_assembly": routed_return_assembly' in source


def test_attempt106_trace_proves_unmodified_control_before_wrappers() -> None:
    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    unmodified = source.index("# Step 1: the unmodified sealed control")
    first_progress = source.index('"completed_steps": [1]', unmodified)
    traced = source.index("# Step 2: attach the trace wrapper", first_progress)
    second_progress = source.index('"completed_steps": [1, 2]', traced)
    resident = source.index("# Step 3 starts only after both control gates", second_progress)
    third_progress = source.index('"completed_steps": [1, 2, 3]', resident)
    assert unmodified < first_progress < traced < second_progress < resident < third_progress
    assert "EXPERT_TRACE_UNMODIFIED_CONTROL_RED" in source
    assert "EXPERT_TRACE_WRAPPER_TRANSPARENCY_RED" in source
    assert 'root / "ATTEMPT106_PROGRESS.json"' in source
    assert "return sorted_output[inverse_order]" in source
    assert "return sorted_output[inverse_order].float()" not in source
    assert "route_rows_per_sample: int | None = None" in source
    assert "route_metadata: Any = None" in source
    assert "return original_forward(hidden_states, top_k_index, top_k_weights)" in source
    assert "weighted = down * route_weight.to(dtype=down.dtype)" in source
    authentic_output = source.index("output, _attention = _run_one_layer_with_attention")
    replay = source.index("replay = replay_resident if resident else replay_control")
    assert authentic_output < replay
    assert "trace_expert = hit[0, 0]" in source
    assert "if expert_index != trace_expert:" in source
    assert "Byte-for-byte source shape of DeepseekV4Experts.forward" in source


def test_attempt106_seals_first_internal_divergence_before_kill_gate() -> None:
    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    resident = source.index("# Step 3 starts only after both control gates")
    first_divergent = source.index("local_first_divergent = next(", resident)
    seal_call = source.index("_seal_attempt106_step3(", first_divergent)
    kill_gate = source.index("A30O_ASSEMBLY_KILL_GATE_RED", seal_call)
    assert resident < first_divergent < seal_call < kill_gate
    assert 'boundaries=local["boundaries"]' in source[seal_call:kill_gate]


def test_attempt106_step3_execution_persists_divergence_before_kill(tmp_path) -> None:
    import json
    from repair_api.resident_full64_accept import _seal_attempt106_step3

    progress = tmp_path / "ATTEMPT106_PROGRESS.json"
    _seal_attempt106_step3(
        progress,
        pin="test-pin",
        control_tap={"sha256": "control"},
        control_flat66=0.00048828125,
        traced_control_tap={"sha256": "control"},
        first_divergent="full_weighted_routed_output",
        pre_gemm_comparison={"exact": True},
        boundaries={"full_weighted_routed_output": {"exact": False}},
    )
    receipt = json.loads(progress.read_text())
    assert receipt["completed_steps"] == [1, 2, 3]
    assert receipt["step3_resident_comparison"]["first_divergent_boundary"] == "full_weighted_routed_output"


def test_post_w2_chain_comparator_self_control_and_first_unequal() -> None:
    from repair_api.resident_full64_accept import _compare_post_w2_chain

    order = (
        "raw_bf16_w2", "fp32_provider_exposure", "route_weight_multiply",
        "reshape_order_gather_buffer", "per_token_reduction", "mlp_return",
        "residual_combine",
    )
    control = {
        name: torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        for name in order
    }
    self_control = _compare_post_w2_chain(
        control, {name: value.clone() for name, value in control.items()}
    )
    assert self_control["comparator_self_control_exact"] is True
    assert self_control["first_unequal_boundary"] is None
    assert tuple(self_control["boundary_order"]) == order
    assert self_control["boundaries"]["raw_bf16_w2"]["control"]["stride"] == [4, 1]

    product = {name: value.clone() for name, value in control.items()}
    product["route_weight_multiply"][1, 2] += 1
    divergent = _compare_post_w2_chain(control, product)
    assert divergent["comparator_self_control_exact"] is True
    assert divergent["first_unequal_boundary"] == "route_weight_multiply"


def test_authentic_static_route_weight_callsite_is_observable_without_arithmetic_mutation() -> None:
    root = Path(__file__).parents[1]
    runner = (root / "resident_full64_accept.py").read_text()
    provider = (root / "assets" / "static_w28_fast_v7_expert_base.py").read_text()

    assert '_route_weight_callsite_trace' in provider
    assert '"co_filename": code.co_filename' in provider
    assert '"co_firstlineno": int(code.co_firstlineno)' in provider
    assert '"input": route_multiply_input.detach()' in provider
    assert '"route_weight": route_multiply_weight.detach()' in provider
    assert '"output": routed_output.detach()' in provider
    assert 'module._route_weight_callsite_trace = route_weight_callsite_capture' in runner
    assert 'module.__dict__.pop("_route_weight_callsite_trace", None)' in runner
    assert 'trace["route_weight_callsite"] = {' in runner
    assert '"route_weight_callsite": variant_trace["route_weight_callsite"]' in runner
    assert 'POST_W2_CHAIN_RECEIPT_NAME' in runner


def test_exact_w28_product_and_planesource_rails_wire_full_hidden_state_call_trees() -> None:
    root = Path(__file__).parents[1]
    tracer = (root / "call_tree_trace.py").read_text()
    runner = (root / "resident_full64_accept.py").read_text()
    builder = (root / "assets" / "builder_B2_PUBLISHED_PRE.py").read_text()

    assert "sys.settrace(self._trace_frame)" in tracer
    assert '"route_weight_multiply_inputs"' in tracer
    assert '"weighted_per_slot_assembly_input"' in tracer
    assert '"residual_combine_inputs"' in tracer
    assert "self._dispatch.__enter__()" not in tracer
    assert "register_forward_pre_hook" not in tracer
    assert "register_forward_hook" not in tracer
    assert "os.fsync" in tracer
    assert 'W28_FULL_CALL_TREE_PATH' in runner
    assert 'rail="product_w28_admission"' in runner
    assert 'W28_FULL_CALL_TREE_PATH' in builder
    assert 'rail="accepted_planesource_w28"' in builder


def test_full_call_tree_resolves_exact_shard_student_module_ownership_shape() -> None:
    from repair_api.call_tree_trace import _module_root

    class ShardStudentShape:
        def __init__(self) -> None:
            self.model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())

    student = ShardStudentShape()
    assert _module_root(student) is student.model
    assert [name for name, _module in _module_root(student).named_modules()] == ["", "0", "1"]


def test_full_call_tree_does_not_recursively_trace_its_own_observer(tmp_path: Path) -> None:
    import json
    from repair_api.call_tree_trace import FullCallTreeTrace

    module = torch.nn.Identity()
    path = tmp_path / "tree.jsonl"
    with FullCallTreeTrace(
        module, path, rail="self_filter_control", basis_sha256="b" * 64,
        canonical_code_commit="c" * 40,
    ):
        module(torch.zeros(1024))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["kind"] for row in rows] == ["header", "footer"]
    assert not any(
        row["kind"] == "python_call" and row["file"].endswith("/call_tree_trace.py")
        for row in rows
    )


def test_full_call_tree_only_extends_transport_timeout_to_3600_seconds() -> None:
    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    assert 'transport_timeout_seconds = 3600 if os.environ.get("W28_FULL_CALL_TREE_PATH") else 900' in source
    assert "timeout=timedelta(seconds=transport_timeout_seconds)" in source
    assert source.count("transport_timeout_seconds") == 2


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


def test_sink_token_sdpa_keeps_sink_visible_with_boolean_mask() -> None:
    torch.manual_seed(17)
    query = torch.randn(1, 2, 4, 8)
    key = torch.randn(1, 2, 4, 8)
    value = torch.randn(1, 2, 4, 8)
    mask = torch.ones((1, 1, 4, 4), dtype=torch.bool).tril()
    module = SimpleNamespace(sinks=torch.tensor([0.75, -0.25]), num_key_value_groups=1)
    scaling = 8 ** -0.5
    observed, _ = ModernGreenResidentEngine._sink_corrected_sdpa_forward(
        module, query, key, value, mask, scaling,
    )
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    scores = scores.masked_fill(~mask, float("-inf"))
    sinks = module.sinks.reshape(1, 2, 1, 1).expand(1, 2, 4, 1)
    probabilities = torch.softmax(torch.cat((scores, sinks), dim=-1), dim=-1)[..., :-1]
    expected = torch.matmul(probabilities, value).transpose(1, 2).contiguous()
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)


def test_sink_token_sdpa_preserves_original_sdpa_problem(monkeypatch) -> None:
    torch.manual_seed(23)
    query = torch.randn(1, 2, 4, 8)
    key = torch.randn(1, 2, 4, 8)
    value = torch.randn(1, 2, 4, 8)
    mask = torch.ones((1, 1, 4, 4), dtype=torch.bool).tril()
    module = SimpleNamespace(sinks=torch.tensor([0.75, -0.25]), num_key_value_groups=1)
    captured = {}

    def fake_sdpa(q, k, v, *, attn_mask, **kwargs):
        captured.update(q=q, k=k, v=v, attn_mask=attn_mask, kwargs=kwargs)
        return torch.zeros_like(q)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)
    ModernGreenResidentEngine._sink_corrected_sdpa_forward(
        module, query, key, value, mask, 8 ** -0.5,
    )
    assert torch.equal(captured["q"], query)
    assert torch.equal(captured["k"], key)
    assert torch.equal(captured["v"], value)
    assert torch.equal(captured["attn_mask"], mask)


def test_sink_corrected_sdpa_builds_additive_mask_for_compressor_bias(monkeypatch) -> None:
    source = (Path(__file__).parents[1] / "modern_green_resident.py").read_text()
    builder = (Path(__file__).parents[1] / "assets" / "t8192_w28_sdpa_teacher_builder.py").read_text()
    assert "_attention_mask_for_layer" in source
    assert '"plain": build_mask("sdpa")' in source
    assert '"compressor": build_mask("eager")' in source
    assert 'mask_kind = "compressor" if getattr(lay.self_attn, "compressor", None) is not None else "plain"' in builder
    config = SimpleNamespace(
        _attn_implementation="official_k2_sink_corrected_sdpa",
        _attn_implementation_internal="official_k2_sink_corrected_sdpa",
    )
    captured: list[str] = []

    def fake_mask(*, config, **kwargs):
        implementation = str(config._attn_implementation)
        captured.append(implementation)
        return f"{implementation}-mask"

    transformers = ModuleType("transformers")
    masking_utils = ModuleType("transformers.masking_utils")
    setattr(masking_utils, "create_sliding_window_causal_mask", fake_mask)
    setattr(transformers, "masking_utils", masking_utils)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.masking_utils", masking_utils)
    rotary = lambda template, **kwargs: (template, kwargs["layer_type"])
    engine = cast(Any, ModernGreenResidentEngine.__new__(ModernGreenResidentEngine))
    engine.torch = torch
    engine.student = SimpleNamespace(
        device=torch.device("cpu"),
        config=config,
        model=SimpleNamespace(
            config=config,
            model=SimpleNamespace(rotary_emb=rotary),
        ),
    )

    _pos, _pe, mask = engine._positional(
        torch.zeros((1, 8), dtype=torch.long),
        torch.zeros((1, 8, 4)),
        object(),
    )

    assert mask == {"plain": "sdpa-mask", "compressor": "eager-mask"}
    assert captured == ["sdpa", "eager"]
    plain = SimpleNamespace(self_attn=SimpleNamespace(compressor=None))
    compressor = SimpleNamespace(self_attn=SimpleNamespace(compressor=object()))
    assert engine._attention_mask_for_layer(plain, mask) == "sdpa-mask"
    assert engine._attention_mask_for_layer(compressor, mask) == "eager-mask"
    assert config._attn_implementation == "official_k2_sink_corrected_sdpa"
    assert config._attn_implementation_internal == "official_k2_sink_corrected_sdpa"


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


def test_chunked_indexer_scorer_is_bitwise_and_bounds_query_workspace() -> None:
    torch.manual_seed(19)
    module = torch.nn.Module()
    module.softmax_scale = 7**-0.5
    module.weights_scaling = 3**-0.5
    module.weights_proj = torch.nn.Linear(13, 3, bias=False).to(torch.bfloat16)
    q = torch.randn((2, 11, 3, 7), dtype=torch.bfloat16)
    compressed_kv = torch.randn((2, 17, 7), dtype=torch.bfloat16)
    hidden_states = torch.randn((2, 11, 13), dtype=torch.bfloat16)
    scores = torch.matmul(q.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1))
    scores = torch.nn.functional.relu(scores) * module.softmax_scale
    weights = module.weights_proj(hidden_states).float() * module.weights_scaling
    expected = (scores * weights.unsqueeze(-1)).sum(dim=2)
    observed_chunks: list[int] = []
    observed = ModernGreenResidentEngine._chunked_indexer_scorer_forward(
        module,
        q,
        compressed_kv,
        hidden_states,
        query_chunk_size=4,
        _chunk_observer=observed_chunks.append,
    )
    assert torch.equal(observed, expected)
    assert observed_chunks == [4, 4, 3]
