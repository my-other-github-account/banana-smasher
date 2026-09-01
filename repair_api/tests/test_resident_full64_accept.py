from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
import torch

from repair_api import ArtifactError, ResidentRepairAPI
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
    _score_admission_windows,
    atomic,
    validate_scheduled_pair_group,
)
from repair_api.modern_green_resident import (
    ModernGreenResidentEngine,
    _builder_frame_readout_logits,
    _score_validation_kld_rows,
)


def test_resident_receipt_atomic_creates_missing_parent() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "receipts" / "receipt.json"
        digest = atomic(path, {"status": "PASS"})
        assert path.exists()
        assert len(digest) == 64


def test_published_pre_selects_source_experts_dispatch() -> None:
    from repair_api.modern_green_resident import _bind_published_pre_experts_dispatch

    model_config = SimpleNamespace(_experts_implementation="grouped_mm")
    expert = SimpleNamespace(
        routed_return_reduction="source_eager_expert_major_index_add"
    )
    student = SimpleNamespace(
        model=SimpleNamespace(config=model_config), experts={0: expert, 42: expert}
    )

    binding = _bind_published_pre_experts_dispatch(
        student, published_pre_recipe=True
    )

    assert model_config._experts_implementation == "eager"
    assert binding == {
        "status": "BOUND_SOURCE_EXPERTS_DISPATCH",
        "previous_implementation": "grouped_mm",
        "selected_implementation": "eager",
        "resident_return_reduction": "source_eager_expert_major_index_add",
    }


def test_published_pre_rejects_resident_provider_bypassing_source_dispatch() -> None:
    from repair_api.modern_green_resident import _bind_published_pre_experts_dispatch

    student = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(_experts_implementation="grouped_mm")
        ),
        experts={0: SimpleNamespace()},
    )
    with pytest.raises(
        ArtifactError, match="resident experts bypass source dispatch"
    ):
        _bind_published_pre_experts_dispatch(
            student, published_pre_recipe=True
        )


def test_w28_trace_wrapper_records_sealed_known_value_control(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    import json

    trace_path = tmp_path / "w28-control.jsonl"
    canonical_pin = "f23b4b19ce96913f4f2a6f302edb61989e697491"
    monkeypatch.setenv("W28_FULL_CALL_TREE_PATH", str(trace_path))
    monkeypatch.setenv("BANANA_SMASHER_CANONICAL_PIN", canonical_pin)
    sealed = {"windows": [28], "kld_mean": 0.1364830042977786, "top1": 880}

    class Api:
        @staticmethod
        def validate(engine: Any, windows: tuple[int, ...], teacher_root: Path) -> dict[str, Any]:
            assert windows == (28,)
            assert teacher_root == tmp_path / "teacher"
            engine.student(torch.zeros(1024))
            return sealed

    observed = _score_admission_windows(
        Api(), SimpleNamespace(student=torch.nn.Identity()), (28,), tmp_path / "teacher"
    )
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    terminal = json.loads(Path(str(trace_path) + ".terminal.json").read_text())

    assert observed == sealed
    assert [row["kind"] for row in rows] == ["header", "footer"]
    assert rows[0]["rail"] == "product_w28_admission"
    assert rows[0]["canonical_code_commit"] == canonical_pin
    assert terminal["status"] == "PASS"
    assert terminal["event_count"] == 2


def test_w28_trace_wrapper_rejects_non_singleton_geometry(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    monkeypatch.setenv("W28_FULL_CALL_TREE_PATH", str(tmp_path / "wrong.jsonl"))
    monkeypatch.setenv("BANANA_SMASHER_CANONICAL_PIN", "f" * 40)
    try:
        _score_admission_windows(
            SimpleNamespace(validate=lambda *_args: {}),
            SimpleNamespace(student=torch.nn.Identity()),
            (27, 28),
            tmp_path,
        )
    except RuntimeError as error:
        assert str(error) == "W28_FULL_CALL_TREE_REQUIRES_EXACT_SINGLETON_28"
    else:
        raise AssertionError("non-singleton trace geometry was accepted")


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


def test_validation_score_can_preserve_gathered_full_softmax_measure() -> None:
    ref_lp = np.log(np.array([[0.6, 0.2]], dtype=np.float64))
    q_lp = np.log(np.array([[0.5, 0.25]], dtype=np.float64))

    direct = _score_validation_kld_rows(
        np, ref_lp, q_lp, preserve_full_softmax=True
    )
    renormalized = _score_validation_kld_rows(
        np, ref_lp, q_lp, preserve_full_softmax=False
    )

    expected = np.sum(np.exp(ref_lp) * (ref_lp - q_lp), axis=1)
    assert np.array_equal(direct, expected)
    assert not np.array_equal(direct, renormalized)


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
    assert 'int(config.get("score_window_batch_size", 0)) != 2' in source
    assert '"score_window_batch_size": 2' in source
    assert '"sealed_builder_window_microbatch": 2' in source
    assert 'config["sealed_builder_window_microbatch"] = 2' in source
    assert 'config["sealed_builder_window_microbatch"] = 1' not in source
    assert 'config["score_window_batch_size"] = 1' in body
    assert "engine.score_pipeline_microbatch = 1" in body
    assert 'config["score_pair_stream_concurrency"] = 1' in body
    assert 'config["score_pipeline_overlap"] = True' in body
    assert 'self.config.get("sealed_builder_window_microbatch", 2)' in engine_source
    assert "published PRE validation requires sealed mb=2 microbatch" in engine_source
    assert "physical_batch_size != 2" in engine_source
    assert "_physical_canary_batch_windows(ordered" not in engine_source
    assert "scheduled PRE pair group requires exact sealed mb=2" not in engine_source
    assert "validate_scheduled_pair_group(" in source[production:]
    assert "validate_full64_admission_pairs(" not in source[production:]
    assert "first_pair_index=0" in source[production:]
    assert (
        'ADOPTED_PROVIDER_EXPERT_SHA256 = "942c3074d89f8872f8c52df78941c908d9fce87edae7c21671d339f3e891d3cb"'
        in source
    )
    assert "bind_combined_gate_up_projection(" not in source[production:]
    assert "bind_routed_return_accumulation(" not in source[production:]


def test_production_installs_provider_global_native_bf16_w2_before_engine() -> None:
    """The shipped main path must pass the runtime provider through the binder gate."""
    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    api_open = source.index("api = ResidentRepairAPI.open(")
    binder = source.index("config = api.bind_combined_gate_up_projection(", api_open)
    engine = source.index("engine = ModernGreenResidentEngine(", binder)

    assert api_open < binder < engine
    assert CURRENT_PROVIDER_EXPERT_SHA256 == (
        "d27ca6c084fcb209ed9d12e3b951a585414d9e3b6b6e62559e869ae374f7079b"
    )
    bound = ResidentRepairAPI.bind_combined_gate_up_projection(
        {}, provider_expert_sha256=CURRENT_PROVIDER_EXPERT_SHA256
    )
    assert bound["resident_gate_up_provider_sha256"] == CURRENT_PROVIDER_EXPERT_SHA256
    assert bound["resident_gate_up_projection"] == "combined_4096_bf16_f_linear_v1"


def test_w28_gate_refreshes_and_requires_provider_activation_after_forward() -> None:
    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    score = source.index("admission = _score_admission_windows(")
    changed_input = source.index("if changed_input_w28_only:", score)
    terminal = source.index('"schema": "banana-smasher-changed-input-w28-terminal-v1"', changed_input)
    activation = source.index(
        "projection_binding = engine.sealed_gate_up_runtime_witness(\n"
        "                require_activation=True\n"
        "            )",
        changed_input,
    )
    assert score < activation < terminal
    assert '"projection_runtime_witness": projection_binding' in source[terminal:terminal + 1200]


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


def test_static_w28_adapter_supplies_model_swiglu_limit_to_historical_provider() -> None:
    """Supply the model clamp before replacing the detached meta expert."""
    from repair_api.modern_green_resident import _bind_historical_swiglu_limit

    provider = (
        Path(__file__).parents[1] / "assets" / "static_w28_modern_green_clean_u0.py"
    ).read_text()
    layer_loop = provider[provider.index("        for layer in range(first, last + 1):") :]
    constructor = layer_loop[
        layer_loop.index("            resident = FullyResidentGroupedV7Experts(") :
        layer_loop.index("            m.model.layers[layer].mlp.experts = resident")
    ]
    assert "swiglu_limit=swiglu_limit" in constructor

    calls = []

    class HistoricalProvider:
        def __init__(self, layer, pilot=True, *, plane_source):
            calls.append((layer, pilot, plane_source))

    adapted = _bind_historical_swiglu_limit(HistoricalProvider, sealed_limit=42.0)
    implicit = adapted(layer=7, plane_source="implicit")
    explicit = adapted(layer=8, plane_source="explicit", swiglu_limit=17.0)
    assert calls == [(7, True, "implicit"), (8, True, "explicit")]
    assert implicit.limit == 42.0
    assert explicit.limit == 17.0


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


def test_exact102_w28_only_consumes_admission_and_returns_before_full64():
    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    gate = source.index('w28_only = os.environ.get("W28_ONLY", "0") == "1"')
    admission = source.index('"banana-smasher-exact102-public-admission-v2"', gate)
    terminal = source.index('"banana-smasher-exact102-imported-w28-v1"', admission)
    full64 = source.index("full_started = time.perf_counter()", terminal)
    assert gate < admission < terminal < full64
    assert 'int(exact102_admission.get("provenance_members", -1)) != 22016' in source
    assert '"exact102_virtual_artifact_sha256": exact102_admission[' in source


def test_static_w28_provider_honors_explicit_immutable_trainer_binding():
    source = (Path(__file__).parents[1] / "modern_green_resident.py").read_text()
    resolver = source[source.index("def _resolve_trainer_source(") :]
    resolver = resolver[: resolver.index("\n\ndef ", 1)]
    configured = resolver.index('configured = config.get("trainer_source")')
    bundled = resolver.index('Path(__file__).resolve().parent / "assets"')
    assert configured < bundled
    assert 'Path(str(configured)).expanduser().resolve()' in resolver
    assert 'str(configured_sha)' in resolver


def test_sealed_pre_binding_owns_accepted_trainer_identity() -> None:
    from repair_api.modern_green_resident import _require_file
    from repair_api.sealed_pre_forward import bind_sealed_pre_resident_config

    authoritative_sha = "b8481000e126d218b8c949eaa0d0a297a95f408bc34f15e46041c9236db0db85"
    stale_sha = "cc0520e00a6cc5b979c638e3f1fd98ae92c882f3cf9f48cbcdf3fa55fad343cc"
    config = {
        "trainer_source": "/stale/inherited/trainer.py",
        "trainer_source_sha256": "0" * 64,
    }
    bind_sealed_pre_resident_config(config)

    expected = (
        Path(__file__).parents[1] / "assets" / "static_w28_modern_green_clean_u0.py"
    ).resolve()
    assert Path(config["trainer_source"]) == expected
    assert config["trainer_source_sha256"] == authoritative_sha
    _require_file(expected, authoritative_sha, "trainer source")
    with pytest.raises(ArtifactError, match="trainer source SHA mismatch"):
        _require_file(expected, stale_sha, "trainer source")


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
    assert 'config.setdefault("resident_validation_expert_implementation", "accepted_static_w28")' in binding
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
    assert "route_observer=route_observer" in source


def test_authentic_route_observer_preserves_all_five_producer_tensors() -> None:
    from repair_api.modern_green_resident import _sealed_builder_accumulate_routes
    from repair_api.resident_full64_accept import _AuthenticRouteCaptureMode

    hidden = torch.zeros((3, 2), dtype=torch.bfloat16)
    routed_output = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],
         [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        dtype=torch.bfloat16,
    )
    top_k_index = torch.tensor([[4, 2], [2, 4], [4, 2]], dtype=torch.int64)
    top_k_weights = torch.tensor(
        [[0.75, 0.25], [0.5, 0.5], [0.125, 0.875]], dtype=torch.float32
    )
    control = _sealed_builder_accumulate_routes(
        hidden, routed_output, top_k_index, top_k_weights
    )
    observer = _AuthenticRouteCaptureMode()
    observed = _sealed_builder_accumulate_routes(
        hidden, routed_output, top_k_index, top_k_weights,
        route_observer=observer,
    )

    expected_experts = top_k_index.reshape(-1).to(torch.int64)
    expected_weights = top_k_weights.reshape(-1, 1).float()
    expected_weighted = (routed_output * expected_weights).to(hidden.dtype)
    routed = expected_weighted.reshape(3, 2, 2)
    expert_order = torch.argsort(top_k_index, dim=1, stable=True)
    expected_ordered = torch.gather(
        routed, 1, expert_order.unsqueeze(-1).expand_as(routed)
    )
    expected = {
        "w2_output": routed_output,
        "expert_indices": expected_experts,
        "route_weights": expected_weights,
        "weighted_buffer": expected_weighted,
        "ordered_weighted_buffer": expected_ordered,
    }

    assert torch.equal(observed, control)
    for field, tensor in expected.items():
        captured = getattr(observer, field)
        assert captured is not None
        assert captured.dtype == tensor.dtype
        assert captured.shape == tensor.shape
        assert captured.stride() == tensor.stride()
        assert torch.equal(
            captured.detach().contiguous().view(torch.uint8),
            tensor.detach().contiguous().view(torch.uint8),
        )


def test_run6519_source_return_assembly_smoke() -> None:
    from repair_api.resident_full64_accept import _replay_source_return_assemblies

    hidden = torch.zeros((2, 1), dtype=torch.bfloat16)
    top_k_index = torch.tensor([[9, 3], [3, 9]], dtype=torch.int64)
    weighted_slot_major = torch.tensor(
        [[1.0], [100.0], [10.0], [1000.0]], dtype=torch.bfloat16
    )
    expected = torch.tensor([[11.0], [1100.0]], dtype=torch.bfloat16)

    observed = _replay_source_return_assemblies(
        hidden, top_k_index, weighted_slot_major, expected, expected,
    )

    assert observed["status"] == "RETURN_ASSEMBLY_PRIMITIVE_LOCALIZED"
    assert observed["instrument_control_self_compare_exact"] is True
    assert observed["accepted_matches_authentic_control"] is True
    assert observed["provider_flat_matches_authentic_provider"] is True
    assert observed["first_assembly_operation_divergence"] is None


def test_run6520_temporal_interleaving_adjudication() -> None:
    from repair_api.resident_full64_accept import _adjudicate_temporal_interleaving

    authentic = torch.tensor([[1.0], [2.0]], dtype=torch.bfloat16)
    source_interleaved = authentic.clone()
    post_materialized = torch.tensor([[1.0], [3.0]], dtype=torch.bfloat16)

    observed = _adjudicate_temporal_interleaving(
        authentic, source_interleaved, post_materialized,
    )

    assert observed["status"] == "TEMPORAL_INTERLEAVING_LOCALIZED"
    assert observed["instrument_control_self_compare_exact"] is True
    assert observed["post_materialized_matches_authentic_control"] is False
    assert observed["first_temporal_operation_divergence"] == (
        "deferred_all_route_materialization_before_index_add"
    )


def test_run6521_source_workspace_lifetime_adjudication() -> None:
    from repair_api.resident_full64_accept import _adjudicate_source_workspace_lifetime

    authentic = torch.tensor([[1.0], [2.0]], dtype=torch.bfloat16)
    retained = torch.tensor([[1.0], [3.0]], dtype=torch.bfloat16)
    fresh = authentic.clone()

    observed = _adjudicate_source_workspace_lifetime(authentic, retained, fresh)

    assert observed["status"] == "SOURCE_PROJECTION_WORKSPACE_LIFETIME_LOCALIZED"
    assert observed["instrument_control_self_compare_exact"] is True
    assert observed["retained_alias_matches_authentic_control"] is False
    assert observed["fresh_workspace_matches_authentic_control"] is True
    assert observed["first_workspace_operation_divergence"] == (
        "source_projection_operand_workspace_before_F.linear"
    )
    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    assert "value.clone(memory_format=torch.contiguous_format)" in source
    assert "weight.clone(memory_format=torch.contiguous_format)" in source


def test_run6522_authentic_source_projection_control_duplicates_exact_call() -> None:
    import types
    import torch.nn.functional as F
    from repair_api.resident_full64_accept import (
        _run_one_layer_with_authentic_projection_control,
    )

    class Experts(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down_proj = torch.nn.Parameter(torch.randn(256, 2, 2))

        def forward(self, hidden, top_k_index, top_k_weights):
            return F.linear(hidden, self.down_proj[204])

    experts = Experts()
    layer = types.SimpleNamespace(mlp=types.SimpleNamespace(experts=experts))
    engine = object()
    hidden = torch.randn(3, 2)
    ids = torch.zeros((1, 3), dtype=torch.int64)

    def fake_layer_run(_engine, _layer, value, _ids):
        index = torch.full((value.shape[0], 1), 204, dtype=torch.int64)
        weights = torch.ones((value.shape[0], 1))
        return _layer.mlp.experts(value, index, weights), None

    with patch(
        "repair_api.resident_full64_accept._run_one_layer_with_attention",
        fake_layer_run,
    ):
        output, control = _run_one_layer_with_authentic_projection_control(
            engine, layer, hidden, ids,
        )

    assert torch.equal(output, F.linear(hidden, experts.down_proj[204]))
    assert control["status"] == "AUTHENTIC_SOURCE_PROJECTION_CONTROL_EXACT"
    assert control["instrument_control_self_compare_exact"] is True
    assert control["source_expert_invocation_count"] == 2
    assert (
        control["authentic_projection_path_return"]
        == control["immediate_duplicate_projection_path_return"]
        == control["post_layer_source_projection_path_return"]
    )
    assert control["source_caller_context"]["status"] == "SOURCE_CALLER_CONTEXT_PARITY"
    assert control["source_implementation_dispatch"]["status"] == (
        "SOURCE_IMPLEMENTATION_DISPATCH_PARITY"
    )


def test_run6524_localizes_experts_implementation_dispatch() -> None:
    import functools
    import types
    from repair_api.resident_full64_accept import (
        _run_one_layer_with_authentic_projection_control,
    )

    class Experts(torch.nn.Module):
        def source(self, hidden, top_k_index, top_k_weights):
            del top_k_index, top_k_weights
            return hidden + 1

        @functools.wraps(source)
        def forward(self, hidden, top_k_index, top_k_weights):
            del top_k_index, top_k_weights
            return hidden + 2

    experts = Experts()
    layer = types.SimpleNamespace(mlp=types.SimpleNamespace(experts=experts))
    hidden = torch.zeros((3, 2))
    ids = torch.zeros((1, 3), dtype=torch.int64)

    def fake_layer_run(_engine, _layer, value, _ids):
        index = torch.zeros((value.shape[0], 1), dtype=torch.int64)
        weights = torch.ones((value.shape[0], 1))
        return _layer.mlp.experts(value, index, weights), None

    with patch(
        "repair_api.resident_full64_accept._run_one_layer_with_attention",
        fake_layer_run,
    ):
        output, control = _run_one_layer_with_authentic_projection_control(
            object(), layer, hidden, ids,
        )

    assert torch.equal(output, hidden + 2)
    assert control["instrument_control_self_compare_exact"] is True
    dispatch = control["source_implementation_dispatch"]
    assert dispatch["status"] == "SOURCE_IMPLEMENTATION_DISPATCH_LOCALIZED"
    assert dispatch["decorated_vs_undecorated_exact"] is False
    assert control["undecorated_source_body_return"]["sha256"] != (
        control["authentic_projection_path_return"]["sha256"]
    )
    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    assert "RUN6524_SOURCE_IMPLEMENTATION_DISPATCH_ONLY" in source
    assert '"SOURCE_IMPLEMENTATION_DISPATCH"' in source


def test_run6873_compares_grouped_mm_operations_with_source_flinear() -> None:
    import sys
    import types
    from repair_api.resident_full64_accept import _call_with_grouped_mm_operation_probe

    class Experts:
        config = types.SimpleNamespace(_experts_implementation="grouped_mm")

    calls = 0

    def fake_grouped(inputs, weights, offsets, *, bias=None, is_transposed=False):
        nonlocal calls
        del offsets, bias, is_transposed
        calls += 1
        return torch.zeros((inputs.shape[0], weights.shape[1]), dtype=inputs.dtype)

    namespace = {"_grouped_linear": fake_grouped}
    exec("def grouped_mm_experts_forward(): pass", namespace)
    moe = types.ModuleType("transformers.integrations.moe")
    moe.grouped_mm_experts_forward = namespace["grouped_mm_experts_forward"]
    integrations = types.ModuleType("transformers.integrations")
    integrations.moe = moe
    transformers = types.ModuleType("transformers")
    transformers.integrations = integrations
    modules = {
        "transformers": transformers,
        "transformers.integrations": integrations,
        "transformers.integrations.moe": moe,
    }
    with patch.dict(sys.modules, modules):
        def forward(hidden, _index, _weights):
            offsets = torch.tensor([hidden.shape[0]], dtype=torch.int32)
            first = namespace["_grouped_linear"](
                hidden, torch.ones((1, 2, 2)), offsets
            )
            return namespace["_grouped_linear"](
                first, torch.ones((1, 2, 2)), offsets
            )

        output, operations = _call_with_grouped_mm_operation_probe(
            Experts(), forward, torch.ones((2, 2)),
            torch.zeros((2, 1), dtype=torch.int64), torch.ones((2, 1)),
        )

    assert calls == 2
    assert namespace["_grouped_linear"] is fake_grouped
    assert torch.equal(output, torch.zeros((2, 2)))
    assert [item["grouped_vs_source_exact"] for item in operations] == [False, True]
    assert operations[0]["source"].endswith("grouped_mm_experts_forward")
    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    assert "expected_world_size = 1 if grouped_mm_operation_probe else 2" in source
    provider_source = (Path(__file__).parents[1] / "modern_green_resident.py").read_text()
    assert "if grouped_mm_singleton_probe:" in provider_source


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


def test_exact102_resident_runner_refuses_unbound_virtual_backpack() -> None:
    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    assert 'config.get("backpack_virtual_terminal_path")' in source
    assert 'config.get("backpack_virtual_terminal_sha256")' in source
    assert 'config.get("backpack_virtual_manifest_sha256")' in source
    assert "EXACT102_MIXED_ARTIFACT_BINDING_MISMATCH" in source


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
