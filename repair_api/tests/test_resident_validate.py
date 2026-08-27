from __future__ import annotations

from contextlib import nullcontext
import inspect
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from repair_api.api import ArtifactError, ResidentRepairAPI
from repair_api.modern_green_resident import (
    ModernGreenResidentEngine,
    _checkpoint_topk_route,
    _install_runtime_modules,
    _published_pre_recipe_policy,
)


def test_published_pre_lr_scale_is_the_only_policy_delta() -> None:
    base = {
        "recipe_id": "published_pre_lower_lr_warmup16_cosine64_v1",
        "published_pre_checkpoint_sha256": (
            "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70"
        ),
    }
    baseline_lrs, baseline_multiplier, baseline_windows = _published_pre_recipe_policy(
        dict(base, lr_scale=0.125), 0
    )
    candidate_lrs, candidate_multiplier, candidate_windows = _published_pre_recipe_policy(
        dict(base, lr_scale=0.03125), 0
    )

    assert candidate_lrs == baseline_lrs
    assert candidate_windows == baseline_windows == [28, 56]
    assert candidate_multiplier == baseline_multiplier * 0.25


def test_validation_forward_reuses_one_sealed_builder_dynamic_cache() -> None:
    positional = inspect.getsource(ModernGreenResidentEngine._positional)
    forward = inspect.getsource(ModernGreenResidentEngine._run_layers)

    assert "past_key_values=cache" in positional
    assert "DynamicCache" not in positional
    assert forward.count("DynamicCache(config=self.student.config)") == 2
    assert forward.index("cache = DynamicCache") < forward.index("for index in range")
    assert "past_key_values=active_cache" in forward
    assert "else cache" in forward
    assert "past_key_values=DynamicCache" not in forward


def test_published_pre_validation_uses_imported_sealed_builder_attention(monkeypatch) -> None:
    import repair_api.official_k2_resident_score as sealed_builder_binding

    calls = []
    monkeypatch.setenv("BR_ATTN_IMPL", "sdpa")
    monkeypatch.delenv("FAST_K2_SEALED_FULL_WEIGHT_BF16", raising=False)
    monkeypatch.delenv("FAST_K2_SEALED_PROJECTION_BF16", raising=False)
    monkeypatch.delenv("FAST_K2_SEALED_NO_SWIGLU_CLAMP", raising=False)
    monkeypatch.setattr(
        sealed_builder_binding,
        "_configured_attention_implementation",
        lambda config: calls.append(dict(config)) or "eager",
    )
    engine = ModernGreenResidentEngine.__new__(ModernGreenResidentEngine)
    engine.config = {
        "resident_validation_proof": True,
        "recipe_id": "published_pre_lower_lr_warmup16_cosine64_v1",
    }
    engine.published_pre_recipe = True
    engine.corpus_path = Path("corpus")
    engine.teacher_root = Path("teacher")
    engine.model_root = Path("model")
    engine.base = SimpleNamespace(T=SimpleNamespace(CKPT=None, DEV=None))
    engine.student = SimpleNamespace()
    engine.torch = SimpleNamespace(
        manual_seed=lambda _seed: None,
        cuda=SimpleNamespace(manual_seed_all=lambda _seed: None),
    )

    engine._configure_base()

    assert calls == [{}]
    assert __import__("os").environ["BR_ATTN_IMPL"] == "eager"
    assert __import__("os").environ["FAST_K2_SEALED_FULL_WEIGHT_BF16"] == "1"
    assert __import__("os").environ["FAST_K2_SEALED_PROJECTION_BF16"] == "0"
    assert __import__("os").environ["FAST_K2_SEALED_NO_SWIGLU_CLAMP"] == "1"
    assert engine.sealed_builder_binding == {
        "builder_source_sha256": "ed6a1d0f0666027372a726ea96d7d6f7c3487b60da8c5d8f8be591330ccb7137",
        "attention_implementation": "eager",
        "expert_arithmetic": "sealed-full-width-bf16-transformed-weight-matmul",
        "expert_swiglu_clamp": "disabled-to-match-run1698-joint-v7-expert",
        "fixture": "canonical-eval-corpus-token_ids-padded-to-2048",
        "readout": "full-softmax-gather-at-teacher-idx-fp16",
    }


def test_static_w28_identical_pre_and_zero_lr_u1_use_same_sealed_provider(monkeypatch) -> None:
    import os
    import torch
    import repair_api.official_k2_resident_score as sealed_builder_binding

    monkeypatch.setattr(
        sealed_builder_binding,
        "_configured_attention_implementation",
        lambda _config: "eager",
    )
    pre_tensors = {"weight": torch.tensor([1.0, -2.0, 3.0])}
    zero_lr_u1_tensors = {name: value.clone() for name, value in pre_tensors.items()}
    assert all(torch.equal(pre_tensors[name], zero_lr_u1_tensors[name]) for name in pre_tensors)

    def bind(identity_config, tensors):
        for key in (
            "BR_ATTN_IMPL",
            "FAST_K2_SEALED_FULL_WEIGHT_BF16",
            "FAST_K2_SEALED_PROJECTION_BF16",
            "FAST_K2_SEALED_NO_SWIGLU_CLAMP",
        ):
            monkeypatch.delenv(key, raising=False)
        engine = ModernGreenResidentEngine.__new__(ModernGreenResidentEngine)
        engine.config = identity_config
        engine.published_pre_recipe = True
        engine.corpus_path = Path("corpus")
        engine.teacher_root = Path("teacher")
        engine.model_root = Path("model")
        engine.base = SimpleNamespace(T=SimpleNamespace(CKPT=None, DEV=None))
        engine.torch = SimpleNamespace(
            manual_seed=lambda _seed: None,
            cuda=SimpleNamespace(manual_seed_all=lambda _seed: None),
        )
        engine._configure_base()
        provider_tensors = {name: value.clone() for name, value in tensors.items()}
        provider_output = {
            "binding": engine.sealed_builder_binding,
            "attention": os.environ["BR_ATTN_IMPL"],
            "full_weight_bf16": os.environ["FAST_K2_SEALED_FULL_WEIGHT_BF16"],
            "projection_bf16": os.environ["FAST_K2_SEALED_PROJECTION_BF16"],
            "no_swiglu_clamp": os.environ["FAST_K2_SEALED_NO_SWIGLU_CLAMP"],
        }
        return provider_tensors, provider_output

    pre_provider_tensors, pre_output = bind(
        {"resident_validation_proof": True}, pre_tensors
    )
    u1_provider_tensors, u1_output = bind(
        {
            "static_w28_gate": {
                "windows": [28],
                "updates": [1, 2, 4],
                "red_kld": 0.20,
                "dead_kld": 0.28,
            }
        },
        zero_lr_u1_tensors,
    )

    assert all(
        torch.equal(pre_provider_tensors[name], u1_provider_tensors[name])
        for name in pre_provider_tensors
    )
    assert pre_output == u1_output


def test_published_pre_w28_uses_sealed_run1698_mb2_fixture(monkeypatch, tmp_path: Path) -> None:
    import repair_api.official_k2_resident_score as sealed_builder_binding

    calls = []
    monkeypatch.setattr(
        sealed_builder_binding,
        "_physical_canary_batch_windows",
        lambda selected, configured, balanced: (
            calls.append((selected, configured, balanced)) or (28, 56)
        ),
    )
    corpus = [{"token_ids": [1] * 1024, "real_len": 1024} for _ in range(57)]
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(__import__("json").dumps(corpus))
    teacher_root = tmp_path / "teacher"
    teacher_root.mkdir()

    class FakeTensor:
        shape = (1024, 8192)

        def __getitem__(self, _key):
            return self

        def to(self, **_kwargs):
            return self

        def contiguous(self):
            return self

    monkeypatch.setattr(
        "repair_api.balanced64._load_torch",
        lambda _path: {"idx": FakeTensor(), "logprob": FakeTensor()},
    )
    ids = []

    class FakeIds:
        def __setitem__(self, _key, value):
            ids.append(value)

    torch = SimpleNamespace(
        long="long", int64="int64", float16="float16",
        full=lambda *_args, **_kwargs: FakeIds(),
        tensor=lambda value, **_kwargs: tuple(value),
    )
    engine = ModernGreenResidentEngine.__new__(ModernGreenResidentEngine)
    engine.published_pre_recipe = True
    engine.rank = 1
    engine.torch = torch
    engine.student = SimpleNamespace(device="cuda")
    engine.corpus_path = corpus_path
    engine.config = {
        "resident_validation_proof": True,
        "validation_corpus": str(corpus_path),
        "sealed_builder_window_microbatch": 2,
    }

    prepared = engine.preload_validation((28,), teacher_root)

    assert calls == [((28,), 2, (28, 56))]
    assert prepared["windows"] == (28,)
    assert prepared["physical_windows"] == (28, 56)
    assert prepared["physical_batch_size"] == 2
    assert set(prepared["ids"]) == {28, 56}
    assert set(prepared["teachers"]) == {28}


def test_published_pre_full64_streams_exact_mb2_groups_without_changing_windows(
    monkeypatch, tmp_path: Path,
) -> None:
    corpus = [{"token_ids": [1] * 1024, "real_len": 1024} for _ in range(84)]
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(__import__("json").dumps(corpus))
    teacher_root = tmp_path / "teacher"
    teacher_root.mkdir()

    class FakeTensor:
        shape = (1024, 8192)

        def __getitem__(self, _key):
            return self

        def to(self, **_kwargs):
            return self

        def contiguous(self):
            return self

    monkeypatch.setattr(
        "repair_api.balanced64._load_torch",
        lambda _path: {"idx": FakeTensor(), "logprob": FakeTensor()},
    )

    class FakeIds:
        def __setitem__(self, _key, _value):
            pass

    torch = SimpleNamespace(
        long="long", int64="int64", float16="float16",
        full=lambda *_args, **_kwargs: FakeIds(),
        tensor=lambda value, **_kwargs: tuple(value),
    )
    engine = ModernGreenResidentEngine.__new__(ModernGreenResidentEngine)
    engine.published_pre_recipe = True
    engine.rank = 1
    engine.torch = torch
    engine.student = SimpleNamespace(device="cuda")
    engine.corpus_path = corpus_path
    engine.config = {
        "resident_validation_proof": True,
        "validation_corpus": str(corpus_path),
        "sealed_builder_window_microbatch": 2,
    }

    windows = tuple(range(20, 84))
    prepared = engine.preload_validation(windows, teacher_root)

    assert prepared["windows"] == windows
    assert prepared["physical_windows"] == windows
    assert prepared["physical_batch_size"] == 2
    assert set(prepared["ids"]) == set(windows)
    assert set(prepared["teachers"]) == set(windows)


def test_published_pre_proof_installs_runtime_assets_from_canonical_pin(monkeypatch, tmp_path: Path) -> None:
    import hashlib
    import repair_api.modern_green_resident as resident

    extension = tmp_path / "grouped.so"
    extension.write_bytes(b"extension")
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_text('{"swiglu_limit": 10.0}')
    loaded = []

    class Expert:
        def __init__(self, layer, pilot=True, *, plane_source, swiglu_limit):
            pass

    def fake_load(name, path):
        loaded.append((name, Path(path)))
        return SimpleNamespace(FullyResidentGroupedV7Experts=Expert)

    monkeypatch.setattr(resident, "_load_source_module", fake_load)
    _install_runtime_modules({
        "resident_validation_proof": True,
        "recipe_id": "published_pre_lower_lr_warmup16_cosine64_v1",
        "model_root": str(model_root),
        "fast_k2_extension": str(extension),
        "fast_k2_extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "fast_k2_module_name": "grouped",
        "fast_k2_wrapper_source": "/foreign/old-wrapper.py",
        "fast_k2_wrapper_source_sha256": "f" * 64,
        "fast_v7_expert_source": "/foreign/old-expert.py",
        "fast_v7_expert_source_sha256": "e" * 64,
    })

    assets = Path(resident.__file__).parent / "assets"
    assert loaded == [
        ("fast_k2_grouped", assets / "fast_k2_grouped.py"),
        ("fast_v7_expert_base", assets / "fast_v7_expert_base.py"),
    ]


def test_activation_checkpoint_recompute_restores_exact_pre_layer_cache_snapshot() -> None:
    snapshot = inspect.getsource(ModernGreenResidentEngine._snapshot_layer_cache)
    restore = inspect.getsource(ModernGreenResidentEngine._restore_layer_cache)
    forward = inspect.getsource(ModernGreenResidentEngine._run_layers)
    route = inspect.getsource(_checkpoint_topk_route)

    assert "layer_cache.keys" in snapshot
    assert "layer_cache.values" in snapshot
    assert "keys.detach().clone()" in snapshot
    assert "values.detach().clone()" in snapshot
    assert "layer_cache.is_initialized" in snapshot
    assert "cumulative_length" in snapshot
    assert "layer_cache.keys = keys" in restore
    assert "layer_cache.values = values" in restore
    assert "layer_cache.is_initialized = initialized" in restore
    assert "layer_cache.cumulative_length = cumulative_length" in restore
    assert "snapshots[index] = self._snapshot_layer_cache" in forward
    assert "self._restore_layer_cache(cache.layers[index], snapshots[index])" in forward
    assert "if snapshots is not None and index in snapshots:" in forward
    assert "active_cache = (" in forward
    assert "past_key_values=active_cache" in forward
    assert 'recompute = _invocation["count"] > 0' in forward
    assert '_invocation["count"] += 1' in forward
    assert "_start: int = start" in forward
    assert "_snapshots: dict[" in forward
    assert "and self.checkpoint_use_reentrant" in forward
    assert "hidden = hidden.detach().requires_grad_(True)" in forward
    assert "route_indices[_index] = result[2].detach()" in forward
    assert "fresh_indices = torch.topk(" in route
    assert "weights = scores.gather(1, indices)" in route
    assert 'ArtifactError("checkpoint route replay geometry drift")' in route
    assert "context_fn=" not in forward
    assert ".crop(" not in forward


def test_public_validate_delegates_to_the_existing_trainer_object(tmp_path: Path) -> None:
    calls = []

    class Trainer:
        def validate(self, windows, teacher_root):
            calls.append((id(self), tuple(windows), Path(teacher_root)))
            return {
                "kld_mean": 0.25,
                "top1": 7,
                "runtime_counters": {
                    "timed_model_payload_reads": 0,
                    "timed_score_file_reads": 0,
                },
            }

    trainer = Trainer()
    api = ResidentRepairAPI.__new__(ResidentRepairAPI)
    receipt = tmp_path / "VALIDATE.json"

    result = api.validate(trainer, [28], tmp_path / "teacher", receipt_path=receipt)

    assert calls == [(id(trainer), (28,), tmp_path / "teacher")]
    assert result["public_api"] == {
        "method": "ResidentRepairAPI.validate",
        "version": "resident-trainer-validate-v1",
    }
    assert result["runtime_counters"]["trainer_object_id"] == id(trainer)
    assert receipt.is_file()


def test_trainer_validate_preloads_before_timing_and_restores_train_mode(tmp_path: Path) -> None:
    events = []

    class Model:
        training = True

        def eval(self):
            events.append("eval")
            self.training = False
            return self

        def train(self, mode=True):
            events.append(("train", mode))
            self.training = bool(mode)
            return self

    engine = ModernGreenResidentEngine.__new__(ModernGreenResidentEngine)
    engine.student = SimpleNamespace(model=Model())
    engine.torch = SimpleNamespace(no_grad=lambda: nullcontext())
    engine.config = {"checkpoint_sha256": "f9bffe04"}

    def preload(self, windows, teacher_root):
        events.append(("preload", self.student.model.training, tuple(windows), Path(teacher_root)))
        return {
            "windows": tuple(windows),
            "teacher_root": Path(teacher_root),
            "corpus_sha256": "corpus-sha",
        }

    def fingerprint(self):
        events.append(("fingerprint", self.student.model.training))
        return {"sha256": "device-sha", "ranks": [{"rank": 0}, {"rank": 1}]}

    def validate_preloaded(self, prepared):
        events.append(("score", self.student.model.training, prepared["windows"]))
        return {
            "kld_mean": 0.125,
            "top1": 11,
            "positions": 1024,
            "support": 8192,
            "windows": list(prepared["windows"]),
            "runtime_counters": {
                "timed_model_payload_reads": 0,
                "timed_score_file_reads": 0,
            },
        }

    engine.preload_validation = MethodType(preload, engine)
    engine._device_parameter_fingerprint = MethodType(fingerprint, engine)
    engine._validate_preloaded = MethodType(validate_preloaded, engine)

    result = engine.validate([28], tmp_path / "teacher")

    assert events == [
        ("preload", True, (28,), tmp_path / "teacher"),
        ("fingerprint", True),
        "eval",
        ("score", False, (28,)),
        ("train", True),
    ]
    assert engine.student.model.training is True
    assert result["checkpoint_sha256"] == "f9bffe04"
    assert result["device_parameter_fingerprint"]["sha256"] == "device-sha"
    assert result["runtime_counters"]["model_mode_restored"] is True
    assert result["runtime_counters"]["timed_model_payload_reads"] == 0
    assert result["runtime_counters"]["timed_score_file_reads"] == 0


def test_published_pre_validation_proof_preloads_only_the_u1_training_dose() -> None:
    loaded_teacher_windows = []

    class Tensor:
        def unsqueeze(self, _dimension):
            return self

        def to(self, _device):
            return self

    class TrainerData:
        @staticmethod
        def load_corpus():
            return object()

        @staticmethod
        def window_ids(_corpus, window):
            return Tensor(), 100 + window

        @staticmethod
        def teacher_rows(window):
            loaded_teacher_windows.append(window)
            return (f"idx-{window}", f"lp-{window}", f"p-{window}")

    engine = ModernGreenResidentEngine.__new__(ModernGreenResidentEngine)
    engine.controlled_arm = False
    engine.published_pre_recipe = True
    engine.global_step = 0
    engine.rank = 1
    engine.student = SimpleNamespace(device="cuda:0")
    engine.base = SimpleNamespace(T=TrainerData)
    engine.config = {
        "resident_validation_proof": True,
        "recipe_id": "published_pre_lower_lr_warmup16_cosine64_v1",
        "published_pre_checkpoint_sha256": (
            "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70"
        ),
    }

    engine._load_training_data()

    assert list(engine.ids_cache) == [28, 56]
    assert engine.real_lengths == {28: 128, 56: 156}
    assert loaded_teacher_windows == [28, 56]
    assert list(engine.teacher_cache) == [28, 56]


def test_published_pre_proof_validates_updates_and_validates_same_engine(monkeypatch, tmp_path: Path) -> None:
    import repair_api.api as api_module
    import repair_api.modern_green_resident as engine_module

    published = "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70"
    events = []

    class Artifact:
        root = tmp_path
        windows = (28,)
        manifest = {
            "checkpoints": {
                "PRE": {"sha256": published, "identity_sha256": "identity", "next_update": 0}
            }
        }

        @staticmethod
        def checkpoint_key(value):
            return str(value)

        @staticmethod
        def checkpoint_path(value):
            return tmp_path / f"{value}.pt"

    class Engine:
        def __init__(self, **kwargs):
            self.config = kwargs["config"]
            events.append(("construct", id(self), kwargs["payload"]))

        def validate(self, windows, teacher_root):
            events.append(("validate", id(self), tuple(windows), Path(teacher_root)))
            ordinal = sum(event[0] == "validate" for event in events)
            return {
                "kld_mean": 0.3 - ordinal * 0.01,
                "top1": ordinal,
                "runtime_counters": {
                    "timed_model_payload_reads": 0,
                    "timed_score_file_reads": 0,
                },
                "device_parameter_fingerprint": {"sha256": f"device-{ordinal}"},
            }

        def advance_to(self, target, *, gather_state=True):
            events.append(("advance", id(self), target, gather_state))
            return {"value": target}, {
                "optimizer_steps": 1,
                "scheduler_steps": 1,
                "gradient_norm": 1.0,
                "parameter_delta_norm": 1.0,
                "loss": 0.5,
                "timings": {"wall_seconds": 1.0},
                "process_gpu_evidence": {},
                "rank_reports": [{"rank": 0}, {"rank": 1}],
                "model_engine": "resident",
                "frozen_surfaces": [],
                "trainable_surfaces": ["luts"],
            }, None

        @staticmethod
        def broadcast_persisted(value):
            return value

        @staticmethod
        def close():
            events.append(("close",))

    api = ResidentRepairAPI.__new__(ResidentRepairAPI)
    api.artifact = Artifact()
    api._identity = MethodType(lambda self, checkpoint, windows: {"basis_sha256": "basis"}, api)
    api._persist_continuation_checkpoint = MethodType(
        lambda self, target, state, report, **kwargs: {
            "checkpoint": "UPDATE_001", "checkpoint_path": "checkpoints/UPDATE_001.pt",
            "checkpoint_sha256": "post-sha", "checkpoint_identity_sha256": "post-identity",
            "state_sha256": "state-sha",
        },
        api,
    )
    monkeypatch.setattr(api_module, "_load_torch", lambda path: {"state": {"luts": {}, "norms": {}, "outputs": {}}})
    monkeypatch.setattr(engine_module, "ModernGreenResidentEngine", Engine)
    config = {
        "authorized_api": True, "world_size": 2, "rank": 0, "local_only": True,
        "trainer_source": "trainer.py", "model_root": "model", "asset_root": "asset",
        "parent_root": "parent", "l034_roster": "roster", "teacher_root": "teacher",
        "corpus": "corpus", "manifest": "manifest", "delta_dir": "delta",
        "vq3b_dir": "vq3b", "master_addr": "spark-2", "master_port": 29598,
        "shared_optimizer_scheduler_lineage": "fresh-published-pre",
        "basis_sha256": "basis", "checkpoint_sha256": published,
        "layer_split": {0: [0, 20], 1: [21, 42]},
        "recipe_id": "published_pre_lower_lr_warmup16_cosine64_v1",
        "published_pre_checkpoint_sha256": published,
        "resident_validation_proof": True,
        "validation_windows": [28],
        "validation_teacher_root": str(tmp_path / "teacher"),
        "expected_pre_validation": {"kld_mean": 0.29, "top1": 1},
    }

    invalid_config = dict(config, validation_windows=[28, 56])
    with pytest.raises(
        ArtifactError,
        match=r"resident validation proof requires validation_windows=\[28\]",
    ):
        api.continue_two_spark_real(
            "PRE", [1], config=invalid_config, receipt_path=tmp_path / "INVALID_PROOF.json"
        )
    assert events == []

    wrong_target_config = dict(
        config,
        expected_pre_validation={"kld_mean": 0.13678686618849925, "top1": 882},
    )
    with pytest.raises(ArtifactError, match="published PRE validation mismatch"):
        api.continue_two_spark_real(
            "PRE", [1], config=wrong_target_config, receipt_path=tmp_path / "WRONG_PRE.json"
        )
    assert [event[0] for event in events] == ["construct", "validate", "close"]
    assert not any(event[0] == "advance" for event in events)
    events.clear()

    result = api.continue_two_spark_real(
        "PRE", [1], config=config, receipt_path=tmp_path / "PROOF.json"
    )

    engine_ids = [event[1] for event in events if event[0] in {"construct", "validate", "advance"}]
    assert len(set(engine_ids)) == 1
    assert [event[0] for event in events] == ["construct", "validate", "advance", "validate", "close"]
    assert [event for event in events if event[0] == "advance"] == [
        ("advance", engine_ids[0], 1, False)
    ]
    milestone = result["milestones"][0]
    assert milestone["resident_state_persisted"] is False
    assert milestone["checkpoint_path"] is None
    assert result["validation"]["pre"]["kld_mean"] == 0.29
    assert round(result["validation"]["post"]["kld_mean"], 2) == 0.28
    assert result["validation"]["post_less_than_pre"] is True
    assert result["validation"]["pre"]["device_parameter_fingerprint"]["sha256"] != result["validation"]["post"]["device_parameter_fingerprint"]["sha256"]


def test_public_fresh_pre_u1_u4_binds_controlled_schedule_and_fresh_steps(
    monkeypatch, tmp_path: Path
) -> None:
    import hashlib
    import json
    import repair_api.api as api_module
    import repair_api.modern_green_resident as engine_module

    published = "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70"
    source_rows = [
        {"global_update": 21, "microbatch_category_order": ["prose", "reasoning", "agentic", "chat", "code", "multilingual"], "windows": [276, 247, 66, 57, 412, 159]},
        {"global_update": 22, "microbatch_category_order": ["reasoning", "agentic", "chat", "code", "multilingual", "prose"], "windows": [404, 226, 37, 421, 445, 81]},
        {"global_update": 23, "microbatch_category_order": ["agentic", "chat", "code", "multilingual", "prose", "reasoning"], "windows": [113, 70, 69, 148, 45, 303]},
        {"global_update": 24, "microbatch_category_order": ["chat", "code", "multilingual", "prose", "reasoning", "agentic"], "windows": [109, 463, 267, 19, 329, 333]},
    ]
    schedule = tmp_path / "NEXT_ADJUSTMENT.json"
    schedule.write_text(json.dumps({
        "schema": "banana-smasher-next-adjustment-v1",
        "unchanged_fields": {
            "train_bank_membership_sha256": "3553fce00efdb6d452171e6d5c429adc31580dedbf63eb821f81bc82406983b3",
        },
        "exact_parameter_delta": {"to": {
            "windows_per_optimizer_update": 6,
            "category_loss_weight": 1 / 6,
            "pipeline_microbatch": 2,
            "pipeline_groups_per_update": 3,
            "group_gradient_scale": 1 / 3,
        }},
        "expected_first_four_update_boundary": {"updates": source_rows},
    }, sort_keys=True))
    schedule_sha = hashlib.sha256(schedule.read_bytes()).hexdigest()
    events = []
    RealEngine = engine_module.ModernGreenResidentEngine

    class Artifact:
        root = tmp_path
        windows = (28,)
        manifest = {"checkpoints": {"PRE": {
            "sha256": published, "identity_sha256": "pre-identity", "next_update": 0,
        }}}

        @staticmethod
        def checkpoint_key(value):
            return str(value)

        @staticmethod
        def checkpoint_path(value):
            return tmp_path / f"{value}.pt"

    class Engine:
        def __init__(self, **kwargs):
            config = kwargs["config"]
            payload = kwargs["payload"]
            assert config["controlled_window_schedule"] == str(schedule)
            assert config["controlled_window_schedule_sha256"] == schedule_sha
            assert "optimizer" not in payload and "optimizer_state" not in payload
            assert "scheduler" not in payload and "scheduler_state" not in payload
            self.policy = RealEngine.__new__(RealEngine)
            self.policy.config = config
            self.policy.controlled_arm = False
            self.policy.controlled_arm_id = None
            self.policy.published_pre_recipe = True
            self.policy.published_pre_controlled_windows = True
            self.policy.global_step = 0
            self.policy.controlled_schedule = json.loads(schedule.read_text())
            self.policy.ids_cache = {
                window: object() for row in source_rows for window in row["windows"]
            }
            self.policy._load_controlled_window_schedule()
            events.append(("construct", config))

        def advance_to(self, target, *, gather_state=True):
            events.append(("advance", target, gather_state))
            windows = self.policy.controlled_windows[target - 1]
            return {"value": target}, {
                "optimizer_steps": 1, "scheduler_steps": 1,
                "gradient_norm": 1.0, "parameter_delta_norm": 1.0, "loss": 0.5,
                "timings": {"wall_seconds": 1.0}, "windows": windows,
                "process_gpu_evidence": {},
                "rank_reports": [{"rank": 0, "windows": windows}, {"rank": 1, "windows": windows}],
                "model_engine": "resident", "frozen_surfaces": [], "trainable_surfaces": ["luts"],
            }, None

        def validate(self, windows, teacher_root):
            ordinal = sum(event[0] == "static_w28" for event in events)
            score = (0.19, 0.21)[ordinal]
            events.append(("static_w28", tuple(windows), teacher_root, score))
            return {"kld_mean": score, "top1": 0.0, "runtime_counters": {}}

        @staticmethod
        def broadcast_persisted(value):
            return value

        @staticmethod
        def close():
            events.append(("close",))

    api = ResidentRepairAPI.__new__(ResidentRepairAPI)
    api.artifact = Artifact()
    api._identity = MethodType(lambda self, checkpoint, windows: {"basis_sha256": "basis"}, api)
    api._persist_continuation_checkpoint = MethodType(
        lambda self, target, state, report, **kwargs: {
            "checkpoint": f"UPDATE_{target:03d}",
            "checkpoint_path": f"checkpoints/UPDATE_{target:03d}.pt",
            "checkpoint_sha256": f"post-sha-{target}",
            "checkpoint_identity_sha256": f"post-identity-{target}",
            "state_sha256": f"state-sha-{target}",
        },
        api,
    )
    monkeypatch.setattr(
        api_module, "_load_torch",
        lambda path: {"state": {"luts": {}, "norms": {}, "outputs": {}}},
    )
    monkeypatch.setattr(engine_module, "ModernGreenResidentEngine", Engine)
    config = {
        "authorized_api": True, "world_size": 2, "rank": 0, "local_only": True,
        "trainer_source": "trainer.py", "model_root": "model", "asset_root": "asset",
        "parent_root": "parent", "l034_roster": "roster", "teacher_root": "teacher",
        "corpus": "corpus", "manifest": "manifest", "delta_dir": "delta",
        "vq3b_dir": "vq3b", "master_addr": "spark-2", "master_port": 29598,
        "shared_optimizer_scheduler_lineage": "fresh-published-pre",
        "basis_sha256": "basis", "checkpoint_sha256": published,
        "layer_split": {0: [0, 20], 1: [21, 42]},
        "recipe_id": "published_pre_lower_lr_warmup16_cosine64_v1",
        "published_pre_checkpoint_sha256": published,
        "fresh_published_pre_lineage": True,
        "controlled_window_schedule": str(schedule),
        "controlled_window_schedule_sha256": schedule_sha,
        "controlled_window_schedule_source_rows": [21, 22, 23, 24],
        "controlled_windows_per_update": 6,
    }

    result = api.continue_two_spark_real(
        "PRE", [1, 2, 3, 4], config=config, receipt_path=tmp_path / "U1_U4.json"
    )

    assert [event[1:3] for event in events if event[0] == "advance"] == [
        (1, True), (2, True), (3, True), (4, True),
    ]
    assert result["controlled_window_schedule_sha256"] == schedule_sha
    assert result["controlled_window_schedule_binding"] == {
        "source_sha256": schedule_sha,
        "source_row_labels": [21, 22, 23, 24],
        "requested_boundaries": [1, 2, 3, 4],
        "windows_per_update": 6,
    }
    assert result["status"] == "PASS"
    assert [row["target_update"] for row in result["milestones"]] == [1, 2, 3, 4]
    assert all(row["optimizer_steps"] == row["scheduler_steps"] == 1 for row in result["milestones"])
    assert [row["controlled_window_schedule_source_row"] for row in result["milestones"]] == [21, 22, 23, 24]
    assert [row["rank_reports"][0]["windows"] for row in result["milestones"]] == [
        row["windows"] for row in source_rows
    ]
