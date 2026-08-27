import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, cast

from repair_api.modern_green_resident import ModernGreenResidentEngine
from repair_api.official_k2_resident_score import (
    OfficialK2ResidentRankEngine,
    OfficialK2ResidentScorer,
    _aggregate_score_phase_profiles,
    _drop_cold_file_cache,
    _drop_cold_model_cache,
    _effective_score_window_batch_size,
    _physical_canary_batch_windows,
    _sealed_pair_groups,
    _rebase_admission_lut_sources,
    _deserialize_resident_storage_ipc,
    _serialize_resident_storage_ipc,
    _validate_public_score_windows,
)
from repair_api import resident_full64_accept


def _write_w28_receipt(path: Path, *, rank: int = 0, task_id: Any = None) -> str:
    value = {
        "schema": "banana-smasher-resident-w28-admission-v1",
        "status": "PASS",
        "task_id": task_id or resident_full64_accept.W28_ADOPTION_TASK,
        "rank": rank,
        "canonical_code_commit": "0eebc78245129bcdc47fbb08964f6c2145b7ff7b",
        "basis_sha256": resident_full64_accept.BASIS,
        "checkpoint_sha256": resident_full64_accept.CHECKPOINT,
        "admission_wall_seconds": 54.0,
        "measurement": {
            "windows": [28],
            "kld_mean": resident_full64_accept.W28_KLD,
            "top1": resident_full64_accept.W28_TOP1,
        },
    }
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_full64_relaunch_adopts_exact_prior_w28_without_replaying_admission() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "W28.json"
        digest = _write_w28_receipt(path, rank=1)
        adopt = getattr(resident_full64_accept, "_adopt_w28_admission")

        row = adopt(path, digest, rank=1)

        assert row["receipt_sha256"] == digest
        assert row["measurement"]["kld_mean"] == resident_full64_accept.W28_KLD
        assert row["admission_adopted"] is True


def test_changed_input_continuation_adopts_its_exact_w28_and_uses_fresh_batch_root() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "W28.json"
        digest = _write_w28_receipt(path, rank=0, task_id="t_5e0f4049")
        adopt = getattr(resident_full64_accept, "_adopt_w28_admission")

        row = adopt(path, digest, rank=0, expected_task_id="t_5e0f4049")

        assert row["task_id"] == "t_5e0f4049"
        source = inspect.getsource(resident_full64_accept.main)
        assert 'ADOPT_W28_KEEP_PROVIDER' in source
        assert 'Path(config["score_resume_root"])' in source
        assert 'root / "receipts",' not in source


def test_full64_relaunch_rejects_tampered_prior_w28_receipt() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "W28.json"
        digest = _write_w28_receipt(path)
        path.write_bytes(path.read_bytes() + b" ")
        adopt = getattr(resident_full64_accept, "_adopt_w28_admission")

        try:
            adopt(path, digest, rank=0)
        except RuntimeError as exc:
            assert "SHA" in str(exc)
        else:
            raise AssertionError("tampered W28 receipt was adopted")


def test_full64_runner_preserves_exact_mb2_pairs_and_only_parallelizes_across_pairs() -> None:
    source = inspect.getsource(resident_full64_accept.main)
    gate = source.index('admission.get("kld_mean") != W28_KLD')
    score_pair = source.index('config["score_window_batch_size"] = 2')
    pair_streams = source.index('config["score_pair_stream_concurrency"] = 1', score_pair)
    rank_pipeline = source.index('config["score_pipeline_overlap"] = True', pair_streams)
    physical_pair = source.index('config["sealed_builder_window_microbatch"] = 2', rank_pipeline)
    engine = source.index("ModernGreenResidentEngine(")

    assert score_pair < pair_streams < rank_pipeline < physical_pair < engine < gate
    assert 'config["score_window_batch_size"] = 4' not in source
    assert 'config["sealed_builder_window_microbatch"] = 4' not in source
    assert 'config["score_window_batch_size"] = 8' not in source
    assert 'config["sealed_builder_window_microbatch"] = 8' not in source


def test_pair_stream_pool_is_bounded_and_reused_across_full64_groups() -> None:
    created: list[object] = []

    def stream(*, device: object) -> object:
        value = object()
        created.append(value)
        return value

    engine = cast(Any, object.__new__(ModernGreenResidentEngine))
    engine.student = SimpleNamespace(device="cuda:0")
    engine.torch = SimpleNamespace(cuda=SimpleNamespace(Stream=stream))

    first = engine._score_pair_stream_pool(3)
    second = engine._score_pair_stream_pool(3)

    assert first is second
    assert tuple(first) == tuple(created)
    assert len(created) == 3
    try:
        engine._score_pair_stream_pool(2)
    except Exception as exc:
        assert "concurrency drift" in str(exc)
    else:
        raise AssertionError("pair stream pool geometry drift was accepted")


def test_pair_stream_handoffs_record_every_cross_stream_tensor_lifetime() -> None:
    source = inspect.getsource(ModernGreenResidentEngine._validate_preloaded)

    # Default-stream inputs consumed on a side stream must retain their storage
    # until that stream completes. Side-stream outputs consumed by the default
    # stream need the reciprocal lifetime binding before the stream pool is reused.
    assert source.count("pair_id_tensor.record_stream(stream)") == 2
    assert "pair_hidden.record_stream(stream)" in source
    assert "pair_hidden.record_stream(launch_stream)" in source
    assert "pair_final.record_stream(launch_stream)" in source


def test_pair_scheduler_keeps_the_sealed_roster_pairing_and_groups_only_whole_pairs() -> None:
    windows = tuple(range(64))
    groups = _sealed_pair_groups(windows, concurrency=3)

    assert groups[0] == ((0, 1), (2, 3), (4, 5))
    assert groups[-1] == ((60, 61), (62, 63))
    assert tuple(window for group in groups for pair in group for window in pair) == windows
    assert all(len(pair) == 2 for group in groups for pair in group)
    # The production runner executes ModernGreenResidentEngine.validate, so the
    # exact pair scheduler and pipeline must live on that active path rather than
    # only on the unused OfficialK2ResidentRankEngine scorer.
    score_source = inspect.getsource(ModernGreenResidentEngine._validate_preloaded)
    workspace_source = inspect.getsource(
        ModernGreenResidentEngine._attention_workspace_for
    )
    send_source = inspect.getsource(ModernGreenResidentEngine._batch_p2p_isend)
    assert "hidden.split(2, dim=0)" in score_source
    assert "for pair_id_tensor, stream in zip(pair_ids, pair_streams)" in score_source
    assert "pair_output = self._run_layers(" in score_source
    assert "pair_hidden, pair_id_tensor, False" in score_source
    assert "score_pair_stream_concurrency" in score_source
    assert "_sealed_pair_groups" in score_source
    assert "_batch_p2p_isend" in score_source
    assert "previous_send.wait()" in score_source
    assert "torch.cuda.current_stream(device=device).cuda_stream" in workspace_source
    assert "workspaces[workspace_key] = current" in workspace_source
    assert "batch_isend_irecv" in send_source

    for rejected in ((1,), (1, 2, 3)):
        try:
            _sealed_pair_groups(rejected, concurrency=3)
        except Exception as exc:
            assert "whole pairs" in str(exc)
        else:
            raise AssertionError("an incomplete sealed pair was scheduled")


def test_production_preserves_the_exact_w28_fwht_backend() -> None:
    source = inspect.getsource(resident_full64_accept.main)
    gate = source.index('admission.get("kld_mean") != W28_KLD')
    production_geometry = source.index('config["score_window_batch_size"] = 2')
    production = source.index("validate_full64_batches(", gate)

    engine = source.index("ModernGreenResidentEngine(")
    assert engine < gate < production_geometry < production
    assert "set_fwht_backend" not in source
    assert 'set_fwht_backend("quack")' not in source


def test_phase_profile_fan_in_binds_every_window_and_the_five_minute_gate() -> None:
    rank_profiles = [
        [
            {"batch_windows": [0, 1, 2, 3], "embedding_ms": 2.0,
             "layer_forward_ms": 30.0, "consumer_wait_ms": 4.0},
            {"batch_windows": [4, 5, 6, 7], "embedding_ms": 3.0,
             "layer_forward_ms": 31.0, "consumer_wait_ms": 5.0},
        ],
        [
            {"batch_windows": [0, 1, 2, 3], "activation_wait_ms": 4.0,
             "layer_forward_ms": 40.0, "readout_ms": 2.0, "logits_ms": 9.0,
             "teacher_gather_ms": 3.0, "binary64_reduce_ms": 6.0, "glue_ms": 1.0},
            {"batch_windows": [4, 5, 6, 7], "activation_wait_ms": 5.0,
             "layer_forward_ms": 41.0, "readout_ms": 2.0, "logits_ms": 10.0,
             "teacher_gather_ms": 4.0, "binary64_reduce_ms": 7.0, "glue_ms": 1.0},
        ],
    ]

    receipt = _aggregate_score_phase_profiles(
        rank_profiles, ordered_windows=tuple(range(8)), post_load_wall_seconds=299.5,
        configured_batch_size=4,
    )

    assert receipt["status"] == "PROFILE_ONLY"
    assert receipt["full64_gate_pass"] is False
    assert receipt["post_load_under_300_seconds"] is True
    assert receipt["window_count"] == 8
    assert receipt["batch_count"] == 2
    assert receipt["phase_milliseconds"]["rank0_layer_forward"] == 61.0
    assert receipt["phase_milliseconds"]["rank1_teacher_gather"] == 7.0
    assert receipt["per_batch"][1]["batch_windows"] == [4, 5, 6, 7]


def test_phase_profile_fan_in_rejects_rank_window_drift() -> None:
    try:
        _aggregate_score_phase_profiles(
            [[{"batch_windows": [0]}], [{"batch_windows": [1]}]],
            ordered_windows=(0,), post_load_wall_seconds=1.0, configured_batch_size=1,
        )
    except Exception as exc:
        assert "window coverage drift" in str(exc)
    else:
        raise AssertionError("rank profile window drift was accepted")


def test_frozen_base_dependencies_are_hash_bound_and_importable() -> None:
    init_source = inspect.getsource(OfficialK2ResidentRankEngine.__init__)
    path_source = inspect.getsource(OfficialK2ResidentRankEngine._prepare_import_paths)

    assert "self.lp4_pack_path: LP4_PACK_SOURCE_SHA256" in init_source
    assert "self.lp4_train_path: LP4_TRAIN_SOURCE_SHA256" in init_source
    assert "self.builder_source_path: T8192_BUILDER_SOURCE_SHA256" in init_source
    assert "self.lp4_pack_path.parent" in path_source
    assert "self.lp4_train_path.parent" in path_source
    assert "self.builder_source_path.parent" in path_source


def test_public_score_window_gate_allows_exact_w28_canary_or_full_balanced64() -> None:
    full = tuple(range(64))
    assert _validate_public_score_windows((28,), full) == "W28_CANARY"
    assert _validate_public_score_windows(full, full) == "FULL64"
    for rejected in ((56,), (28, 56), tuple(reversed(full))):
        try:
            _validate_public_score_windows(rejected, full)
        except Exception as exc:
            assert "exact W28 canary or 64 ordered Balanced64 windows" in str(exc)
        else:
            raise AssertionError(f"unsupported public score geometry accepted: {rejected}")


def test_w28_canary_clamps_fullrail_batch_without_changing_configured_geometry() -> None:
    assert _effective_score_window_batch_size(4, 1) == 1
    assert _effective_score_window_batch_size(4, 64) == 4
    for configured, count in ((0, 1), (4, 0)):
        try:
            _effective_score_window_batch_size(configured, count)
        except Exception as exc:
            assert "positive" in str(exc)
        else:
            raise AssertionError("invalid score window geometry was accepted")


def test_w28_canary_executes_the_exact_aligned_batch4_shape() -> None:
    balanced64 = tuple(range(64))
    assert _physical_canary_batch_windows((28,), 4, balanced64) == (28, 29, 30, 31)
    assert _physical_canary_batch_windows((28,), 1, balanced64) == (28,)
    assert _physical_canary_batch_windows(balanced64, 4, balanced64) == balanced64
    try:
        _physical_canary_batch_windows((28,), 4, tuple(range(30)))
    except Exception as exc:
        assert "complete aligned batch" in str(exc)
    else:
        raise AssertionError("incomplete W28 physical batch was accepted")


def test_resident_validation_profiles_forward_p2p_and_readout_per_batch() -> None:
    source = inspect.getsource(ModernGreenResidentEngine._validate_preloaded)
    assert '"weight_reconstruction_ms": 0.0' in source
    assert '"forward_ms"' in source
    assert '"p2p_ms"' in source
    assert '"readout_ms"' in source
    assert '"phase_profiles_by_rank"' in source
    assert '"mechanism_counter_delta"' in source
    assert "resident validation reconstructed weights inside a batch" in source
    assert "previous_hidden = hidden" in source
    assert "previous_send.wait()" in source
    assert "del ids" in source
    assert "del hidden, ids, final" in source


def test_resident_mechanism_counter_delta_is_exact() -> None:
    before = {"counters": {"projection_calls": 9, "reconstruction_calls": 0}}
    after = {"counters": {"projection_calls": 21, "reconstruction_calls": 0}}
    assert ModernGreenResidentEngine._mechanism_counter_delta(before, after) == {
        "projection_calls": 12,
        "reconstruction_calls": 0,
    }


def test_resident_mechanism_counter_delta_is_exact() -> None:
    before = {"counters": {"projection_calls": 9, "reconstruction_calls": 0}}
    after = {"counters": {"projection_calls": 21, "reconstruction_calls": 0}}
    assert ModernGreenResidentEngine._mechanism_counter_delta(before, after) == {
        "projection_calls": 12,
        "reconstruction_calls": 0,
    }


def test_one_batch_profile_is_fail_closed_before_full64() -> None:
    source = (Path(__file__).parents[1] / "physical_profile_batch.py").read_text()
    assert "WINDOWS = (28, 56, 68, 71)" in source
    assert '"PASS_PROFILE_ONLY"' in source
    assert '"scientific_acceptance": False' in source
    assert 'PHYSICAL_PROFILE_ATTEMPT' in source
    assert "PROFILE_RESIDENT_PROVIDER_NOT_OBSERVED" in source
    assert "PROFILE_WEIGHT_RECONSTRUCTION_CALL_NONZERO" in source
    assert "PROFILE_PACKED_PROJECTION_NOT_EXECUTED" in source


def test_resident_layers_share_the_sealed_builder_cache_across_layers() -> None:
    source = inspect.getsource(OfficialK2ResidentRankEngine._run_layers)
    assert source.count("layer_cache = DynamicCache") == 1
    assert source.index("layer_cache = DynamicCache") < source.index("for index in range")
    assert "pe, pos, mask, layer_cache, output, residual" in source
    assert "del layer_cache" in source


def test_gpu_resident_storage_descriptor_roundtrip_preserves_shared_lifetime() -> None:
    import torch

    owner = torch.arange(8, dtype=torch.float32)
    encoded = _serialize_resident_storage_ipc({"layer": 7, "master": owner})
    consumer = _deserialize_resident_storage_ipc(encoded)

    assert len(encoded) < 1_048_576
    assert consumer["layer"] == 7
    owner.add_(3)
    assert torch.equal(consumer["master"], owner)


def test_gpu_resident_storage_broker_retains_owner_and_avoids_consumer_allocations() -> None:
    publish = inspect.getsource(OfficialK2ResidentRankEngine._publish_brokered_resident_storage)
    consume = inspect.getsource(OfficialK2ResidentRankEngine._consume_brokered_resident_storage)
    asset = (Path(__file__).parents[1] / "assets" / "fast_v7_expert_base.py").read_text()
    provider_branch = asset.index('storage_provider = globals().get("_resident_storage_provider")')
    ordinary_allocation = asset.index("packed = torch.empty(")

    assert "self._gpu_storage_broker_owned[layer] = (source, expert)" in publish
    assert publish.index("self._gpu_storage_broker_owned[layer]") < publish.index(
        "self.rendezvous_store.set(key, _serialize_resident_storage_ipc(payload))"
    )
    assert 'self.rendezvous_store.get(key + "-consumed")' in publish
    assert 'self.rendezvous_store.set(key + "-consumed", b"1")' in consume
    assert provider_branch < ordinary_allocation


def test_shared_cache_releases_each_completed_layer_without_replacing_cache_identity() -> None:
    import torch

    entry = SimpleNamespace(
        keys=torch.ones((4, 2, 8, 3)),
        values=torch.ones((4, 2, 8, 3)),
        is_initialized=True,
    )
    cache = SimpleNamespace(layers=[entry])

    OfficialK2ResidentRankEngine._release_completed_layer_cache(cache, 0)

    assert cache.layers[0] is entry
    assert entry.keys.numel() == 0
    assert entry.values.numel() == 0
    assert entry.is_initialized is False
    source = inspect.getsource(OfficialK2ResidentRankEngine._run_layers)
    assert source.index("self._streamed_decoder_layer(") < source.index(
        "self._release_completed_layer_cache(layer_cache, index)"
    )


def test_score_resume_namespace_and_identity_bind_attention_mode() -> None:
    with tempfile.TemporaryDirectory() as directory:
        engine = object.__new__(OfficialK2ResidentRankEngine)
        engine.parent_root = Path(directory) / "parent"
        engine.checkpoint_sha256 = "a" * 64
        engine.rank = 0
        resume_root = Path(directory) / "rank-local-resume"
        engine.config = {
            "attention_implementation_override": "eager",
            "score_resume_root": str(resume_root),
        }
        eager = engine._score_resume_path()
        engine.config = {
            "attention_implementation_override": "sdpa",
            "score_resume_root": str(resume_root),
        }
        sdpa = engine._score_resume_path()
        assert eager.parent == resume_root.resolve()
        assert sdpa.parent == resume_root.resolve()
        assert eager != sdpa
        assert "EAGER" in eager.name
        assert "SDPA" in sdpa.name
        assert eager.name != f"SCORE_RESUME_{'a' * 64}_RANK0.json"
        assert sdpa.name != f"SCORE_RESUME_{'a' * 64}_RANK0.json"
    persist = inspect.getsource(OfficialK2ResidentRankEngine._persist_score_resume)
    load = inspect.getsource(OfficialK2ResidentRankEngine._load_score_resume)
    assert '"attention_implementation"' in persist
    assert '"attention_implementation"' in load


def test_reboot_volatile_lut_paths_rebind_only_to_hash_matched_durable_bytes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        parent = Path(directory) / "L007" / "parent"
        parent.mkdir(parents=True)
        manifest = parent / "QTIP_V7_MANIFEST.json"
        wire = parent / "L007.tlut.f16"
        manifest.write_bytes(b"manifest")
        wire.write_bytes(b"wire")
        admission = {
            "trainable_roster": {
                "luts": [{
                    "layer": 7,
                    "source_manifest": {
                        "path": "/dev/shm/reboot-stale/QTIP_V7_MANIFEST.json",
                        "sha256": hashlib.sha256(b"manifest").hexdigest(),
                    },
                    "wire": {
                        "source_path": "/dev/shm/reboot-stale/L007.tlut.f16",
                        "sha256": hashlib.sha256(b"wire").hexdigest(),
                    },
                }]
            }
        }

        rebound = _rebase_admission_lut_sources(admission, directory)

        row = rebound["trainable_roster"]["luts"][0]
        assert row["source_manifest"]["path"] == str(manifest.resolve())
        assert row["wire"]["source_path"] == str(wire.resolve())
        assert admission["trainable_roster"]["luts"][0]["source_manifest"]["path"].startswith("/dev/shm/")
        wire.write_bytes(b"drift")
        try:
            _rebase_admission_lut_sources(admission, directory)
        except Exception as exc:
            assert "SHA mismatch" in str(exc)
        else:
            raise AssertionError("digest drift was accepted")


def test_hash_bound_grouped_extension_loads_before_resident_expert() -> None:
    source = inspect.getsource(OfficialK2ResidentRankEngine.__init__)
    extension_env = source.index('os.environ["FAST_K2_EXTENSION"]')
    wrapper_load = source.index('self._load_module("fast_k2_grouped", self.fast_k2_wrapper_source)')
    expert_load = source.index('self._load_module("fast_v7_expert_base", self.expert_source)')
    trainer_load = source.index("self.trainer = self._load_module(")
    assert "self.fast_k2_extension: str(config[\"fast_k2_extension_sha256\"])" in source
    assert "self.fast_k2_wrapper_source: str(config[\"fast_k2_wrapper_source_sha256\"])" in source
    assert extension_env < wrapper_load < expert_load < trainer_load


def test_cold_source_cache_drop_preserves_immutable_file(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "member.q2v7wire"
        path.write_bytes(b"immutable-wire")
        calls = []
        monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
        monkeypatch.setattr(
            os,
            "posix_fadvise",
            lambda fd, offset, length, advice: calls.append((offset, length, advice)),
            raising=False,
        )

        assert _drop_cold_file_cache([path]) == (1, len(b"immutable-wire"))
        assert path.read_bytes() == b"immutable-wire"
        assert calls == [(0, 0, 4)]


def test_model_cache_drop_covers_only_root_safetensor_shards(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        shards = [root / "model-00001.safetensors", root / "model-00002.safetensors"]
        ignored = root / "model.safetensors.index.json"
        nested = root / "nested" / "other.safetensors"
        nested.parent.mkdir()
        for path in (*shards, ignored, nested):
            path.write_bytes(path.name.encode())
        seen = []
        monkeypatch.setattr(
            "repair_api.official_k2_resident_score._drop_cold_file_cache",
            lambda paths: seen.extend(paths) or (len(seen), sum(path.stat().st_size for path in seen)),
        )

        count, byte_count = _drop_cold_model_cache(root)

        assert seen == [path.resolve() for path in shards]
        assert count == 2
        assert byte_count == sum(path.stat().st_size for path in shards)


def test_model_cache_is_dropped_after_student_load_before_input_allocation() -> None:
    source = inspect.getsource(OfficialK2ResidentRankEngine.__init__)
    student = source.index("self.student = self.trainer.ShardStudent(")
    cache_drop = source.index("_drop_cold_model_cache(self.model_root)")
    load_inputs = source.index("self._load_inputs()")
    assert student < cache_drop < load_inputs


def test_incremental_cache_drop_is_receipted_per_loaded_layer(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        loaded = root / "L007" / "member.q2v7wire"
        pending = root / "L008" / "member.q2v7wire"
        for path in (loaded, pending):
            path.parent.mkdir(parents=True)
            path.write_bytes(b"wire")
        monkeypatch.setattr(
            "repair_api.official_k2_resident_score._drop_cold_file_cache",
            lambda paths: (len(tuple(paths)), sum(path.stat().st_size for path in paths)),
        )
        engine = object.__new__(OfficialK2ResidentRankEngine)
        engine.config = {"drop_parent_cache_incrementally_after_layer_load": True}
        engine.parent_root = root
        engine.status = {}
        engine.cold_source_cache_drop_files = 0
        engine.cold_source_cache_drop_bytes = 0
        engine.cold_source_files_pruned = 0
        engine.cold_source_bytes_pruned = 0

        engine._status(phase="loading", loaded_layer=7)

        assert engine.cold_source_cache_drop_files == 1
        assert engine.cold_source_cache_drop_bytes == 4
        assert loaded.read_bytes() == b"wire"
        assert pending.read_bytes() == b"wire"


def test_incremental_cache_drop_skips_roster_layer_absent_from_parent_root(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        root.mkdir(exist_ok=True)
        calls = []
        monkeypatch.setattr(
            "repair_api.official_k2_resident_score._drop_cold_file_cache",
            lambda paths: calls.append(tuple(paths)) or (0, 0),
        )
        engine = object.__new__(OfficialK2ResidentRankEngine)
        engine.config = {"drop_parent_cache_incrementally_after_layer_load": True}
        engine.parent_root = root
        engine.status = {}
        engine.cold_source_cache_drop_files = 0
        engine.cold_source_cache_drop_bytes = 0
        engine.cold_source_files_pruned = 0
        engine.cold_source_bytes_pruned = 0

        engine._status(phase="loading", loaded_layer=34)

        assert calls == []
        assert engine.status["loaded_layer"] == 34
        assert engine.cold_source_cache_drop_files == 0
        assert engine.cold_source_cache_drop_bytes == 0


def test_cache_drop_can_release_staggered_rank1_without_deleting_warm_sources() -> None:
    with tempfile.TemporaryDirectory() as directory:
        gate_dir = Path(directory)
        config = {
            "cold_load_gate_dir": str(gate_dir),
            "cold_load_generation": "cache-drop-generation",
            "cold_load_gate_timeout_seconds": 0.02,
        }
        rank0 = object.__new__(OfficialK2ResidentRankEngine)
        rank0.rank = 0
        rank0.config = dict(config)
        rank0.cold_source_files_pruned = 0
        rank0.cold_source_bytes_pruned = 0
        rank0.cold_source_cache_drop_files = 42
        rank0.cold_source_cache_drop_bytes = 34_021_112_832
        rank1 = object.__new__(OfficialK2ResidentRankEngine)
        rank1.rank = 1
        rank1.config = dict(config)

        rank0._publish_cold_load_pruned()
        rank1._wait_for_cold_load_turn()

        row = json.loads((gate_dir / "COLD_LOAD_RANK0_PRUNED.json").read_text())
        assert row["generation"] == "cache-drop-generation"
        assert row["cold_source_bytes_pruned"] == 0
        assert row["cold_source_cache_drop_bytes"] == 34_021_112_832


def test_rank1_gate_waits_for_rank0_transient_allocator_release(monkeypatch) -> None:
    events: list[str] = []
    reserved = iter((29 << 30, 41 << 30, 41 << 30, 18 << 30))
    cuda = SimpleNamespace(
        synchronize=lambda: events.append("synchronize"),
        empty_cache=lambda: events.append("empty_cache"),
        memory_reserved=lambda device=None: next(reserved),
        memory_allocated=lambda device=None: next(reserved),
    )
    engine = cast(Any, object.__new__(OfficialK2ResidentRankEngine))
    engine.rank = 0
    engine.device = "cuda:0"
    engine.torch = SimpleNamespace(cuda=cuda)
    monkeypatch.setattr("repair_api.official_k2_resident_score.gc.collect", lambda: events.append("gc"))

    engine._release_transient_resident_load_workspace()

    assert events == ["synchronize", "gc", "empty_cache", "synchronize"]
    assert engine.transient_load_memory_release == {
        "reserved_before_bytes": 29 << 30,
        "allocated_before_bytes": 41 << 30,
        "reserved_after_bytes": 41 << 30,
        "allocated_after_bytes": 18 << 30,
    }
    source = inspect.getsource(OfficialK2ResidentRankEngine.__init__)
    assert source.index("self._load_inputs()") < source.index(
        "self._release_transient_resident_load_workspace()"
    ) < source.index("self._publish_cold_load_pruned()")


def test_rank1_gate_precedes_checkpoint_payload_materialization() -> None:
    source = inspect.getsource(OfficialK2ResidentScorer.score)
    assert source.index("_wait_for_cold_load_gate(") < source.index(
        "payload = _load_score_checkpoint(checkpoint_path, checkpoint_sha, self.config)"
    )


def test_checkpoint_payload_reference_dies_before_resident_scoring() -> None:
    source = inspect.getsource(OfficialK2ResidentScorer.score)
    assert source.index("del payload") < source.index(
        "engine._release_post_bind_checkpoint_workspace()"
    ) < source.index("measured = engine.score()")


def test_rank0_checkpoint_payload_lifetime_is_resolved_before_rank1_gate_opens() -> None:
    source = inspect.getsource(OfficialK2ResidentRankEngine.__init__)
    assert source.index("self._bind_checkpoint_state(payload, admission)") < source.index(
        "_release_or_retain_checkpoint_payload("
    ) < source.index("self._release_post_bind_checkpoint_workspace()") < source.index(
        "self._publish_cold_load_pruned()"
    )


def test_rank0_rendezvous_arms_without_waiting_for_gated_rank1() -> None:
    class FakeDist:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def TCPStore(self, *args: Any, **kwargs: Any) -> object:
            self.calls.append(kwargs)
            return object()

    for rank in (0, 1):
        engine = cast(Any, object.__new__(OfficialK2ResidentRankEngine))
        engine.config = {
            "master_addr": "127.0.0.1",
            "master_port": 30293,
            "rendezvous_timeout_seconds": 120,
        }
        engine.rank = rank
        engine.dist = FakeDist()
        engine._init_rendezvous()
        assert engine.dist.calls[0]["wait_for_workers"] is (rank != 0)


def test_configured_resident_trainer_does_not_shadow_frozen_base_trainer() -> None:
    source = inspect.getsource(OfficialK2ResidentRankEngine.__init__)

    assert 'sys.modules["lp4_train"] = self.trainer' not in source
