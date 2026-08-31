from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from repair_api import ArtifactError, ResidentRepairAPI, ScoreResult
from repair_api.cli import main as cli_main
from repair_api.official_k2_resident_score import (
    ALTERNATE_PRE_CHECKPOINT_SHA256,
    BUILDER_EVAL_CORPUS_SHA256,
    CANONICAL_U0_CHECKPOINT_SHA256,
    CANONICAL_U0_LOCK_CORPUS_SHA256,
    CANONICAL_U1_CHECKPOINT_SHA256,
    SCORE_TRAIN_CORPUS_SHA256,
    TEACHER_INVENTORY_SHA256,
    ROUTED_K2_API_METHOD,
    ROUTED_K2_API_VERSION,
    ROUTED_K2_CLOSURE,
    ROUTED_K2_ROUTE_KIND,
    PUBLISHED_PRE_IDENTITY_SHA256,
    PUBLISHED_PRE_PAYLOAD_IDENTITY_SHA256,
    PUBLISHED_PRE_OPTIMIZER_SCHEDULER_LINEAGE,
    OfficialK2ResidentRankEngine,
    OfficialK2ResidentScorer,
    _published_pre_production_admitted,
    _write_q_lp_capture,
    _canonical_causal_score_tokens,
    _prune_loaded_parent_members,
    _validate_raw_u0_gates,
    _validate_qsfp_pin,
    authorize_production_score,
    validate_payload_identity,
)


PRE_SHA = CANONICAL_U0_CHECKPOINT_SHA256
PRE_KLD = 0.22939197531977115
PRE_TOP1 = 56533
WINDOWS = list(range(64))


class FakeOfficialBackend:
    calls = []

    def __init__(self, artifact, config):
        self.artifact = artifact
        self.config = config

    def score(self, checkpoint, windows):
        self.calls.append((checkpoint, tuple(windows)))
        return ScoreResult(
            checkpoint=checkpoint,
            windows=tuple(windows),
            positions=len(windows) * 1024,
            support=8192,
            kld=PRE_KLD,
            top1=PRE_TOP1,
            top1_rate=PRE_TOP1 / (len(windows) * 1024),
            artifact_root=str(self.artifact.root),
            spec="balanced64-v1",
            candidate_dir="fully-resident-official-k2",
            execution_mode="resident_in_memory",
            resident_load_seconds=1.25,
            timed_wall_seconds=2.5,
            runtime_counters={
                "timed_score_file_reads": 0,
                "file_reads_during_timed_score": 0,
            },
        )


class OfficialK2ResidentScoreTests(unittest.TestCase):
    def test_resident_rank_resolves_unused_missing_delta_input_without_restage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            vq3b = root / "vq3b"
            missing_delta = root / "dummy_delta"
            manifest.write_text("{}")
            engine = cast(Any, object.__new__(OfficialK2ResidentRankEngine))
            engine.asset_root = root
            engine.windows = (28,)
            engine.rank = 0
            engine.model_root = root / "model"
            engine.config = {
                "binrepair_manifest": str(manifest),
                "binrepair_delta_dir": str(missing_delta),
                "binrepair_vq3b_dir": str(vq3b),
                "attention_implementation": "eager",
            }

            engine._configure_base_environment()

            resolved = Path(os.environ["BR_DELTA_DIR"])
            self.assertFalse(missing_delta.exists())
            self.assertFalse(vq3b.exists())
            self.assertNotEqual(resolved, missing_delta)
            self.assertEqual(Path(os.environ["BR_VQ3B_DIR"]), resolved)
            self.assertEqual(
                (resolved / "DELTA_PACK.COMPLETE").read_text(),
                "RESIDENT_GROUPED_PROVIDER_UNUSED\n",
            )
            self.assertEqual(engine._resident_base_input_dir, resolved)
            loaded = {}
            engine._load_module = lambda name, path: loaded.setdefault(
                "module", SimpleNamespace(T=SimpleNamespace(CKPT=None, DEV=None), path=path)
            )
            base = engine._load_base()
            self.assertEqual(
                base.path,
                Path(__file__).parents[2] / "runtime" / "v7" / "runner" / "base_binrepair_e2e.py",
            )
            self.assertEqual(base.T.CKPT, str(engine.model_root))
            self.assertEqual(base.T.DEV, "cuda")
            (resolved / "DELTA_PACK.COMPLETE").unlink()
            resolved.rmdir()

    def test_q_lp_capture_is_complete_immutable_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q_lp.npy"
            array = np.zeros((1024, 8192), dtype=np.float16)
            receipt = _write_q_lp_capture(path, array)
            self.assertEqual(receipt["shape"], [1024, 8192])
            self.assertEqual(np.load(path, allow_pickle=False).shape, (1024, 8192))
            self.assertEqual(path.stat().st_mode & 0o222, 0)
            with self.assertRaisesRegex(ArtifactError, "already exists"):
                _write_q_lp_capture(path, array)

    def test_parity_tap_cli_calls_public_facade(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "tap.json"
            api = Mock()
            api.parity_tap.return_value = {"status": "PASS"}
            with patch("repair_api.cli.ResidentRepairAPI.open", return_value=api):
                rc = cli_main([
                    "parity-tap", "--artifact-root", str(root), "--checkpoint", "UPDATE_000",
                    "--window", "28", "--receipt", str(receipt), "--task-id", "t_5c0ea842",
                ])
            self.assertEqual(rc, 0)
            api.parity_tap.assert_called_once_with(
                "UPDATE_000", window=28, mode="current", receipt_path=receipt,
                preflight={"claim_path": None, "task_id": "t_5c0ea842", "peak_gib": 0.0},
            )

    def test_cli_scores_two_checkpoints_on_one_resident_api_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_receipt = root / "first.json"
            second_receipt = root / "second.json"
            first = SimpleNamespace(as_dict=lambda: {"checkpoint": "UPDATE_000", "kld": 0.1})
            second = SimpleNamespace(as_dict=lambda: {"checkpoint": "PRE", "kld": 0.2})
            api = Mock()
            api.score.side_effect = [first, second]
            with patch("repair_api.cli.ResidentRepairAPI.open", return_value=api):
                rc = cli_main([
                    "score", "--artifact-root", str(root), "--checkpoint", "UPDATE_000",
                    "--receipt", str(first_receipt), "--then-checkpoint", "PRE",
                    "--then-receipt", str(second_receipt),
                ])
            self.assertEqual(rc, 0)
            self.assertEqual([call.args[0] for call in api.score.call_args_list], ["UPDATE_000", "PRE"])
            self.assertEqual(api.score.call_args_list[0].kwargs["receipt_path"], first_receipt)
            self.assertEqual(api.score.call_args_list[1].kwargs["receipt_path"], second_receipt)

    def test_canonical_score_preserves_2048_context_and_pads_short_rows(self):
        tokens = list(range(2048))
        selected = _canonical_causal_score_tokens(tokens, real_len=2048, pad_token_id=1)
        self.assertEqual(selected, tokens)
        self.assertEqual(len(selected), 2048)

        short = list(range(1500))
        padded = _canonical_causal_score_tokens(short, real_len=1500, pad_token_id=7)
        self.assertEqual(padded[:1500], short)
        self.assertEqual(padded[1500:], [7] * 548)
        with self.assertRaisesRegex(ArtifactError, "fewer than 1024 real tokens"):
            _canonical_causal_score_tokens(tokens[:1000], real_len=1000, pad_token_id=1)

    def test_canonical_input_identities_are_distinct_and_frozen(self):
        self.assertEqual(BUILDER_EVAL_CORPUS_SHA256, "5aadaacbb486ae4f528c5e51ae70beff863337bd908fc727e6e49fc3ac520ebd")
        self.assertEqual(SCORE_TRAIN_CORPUS_SHA256, "16575db7fd180ca193aa13c4e642400b9ed416dbd0c36c3c5302422b31f5cbae")
        self.assertEqual(TEACHER_INVENTORY_SHA256, "017c7e9261b3e3701bd2f2dd53a03e46466b1dd2a3c5b4ecfb55b4c0aad04a92")
        self.assertEqual(CANONICAL_U0_LOCK_CORPUS_SHA256, "434a3f9eec14e54d348efde3265998c9521bb3579cba0d976b3e0a9b93d184c5")
        self.assertNotEqual(CANONICAL_U0_LOCK_CORPUS_SHA256, SCORE_TRAIN_CORPUS_SHA256)
        self.assertNotEqual(BUILDER_EVAL_CORPUS_SHA256, SCORE_TRAIN_CORPUS_SHA256)
        self.assertEqual(
            ROUTED_K2_CLOSURE["official_class_sha256"],
            "7687e39fc5b6bb34b30e8d4a79771affb472497f4d2f323adbe1e8e277746729",
        )
        from repair_api.official_k2_resident_score import OfficialK2ResidentScorer
        self.assertIn(
            '"official_physical_layer_sha256": _configured_expert_source_sha256(self.config)',
            inspect.getsource(OfficialK2ResidentScorer.score),
        )

    def test_raw_u0_payload_retains_immutable_lock_corpus_identity(self):
        payload = {
            "identity": {"corpus_sha256": "historical-lock"},
            "state": {},
            "checkpoint_loaded": False,
            "optimizer_state": {"state": {}},
            "scheduler_state": {"last_epoch": 0},
        }
        lock = {
            "checkpoint_loaded": False,
            "optimizer_state_entries": 0,
            "scheduler_epoch": 0,
        }
        with patch("repair_api.official_k2_resident_score.CANONICAL_U0_LOCK_CORPUS_SHA256", "historical-lock"), \
             patch("repair_api.official_k2_resident_score.CANONICAL_CORPUS_SHA256", "runtime-score"):
            _validate_raw_u0_gates(payload, lock)

    def test_two_host_route_requires_explicit_qsfp_pin(self):
        route = {
            "master_addr": "192.168.200.4",
            "qsfp_host_ip_by_rank": {"0": "192.168.200.4", "1": "192.168.200.6"},
            "distributed_socket_interface": "enp1s0f1np1",
        }
        self.assertEqual(_validate_qsfp_pin(route, rank=1)["local_qsfp_ip"], "192.168.200.6")
        for key, value in (
            ("master_addr", "192.168.88.14"),
            ("distributed_socket_interface", "enP7s7"),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ArtifactError, "QSFP"):
                _validate_qsfp_pin({**route, key: value}, rank=0)
        with self.assertRaisesRegex(ArtifactError, "QSFP"):
            _validate_qsfp_pin({**route, "qsfp_host_ip_by_rank": {"0": "192.168.200.4"}}, rank=1)

    def test_memory_preflight_credits_only_task_local_prunable_cold_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "parent"
            member = root / "L000" / "E000_w1.q2v7wire"
            member.parent.mkdir(parents=True)
            member.write_bytes(b"x" * 12)
            engine = cast(Any, object.__new__(OfficialK2ResidentRankEngine))
            engine.config = {
                "estimated_resident_bytes_by_rank": {0: 20},
                "estimated_peak_bytes_by_rank": {0: 20},
                "cuda_reserve_bytes": {0: 1},
                "prune_parent_after_resident_load": True,
            }
            engine.rank = 0
            engine.parent_root = root
            engine.torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: (10, 100)))
            OfficialK2ResidentRankEngine._preflight_memory(engine)
            self.assertEqual(engine.memory_preflight["prunable_cold_source_bytes"], 12)
            self.assertEqual(engine.memory_preflight["effective_free_after_cold_source_prune_bytes"], 22)
            self.assertEqual(engine.memory_preflight["margin_bytes"], 1)

    def test_prune_loaded_parent_members_is_root_bound_and_skips_l034(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "parent"
            inside = root / "L000" / "E000_w1.q2v7wire"
            l034 = Path(directory) / "l034" / "E000_w1.q2v7wire"
            outside = Path(directory) / "outside.q2v7wire"
            for path, payload in ((inside, b"inside"), (l034, b"l034"), (outside, b"outside")):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            count, byte_count = _prune_loaded_parent_members(
                parent_root=root,
                sources={
                    0: SimpleNamespace(layer=0, member_paths={(0, "w1"): inside}),
                    34: SimpleNamespace(layer=34, member_paths={(0, "w1"): l034}),
                },
            )
            self.assertEqual((count, byte_count), (1, len(b"inside")))
            self.assertFalse(inside.exists())
            self.assertTrue(l034.exists())
            with self.assertRaisesRegex(ArtifactError, "escapes staged root"):
                _prune_loaded_parent_members(
                    parent_root=root,
                    sources={1: SimpleNamespace(layer=1, member_paths={(0, "w1"): outside})},
                )
            self.assertTrue(outside.exists())

    def test_loading_status_incrementally_prunes_only_the_fully_loaded_parent_layer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "parent"
            loaded = root / "L000" / "E000_w1.q2v7wire"
            pending = root / "L001" / "E000_w1.q2v7wire"
            for path in (loaded, pending):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"wire")
            engine = cast(Any, object.__new__(OfficialK2ResidentRankEngine))
            engine.config = {"prune_parent_incrementally_after_layer_load": True}
            engine.parent_root = root
            engine.status = {}
            engine.cold_source_files_pruned = 0
            engine.cold_source_bytes_pruned = 0
            OfficialK2ResidentRankEngine._status(
                engine, phase="loading", loaded_layer=0, loaded_layers=1
            )
            self.assertFalse(loaded.exists())
            self.assertTrue(pending.exists())
            self.assertEqual(engine.cold_source_files_pruned, 1)
            self.assertEqual(engine.cold_source_bytes_pruned, 4)

    def test_cold_load_gate_rejects_stale_generation_and_releases_rank1_after_rank0_prune(self):
        with tempfile.TemporaryDirectory() as directory:
            gate_dir = Path(directory)
            rank1 = cast(Any, object.__new__(OfficialK2ResidentRankEngine))
            rank1.rank = 1
            rank1.config = {
                "cold_load_gate_dir": str(gate_dir),
                "cold_load_generation": "current",
                "cold_load_gate_timeout_seconds": 0.02,
            }
            (gate_dir / "COLD_LOAD_RANK0_PRUNED.json").write_text(
                json.dumps({"status": "PASS", "generation": "stale"}) + "\n"
            )
            with self.assertRaisesRegex(ArtifactError, "cold-load gate timeout"):
                OfficialK2ResidentRankEngine._wait_for_cold_load_turn(rank1)

            (gate_dir / "COLD_LOAD_RANK0_PRUNED.json").unlink()
            with self.assertRaisesRegex(ArtifactError, "cold-load gate timeout"):
                OfficialK2ResidentRankEngine._wait_for_cold_load_turn(rank1)

            rank0 = cast(Any, object.__new__(OfficialK2ResidentRankEngine))
            rank0.rank = 0
            rank0.config = dict(rank1.config)
            rank0.cold_source_files_pruned = 16128
            rank0.cold_source_bytes_pruned = 34021112832
            OfficialK2ResidentRankEngine._publish_cold_load_pruned(rank0)
            OfficialK2ResidentRankEngine._wait_for_cold_load_turn(rank1)
            row = json.loads((gate_dir / "COLD_LOAD_RANK0_PRUNED.json").read_text())
            self.assertEqual(row["generation"], "current")
            self.assertEqual(row["cold_source_bytes_pruned"], 34021112832)

    def test_rank0_tcpstore_listener_is_armed_before_heavy_resident_load(self):
        calls = []

        class FakeDist:
            def TCPStore(self, *args, **kwargs):
                store = object()
                calls.append(("TCPStore", args, kwargs, store))
                return store

            def is_initialized(self):
                return False

            def init_process_group(self, **kwargs):
                calls.append(("init_process_group", kwargs))

        engine = SimpleNamespace(
            config={
                "master_addr": "192.168.200.3",
                "master_port": 29698,
                "distributed_backend": "nccl",
                "rendezvous_timeout_seconds": 120,
                "process_group_timeout_seconds": 3600,
            },
            dist=FakeDist(),
            rank=0,
            _warm_p2p_communicator=lambda: calls.append(("warm_p2p",)),
        )
        OfficialK2ResidentRankEngine._init_rendezvous(engine)
        OfficialK2ResidentRankEngine._init_distributed(engine)
        self.assertEqual(calls[0][0], "TCPStore")
        self.assertEqual(calls[0][1][:4], ("192.168.200.3", 29698, 2, True))
        self.assertEqual(calls[0][2]["timeout"].total_seconds(), 120)
        self.assertIs(calls[1][1]["store"], calls[0][3])
        self.assertEqual(calls[1][1]["timeout"].total_seconds(), 3600)
        self.assertEqual(calls[2], ("warm_p2p",))
        source = inspect.getsource(OfficialK2ResidentRankEngine.__init__)
        self.assertLess(source.index("self._init_rendezvous()"), source.index("self.trainer.ShardStudent("))

    def test_host_local_shared_cuda_device_uses_gloo_control_group(self):
        calls = []

        class FakeDist:
            def is_initialized(self):
                return False

            def init_process_group(self, **kwargs):
                calls.append(kwargs)

        engine = SimpleNamespace(
            config={
                "master_addr": "127.0.0.1",
                "distributed_backend": "nccl",
                "shared_cuda_device_process_group": True,
                "process_group_timeout_seconds": 3600,
            },
            dist=FakeDist(),
            rank=0,
            rendezvous_store=object(),
        )
        OfficialK2ResidentRankEngine._init_distributed(engine)
        self.assertEqual(calls[0]["backend"], "gloo")

    def test_cross_host_score_collectively_warms_p2p_before_first_batch(self):
        events = []

        class Tensor:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        class Work:
            def wait(self):
                events.append("wait")

        class FakeDist:
            isend = object()
            irecv = object()

            def P2POp(self, operation, tensor, peer):
                events.append(("op", operation, peer))
                return operation, tensor, peer

            def batch_isend_irecv(self, operations):
                events.append(("batch", len(operations)))
                return [Work(), Work()]

        torch = SimpleNamespace(
            int32=object(),
            full=lambda *args, **kwargs: Tensor(0),
            empty_like=lambda tensor: Tensor(1),
            cuda=SimpleNamespace(synchronize=lambda: events.append("sync")),
        )
        engine = cast(Any, SimpleNamespace(
            rank=0,
            torch=torch,
            dist=FakeDist(),
            student=SimpleNamespace(device="cuda:0"),
        ))

        OfficialK2ResidentRankEngine._warm_p2p_communicator(engine)

        self.assertEqual(events[2], ("batch", 2))
        self.assertEqual(events[3:6], ["wait", "wait", "sync"])
        self.assertEqual(engine.p2p_communicator_warmup["status"], "PASS")
        source = inspect.getsource(OfficialK2ResidentRankEngine._init_distributed)
        self.assertIn("self._warm_p2p_communicator()", source)
        self.assertIn("shared_cuda_device_process_group", source)

    def test_cross_host_score_keeps_activation_transport_on_batched_p2p_api(self):
        events = []

        class Work:
            def wait(self):
                events.append("wait")

        class FakeDist:
            isend = object()
            irecv = object()

            def P2POp(self, operation, tensor, peer):
                events.append(("op", operation, tensor, peer))
                return operation, tensor, peer

            def batch_isend_irecv(self, operations):
                events.append(("batch", tuple(operations)))
                return [Work()]

        engine = cast(Any, SimpleNamespace(dist=FakeDist()))
        send_work = OfficialK2ResidentRankEngine._batch_p2p_isend(
            engine, "activation", dst=1
        )
        OfficialK2ResidentRankEngine._batch_p2p_recv(
            engine, "receive-buffer", src=0
        )

        self.assertIsInstance(send_work, Work)
        self.assertEqual(events[-1], "wait")
        self.assertEqual(
            [event[0] for event in events if isinstance(event, tuple)],
            ["op", "batch", "op", "batch"],
        )
        source = inspect.getsource(OfficialK2ResidentRankEngine._score_window)
        self.assertIn("self._batch_p2p_isend(hidden, dst=1)", source)
        self.assertIn("self._batch_p2p_recv(hidden, src=0)", source)
        self.assertNotIn("self.dist.isend(hidden", source)
        self.assertNotIn("self.dist.send(hidden", source)
        self.assertNotIn("self.dist.recv(hidden", source)

    def test_shared_cuda_device_transport_uses_ipc_descriptor_not_dist_send(self):
        source = inspect.getsource(OfficialK2ResidentRankEngine._score_window)
        self.assertIn("ForkingPickler.dumps(hidden)", source)
        self.assertIn('self.rendezvous_store.set(ipc_key, payload)', source)
        self.assertIn('self.rendezvous_store.get(previous_key + "-consumed")', source)
        self.assertIn("len(payload) > 65536", source)

    def test_full_prefill_matches_sealed_dynamic_cache_semantics(self):
        source = inspect.getsource(OfficialK2ResidentRankEngine._run_layers)
        self.assertEqual(source.count("DynamicCache(config=self.student.config)"), 1)
        self.assertIn("past_key_values=cache", source)
        self.assertNotIn("past_key_values=DynamicCache(config=self.student.config)", source)
        positional = inspect.getsource(OfficialK2ResidentRankEngine._positional)
        self.assertIn("past_key_values=cache", positional)
        self.assertNotIn("DynamicCache", positional)
        self.assertIn("unsqueeze(0)", positional)
        self.assertNotIn("expand(ids.shape[0]", positional)

    def test_causal_mask_is_fresh_per_window_not_cached_after_cache_mutation(self):
        source = inspect.getsource(OfficialK2ResidentRankEngine._positional)
        self.assertNotIn("_positional_cache", source)
        self.assertIn("create_sliding_window_causal_mask", source)

    def test_score_publishes_per_window_hot_path_progress(self):
        source = inspect.getsource(OfficialK2ResidentRankEngine.score)
        self.assertIn("SCORE_PROGRESS_RANK", source)
        self.assertIn('"completed_windows"', source)
        self.assertIn('"last_window_profile"', source)

    def test_official_scorer_binds_source_moe_and_enforces_u0_calibration(self):
        constructor = inspect.getsource(OfficialK2ResidentRankEngine.__init__)
        student = constructor.index("self.student = self.trainer.ShardStudent(")
        source_dispatch = constructor.index(
            "_bind_published_pre_experts_dispatch(", student
        )
        inputs = constructor.index("self._load_inputs()", source_dispatch)
        assert student < source_dispatch < inputs

        backend_score = inspect.getsource(OfficialK2ResidentScorer.score)
        measured = backend_score.index("measured = engine.score()")
        calibration = backend_score.index(
            "checkpoint_sha == CANONICAL_U0_CHECKPOINT_SHA256", measured
        )
        receipt = backend_score.index("quality_status =", calibration)
        assert measured < calibration < receipt

    def test_score_resume_round_trips_exact_binary64_terms_before_next_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = cast(Any, object.__new__(OfficialK2ResidentRankEngine))
            engine.parent_root = Path(directory) / "parent"
            engine.parent_root.mkdir()
            engine.rank = 1
            engine.windows = (28, 56)
            engine.checkpoint_sha256 = CANONICAL_U0_CHECKPOINT_SHA256
            terms = [float(index) / 1024.0 for index in range(1024)]
            per_window = [
                {"ordinal": 0, "window": 28, "positions": 1024, "support": 8192,
                 "kld_sum_binary64": sum(terms), "top1": 3}
            ]
            OfficialK2ResidentRankEngine._persist_score_resume(
                engine,
                completed_windows=1,
                terms=terms,
                top1=3,
                per_window=per_window,
                cumulative_scoring_wall_seconds=22.5,
            )
            loaded = OfficialK2ResidentRankEngine._load_score_resume(engine)
            self.assertEqual(loaded["completed_windows"], 1)
            self.assertEqual(loaded["terms"], terms)
            self.assertEqual(loaded["top1"], 3)
            self.assertEqual(loaded["per_window"], per_window)
            self.assertEqual(loaded["cumulative_scoring_wall_seconds"], 22.5)
            source_file = inspect.getsourcefile(OfficialK2ResidentRankEngine)
            self.assertIsNotNone(source_file)
            self.assertEqual(
                loaded["implementation_sha256"],
                OfficialK2ResidentRankEngine._score_implementation_sha256(),
            )
            resume_path = OfficialK2ResidentRankEngine._score_resume_path(engine)
            legacy = json.loads(resume_path.read_text())
            legacy["implementation_sha256"] = "ba94e819badadeace56ff0c48b780a1f4129f0d58daffdd2759de1d25bd98236"
            resume_path.write_text(json.dumps(legacy, sort_keys=True) + "\n")
            migrated = OfficialK2ResidentRankEngine._load_score_resume(engine)
            self.assertEqual(
                migrated["source_implementation_sha256"],
                "ba94e819badadeace56ff0c48b780a1f4129f0d58daffdd2759de1d25bd98236",
            )
            self.assertEqual(migrated["implementation_sha256"], loaded["implementation_sha256"])


    def test_score_resumes_at_next_window_and_reduces_saved_terms_in_order(self):
        class FakeDist:
            @staticmethod
            def broadcast_object_list(rows, src):
                return None

            @staticmethod
            def all_gather_object(rows, value):
                rows[:] = [dict(value), dict(value)]

        with tempfile.TemporaryDirectory() as directory:
            engine = cast(Any, object.__new__(OfficialK2ResidentRankEngine))
            engine.parent_root = Path(directory) / "parent"
            engine.parent_root.mkdir()
            engine.rank = 1
            engine.windows = (28, 56)
            engine.checkpoint_sha256 = CANONICAL_U0_CHECKPOINT_SHA256
            engine.config = {"score_window_batch_size": 1}
            engine.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
            engine.dist = FakeDist()
            engine.read_counter = SimpleNamespace(delta=lambda start: 0, paths=[])
            engine.ready_counter = 0
            engine.status = {}
            engine.resident_ready = [{"resident_load_seconds": 1.0, "memory_preflight": {}},
                                     {"resident_load_seconds": 1.0, "memory_preflight": {}}]
            engine.last_window_profile = {}
            saved_terms = [0.25] * 1024
            saved_window = [{"ordinal": 0, "window": 28, "positions": 1024, "support": 8192,
                             "kld_sum_binary64": 256.0, "top1": 5}]
            OfficialK2ResidentRankEngine._persist_score_resume(
                engine, completed_windows=1, terms=saved_terms, top1=5,
                per_window=saved_window, cumulative_scoring_wall_seconds=22.0,
            )
            calls = []

            def score_window(batch):
                calls.append(tuple(batch))
                engine.last_window_profile = {"batch_windows": list(batch)}
                return [([0.5] * 1024, 7)]

            engine._score_window = score_window
            result = OfficialK2ResidentRankEngine.score(engine)
            self.assertEqual(calls, [(56,)])
            self.assertEqual(result["positions"], 2048)
            self.assertEqual(result["kld_mean"], 0.375)
            self.assertEqual(result["top1"], 12)
            self.assertEqual(result["per_window"][0], saved_window[0])
            self.assertEqual(result["per_window"][1]["ordinal"], 1)
            self.assertGreaterEqual(result["cumulative_scoring_wall_seconds"], 22.0)

    def test_shared_device_score_batches_four_independent_windows_for_weight_reuse(self):
        source = inspect.getsource(OfficialK2ResidentRankEngine.score)
        self.assertIn("score_window_batch_size", source)
        self.assertIn("range(completed_before, len(self.windows), batch_size)", source)
        batch_source = inspect.getsource(OfficialK2ResidentRankEngine._score_window)
        self.assertIn("torch.cat", batch_source)
        self.assertIn("batch_windows", batch_source)

    def test_score_pipelines_resident_rank_halves_on_distinct_hosts(self):
        batch_source = inspect.getsource(OfficialK2ResidentRankEngine._score_window)
        score_source = inspect.getsource(OfficialK2ResidentRankEngine.score)
        drain_source = inspect.getsource(OfficialK2ResidentRankEngine._drain_pipeline_inflight)
        self.assertIn("network_pipeline", batch_source)
        self.assertIn("self.dist.isend", batch_source)
        self.assertIn("self._pipeline_inflight.append", batch_source)
        self.assertIn("if len(self._pipeline_inflight) > 1", batch_source)
        self.assertIn("previous_key", batch_source)
        self.assertIn('self.rendezvous_store.set(ipc_key + "-consumed", "1")', batch_source)
        self.assertIn("previous_work.wait()", batch_source)
        self.assertIn("work.wait()", drain_source)
        self.assertIn("self._drain_pipeline_inflight()", score_source)
        self.assertIn('score_window_batch_size", 1', score_source)

        events = []
        engine = cast(Any, object.__new__(OfficialK2ResidentRankEngine))
        engine.rendezvous_store = SimpleNamespace(
            get=lambda key: events.append(("ack", key)) or b"1"
        )
        work = SimpleNamespace(wait=lambda: events.append(("wait", "send")))
        engine._pipeline_inflight = [("batch-0", object(), work)]
        waited = engine._drain_pipeline_inflight()
        self.assertEqual(events, [("ack", "batch-0-consumed"), ("wait", "send")])
        self.assertEqual(engine._pipeline_inflight, [])
        self.assertGreaterEqual(waited, 0.0)

    def test_positional_state_is_fresh_and_cache_coupled_like_sealed_builder(self):
        source = inspect.getsource(OfficialK2ResidentRankEngine._positional)
        self.assertNotIn("_positional_cache", source)
        self.assertIn("past_key_values=cache", source)
        self.assertNotIn("DynamicCache", source)

    def make_artifact(self, root: Path, *, update: int = 0, checkpoint_sha: str = PRE_SHA):
        (root / "checkpoints").mkdir()
        (root / "score" / "teacher").mkdir(parents=True)
        (root / "score" / "candidates" / f"UPDATE_{update:03d}").mkdir(parents=True)
        checkpoint = root / "checkpoints" / f"UPDATE_{update:03d}.pt"
        checkpoint.write_bytes(b"checkpoint")
        manifest = {
            "schema": "repair-artifact-v1",
            "identity": {
                "basis_sha256": "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b",
                "builder_eval_corpus_sha256": "builder" * 16,
                "train_score_corpus_sha256": "train" * 16,
                "teacher_inventory": ["teacher-v1"],
            },
            "checkpoints": {
                f"UPDATE_{update:03d}": {
                    "path": f"checkpoints/UPDATE_{update:03d}.pt",
                    "sha256": checkpoint_sha,
                    "identity_sha256": "identity" * 8,
                    "parent_sha256": "parent" * 8 if update else None,
                    "next_update": update,
                }
            },
            "score": {
                "spec": "balanced64-v1",
                "teacher_dir": "score/teacher",
                "candidate_dir_template": "score/candidates/{checkpoint}",
                "window_ids": WINDOWS,
                "positions_per_window": 1024,
                "support": 8192,
                "official_k2_resident": {
                    "basis_sha256": "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b",
                    "pre_checkpoint_sha256": CANONICAL_U0_CHECKPOINT_SHA256,
                },
            },
        }
        (root / "ARTIFACT.json").write_text(json.dumps(manifest))
        return checkpoint

    def make_routed_artifact(self, root: Path):
        (root / "checkpoints").mkdir()
        (root / "score" / "teacher").mkdir(parents=True)
        (root / "score" / "candidates" / "PRE").mkdir(parents=True)
        (root / "score" / "candidates" / "POST").mkdir(parents=True)
        pre = root / "checkpoints" / "PRE.pt"
        post = root / "checkpoints" / "POST.pt"
        pre.write_bytes(b"routed-pre")
        post.write_bytes(b"routed-post")
        pre_identity = "pre-identity" * 5 + "p"
        post_identity = "post-identity" * 4 + "post"
        manifest_sha = "manifest" * 8
        teacher_manifest = "teacher-manifest" * 4
        corpus_manifest = "corpus-manifest" * 4
        window_manifest = "window-manifest" * 4
        config = {
            "basis_sha256": ROUTED_K2_CLOSURE["basis_model_index_sha256"],
            "pre_checkpoint_sha256": ALTERNATE_PRE_CHECKPOINT_SHA256,
            "teacher_manifest_sha256": teacher_manifest,
            "corpus_manifest_sha256": corpus_manifest,
            "window_manifest_sha256": window_manifest,
        }
        manifest = {
            "schema": "repair-artifact-v1",
            "identity": {
                "basis_sha256": ROUTED_K2_CLOSURE["basis_model_index_sha256"],
                "builder_eval_corpus_sha256": corpus_manifest,
                "train_score_corpus_sha256": corpus_manifest,
                "teacher_inventory": [teacher_manifest],
            },
            "checkpoints": {
                "PRE": {
                    "path": "checkpoints/PRE.pt",
                    "sha256": ALTERNATE_PRE_CHECKPOINT_SHA256,
                    "identity_sha256": pre_identity,
                    "parent_sha256": None,
                    "next_update": 0,
                },
                "POST": {
                    "path": "checkpoints/POST.pt",
                    "sha256": ROUTED_K2_CLOSURE["post_checkpoint_sha256"],
                    "identity_sha256": post_identity,
                    "parent_sha256": ALTERNATE_PRE_CHECKPOINT_SHA256,
                    "next_update": 1,
                },
            },
            "score": {
                "spec": "balanced64-v1",
                "teacher_dir": "score/teacher",
                "candidate_dir_template": "score/candidates/{checkpoint}",
                "window_ids": WINDOWS,
                "positions_per_window": 1024,
                "support": 8192,
                "official_k2_resident": config,
            },
        }
        (root / "ARTIFACT.json").write_text(json.dumps(manifest))
        route = {
            **ROUTED_K2_CLOSURE,
            "route_kind": ROUTED_K2_ROUTE_KIND,
            "pre_checkpoint_identity_sha256": pre_identity,
            "post_checkpoint_identity_sha256": post_identity,
            "post_parent_checkpoint_sha256": ALTERNATE_PRE_CHECKPOINT_SHA256,
            "teacher_manifest_sha256": teacher_manifest,
            "corpus_manifest_sha256": corpus_manifest,
            "window_manifest_sha256": window_manifest,
        }
        return route

    def test_api_score_rejects_alternate_pre_even_when_backend_is_declared(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root, checkpoint_sha=ALTERNATE_PRE_CHECKPOINT_SHA256)
            api = ResidentRepairAPI.open(root, official_backend_factory=FakeOfficialBackend)
            api.artifact.manifest["checkpoints"]["UPDATE_000"]["sha256"] = ALTERNATE_PRE_CHECKPOINT_SHA256
            with self.assertRaisesRegex(ArtifactError, "quarantine-only"):
                api.score("UPDATE_000", windows=WINDOWS)

    def test_published_pre_admits_authentic_payload_identity_lineage(self):
        manifest = {
            "checkpoints": {
                "UPDATE_000": {
                    "sha256": ALTERNATE_PRE_CHECKPOINT_SHA256,
                    "identity_sha256": PUBLISHED_PRE_PAYLOAD_IDENTITY_SHA256,
                    "parent_sha256": None,
                    "next_update": 0,
                },
                "UPDATE_001": {
                    "parent_sha256": ALTERNATE_PRE_CHECKPOINT_SHA256,
                    "parent_identity_sha256": PUBLISHED_PRE_PAYLOAD_IDENTITY_SHA256,
                    "optimizer_scheduler_lineage": PUBLISHED_PRE_OPTIMIZER_SCHEDULER_LINEAGE,
                    "next_update": 1,
                },
            }
        }

        self.assertTrue(_published_pre_production_admitted(manifest))

    def test_api_score_admits_only_identity_exact_fresh_published_pre_lineage(self):
        def manifest_for(root: Path) -> dict[str, Any]:
            self.make_artifact(root, checkpoint_sha=ALTERNATE_PRE_CHECKPOINT_SHA256)
            manifest = json.loads((root / "ARTIFACT.json").read_text())
            pre = manifest["checkpoints"]["UPDATE_000"]
            pre["identity_sha256"] = PUBLISHED_PRE_PAYLOAD_IDENTITY_SHA256
            pre["parent_sha256"] = None
            manifest["checkpoints"]["UPDATE_001"] = {
                "path": "checkpoints/UPDATE_001.pt",
                "sha256": "candidate-sha",
                "identity_sha256": "candidate-identity",
                "parent_sha256": ALTERNATE_PRE_CHECKPOINT_SHA256,
                "parent_identity_sha256": PUBLISHED_PRE_PAYLOAD_IDENTITY_SHA256,
                "optimizer_scheduler_lineage": PUBLISHED_PRE_OPTIMIZER_SCHEDULER_LINEAGE,
                "next_update": 1,
            }
            manifest["score"]["official_k2_resident"]["pre_checkpoint_sha256"] = (
                ALTERNATE_PRE_CHECKPOINT_SHA256
            )
            return manifest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = manifest_for(root)
            (root / "ARTIFACT.json").write_text(json.dumps(manifest))
            api = ResidentRepairAPI.open(root, official_backend_factory=FakeOfficialBackend)
            self.assertEqual(api.score("UPDATE_000", windows=WINDOWS).checkpoint, "UPDATE_000")
            admitted = authorize_production_score(
                0,
                checkpoint_sha256=ALTERNATE_PRE_CHECKPOINT_SHA256,
                checkpoint_parent_sha256=None,
                allow_published_pre_production=True,
            )
            self.assertEqual(admitted["scope"], "PUBLISHED_PRE_PRODUCTION")
            validate_payload_identity(
                {"identity": {
                    "identity_sha256": PUBLISHED_PRE_PAYLOAD_IDENTITY_SHA256,
                    "next_update": 0,
                    "checkpoint_loaded": True,
                }},
                checkpoint_sha256=ALTERNATE_PRE_CHECKPOINT_SHA256,
                checkpoint_identity_sha256=PUBLISHED_PRE_IDENTITY_SHA256,
                next_update=0,
                allow_published_pre_identity_alias=True,
            )

        for field, bad_value in (
            ("identity_sha256", "wrong-pre-identity"),
            ("parent_identity_sha256", "wrong-parent-identity"),
            ("optimizer_scheduler_lineage", "quarantine-lineage"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = manifest_for(root)
                target = manifest["checkpoints"][
                    "UPDATE_000" if field == "identity_sha256" else "UPDATE_001"
                ]
                target[field] = bad_value
                (root / "ARTIFACT.json").write_text(json.dumps(manifest))
                api = ResidentRepairAPI.open(root, official_backend_factory=FakeOfficialBackend)
                with self.assertRaisesRegex(ArtifactError, "quarantine-only"):
                    api.score("UPDATE_000", windows=WINDOWS)

    def test_api_score_allows_exact_alternate_pre_sealed_reference_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root, checkpoint_sha=ALTERNATE_PRE_CHECKPOINT_SHA256)
            api = ResidentRepairAPI.open(root, official_backend_factory=FakeOfficialBackend)
            api.artifact.manifest["checkpoints"]["UPDATE_000"]["sha256"] = ALTERNATE_PRE_CHECKPOINT_SHA256
            api.artifact.manifest["score"]["official_k2_resident"]["parity_tap_mode"] = "sealed_reference"
            result = api.score("UPDATE_000", windows=WINDOWS)
            self.assertEqual(result.checkpoint, "UPDATE_000")

    def test_alternate_pre_diagnostic_admission_is_exact_and_parentless(self):
        admitted = authorize_production_score(
            0,
            checkpoint_sha256=ALTERNATE_PRE_CHECKPOINT_SHA256,
            checkpoint_parent_sha256=None,
            allow_alternate_pre_diagnostic=True,
        )
        self.assertEqual(admitted["scope"], "ALTERNATE_PRE_DIAGNOSTIC_ONLY")
        with self.assertRaisesRegex(ArtifactError, "parentless update 0"):
            authorize_production_score(
                0,
                checkpoint_sha256=ALTERNATE_PRE_CHECKPOINT_SHA256,
                checkpoint_parent_sha256="not-parentless",
                allow_alternate_pre_diagnostic=True,
            )

    def test_score_phase_profile_uses_physical_canary_coverage(self):
        source = inspect.getsource(OfficialK2ResidentRankEngine.score)
        self.assertIn('"physical_canary_windows", self.windows[completed_before:]', source)
        self.assertIn("ordered_windows=profile_windows", source)

    def test_score_routed_k2_accepts_exact_closure_and_emits_resident_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route = self.make_routed_artifact(root)

            class RoutedFakeBackend(FakeOfficialBackend):
                instances = 0

                def __init__(self, artifact, config):
                    super().__init__(artifact, config)
                    type(self).instances += 1

                def score(self, checkpoint, windows):
                    self.calls.append((checkpoint, tuple(windows)))
                    is_post = checkpoint == "POST"
                    kld = PRE_KLD + (0.125 if is_post else 0.0)
                    top1 = PRE_TOP1 + (7 if is_post else 0)
                    return ScoreResult(
                        checkpoint=checkpoint, windows=tuple(windows),
                        positions=65536, support=8192, kld=kld, top1=top1,
                        top1_rate=top1 / 65536, artifact_root=str(self.artifact.root),
                        spec="balanced64-v1", candidate_dir="routed-k2",
                        execution_mode="resident_in_memory", resident_load_seconds=1.0,
                        timed_wall_seconds=2.0,
                        runtime_counters={"timed_score_file_reads": 0,
                                          "file_reads_during_timed_score": 0,
                                          "resident_ready": [{"rank": 0}, {"rank": 1}],
                                          "rank_terminal": [{"rank": 0, "timed_score_file_reads": 0,
                                                              "fallback_calls": 0, "reconstruction_calls": 0,
                                                              "reference_fwht_calls": 0, "cpu_relay_bytes": 0},
                                                             {"rank": 1, "timed_score_file_reads": 0,
                                                              "fallback_calls": 0, "reconstruction_calls": 0,
                                                              "reference_fwht_calls": 0, "cpu_relay_bytes": 0}],
                                          "payload_model_file_read_delta": 0,
                                          "fallback_calls": 0, "reconstruction_calls": 0,
                                          "reference_fwht_calls": 0, "cpu_relay_bytes": 0},
                    )

            api = ResidentRepairAPI.open(root, official_backend_factory=RoutedFakeBackend)
            RoutedFakeBackend.calls.clear()
            result = api.score_routed_k2("PRE", "POST", route=route, windows=WINDOWS)
            self.assertEqual(RoutedFakeBackend.instances, 1)
            self.assertEqual(RoutedFakeBackend.calls, [("PRE", tuple(WINDOWS)), ("POST", tuple(WINDOWS))])
            self.assertEqual(result["public_method"], ROUTED_K2_API_METHOD)
            self.assertEqual(result["api_version"], ROUTED_K2_API_VERSION)
            self.assertEqual(result["route_kind"], ROUTED_K2_ROUTE_KIND)
            self.assertEqual(result["pre"]["positions"], 65536)
            self.assertEqual(result["post"]["positions"], 65536)
            self.assertEqual(result["details_64_64"], {"pre": True, "post": True})
            self.assertTrue(result["resident_proof"]["two_rank_resident_ready"])
            self.assertTrue(result["zero_read_proof"]["timed_model_payload_reads_zero"])
            self.assertEqual(result["package"]["package_identity_sha256"], ROUTED_K2_CLOSURE["package_identity_sha256"])

    def test_score_routed_k2_rejects_any_sealed_identity_drift(self):
        fields = (
            "package_identity_sha256", "selected_roster_sha256", "selected_binding_sha256",
            "official_class_sha256", "basis_model_index_sha256",
            "pre_checkpoint_sha256", "post_checkpoint_sha256",
        )
        for field in fields:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                route = self.make_routed_artifact(root)
                route[field] = "wrong-" + field
                api = ResidentRepairAPI.open(root, official_backend_factory=FakeOfficialBackend)
                with self.assertRaisesRegex(ArtifactError, "routed-K2|routed_k2"):
                    api.score_routed_k2("PRE", "POST", route=route, windows=WINDOWS)

    def test_public_parity_tap_has_exact_tensor_schema_and_is_diagnostic_only(self):
        required = (
            "ids", "embeddings", *(f"L{layer:03d}" for layer in range(43)),
            "hc_head", "norm", "logits", "q_lp_at_ref", "q_argmax",
        )

        class ParityBackend(FakeOfficialBackend):
            calls = []

            def parity_tap(self, checkpoint, window):
                self.calls.append((checkpoint, window))
                return {
                    "taps": {
                        name: {
                            "sha256": hashlib.sha256(name.encode()).hexdigest(),
                            "dtype": "torch.bfloat16",
                            "shape": [1],
                            "sample": [0.0],
                        }
                        for name in required
                    },
                    "diagnostic_metrics": {
                        "window": 28,
                        "positions": 1024,
                        "support": 8192,
                        "kld_sum": 1.0,
                        "kld_mean": 1.0 / 1024,
                        "top1": 1000,
                        "mass_p_mean": 0.9999,
                        "mass_p_sum": 1023.9,
                        "mass_q_mean": 0.9986,
                        "mass_q_sum": 1022.6,
                    },
                    "runtime_counters": {
                        "timed_model_payload_reads": 0,
                        "fallback_calls": 0,
                        "reconstruction_calls": 0,
                        "reference_fwht_calls": 0,
                        "cpu_relay_bytes": 0,
                        "layer_streaming_calls": 0,
                        "resident_ready": [{"rank": 0}, {"rank": 1}],
                    },
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root)
            api = ResidentRepairAPI.open(root, official_backend_factory=ParityBackend)
            api.artifact.manifest["checkpoints"]["UPDATE_000"]["sha256"] = hashlib.sha256(b"checkpoint").hexdigest()
            result = api.parity_tap("UPDATE_000", window=28)

        self.assertEqual(ParityBackend.calls, [("UPDATE_000", 28)])
        self.assertEqual(tuple(result["taps"]), required)
        self.assertEqual(result["quality_status"], "DIAGNOSTIC_ONLY_UNPROMOTED")
        self.assertEqual(result["public_api"]["method"], "ResidentRepairAPI.parity_tap")
        self.assertNotIn("target_kld", result)
        self.assertNotIn("target_top1", result)

    def test_public_parity_tap_rejects_non_scalar_windows_before_backend(self):
        class NeverCalledBackend(FakeOfficialBackend):
            calls = 0

            def parity_tap(self, checkpoint, window):
                self.calls += 1
                raise AssertionError("backend must not run for an invalid one-window request")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root)
            api = ResidentRepairAPI.open(root, official_backend_factory=NeverCalledBackend)
            api.artifact.manifest["checkpoints"]["UPDATE_000"]["sha256"] = hashlib.sha256(
                b"checkpoint"
            ).hexdigest()
            for invalid in (True, [28], (28,), "28,56"):
                with self.subTest(window=invalid), self.assertRaisesRegex(
                    ArtifactError, "exactly one integer window"
                ):
                    api.parity_tap("UPDATE_000", window=cast(Any, invalid))
        self.assertEqual(NeverCalledBackend.calls, 0)

    def test_public_parity_tap_is_repeatable_with_fresh_diagnostic_backend(self):
        required = (
            "ids", "embeddings", *(f"L{layer:03d}" for layer in range(43)),
            "hc_head", "norm", "logits", "q_lp_at_ref", "q_argmax",
        )

        class SingleUseBackend(FakeOfficialBackend):
            instances = 0

            def __init__(self, artifact, config):
                super().__init__(artifact, config)
                type(self).instances += 1
                self.used = False

            def parity_tap(self, checkpoint, window):
                if self.used:
                    raise AssertionError("a one-shot parity backend was reused")
                self.used = True
                return {
                    "taps": {
                        name: {
                            "sha256": hashlib.sha256(name.encode()).hexdigest(),
                            "dtype": "torch.bfloat16",
                            "shape": [1],
                            "sample": [0.0],
                        }
                        for name in required
                    },
                    "diagnostic_metrics": {
                        "window": window, "positions": 1024, "support": 8192,
                        "kld_sum": 0.0, "kld_mean": 0.0, "top1": 1024,
                    },
                    "runtime_counters": {
                        "timed_model_payload_reads": 0, "fallback_calls": 0,
                        "reconstruction_calls": 0, "reference_fwht_calls": 0,
                        "cpu_relay_bytes": 0, "layer_streaming_calls": 0,
                        "resident_ready": [{"rank": 0}, {"rank": 1}],
                    },
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root)
            api = ResidentRepairAPI.open(root, official_backend_factory=SingleUseBackend)
            api.artifact.manifest["checkpoints"]["UPDATE_000"]["sha256"] = hashlib.sha256(
                b"checkpoint"
            ).hexdigest()
            first = api.parity_tap("UPDATE_000", window=28)
            second = api.parity_tap("UPDATE_000", window=28)
        self.assertEqual(first["taps"], second["taps"])
        self.assertEqual(SingleUseBackend.instances, 2)

    def test_public_parity_tap_rolls_back_promotion_state_mutation(self):
        required = (
            "ids", "embeddings", *(f"L{layer:03d}" for layer in range(43)),
            "hc_head", "norm", "logits", "q_lp_at_ref", "q_argmax",
        )

        class MutatingBackend(FakeOfficialBackend):
            def parity_tap(self, checkpoint, window):
                self.artifact.manifest["target_ladder"].append("promoted")
                self.artifact.manifest["best_score"] = -1.0
                self.artifact.manifest["candidate_status"] = "PROMOTED"
                return {
                    "taps": {
                        name: {
                            "sha256": hashlib.sha256(name.encode()).hexdigest(),
                            "dtype": "torch.bfloat16", "shape": [1], "sample": [0.0],
                        }
                        for name in required
                    },
                    "diagnostic_metrics": {
                        "window": window, "positions": 1024, "support": 8192,
                        "kld_sum": 0.0, "kld_mean": 0.0, "top1": 1024,
                    },
                    "runtime_counters": {
                        "timed_model_payload_reads": 0, "fallback_calls": 0,
                        "reconstruction_calls": 0, "reference_fwht_calls": 0,
                        "cpu_relay_bytes": 0, "layer_streaming_calls": 0,
                        "resident_ready": [{"rank": 0}, {"rank": 1}],
                    },
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root)
            api = ResidentRepairAPI.open(root, official_backend_factory=MutatingBackend)
            api.artifact.manifest["checkpoints"]["UPDATE_000"]["sha256"] = hashlib.sha256(
                b"checkpoint"
            ).hexdigest()
            api.artifact.manifest.update({
                "target_ladder": [0.3, 0.2],
                "best_score": 0.2,
                "candidate_status": "CANDIDATE",
            })
            before = json.loads(json.dumps(api.artifact.manifest))
            with self.assertRaisesRegex(ArtifactError, "diagnostic mutated artifact state"):
                api.parity_tap("UPDATE_000", window=28)
            self.assertEqual(api.artifact.manifest, before)

    def test_public_parity_tap_rejects_non_numeric_or_nonfinite_samples(self):
        required = (
            "ids", "embeddings", *(f"L{layer:03d}" for layer in range(43)),
            "hc_head", "norm", "logits", "q_lp_at_ref", "q_argmax",
        )

        class BadSampleBackend(FakeOfficialBackend):
            sample: list[Any] = ["not-numeric"]

            def parity_tap(self, checkpoint, window):
                return {
                    "taps": {
                        name: {
                            "sha256": hashlib.sha256(name.encode()).hexdigest(),
                            "dtype": "torch.bfloat16", "shape": [1],
                            "sample": list(type(self).sample) if name == "L000" else [0.0],
                        }
                        for name in required
                    },
                    "diagnostic_metrics": {
                        "window": window, "positions": 1024, "support": 8192,
                        "kld_sum": 0.0, "kld_mean": 0.0, "top1": 1024,
                    },
                    "runtime_counters": {
                        "timed_model_payload_reads": 0, "fallback_calls": 0,
                        "reconstruction_calls": 0, "reference_fwht_calls": 0,
                        "cpu_relay_bytes": 0, "layer_streaming_calls": 0,
                        "resident_ready": [{"rank": 0}, {"rank": 1}],
                    },
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root)
            api = ResidentRepairAPI.open(root, official_backend_factory=BadSampleBackend)
            api.artifact.manifest["checkpoints"]["UPDATE_000"]["sha256"] = hashlib.sha256(
                b"checkpoint"
            ).hexdigest()
            for sample in (["not-numeric"], [float("nan")], [float("inf")], [True]):
                BadSampleBackend.sample = sample
                with self.subTest(sample=sample), self.assertRaisesRegex(
                    ArtifactError, "finite numeric sample"
                ):
                    api.parity_tap("UPDATE_000", window=28)

    def test_tensor_tap_hash_and_bf16_sample_are_deterministic_and_bounded(self):
        tensor = torch.tensor(
            [0.1, -0.2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            dtype=torch.bfloat16,
        )
        first = OfficialK2ResidentRankEngine._tensor_tap(tensor)
        second = OfficialK2ResidentRankEngine._tensor_tap(tensor.clone())
        self.assertEqual(first, second)
        self.assertEqual(first["dtype"], "torch.bfloat16")
        self.assertEqual(first["shape"], [9])
        self.assertEqual(len(first["sample"]), 8)
        torch.testing.assert_close(
            torch.tensor(first["sample"], dtype=torch.bfloat16), tensor[:8], rtol=0, atol=0
        )

    def test_public_parity_current_layer0_matches_sealed_eager_reference(self):
        required = (
            "ids", "embeddings", *(f"L{layer:03d}" for layer in range(43)),
            "hc_head", "norm", "logits", "q_lp_at_ref", "q_argmax",
        )
        sealed_l000 = [
            -0.2412109375, -0.10498046875, 0.1259765625, 0.045166015625,
            -0.0693359375, 0.1357421875, 0.0240478515625, 0.2197265625,
        ]
        divergent_l000 = [
            -0.28515625, -0.07275390625, 0.08251953125, -0.015380859375,
            -0.169921875, 0.1552734375, -0.049560546875, 0.201171875,
        ]

        class LayerZeroParityBackend(FakeOfficialBackend):
            configs = []

            def __init__(self, artifact, config):
                super().__init__(artifact, config)
                self.configs.append(dict(config))

            def parity_tap(self, checkpoint, window):
                implementation = self.config.get(
                    "attention_implementation_override",
                    self.config.get("attention_implementation", "eager"),
                )
                l000 = sealed_l000 if implementation == "eager" else divergent_l000
                taps = {}
                for name in required:
                    sample = l000 if name == "L000" else [0.0]
                    taps[name] = {
                        "sha256": hashlib.sha256(
                            f"{name}:{implementation}".encode()
                        ).hexdigest(),
                        "dtype": "torch.bfloat16",
                        "shape": [1, 2048, 4, 4096] if name == "L000" else [1],
                        "sample": sample,
                    }
                return {
                    "taps": taps,
                    "diagnostic_metrics": {
                        "window": window,
                        "positions": 1024,
                        "support": 8192,
                        "kld_sum": 1.0,
                        "kld_mean": 1.0 / 1024,
                        "top1": 1000,
                    },
                    "runtime_counters": {
                        "timed_model_payload_reads": 0,
                        "fallback_calls": 0,
                        "reconstruction_calls": 0,
                        "reference_fwht_calls": 0,
                        "cpu_relay_bytes": 0,
                        "layer_streaming_calls": 0,
                        "resident_ready": [{"rank": 0}, {"rank": 1}],
                    },
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root)
            api = ResidentRepairAPI.open(root, official_backend_factory=LayerZeroParityBackend)
            api.artifact.manifest["checkpoints"]["UPDATE_000"]["sha256"] = hashlib.sha256(
                b"checkpoint"
            ).hexdigest()
            api.artifact.manifest["score"]["official_k2_resident"][
                "attention_implementation"
            ] = "sdpa"
            current = api.parity_tap("UPDATE_000", window=28, mode="current")
            sealed = api.parity_tap("UPDATE_000", window=28, mode="sealed_reference")

        torch.testing.assert_close(
            torch.tensor(current["taps"]["L000"]["sample"], dtype=torch.bfloat16),
            torch.tensor(sealed["taps"]["L000"]["sample"], dtype=torch.bfloat16),
            rtol=0.016,
            atol=1.0e-5,
        )
        self.assertEqual(
            [config["attention_implementation_override"] for config in LayerZeroParityBackend.configs],
            ["eager", "eager"],
        )

    def test_api_score_routes_declared_official_backend_without_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root)
            FakeOfficialBackend.calls.clear()
            api = ResidentRepairAPI.open(root, official_backend_factory=FakeOfficialBackend)
            # Avoid fixture bytes failing the normal production SHA readback; the
            # route itself receives the already manifest-bound checkpoint key.
            api.artifact.manifest["checkpoints"]["UPDATE_000"]["sha256"] = hashlib.sha256(b"checkpoint").hexdigest()
            result = api.score("UPDATE_000", windows=WINDOWS)
            self.assertEqual(FakeOfficialBackend.calls, [("UPDATE_000", tuple(WINDOWS))])
            self.assertEqual(result.execution_mode, "resident_in_memory")
            self.assertEqual(result.runtime_counters["timed_score_file_reads"], 0)
            self.assertEqual(result.positions, 65536)
            self.assertEqual(result.identity["public_api"], {
                "method": "ResidentRepairAPI.score",
                "version": "official-k2-resident-v2",
            })
            api.artifact.manifest["checkpoints"]["UPDATE_001"] = dict(
                api.artifact.manifest["checkpoints"]["UPDATE_000"]
            )
            api.score("UPDATE_001", windows=WINDOWS)
            self.assertEqual(len(api._official_backends), 1)
            self.assertEqual(FakeOfficialBackend.calls[-1], ("UPDATE_001", tuple(WINDOWS)))

    def test_canonical_u0_requires_unskippable_calibration(self):
        with self.assertRaisesRegex(ArtifactError, "canonical resident calibration receipt"):
            authorize_production_score(
                0,
                checkpoint_sha256=CANONICAL_U0_CHECKPOINT_SHA256,
            )
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "CANONICAL_RESIDENT_CALIBRATION.json"
            receipt.write_text(json.dumps({
                "schema": "official-k2-resident-canonical-calibration-v1",
                "status": "PASS",
                "checkpoint_sha256": CANONICAL_U0_CHECKPOINT_SHA256,
                "lane": "official-k2-resident",
            }))
            row = authorize_production_score(
                0,
                checkpoint_sha256=CANONICAL_U0_CHECKPOINT_SHA256,
                pre_calibration_receipt=receipt,
            )
        self.assertEqual(row["scope"], "CANONICAL_U0")
        self.assertEqual(row["checkpoint_sha256"], CANONICAL_U0_CHECKPOINT_SHA256)
        self.assertEqual(row["calibration"]["status"], "PASS")

    def test_canonical_u1_requires_exact_admitted_immediate_u0_parent(self):
        row = authorize_production_score(
            1,
            checkpoint_sha256=CANONICAL_U1_CHECKPOINT_SHA256,
            checkpoint_parent_sha256=CANONICAL_U0_CHECKPOINT_SHA256,
        )
        self.assertEqual(row["scope"], "CANONICAL_U1_IMMEDIATE_PARENT")
        published_pre_row = authorize_production_score(
            1,
            checkpoint_sha256=CANONICAL_U1_CHECKPOINT_SHA256,
            checkpoint_parent_sha256=ALTERNATE_PRE_CHECKPOINT_SHA256,
        )
        self.assertEqual(published_pre_row["scope"], "CANONICAL_U1_IMMEDIATE_PARENT")
        with self.assertRaisesRegex(ArtifactError, "exact admitted immediate U0 parent"):
            authorize_production_score(
                1,
                checkpoint_sha256=CANONICAL_U1_CHECKPOINT_SHA256,
                checkpoint_parent_sha256="wrong-parent",
            )

    def test_alternate_pre_is_quarantine_only_for_canonical_lane(self):
        with self.assertRaisesRegex(ArtifactError, "quarantine-only"):
            authorize_production_score(
                0,
                checkpoint_sha256=ALTERNATE_PRE_CHECKPOINT_SHA256,
            )

        from repair_api.official_k2_resident_score import enforce_pre_canary

        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "PRE_CALIBRATION.json"
            passed = enforce_pre_canary(
                checkpoint_sha256=PRE_SHA,
                kld=PRE_KLD,
                top1=PRE_TOP1,
                receipt_path=receipt,
            )
            self.assertEqual(passed["status"], "PASS")
            self.assertEqual(json.loads(receipt.read_text())["quality_status"], "ACCEPTED_PRE_CANARY")
            quarantine = Path(directory) / "PRE_MISMATCH.json"
            with self.assertRaisesRegex(ArtifactError, "PRE canary mismatch"):
                enforce_pre_canary(
                    checkpoint_sha256=PRE_SHA,
                    kld=PRE_KLD + 1e-8,
                    top1=PRE_TOP1,
                    receipt_path=quarantine,
                )
            self.assertEqual(json.loads(quarantine.read_text())["status"], "QUARANTINED")

    def test_u3_plus_requires_pre_calibration_and_preregistered_question(self):
        from repair_api.official_k2_resident_score import authorize_production_score

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pre = root / "PRE_CALIBRATION.json"
            question = root / "SCIENTIFIC_QUESTION.json"
            with self.assertRaisesRegex(ArtifactError, "canonical resident calibration receipt"):
                authorize_production_score(3, pre_calibration_receipt=pre)
            pre.write_text(json.dumps({
                "schema": "official-k2-resident-canonical-calibration-v1",
                "status": "PASS",
                "checkpoint_sha256": CANONICAL_U0_CHECKPOINT_SHA256,
                "lane": "official-k2-resident",
            }))
            # Canonical U0/U1 are admitted by immutable SHA; an explicitly
            # calibrated non-canonical update remains separately gated.
            self.assertEqual(authorize_production_score(1, pre_calibration_receipt=pre)["scope"], "CALIBRATED_PRE_OR_U1")
            with self.assertRaisesRegex(ArtifactError, "pre-registered scientific question"):
                authorize_production_score(3, pre_calibration_receipt=pre)
            question.write_text(json.dumps({
                "schema": "official-k2-resident-scientific-question-v1",
                "status": "PRE_REGISTERED",
                "checkpoint_update": 3,
                "matched_parent_sha256": "parent" * 8,
                "ordered_windows_sha256": "windows" * 8,
                "dose": 3,
            }))
            row = authorize_production_score(
                3,
                pre_calibration_receipt=pre,
                scientific_question_receipt=question,
                checkpoint_parent_sha256="parent" * 8,
                ordered_windows_sha256="windows" * 8,
            )
            self.assertEqual(row["scope"], "SEPARATE_PRE_REGISTERED_QUESTION")

    def test_payload_identity_is_bound_before_resident_construction(self):
        from repair_api.official_k2_resident_score import validate_payload_identity

        valid = {
            "identity": {
                "identity_sha256": "identity" * 8,
                "checkpoint_sha256": "checkpoint" * 6 + "ck",
                "next_update": 1,
                "checkpoint_loaded": True,
            }
        }
        validate_payload_identity(
            valid,
            checkpoint_sha256="checkpoint" * 6 + "ck",
            checkpoint_identity_sha256="identity" * 8,
            next_update=1,
        )
        without_circular_file_sha = {
            "identity": {
                key: value
                for key, value in valid["identity"].items()
                if key != "checkpoint_sha256"
            }
        }
        validate_payload_identity(
            without_circular_file_sha,
            checkpoint_sha256="checkpoint" * 6 + "ck",
            checkpoint_identity_sha256="identity" * 8,
            next_update=1,
        )
        with self.assertRaisesRegex(ArtifactError, "checkpoint_sha256"):
            validate_payload_identity(
                {"identity": {**valid["identity"], "checkpoint_sha256": "wrong"}},
                checkpoint_sha256="checkpoint" * 6 + "ck",
                checkpoint_identity_sha256="identity" * 8,
                next_update=1,
            )
        with self.assertRaisesRegex(ArtifactError, "payload checkpoint identity"):
            validate_payload_identity(
                {"identity": {**valid["identity"], "next_update": 3}},
                checkpoint_sha256="checkpoint" * 6 + "ck",
                checkpoint_identity_sha256="identity" * 8,
                next_update=1,
            )

    def _make_raw_u0_adapter_fixture(self, root: Path):
        import torch

        checkpoint = root / "checkpoints" / "UPDATE_000.pt"
        lock = root / "receipts" / "CLEAN_U0_LOCK.json"
        checkpoint.parent.mkdir(parents=True)
        lock.parent.mkdir(parents=True)
        torch.save({
            "state": {"luts": {}, "norms": {}, "outputs": {}},
            "optimizer_state": {"state": {}, "param_groups": []},
            "scheduler_state": {"last_epoch": 0},
        }, checkpoint)
        lock.write_text(json.dumps({
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "checkpoint_loaded": False,
            "optimizer_state": {"state": {}, "param_groups": []},
            "scheduler_epoch": 0,
            "basis_sha256": "basis-fixture",
            "corpus_sha256": "score-corpus-fixture",
            "trajectory_sha256": "trajectory-fixture",
        }, sort_keys=True) + "\n")
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest()
        manifest = {
            "schema": "repair-artifact-v1",
            "identity": {
                "basis_sha256": "basis-fixture",
                "builder_eval_corpus_sha256": "builder-corpus-fixture",
                "train_score_corpus_sha256": "score-corpus-fixture",
                "teacher_inventory_sha256": "teacher-inventory-fixture",
            },
            "checkpoints": {
                "UPDATE_000": {
                    "path": "checkpoints/UPDATE_000.pt",
                    "sha256": checkpoint_sha,
                    "identity_sha256": "identity-fixture",
                    "parent_sha256": None,
                    "next_update": 0,
                    "canonical_source_path": "/home/dnola/missions/MODERN_GREEN_t_6bc398da/run_clean_u0_attempt4/checkpoints/UPDATE_000.pt",
                }
            },
            "score": {
                "spec": "balanced64-v1",
                "teacher_dir": "score/teacher",
                "candidate_dir_template": "score/candidates/{checkpoint}",
                "window_ids": WINDOWS,
                "positions_per_window": 1024,
                "support": 8192,
            },
            "canonical_raw_u0": {
                "clean_u0_lock_path": "receipts/CLEAN_U0_LOCK.json",
                "clean_u0_lock_sha256": lock_sha,
                "trajectory_sha256": "trajectory-fixture",
            },
        }
        (root / "ARTIFACT.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
        return manifest, checkpoint, lock, checkpoint_sha, lock_sha

    def test_canonical_raw_u0_adapter_passes_without_embedded_identity_fields(self):
        from repair_api.official_k2_resident_score import load_canonical_raw_u0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, checkpoint, _lock, checkpoint_sha, lock_sha = self._make_raw_u0_adapter_fixture(root)
            before = checkpoint.read_bytes()
            with patch("repair_api.official_k2_resident_score.CANONICAL_U0_CHECKPOINT_SHA256", checkpoint_sha), \
                 patch("repair_api.official_k2_resident_score.CANONICAL_U0_LOCK_SHA256", lock_sha), \
                 patch("repair_api.official_k2_resident_score.CANONICAL_U0_IDENTITY_SHA256", "identity-fixture"), \
                 patch("repair_api.official_k2_resident_score.BASIS_SHA256", "basis-fixture"), \
                 patch("repair_api.official_k2_resident_score.CANONICAL_U0_TRAJECTORY_SHA256", "trajectory-fixture"), \
                 patch("repair_api.official_k2_resident_score.BUILDER_EVAL_CORPUS_SHA256", "builder-corpus-fixture"), \
                 patch("repair_api.official_k2_resident_score.SCORE_TRAIN_CORPUS_SHA256", "score-corpus-fixture"), \
                 patch("repair_api.official_k2_resident_score.TEACHER_INVENTORY_SHA256", "teacher-inventory-fixture"), \
                 patch("repair_api.official_k2_resident_score.CANONICAL_CORPUS_SHA256", "score-corpus-fixture"), \
                 patch("repair_api.official_k2_resident_score.CANONICAL_U0_LOCK_CORPUS_SHA256", "score-corpus-fixture"):
                payload = load_canonical_raw_u0(root, manifest=manifest)
            identity = payload["identity"]
            self.assertEqual(identity["source"], "canonical_raw_manifest_adapter")
            self.assertEqual(identity["runtime_load_provenance"]["source"], "canonical_raw_manifest_adapter")
            self.assertTrue(identity["checkpoint_loaded"])
            self.assertFalse(identity["embedded_identity_fields"])
            self.assertEqual(identity["checkpoint_sha256"], checkpoint_sha)
            self.assertEqual(payload["optimizer_state"]["state"], {})
            self.assertEqual(checkpoint.read_bytes(), before)

    def test_canonical_raw_u0_adapter_rejects_sha_path_lock_and_basis_drift(self):
        from repair_api.official_k2_resident_score import load_canonical_raw_u0

        for mutation in ("sha", "path", "lock", "basis"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, _checkpoint, _lock, checkpoint_sha, lock_sha = self._make_raw_u0_adapter_fixture(root)
                if mutation == "sha":
                    manifest["checkpoints"]["UPDATE_000"]["sha256"] = "wrong"
                elif mutation == "path":
                    manifest["checkpoints"]["UPDATE_000"]["path"] = "checkpoints/other.pt"
                elif mutation == "lock":
                    manifest["canonical_raw_u0"]["clean_u0_lock_sha256"] = "wrong"
                else:
                    manifest["identity"]["basis_sha256"] = "wrong"
                with patch("repair_api.official_k2_resident_score.CANONICAL_U0_CHECKPOINT_SHA256", checkpoint_sha), \
                     patch("repair_api.official_k2_resident_score.CANONICAL_U0_LOCK_SHA256", lock_sha), \
                     patch("repair_api.official_k2_resident_score.CANONICAL_U0_IDENTITY_SHA256", "identity-fixture"), \
                     patch("repair_api.official_k2_resident_score.BASIS_SHA256", "basis-fixture"), \
                     patch("repair_api.official_k2_resident_score.CANONICAL_U0_TRAJECTORY_SHA256", "trajectory-fixture"), \
                     patch("repair_api.official_k2_resident_score.BUILDER_EVAL_CORPUS_SHA256", "builder-corpus-fixture"), \
                 patch("repair_api.official_k2_resident_score.SCORE_TRAIN_CORPUS_SHA256", "score-corpus-fixture"), \
                 patch("repair_api.official_k2_resident_score.TEACHER_INVENTORY_SHA256", "teacher-inventory-fixture"), \
                 patch("repair_api.official_k2_resident_score.CANONICAL_CORPUS_SHA256", "score-corpus-fixture"), \
                 patch("repair_api.official_k2_resident_score.CANONICAL_U0_LOCK_CORPUS_SHA256", "score-corpus-fixture"):
                    with self.assertRaises(ArtifactError):
                        load_canonical_raw_u0(root, manifest=manifest)

    def test_canonical_raw_u1_is_adapted_before_payload_identity_validation(self):
        import repair_api.official_k2_resident_score as scorer_module

        u1_identity_sha256 = "53cb15a23aa2c695b2ff1ca5d0bcb6dabc7848d154785c2ffd32faec18ba3faf"
        raw = {
            "state": {"luts": {}, "norms": {}, "outputs": {}},
            "identity": {
                "framework": "banana-smasher",
                "model_index_sha256": scorer_module.BASIS_SHA256,
                "continuous_parent_checkpoint_sha256": CANONICAL_U0_CHECKPOINT_SHA256,
            },
            "identity_sha256": u1_identity_sha256,
            "next_update": 1,
        }
        adapted = dict(raw)
        adapted["identity"] = {
            "identity_sha256": u1_identity_sha256,
            "checkpoint_sha256": CANONICAL_U1_CHECKPOINT_SHA256,
            "next_update": 1,
            "checkpoint_loaded": True,
        }
        events = []

        def adapt(payload, **kwargs):
            events.append("adapt")
            self.assertIs(payload, raw)
            self.assertEqual(kwargs["checkpoint_key"], "UPDATE_001")
            return adapted

        def validate(payload, **kwargs):
            events.append("validate")
            self.assertIs(payload, adapted)
            self.assertEqual(kwargs["checkpoint_sha256"], CANONICAL_U1_CHECKPOINT_SHA256)
            self.assertEqual(kwargs["checkpoint_identity_sha256"], u1_identity_sha256)
            self.assertEqual(kwargs["next_update"], 1)
            raise RuntimeError("STOP_AFTER_IDENTITY_VALIDATION")

        artifact = SimpleNamespace(
            root=Path("/authentic-u1-fixture"),
            windows=tuple(WINDOWS),
            manifest={"checkpoints": {"UPDATE_001": {
                "sha256": CANONICAL_U1_CHECKPOINT_SHA256,
                "identity_sha256": u1_identity_sha256,
                "parent_sha256": CANONICAL_U0_CHECKPOINT_SHA256,
                "next_update": 1,
            }}},
            checkpoint_path=lambda _checkpoint: Path("/authentic-u1-fixture/checkpoints/UPDATE_001.pt"),
        )
        scorer = scorer_module.OfficialK2ResidentScorer(
            cast(Any, artifact), {"basis_sha256": scorer_module.BASIS_SHA256}
        )
        with patch.object(scorer_module, "authorize_production_score", return_value={
                 "scope": "CANONICAL_U1_IMMEDIATE_PARENT"
             }), patch.object(scorer_module, "_load_torch", return_value=raw), \
             patch.object(scorer_module, "adapt_canonical_raw_u1_payload", side_effect=adapt, create=True), \
             patch.object(scorer_module, "validate_payload_identity", side_effect=validate):
            with self.assertRaisesRegex(RuntimeError, "STOP_AFTER_IDENTITY_VALIDATION"):
                scorer.score("UPDATE_001", WINDOWS)
        self.assertEqual(events, ["adapt", "validate"])

    def test_u1_adapter_retains_expert_plane_surface(self):
        from repair_api.official_k2_resident_score import adapt_canonical_raw_u1_payload
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoints" / "UPDATE_001.pt"
            checkpoint.parent.mkdir(parents=True)
            source_identity = {
                "schema": "resident-continuation-checkpoint-identity-v1",
                "basis_sha256": "basis-fixture",
                "checkpoint": "UPDATE_001",
                "checkpoint_loaded": True,
                "identity_sha256": "u1-identity-fixture",
                "next_update": 1,
                "parent_checkpoint_sha256": "published-pre-fixture",
            }
            raw = {
                "state": {
                    "luts": {}, "norms": {}, "outputs": {},
                    "expert_planes_l028_su_sv": {},
                },
                "identity": source_identity,
                "optimizer": {"state": {"one": {}}, "param_groups": []},
            }
            torch.save(raw, checkpoint)
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            manifest = {
                "schema": "repair-artifact-v1",
                "identity": {"basis_sha256": "basis-fixture"},
                "checkpoints": {"UPDATE_001": {
                    "path": "checkpoints/UPDATE_001.pt",
                    "sha256": checkpoint_sha,
                    "identity_sha256": "u1-identity-fixture",
                    "parent_sha256": "published-pre-fixture",
                    "next_update": 1,
                }},
            }
            (root / "ARTIFACT.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
            before = checkpoint.read_bytes()
            with patch("repair_api.official_k2_resident_score.CANONICAL_U1_CHECKPOINT_SHA256", checkpoint_sha), \
                 patch("repair_api.official_k2_resident_score.CANONICAL_U1_IDENTITY_SHA256", "u1-identity-fixture"), \
                 patch("repair_api.official_k2_resident_score.CANONICAL_U0_CHECKPOINT_SHA256", "u0-fixture"), \
                 patch("repair_api.official_k2_resident_score.ALTERNATE_PRE_CHECKPOINT_SHA256", "published-pre-fixture"), \
                 patch("repair_api.official_k2_resident_score.BASIS_SHA256", "basis-fixture"):
                adapted = adapt_canonical_raw_u1_payload(
                    raw, artifact_root=root, manifest=manifest, checkpoint_path=checkpoint
                )
            identity = adapted["identity"]
            self.assertEqual(identity["source"], "canonical_raw_u1_manifest_adapter")
            self.assertEqual(identity["checkpoint_sha256"], checkpoint_sha)
            self.assertEqual(identity["identity_sha256"], "u1-identity-fixture")
            self.assertEqual(identity["next_update"], 1)
            self.assertTrue(identity["checkpoint_loaded"])
            self.assertEqual(identity["parent_checkpoint_sha256"], "published-pre-fixture")
            self.assertEqual(identity["runtime_load_provenance"]["source_identity"], source_identity)
            self.assertEqual(
                set(adapted["state"]),
                {"luts", "norms", "outputs", "expert_planes_l028_su_sv"},
            )
            self.assertIs(
                adapted["state"]["expert_planes_l028_su_sv"],
                raw["state"]["expert_planes_l028_su_sv"],
            )
            self.assertIn("expert_planes_l028_su_sv", raw["state"])
            self.assertEqual(checkpoint.read_bytes(), before)

    def test_resident_backend_rejects_streaming_fallback_configuration(self):
        from repair_api.official_k2_resident_score import OfficialK2ResidentScorer

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifact(root)
            artifact = ResidentRepairAPI.open(root).artifact
            for field in ("remote", "shard_buf", "planes_dir", "builder", "reference_fwht"):
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ArtifactError, "forbidden non-resident"):
                        OfficialK2ResidentScorer(artifact, {field: "forbidden"})

    def test_production_standalone_builder_runners_are_hard_rejected(self):
        from repair_api.production_score_guard import reject_standalone_score_runner

        for entrypoint in (
            "repair_api.official_resident_campaign",
            "repair_api.resident_campaign",
            "repair_api.score_u16",
            "rail.score_outputs",
            "builder.main",
        ):
            with self.subTest(entrypoint=entrypoint):
                with self.assertRaisesRegex(ArtifactError, "ResidentRepairAPI.score"):
                    reject_standalone_score_runner(entrypoint)

    def test_score_cli_routes_exact_pre_post_through_public_routed_method(self):
        class FakeAPI:
            def __init__(self):
                self.calls = []

            def score_routed_k2(self, pre, post, *, route, windows, receipt_path):
                self.calls.append((pre, post, route, windows, receipt_path))
                return {"status": "PASS", "public_method": ROUTED_K2_API_METHOD}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route_path = root / "route.json"
            receipt_path = root / "receipt.json"
            route_path.write_text(json.dumps({"route_kind": ROUTED_K2_ROUTE_KIND}) + "\n")
            fake = FakeAPI()
            with patch("repair_api.cli.ResidentRepairAPI.open", return_value=fake), patch("builtins.print"):
                rc = cli_main([
                    "score", "--artifact-root", str(root), "--checkpoint", "PRE",
                    "--routed-post", "POST", "--route", str(route_path),
                    "--receipt", str(receipt_path), "--windows", "1,2,3",
                ])
            self.assertEqual(rc, 0)
            self.assertEqual(
                fake.calls,
                [("PRE", "POST", {"route_kind": ROUTED_K2_ROUTE_KIND}, [1, 2, 3], receipt_path)],
            )

    def test_supported_production_entrypoints_install_the_hard_guard(self):
        source_root = Path(__file__).resolve().parents[1]
        for name in ("official_resident_campaign.py", "resident_campaign.py", "score_u16.py"):
            text = (source_root / name).read_text()
            self.assertIn("reject_standalone_score_runner", text, name)
        api_text = (source_root / "api.py").read_text()
        cli_text = (source_root / "cli.py").read_text()
        integration_text = (source_root / "integration.py").read_text()
        remote_text = (source_root / "remote_integration.py").read_text()
        self.assertIn("def score(", api_text)
        self.assertIn("api.score(", cli_text)
        self.assertIn("api.score_routed_k2(", cli_text)
        self.assertIn("api.score(args.checkpoint", integration_text)
        self.assertIn("api.score({checkpoint!r}", remote_text)


if __name__ == "__main__":
    unittest.main()
