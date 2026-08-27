from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from repair_api import ArtifactError, ResidentRepairAPI, ScoreResult
from repair_api.official_k2_resident_score import (
    CANONICAL_U0_CHECKPOINT_SHA256,
    EXPECTED_RESIDENT_BYTES,
    DEFAULT_CUDA_RESERVE_BYTES,
    resolve_rank_local_bytes,
)


class _FakeCuda:
    def __init__(self, free_bytes: int):
        self.free_bytes = free_bytes

    def mem_get_info(self):
        return self.free_bytes, self.free_bytes + 1


class _FakeTorch:
    def __init__(self, free_bytes: int):
        self.cuda = _FakeCuda(free_bytes)


class MemoryFitPolicyTests(unittest.TestCase):

    def test_public_resident_binds_base_loader_environment_from_manifest(self):
        source = Path(__file__).resolve().parents[1] / "official_k2_resident_score.py"
        text = source.read_text()
        self.assertIn("_configure_base_environment", text)
        self.assertIn("binrepair_manifest", text)
        self.assertIn("binrepair_delta_dir", text)
        self.assertIn("binrepair_vq3b_dir", text)
        self.assertIn("binrepair_train_windows", text)
        self.assertIn("binrepair_probe_windows", text)

    def test_base_loader_window_environment_is_not_treated_as_a_path(self):
        from repair_api.official_k2_resident_score import OfficialK2ResidentRankEngine

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            delta = root / "delta"
            vq3b = root / "vq3b"
            manifest.write_text("{}")
            delta.mkdir()
            vq3b.mkdir()
            engine = OfficialK2ResidentRankEngine.__new__(OfficialK2ResidentRankEngine)
            engine.asset_root = root
            engine.windows = (28, 56, 68, 71)
            engine.config = {
                "binrepair_manifest": manifest,
                "binrepair_delta_dir": delta,
                "binrepair_vq3b_dir": vq3b,
                "binrepair_train_windows": "28,56",
                "binrepair_probe_windows": "68,71",
            }
            engine._configure_base_environment()
            self.assertEqual(os.environ["BR_MANIFEST"], str(manifest.resolve()))
            self.assertEqual(os.environ["BR_DELTA_DIR"], str(delta.resolve()))
            self.assertEqual(os.environ["BR_VQ3B_DIR"], str(vq3b.resolve()))
            self.assertEqual(os.environ["BR_TRAIN"], "28,56")
            self.assertEqual(os.environ["BR_PROBE"], "68,71")
    def make_engine(self, *, free_bytes: int, config: dict):
        from repair_api.official_k2_resident_score import OfficialK2ResidentRankEngine

        engine = OfficialK2ResidentRankEngine.__new__(OfficialK2ResidentRankEngine)
        engine.rank = 0
        engine.torch = _FakeTorch(free_bytes)
        engine.config = config
        return engine

    def test_default_four_gib_reserve_rejects_when_margin_is_insufficient(self):
        resident = EXPECTED_RESIDENT_BYTES[0]
        engine = self.make_engine(
            free_bytes=resident + DEFAULT_CUDA_RESERVE_BYTES - 1,
            config={"estimated_resident_bytes_by_rank": {"0": resident}},
        )
        with self.assertRaisesRegex(ArtifactError, r"free=.*required=.*reserve=.*margin=-1") as caught:
            engine._preflight_memory()
        self.assertEqual(caught.exception.__dict__["memory_preflight"]["margin_bytes"], -1)
        self.assertEqual(caught.exception.__dict__["memory_preflight"]["reserve_bytes"], DEFAULT_CUDA_RESERVE_BYTES)

    def test_explicit_bounded_rank_local_reserve_accepts_positive_margin(self):
        resident = EXPECTED_RESIDENT_BYTES[0]
        reserve = 1 << 30
        margin = 123456789
        engine = self.make_engine(
            free_bytes=resident + reserve + margin,
            config={
                "estimated_resident_bytes_by_rank": {"0": resident},
                "estimated_peak_bytes_by_rank": {"0": resident},
                "cuda_reserve_bytes": {"0": reserve, "1": reserve},
            },
        )
        engine._preflight_memory()
        self.assertEqual(engine.memory_preflight["cuda_free_bytes"], resident + reserve + margin)
        self.assertEqual(engine.memory_preflight["estimated_resident_bytes"], resident)
        self.assertEqual(engine.memory_preflight["peak_estimate_bytes"], resident)
        self.assertEqual(engine.memory_preflight["reserve_bytes"], reserve)
        self.assertEqual(engine.memory_preflight["margin_bytes"], margin)
        self.assertEqual(engine.memory_preflight["reserve_policy"], "rank_local_explicit")

    def test_explicit_rank_local_reserve_rejects_insufficient_margin(self):
        resident = EXPECTED_RESIDENT_BYTES[0]
        reserve = 1 << 30
        engine = self.make_engine(
            free_bytes=resident + reserve - 1,
            config={
                "estimated_resident_bytes_by_rank": {"0": resident},
                "cuda_reserve_bytes": {"0": reserve, "1": reserve},
            },
        )
        with self.assertRaisesRegex(ArtifactError, r"margin=-1"):
            engine._preflight_memory()

    def test_rank_local_policy_rejects_missing_or_nonpositive_reserve(self):
        with self.assertRaisesRegex(ArtifactError, "rank-local"):
            resolve_rank_local_bytes({"1": 1 << 30}, 0, default=DEFAULT_CUDA_RESERVE_BYTES, field="cuda_reserve_bytes")
        with self.assertRaisesRegex(ArtifactError, "positive"):
            resolve_rank_local_bytes({"0": 0}, 0, default=DEFAULT_CUDA_RESERVE_BYTES, field="cuda_reserve_bytes")

    def test_api_receipt_provenance_and_resident_guards(self):
        class Backend:
            fallback_calls = 0

            def __init__(self, artifact, config):
                self.artifact = artifact

            def score(self, checkpoint, windows):
                return ScoreResult(
                    checkpoint=checkpoint,
                    windows=tuple(windows),
                    positions=65536,
                    support=8192,
                    kld=0.1,
                    top1=10,
                    top1_rate=10 / 65536,
                    artifact_root=str(self.artifact.root),
                    spec="balanced64-v1",
                    candidate_dir="fully-resident-official-k2",
                    execution_mode="resident_in_memory",
                    resident_load_seconds=0.1,
                    timed_wall_seconds=0.2,
                    runtime_counters={
                        "timed_score_file_reads": 0,
                        "file_reads_during_timed_score": 0,
                        "fallback_calls": self.fallback_calls,
                        "reconstruction_calls": 0,
                        "reference_fwht_calls": 0,
                        "cpu_relay_bytes": 0,
                        "memory_preflight": {
                            "rank": 0,
                            "cuda_free_bytes": 10,
                            "estimated_resident_bytes": 5,
                            "peak_estimate_bytes": 5,
                            "reserve_bytes": 1,
                            "margin_bytes": 4,
                            "predicate": "resident_estimate_bytes + reserve_bytes <= cuda_free_bytes",
                        },
                    },
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoints").mkdir()
            (root / "checkpoints" / "UPDATE_000.pt").write_bytes(b"checkpoint")
            manifest = {
                "schema": "repair-artifact-v1",
                "identity": {
                    "basis_sha256": "b" * 64,
                    "builder_eval_corpus_sha256": "c" * 64,
                    "train_score_corpus_sha256": "d" * 64,
                    "teacher_inventory": ["teacher"],
                },
                "checkpoints": {
                    "UPDATE_000": {
                        "path": "checkpoints/UPDATE_000.pt",
                        "sha256": CANONICAL_U0_CHECKPOINT_SHA256,
                        "identity_sha256": "identity",
                        "next_update": 0,
                    }
                },
                "score": {
                    "spec": "balanced64-v1",
                    "teacher_dir": "teacher",
                    "candidate_dir_template": "score/candidates/{checkpoint}",
                    "window_ids": list(range(64)),
                    "positions_per_window": 1024,
                    "support": 8192,
                    "official_k2_resident": {"basis_sha256": "basis"},
                },
            }
            (root / "ARTIFACT.json").write_text(json.dumps(manifest))
            receipt = root / "receipt.json"
            api = ResidentRepairAPI.open(root, official_backend_factory=Backend)
            api.artifact.manifest["checkpoints"]["UPDATE_000"]["sha256"] = CANONICAL_U0_CHECKPOINT_SHA256
            result = api.score("UPDATE_000", receipt_path=receipt)
            self.assertEqual(result.runtime_counters["timed_score_file_reads"], 0)
            payload = json.loads(receipt.read_text())
            self.assertEqual(payload["identity"]["public_api"]["method"], "ResidentRepairAPI.score")
            self.assertIn("memory_preflight", payload["runtime_counters"])

            class BadBackend(Backend):
                fallback_calls = 1

            bad_api = ResidentRepairAPI.open(root, official_backend_factory=BadBackend)
            with self.assertRaisesRegex(ArtifactError, "terminal closure failed"):
                bad_api.score("UPDATE_000")

            class MemoryErrorBackend:
                def __init__(self, artifact, config):
                    pass

                def score(self, checkpoint, windows):
                    error = ArtifactError("CUDA OOM during resident admission")
                    error.__dict__["memory_preflight"] = {
                        "cuda_free_bytes": 9,
                        "estimated_resident_bytes": 8,
                        "peak_estimate_bytes": 8,
                        "reserve_bytes": 1,
                        "margin_bytes": 0,
                    }
                    raise error

            error_receipt = root / "error_receipt.json"
            error_api = ResidentRepairAPI.open(root, official_backend_factory=MemoryErrorBackend)
            with self.assertRaisesRegex(ArtifactError, "CUDA OOM"):
                error_api.score("UPDATE_000", receipt_path=error_receipt)
            error_payload = json.loads(error_receipt.read_text())
            self.assertEqual(error_payload["status"], "ERROR")
            self.assertEqual(error_payload["memory_preflight"]["reserve_bytes"], 1)


if __name__ == "__main__":
    unittest.main()
