from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

from repair_api.api import ResidentRepairAPI
from repair_api.balanced64 import ScoreResult


class U20OfficialRouteRegression(unittest.TestCase):
    def test_official_config_without_candidate_rows_never_uses_row_loader(self):
        windows = (28, 56)
        artifact = SimpleNamespace(
            root="/artifact",
            windows=windows,
            manifest={
                "checkpoints": {
                    "UPDATE_020": {
                        "sha256": "1" * 64,
                        "identity_sha256": "2" * 64,
                        "next_update": 20,
                        "parent_sha256": "3" * 64,
                    }
                },
                "score": {"official_k2_resident": {"basis_sha256": "4" * 64}},
            },
            checkpoint_key=lambda checkpoint: "UPDATE_020",
        )
        backend = mock.Mock()
        backend.score.return_value = ScoreResult(
            checkpoint="UPDATE_020",
            windows=windows,
            positions=len(windows) * 1024,
            support=8192,
            kld=0.1,
            top1=1,
            top1_rate=1 / (len(windows) * 1024),
            artifact_root="/artifact",
            spec="balanced64-v1",
            candidate_dir="fully-resident-official-k2",
            execution_mode="resident_in_memory",
            resident_load_seconds=0.1,
            timed_wall_seconds=0.2,
            runtime_counters={
                "timed_score_file_reads": 0,
                "payload_model_file_read_delta": 0,
                "fallback_calls": 0,
                "reconstruction_calls": 0,
                "reference_fwht_calls": 0,
                "cpu_relay_bytes": 0,
            },
        )
        factory = mock.Mock(return_value=backend)
        api = ResidentRepairAPI.__new__(ResidentRepairAPI)
        api.artifact = artifact
        api._shared_preflight = mock.Mock()
        api._last_preflight = {}
        api._official_backend_factory = factory
        api._official_backends = {}
        api._resident_for = mock.Mock(side_effect=AssertionError("candidate-row loader must not run"))
        api._validate_scientific_identity = mock.Mock()

        result = api.score("UPDATE_020", windows=windows)

        factory.assert_called_once_with(artifact, artifact.manifest["score"]["official_k2_resident"])
        backend.score.assert_called_once_with("UPDATE_020", windows)
        api._resident_for.assert_not_called()
        self.assertEqual(result.execution_mode, "resident_in_memory")
        self.assertEqual(result.runtime_counters["public_api_method"], "ResidentRepairAPI.score")


if __name__ == "__main__":
    unittest.main()
