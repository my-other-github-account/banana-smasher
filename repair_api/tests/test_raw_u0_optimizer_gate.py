import unittest

from repair_api.balanced64 import ArtifactError
from repair_api.official_k2_resident_score import (
    BASIS_SHA256,
    CANONICAL_U0_LOCK_CORPUS_SHA256,
    _validate_raw_u0_gates,
)


class CanonicalRawU0OptimizerGateTests(unittest.TestCase):
    def payload(self):
        return {
            "state": {"luts": {}, "norms": {}, "outputs": {}},
            "identity": {
                "fresh_u0": True,
                "input_checkpoint_sha256": None,
                "parent_checkpoint_sha256": None,
                "model_index_sha256": BASIS_SHA256,
                "corpus_sha256": CANONICAL_U0_LOCK_CORPUS_SHA256,
                "optimizer_state_entries": 0,
            },
            "identity_sha256": "d602de92d998c0e649b0bc4fdf35a857384ff3cf6d1021bdbb76a8070af73a88",
            "next_update": 0,
            "optimizer": {
                "state": {},
                "param_groups": [
                    {"group_name": "luts", "params": [0, 1]},
                    {"group_name": "norms", "params": [2]},
                    {"group_name": "outputs", "params": [3]},
                ],
            },
            "scheduler": {"last_epoch": 0},
        }

    def lock(self):
        return {
            "checkpoint_loaded": False,
            "optimizer_state_entries": 0,
            "scheduler_epoch": 0,
        }

    def test_fresh_optimizer_parameter_groups_are_metadata_not_state(self):
        _validate_raw_u0_gates(self.payload(), self.lock())

    def test_nonempty_optimizer_state_is_rejected(self):
        payload = self.payload()
        payload["optimizer"]["state"] = {0: {"step": 1}}
        with self.assertRaisesRegex(ArtifactError, "optimizer state is not empty"):
            _validate_raw_u0_gates(payload, self.lock())

    def test_nonzero_declared_optimizer_state_entries_are_rejected(self):
        payload = self.payload()
        payload["identity"]["optimizer_state_entries"] = 1
        with self.assertRaisesRegex(ArtifactError, "optimizer state entries drift"):
            _validate_raw_u0_gates(payload, self.lock())


if __name__ == "__main__":
    unittest.main()
