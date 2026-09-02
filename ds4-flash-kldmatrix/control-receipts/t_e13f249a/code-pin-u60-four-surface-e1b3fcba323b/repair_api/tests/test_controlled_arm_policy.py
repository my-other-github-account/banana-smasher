import unittest

from repair_api.api import _validate_controlled_arm_start
from repair_api.balanced64 import ArtifactError
from repair_api.modern_green_resident import (
    HISTORICAL_BASE_LRS,
    _controlled_arm_origin,
    _controlled_arm_policy,
    _controlled_scheduler_step,
)


class ControlledArmPolicyTest(unittest.TestCase):
    def test_exact_registered_policies(self):
        base, multiplier, windows = _controlled_arm_policy({"controlled_arm_id": "lr_scale_only"}, 16)
        self.assertEqual(base, HISTORICAL_BASE_LRS)
        self.assertEqual(multiplier, 0.125)
        self.assertEqual(windows, 6)

        _, multiplier, windows = _controlled_arm_policy({"controlled_arm_id": "cosine_restart_only"}, 16)
        self.assertEqual(multiplier, 1.0)
        self.assertEqual(windows, 6)
        _, multiplier, _ = _controlled_arm_policy({"controlled_arm_id": "cosine_restart_only"}, 64 - 1)
        self.assertGreater(multiplier, 0.1)

        expected = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
        observed = [_controlled_arm_policy({"controlled_arm_id": "warmup_restart_only"}, step)[1] for step in range(16, 24)]
        self.assertEqual(observed, expected)

        _, multiplier, windows = _controlled_arm_policy({"controlled_arm_id": "window_dose_only"}, 16)
        self.assertEqual(multiplier, 1.0)
        self.assertEqual(windows, 16)

    def test_unknown_arm_and_cursor_fail_closed(self):
        with self.assertRaises(ArtifactError):
            _controlled_arm_policy({"controlled_arm_id": "combo"}, 16)
        with self.assertRaises(ArtifactError):
            _controlled_arm_policy({"controlled_arm_id": "lr_scale_only"}, 15)

    def test_from_u0_registered_policies_diverge_before_u16(self):
        base, multiplier, windows = _controlled_arm_policy(
            {"controlled_arm_id": "from_u0_historical_control"}, 0
        )
        self.assertEqual(base, HISTORICAL_BASE_LRS)
        self.assertEqual(multiplier, 1.0 / 16.0)
        self.assertEqual(windows, 2)

        _, multiplier, windows = _controlled_arm_policy(
            {"controlled_arm_id": "from_u0_lr_scale_only"}, 0
        )
        self.assertEqual(multiplier, 0.125 / 16.0)
        self.assertEqual(windows, 2)

        _, first, windows = _controlled_arm_policy(
            {"controlled_arm_id": "from_u0_cosine_only"}, 0
        )
        _, later, _ = _controlled_arm_policy(
            {"controlled_arm_id": "from_u0_cosine_only"}, 15
        )
        self.assertEqual(first, 1.0)
        self.assertLess(later, first)
        self.assertEqual(windows, 2)

        _, multiplier, windows = _controlled_arm_policy(
            {"controlled_arm_id": "from_u0_window_dose6_only"}, 0
        )
        self.assertEqual(multiplier, 1.0 / 16.0)
        self.assertEqual(windows, 6)

    def test_from_u0_policy_rejects_out_of_range_cursor(self):
        with self.assertRaises(ArtifactError):
            _controlled_arm_policy({"controlled_arm_id": "from_u0_lr_scale_only"}, -1)
        with self.assertRaises(ArtifactError):
            _controlled_arm_policy({"controlled_arm_id": "from_u0_lr_scale_only"}, 64)

    def test_from_u0_start_and_same_recipe_scored_resume_are_admitted(self):
        _validate_controlled_arm_start(
            "from_u0_lr_scale_only", 0, {}, controlled_config_sha256="cfg"
        )
        _validate_controlled_arm_start(
            "from_u0_lr_scale_only", 4,
            {"controlled_arm_id": "from_u0_lr_scale_only", "controlled_config_sha256": "cfg"},
            controlled_config_sha256="cfg",
        )

    def test_from_u0_midpoint_recipe_switch_is_rejected_until_kld_resume_seal(self):
        with self.assertRaises(ArtifactError):
            _validate_controlled_arm_start(
                "from_u0_cosine_only", 16,
                {"controlled_arm_id": "from_u0_lr_scale_only", "controlled_config_sha256": "lr-cfg"},
                controlled_config_sha256="cos-cfg",
            )

    def test_from_u16_recipe_cannot_claim_a_u0_start(self):
        with self.assertRaises(ArtifactError):
            _validate_controlled_arm_start("lr_scale_only", 0, {}, controlled_config_sha256="cfg")

    def test_schedule_origin_and_lambda_cursor_follow_trajectory_origin(self):
        self.assertEqual(_controlled_arm_origin("lr_scale_only"), 16)
        self.assertEqual(_controlled_scheduler_step("lr_scale_only", 0), 16)
        self.assertEqual(_controlled_scheduler_step("lr_scale_only", 4), 20)
        self.assertEqual(_controlled_arm_origin("from_u0_lr_scale_only"), 0)
        self.assertEqual(_controlled_scheduler_step("from_u0_lr_scale_only", 0), 0)
        self.assertEqual(_controlled_scheduler_step("from_u0_lr_scale_only", 4), 4)


if __name__ == "__main__":
    unittest.main()
