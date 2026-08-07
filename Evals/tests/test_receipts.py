from __future__ import annotations

import copy
import json
import unittest
from decimal import Decimal, localcontext
from pathlib import Path

from Evals.tools.receipts import ReceiptError, verify_result_receipt


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "Evals" / "results" / "deepseek-v4-flash-0731-balanced64-v1.json"
SUITE_LOCK = ROOT / "Evals" / "configs" / "balanced64-v1.json"
README = ROOT / "Evals" / "README.md"


class Balanced64ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.receipt = json.loads(RESULT.read_text(encoding="utf-8"))
        cls.suite_lock = json.loads(SUITE_LOCK.read_text(encoding="utf-8"))

    def test_verifier_reports_eight_rows_and_expected_rankings(self) -> None:
        summary = verify_result_receipt(self.receipt, self.suite_lock)

        self.assertEqual(summary["models"], 8)
        self.assertEqual(
            summary["top1_ranking"],
            [
                "UD-IQ4_XS",
                "QTIP3-uniform-exact",
                "BQ23-PRE-FF0731",
                "QTIP2.5-all43-FF0731",
                "UD-IQ3_XXS",
                "QTIP2-corrected-all43",
                "UD-IQ2_XXS",
                "DwarfStar-Q2-0731",
            ],
        )
        self.assertEqual(
            summary["kld_ranking"],
            [
                "UD-IQ4_XS",
                "QTIP3-uniform-exact",
                "BQ23-PRE-FF0731",
                "UD-IQ3_XXS",
                "QTIP2.5-all43-FF0731",
                "QTIP2-corrected-all43",
                "UD-IQ2_XXS",
                "DwarfStar-Q2-0731",
            ],
        )

    def test_bq23_row_preserves_exact_measurement_and_accounting(self) -> None:
        row = next(
            row for row in self.receipt["results"] if row["model_id"] == "BQ23-PRE-FF0731"
        )

        self.assertEqual(row["top1"], {
            "matches": 59465,
            "positions": 65536,
            "rate": "0.9073638916015625",
        })
        self.assertEqual(row["kld"]["mean"], "0.17548247979523035")
        self.assertEqual(sum(item["top1_matches"] for item in row["classes"].values()), 59465)
        self.assertEqual(sum(item["positions"] for item in row["classes"].values()), 65536)
        weighted_kld = sum(
            Decimal(item["kld_mean"]) * item["positions"]
            for item in row["classes"].values()
        ) / Decimal(65536)
        self.assertEqual(repr(float(weighted_kld)), row["kld"]["mean"])

        components = row["weight_components"]
        expert = components["expert_plane_payload"]
        retained = components["retained_non_routed_payload"]
        metadata = components["weight_pack_index_metadata"]
        repair = components["repair_payload"]
        self.assertEqual(sum(expert["components"].values()), expert["bytes"])
        self.assertEqual(sum(retained["components"].values()), retained["bytes"])
        self.assertEqual(
            expert["bytes"] + retained["bytes"] + metadata["bytes"] + repair["bytes"],
            row["wire"]["bytes"],
        )
        self.assertEqual(row["wire"]["bytes"], 121051240695)
        self.assertEqual(row["wire"]["decimal_gb"], "121.051240695")
        self.assertEqual(
            row["wire"]["normalized_bpw"],
            "3.4058817893203762511833031389578886340348442685415543540833912222265507571657039",
        )

        expert_accounting = row["expert_plane_accounting"]
        self.assertEqual(expert_accounting["bytes"], 101334321540)
        self.assertEqual(expert_accounting["parameter_denominator"], 277025390592)
        self.assertEqual(
            expert_accounting["bpw"],
            "2.9263547669316447058389353197674418604651162790697674418604651162790697674418605",
        )
        with localcontext() as context:
            context.prec = len(Decimal(expert_accounting["bpw"]).as_tuple().digits)
            expected_expert_bpw = Decimal(expert_accounting["bytes"] * 8) / Decimal(
                expert_accounting["parameter_denominator"]
            )
        self.assertEqual(Decimal(expert_accounting["bpw"]), expected_expert_bpw)

        tier_counts = row["assignment"]["tier_counts"]
        self.assertEqual(sum(tier_counts.values()), 22016)
        self.assertEqual(tier_counts["native_mxfp4"], 1951)
        self.assertEqual(tier_counts["qtip2"], 4236)
        self.assertEqual(tier_counts["qtip3"], 15829)
        self.assertEqual(row["assignment"]["assignment_map_sha256"], "020a8bd13281b8aac1988f62e5570772549effc39391d6650f2a3c47b54c394a")
        self.assertEqual(row["artifact"]["physical_pack_sha256"], "1f844a19bddc77e0cf978510ac975ae7133c6cf79289e84b1c5de69f84a99628")
        self.assertFalse(row["measurement"]["repair_applied"])
        self.assertFalse(row["measurement"]["holdout_used"])
        self.assertFalse(row["measurement"]["fallback_used"])

    def test_verifier_rejects_bq23_tier_redistribution_with_same_total(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        row = next(
            row for row in receipt["results"] if row["model_id"] == "BQ23-PRE-FF0731"
        )
        row["assignment"]["tier_counts"]["native_mxfp4"] += 1
        row["assignment"]["tier_counts"]["qtip3"] -= 1

        with self.assertRaisesRegex(ReceiptError, "tier counts differ"):
            verify_result_receipt(receipt, self.suite_lock)

    def test_verifier_rejects_self_consistent_bq23_identity_replacement(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        row = next(
            row for row in receipt["results"] if row["model_id"] == "BQ23-PRE-FF0731"
        )
        replacement = "0" * 64
        row["artifact"]["assignment_map_sha256"] = replacement
        row["assignment"]["assignment_map_sha256"] = replacement
        row["measurement"]["assignment_map_sha256"] = replacement

        with self.assertRaisesRegex(ReceiptError, "published identity"):
            verify_result_receipt(receipt, self.suite_lock)

    def test_verifier_rejects_bq23_physical_receipt_hash_drift(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        row = next(
            row for row in receipt["results"] if row["model_id"] == "BQ23-PRE-FF0731"
        )
        row["measurement"]["artifacts"]["canonical_aggregate_sha256"] = "0" * 64

        with self.assertRaisesRegex(ReceiptError, "physical receipt identities differ"):
            verify_result_receipt(receipt, self.suite_lock)

    def test_readme_publishes_bq23_as_eighth_ranked_row(self) -> None:
        readme = README.read_text(encoding="utf-8")

        self.assertIn("for eight quants", readme)
        self.assertIn(
            "| **BQ23-PRE** | **90.74%** (59,465/65,536) | **0.175482** | 121.051 | 3.406 | FP8 e4m3 dynamic own-base |",
            readme,
        )
        self.assertIn("IQ4 > QTIP3 > BQ23-PRE > QTIP2.5 > IQ3", readme)
        self.assertIn("IQ4 > QTIP3 > BQ23-PRE > IQ3 > QTIP2.5", readme)
        self.assertIn("90.68% (17,643/19,456)", readme)
        self.assertIn("0.2312 | 0.0737 | 0.1428 | 0.1933 | 0.2339 | 0.0849", readme)


if __name__ == "__main__":
    unittest.main()
