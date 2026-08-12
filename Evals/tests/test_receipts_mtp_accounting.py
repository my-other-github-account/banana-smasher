from __future__ import annotations

import hashlib
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from Evals.tools.receipts import ReceiptError, _verify_classes, main, verify_result_receipt


class PublishedMtpAccountingTest(unittest.TestCase):
    def test_published_result_verifies_with_explicit_mtp_scopes(self) -> None:
        evals_dir = Path(__file__).resolve().parents[1]
        result = json.loads(
            (evals_dir / "results/deepseek-v4-flash-0731-balanced64-v1.json").read_text()
        )
        suite_lock = json.loads(
            (evals_dir / "configs/balanced64-v1.json").read_text()
        )

        summary = verify_result_receipt(result, suite_lock)

        self.assertEqual(summary["models"], 13)
        model_ids = {row["model_id"] for row in result["results"]}
        self.assertIn("EXL3-K2P5-greedy-full", model_ids)
        self.assertIn("EXL3-K2P5-greedy-routed-native-rest", model_ids)
        self.assertIn("EXL3-K3-routed-native-rest", model_ids)
        self.assertIn("Physical-K2K3-2P5-alternating-comparator", model_ids)
        self.assertNotIn("QTIP2-corrected-all43", model_ids)
        self.assertNotIn("EXL3-K2P5-physical-alternating", model_ids)

        full_greedy = next(
            row for row in result["results"] if row["model_id"] == "EXL3-K2P5-greedy-full"
        )
        self.assertEqual(full_greedy["display_name"], "EXL3 K2.5 greedy optimizer full")
        self.assertEqual(full_greedy["top1"]["matches"], 54_732)
        self.assertEqual(full_greedy["kld"]["mean"], "0.30277489559979315")
        self.assertEqual(full_greedy["wire"]["bytes"], 94_832_865_520)
        self.assertEqual(
            full_greedy["artifact"]["candidate_artifact_sha256"],
            "7c8d1aa6d5fea5c22374346b0e18450881cc97cee118f7bc75f064f56f828044",
        )
        self.assertEqual(
            full_greedy["artifact"]["candidate_manifest_sha256"],
            "a226f60c6193f6fb2a8b1240cbf83b8ecea3bea3de9d905460244501545cc503",
        )

        routed_greedy = next(
            row
            for row in result["results"]
            if row["model_id"] == "EXL3-K2P5-greedy-routed-native-rest"
        )
        self.assertEqual(
            routed_greedy["display_name"],
            "EXL3 K2.5 greedy-upcast routed-only + native rest",
        )
        self.assertEqual(routed_greedy["top1"]["matches"], 57_885)
        self.assertEqual(routed_greedy["kld"]["mean"], "0.1746041415211709")
        self.assertEqual(routed_greedy["wire"]["bytes"], 106_282_510_072)
        self.assertEqual(
            routed_greedy["artifact"]["candidate_artifact_sha256"],
            "5bedb489dfe62bad9107948d011a42cf888f7e2789a6386b680b2da7681be051",
        )
        self.assertEqual(
            routed_greedy["artifact"]["candidate_manifest_sha256"],
            "6e77d799bbc6516375fddeda848df972143639880140099b840ef364b035aad7",
        )
        self.assertNotEqual(
            routed_greedy["artifact"]["candidate_artifact_sha256"],
            full_greedy["artifact"]["candidate_artifact_sha256"],
        )

        routed_k3 = next(
            row for row in result["results"] if row["model_id"] == "EXL3-K3-routed-native-rest"
        )
        self.assertEqual(routed_k3["display_name"], "EXL3 K3 routed-only + native rest")
        self.assertEqual(routed_k3["top1"]["matches"], 60_447)
        self.assertEqual(routed_k3["kld"]["mean"], "0.07686796725357639")
        self.assertEqual(routed_k3["wire"]["bytes"], 123_999_250_168)
        self.assertEqual(
            routed_k3["artifact"]["candidate_manifest_sha256"],
            "42f3d57f5f112a9dbb7badd4dc76536f0ef6da7a3fe0422bc271891b487f83c8",
        )

        alternating = next(
            row
            for row in result["results"]
            if row["model_id"] == "Physical-K2K3-2P5-alternating-comparator"
        )
        self.assertEqual(
            alternating["display_name"],
            "Physical alternating K2/K3 2.5-BPW comparator",
        )
        self.assertEqual(alternating["top1"]["matches"], 54_585)
        self.assertEqual(alternating["kld"]["mean"], "0.29960352599248635")

        matrix_text = (evals_dir / "README.md").read_text()
        public_table = matrix_text.split("## EXL 2×3 scope/rate matrix", 1)[0]
        self.assertIn("| Comparison BPW | FP basis |", public_table)
        self.assertNotIn("Matched physical BPW", public_table)
        self.assertNotIn("| Base-equivalent BPW |", public_table)
        self.assertIn(
            "| **EXL3 K2.5 greedy-upcast routed-only + native rest** | "
            "**88.33%** (57,885/65,536) | **0.174604** | **84.80%** (424/500) | "
            "**19.998** | **28.358** | **0.563** | 106.283 | "
            "MTP included | 2.990 | FP8 e4m3 dynamic own-base |",
            public_table,
        )
        self.assertIn(
            "| **DwarfStar Q2** | **83.69%** (54,845/65,536) | "
            "**0.309521** | **80.60%** (403/500) | **21.092** | **30.576** | "
            "**0.593** | 93.691 | Stock MTP excluded; separate drafter included | "
            "2.636 | FP8 e4m3 dynamic own-base |",
            public_table,
        )
        self.assertIn("## EXL 2×3 scope/rate matrix", matrix_text)
        self.assertNotIn("measurement in progress", matrix_text)
        self.assertIn("57,885/65,536; KLD 0.174604", matrix_text)
        self.assertIn("Physical alternating K2/K3 2.5-BPW comparator", matrix_text)

        accounting_receipt = (
            evals_dir / "results/deepseek-v4-flash-0731-mtp-size-accounting-v1.json"
        )
        self.assertEqual(
            hashlib.sha256(accounting_receipt.read_bytes()).hexdigest(),
            result["size_accounting"]["receipt_sha256"],
        )
        self.assertEqual(result["mtp_inclusive_parameter_denominator"], 294_550_374_339)
        scopes = {
            row["model_id"]: row["wire"]["artifact_payload_scope"]
            for row in result["results"]
        }
        self.assertEqual(scopes["UD-IQ4_XS"], "base-model-only")
        self.assertEqual(scopes["DwarfStar-Q2-0731"], "base-plus-separate-drafter")
        self.assertEqual(scopes["EXL3-K2-routed-native-rest"], "base-plus-native-mtp")
        self.assertEqual(
            scopes["EXL3-K2P5-greedy-routed-native-rest"],
            "base-plus-native-mtp",
        )
        accounting = json.loads(accounting_receipt.read_text())
        routed_greedy_accounting = accounting["rows"][
            "EXL3-K2P5-greedy-routed-native-rest"
        ]
        self.assertEqual(routed_greedy_accounting["shipping_bytes"], 106_282_510_072)
        self.assertEqual(
            routed_greedy_accounting["routed_optimizer_payload_bytes"],
            86_573_712_384,
        )
        self.assertEqual(
            routed_greedy_accounting["retained_native_nonrouted_payload_bytes"],
            19_708_797_688,
        )
        self.assertEqual(
            routed_greedy_accounting["routed_payload_bytes_by_source"],
            {"K2": 35_641_165_824, "K3": 50_932_546_560},
        )
        self.assertEqual(
            routed_greedy_accounting["solution_group_counts"],
            {"K2": 66, "K3": 63},
        )
        self.assertEqual(
            routed_greedy_accounting["routed_group_counts"],
            {"K2": 42, "K3": 44},
        )
        routed_k2 = next(
            row
            for row in result["results"]
            if row["model_id"] == "EXL3-K2-routed-native-rest"
        )
        self.assertEqual(routed_k2["display_name"], "EXL3 K2 routed-only + native rest")
        self.assertEqual(routed_k2["wire"]["bytes"], 89_371_076_344)
        self.assertEqual(
            routed_k2["wire"]["normalized_bpw"],
            "2.5145328512486971484262613667868966546438084310621785627887683259240843040121566",
        )
        self.assertEqual(
            routed_k2["wire"]["total_model_bpw"],
            "2.4273220238013951186880076912235235955776634019794933505294131935834137741587886",
        )
        self.assertEqual(routed_k2["top1"]["matches"], 56_579)
        self.assertEqual(routed_k2["kld"]["mean"], "0.23428769710091882")
        self.assertEqual(
            routed_k2["artifact"]["mechanism"].split(";")[0],
            "homogeneous K2/mul1 only for layers.*.ffn.experts.*",
        )

    def test_exl_matrix_protected_receipt_hash_drift_is_rejected(self) -> None:
        evals_dir = Path(__file__).resolve().parents[1]
        result = json.loads(
            (evals_dir / "results/deepseek-v4-flash-0731-balanced64-v1.json").read_text()
        )
        suite_lock = json.loads(
            (evals_dir / "configs/balanced64-v1.json").read_text()
        )
        full_greedy = next(
            row for row in result["results"] if row["model_id"] == "EXL3-K2P5-greedy-full"
        )
        solution_receipt = next(
            source
            for source in full_greedy["source_receipts"]
            if source["label"] == "EXL3 K2.5 greedy exact-rate optimizer solution"
        )
        solution_receipt["sha256"] = "f" * 64

        with self.assertRaisesRegex(
            ReceiptError,
            "EXL3-K2P5-greedy-full: protected EXL publication value drift",
        ):
            verify_result_receipt(result, suite_lock)

    def test_routed_greedy_protected_receipt_hash_drift_is_rejected(self) -> None:
        evals_dir = Path(__file__).resolve().parents[1]
        result = json.loads(
            (evals_dir / "results/deepseek-v4-flash-0731-balanced64-v1.json").read_text()
        )
        suite_lock = json.loads(
            (evals_dir / "configs/balanced64-v1.json").read_text()
        )
        routed_greedy = next(
            row
            for row in result["results"]
            if row["model_id"] == "EXL3-K2P5-greedy-routed-native-rest"
        )
        terminal_receipt = next(
            source
            for source in routed_greedy["source_receipts"]
            if source["label"] == "EXL3 K2.5 greedy routed-native Exact64 terminal"
        )
        terminal_receipt["sha256"] = "f" * 64

        with self.assertRaisesRegex(
            ReceiptError,
            "EXL3-K2P5-greedy-routed-native-rest: protected EXL publication value drift",
        ):
            verify_result_receipt(result, suite_lock)

    def test_class_kld_reaggregation_accepts_one_binary64_ulp_from_rounded_means(
        self,
    ) -> None:
        evals_dir = Path(__file__).resolve().parents[1]
        suite_lock = json.loads(
            (evals_dir / "configs/balanced64-v1.json").read_text()
        )
        classes = {
            "agentic": {
                "windows": 19,
                "positions": 19456,
                "kld_mean": "0.3231889470175587",
                "top1_matches": 16815,
                "top1_rate": "0.8642578125",
            },
            "chat": {
                "windows": 7,
                "positions": 7168,
                "kld_mean": "0.0989122228686422",
                "top1_matches": 6401,
                "top1_rate": "0.8929966517857142857142857142857142857142857142857142857142857142857142857142857142857142857142857143",
            },
            "code": {
                "windows": 9,
                "positions": 9216,
                "kld_mean": "0.1223480211377152",
                "top1_matches": 8302,
                "top1_rate": "0.9008246527777777777777777777777777777777777777777777777777777777777777777777777777777777777777777778",
            },
            "multilingual": {
                "windows": 10,
                "positions": 10240,
                "kld_mean": "0.3636492967406757",
                "top1_matches": 8344,
                "top1_rate": "0.81484375",
            },
            "prose": {
                "windows": 10,
                "positions": 10240,
                "kld_mean": "0.2959973654754073",
                "top1_matches": 8210,
                "top1_rate": "0.8017578125",
            },
            "reasoning": {
                "windows": 9,
                "positions": 9216,
                "kld_mean": "0.051537583182714196",
                "top1_matches": 8507,
                "top1_rate": "0.9230685763888888888888888888888888888888888888888888888888888888888888888888888888888888888888888889",
            },
        }

        _verify_classes(
            {"classes": classes},
            suite_lock,
            model_id="EXL3-K2-routed-native-rest",
            kld_mean=Decimal("0.23428769710091882"),
            matches=56579,
            positions=65536,
        )
        with self.assertRaisesRegex(ReceiptError, "more than one binary64 ULP"):
            _verify_classes(
                {"classes": classes},
                suite_lock,
                model_id="EXL3-K2-routed-native-rest",
                kld_mean=Decimal("0.2342876971009187"),
                matches=56579,
                positions=65536,
            )

    def test_cli_rejects_tampered_mtp_accounting_receipt(self) -> None:
        evals_dir = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as temporary_directory:
            temporary_evals = Path(temporary_directory) / "Evals"
            results_dir = temporary_evals / "results"
            configs_dir = temporary_evals / "configs"
            results_dir.mkdir(parents=True)
            configs_dir.mkdir(parents=True)

            comparison_name = "deepseek-v4-flash-0731-balanced64-v1.json"
            correction_name = "deepseek-v4-flash-0731-mtp-size-accounting-v1.json"
            (results_dir / comparison_name).write_bytes(
                (evals_dir / "results" / comparison_name).read_bytes()
            )
            correction = (evals_dir / "results" / correction_name).read_text()
            (results_dir / correction_name).write_text(
                correction.replace('"status": "PASS"', '"status": "FAIL"', 1)
            )
            suite_lock = configs_dir / "balanced64-v1.json"
            suite_lock.write_bytes((evals_dir / "configs/balanced64-v1.json").read_bytes())

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(
                        [
                            "verify",
                            str(results_dir / comparison_name),
                            "--suite-lock",
                            str(suite_lock),
                        ]
                    )


if __name__ == "__main__":
    unittest.main()
