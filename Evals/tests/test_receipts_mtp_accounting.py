from __future__ import annotations

import hashlib
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from Evals.tools.receipts import main, verify_result_receipt


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

        self.assertEqual(summary["models"], 10)
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
        self.assertEqual(scopes["QTIP2-corrected-all43"], "base-plus-native-mtp")
        self.assertEqual(scopes["DwarfStar-Q2-0731"], "base-plus-separate-drafter")

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
