#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import math
import unittest
from decimal import Decimal, getcontext, localcontext
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "Evals/tools/mmlu_density/aggregate_kimi_iq1s.py"
spec = importlib.util.spec_from_file_location("aggregate_kimi_iq1s", SCRIPT)
assert spec is not None
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class KimiIQ1SAggregateTest(unittest.TestCase):
    def make_fixture(self):
        ledger = []
        rows = []
        for ordinal in range(500):
            answer_index = ordinal % 4
            answer = "ABCD"[answer_index]
            logits = {label: -2.0 for label in "ABCD"}
            prediction_index = answer_index if ordinal % 5 else (answer_index + 1) % 4
            logits["ABCD"[prediction_index]] = 2.0
            probabilities = module.softmax([logits[label] for label in "ABCD"])
            ledger.append({
                "sample_ordinal": ordinal,
                "answer": answer,
                "answer_index": answer_index,
                "row_sha256": "1" * 64,
                "selection_sha256": "2" * 64,
                "prompt_sha256": "3" * 64,
                "packed_uint32_le_sha256": "4" * 64,
            })
            rows.append({
                "id": f"qrow-{ordinal:03d}",
                "prediction": "ABCD"[prediction_index],
                "candidate_logits": logits,
                "candidate_probabilities": dict(zip("ABCD", probabilities)),
            })
        return ledger, rows

    def test_validates_and_aggregates_exact_500_rows(self):
        ledger, rows = self.make_fixture()
        module.validate_kimi_rows(rows, ledger, "qrow-{ordinal:03d}")
        result = module.aggregate_kimi(rows, ledger, module.SOURCE_SPECS[0])
        self.assertEqual(result["correct"], 400)
        self.assertEqual(result["mmlu_percent"], 80.0)
        self.assertTrue(math.isfinite(result["gold_cross_entropy_bits"]))
        getcontext().prec = 80
        expected_bpw = Decimal(8 * module.SOURCE_SPECS[0]["complete_artifact_bytes"]) / Decimal(module.BASE_PARAMETER_COUNT)
        self.assertEqual(result["base_equivalent_bpw"], str(expected_bpw))
        with localcontext() as context:
            context.prec = 100
            expected_per_gb = (Decimal("80.0") - Decimal(25)) / (
                Decimal(module.SOURCE_SPECS[0]["complete_artifact_bytes"]) / Decimal(1_000_000_000)
            )
        self.assertEqual(result["mmlu_per_gb"], str(expected_per_gb))
        self.assertEqual(result["scope"], "base model only / no drafter-MTP claim")

    def test_rejects_duplicate_or_out_of_order_ids(self):
        ledger, rows = self.make_fixture()
        rows[1]["id"] = rows[0]["id"]
        with self.assertRaisesRegex(ValueError, "ordinals 0..499 exactly once"):
            module.validate_kimi_rows(rows, ledger, "qrow-{ordinal:03d}")

    def test_rejects_nonfinite_or_inconsistent_probabilities(self):
        ledger, rows = self.make_fixture()
        rows[10]["candidate_logits"]["A"] = math.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            module.validate_kimi_rows(rows, ledger, "qrow-{ordinal:03d}")
        ledger, rows = self.make_fixture()
        rows[10]["candidate_probabilities"]["A"] += 0.01
        with self.assertRaisesRegex(ValueError, "do not normalize logits"):
            module.validate_kimi_rows(rows, ledger, "qrow-{ordinal:03d}")

    def test_published_companion_is_public_safe_and_formula_exact(self):
        result_path = REPO / "archive/notes/benchmarks/mmlu-density/mmlu500-v1/kimi-iq1s-results.json"
        report_path = REPO / "archive/notes/benchmarks/mmlu-density/mmlu500-v1/kimi-iq1s-results.md"
        evidence_path = REPO / "archive/notes/benchmarks/mmlu-density/mmlu500-v1/kimi-iq1s-evidence-manifest.json"
        for path in (result_path, report_path, evidence_path):
            self.assertTrue(path.is_file(), path)
        result = json.loads(result_path.read_text())
        evidence = json.loads(evidence_path.read_text())
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["base_parameter_authority"]["base_parameter_count"], 2_779_931_837_184)
        self.assertEqual([row["correct"] for row in result["rows"]], [342, 412])
        self.assertEqual(result["reference_context"]["correct"], 417)
        self.assertEqual(result["reference_context"]["relative_density"], 1.0)
        self.assertEqual(evidence["results_sha256"], hashlib.sha256(result_path.read_bytes()).hexdigest())
        for row in result["rows"]:
            recomputed = (row["mmlu_percent"] - 25.0) / row["complete_decimal_gb"]
            self.assertAlmostEqual(row["complete_size_intelligence_density"], recomputed, places=15)
            self.assertAlmostEqual(float(row["mmlu_per_gb"]), recomputed, places=15)
            self.assertEqual(row["scope"], "base model only / no drafter-MTP claim")
        public_text = "\n".join(path.read_text() for path in (result_path, report_path, evidence_path))
        for forbidden in ("/home/", "/Users/", "task_id", "dnola", "macmini", "spark-"):
            self.assertNotIn(forbidden.lower(), public_text.lower())


if __name__ == "__main__":
    unittest.main()
