#!/usr/bin/env python3
import unittest
from pathlib import Path

from tools.mmlu_density.publish_four_row_results import render


class PublishFourRowResultsTest(unittest.TestCase):
    def terminal(self):
        variants = ["UD-IQ4_XS", "UD-IQ3_XXS", "UD-IQ2_XXS", "DwarfStar-Q2-0731"]
        labels = ["Unsloth IQ4", "Unsloth IQ3", "Unsloth IQ2", "DwarfStar Q2 0731"]
        rows = []
        for index, (variant, label) in enumerate(zip(variants, labels)):
            rows.append({
                "variant": variant, "label": label, "n": 500, "correct": 400 + index,
                "mmlu_percent": 80.0 + index / 5,
                "gold_cross_entropy_bits": 1.123456789,
                "complete_artifact_bytes": 136662446656 - index,
                "complete_decimal_gb": (136662446656 - index) / 1e9,
                "base_equivalent_bpw": 3.8451166272834687 - index,
                "mmlu_capability_density": 0.40123456789012345 + index,
                "relative_density": 1.0 + index,
            })
        return {
            "schema": "banana-smasher.mmlu500-four-row-density-terminal.v1",
            "status": "PASS", "independent_recomputation": "PASS",
            "source_scoring_basis_sha256": "e" * 64,
            "relative_density_reference": "UD-IQ4_XS", "rows": rows,
        }

    def test_exact_public_rows_and_full_density_inputs(self):
        text = render(self.terminal())
        self.assertEqual(sum(f"| {label} |" in text for label in ("Unsloth IQ4", "Unsloth IQ3", "Unsloth IQ2", "DwarfStar Q2 0731")), 4)
        self.assertIn("80.00%", text)
        self.assertIn("136662446656", text)
        self.assertIn("3.8451166272834687", text)
        self.assertIn("0.40123456789012346", text)
        self.assertIn("complete base-plus-drafter", text)
        self.assertIn("independently recomputed", text)
        self.assertIn("source-scoring basis", text)
        self.assertIn("`" + "e" * 64 + "`", text)
        for forbidden in ("/home/", "/Users/", "spark-"):
            self.assertNotIn(forbidden, text)

    def test_refuses_nonsealed_terminal(self):
        terminal = self.terminal()
        terminal["status"] = "INCOMPLETE"
        with self.assertRaisesRegex(ValueError, "unsealed"):
            render(terminal)

    def test_live_evals_page_links_exact_four_row_table(self):
        readme = (Path(__file__).resolve().parents[1] / "Evals/README.md").read_text()
        self.assertIn("MMLU-500 capability-density table", readme)
        self.assertIn("notes/benchmarks/mmlu-density/mmlu500-v1/four-row-results.md", readme)
        self.assertIn("exactly the Unsloth IQ4, Unsloth IQ3, Unsloth IQ2, and DwarfStar Q2", readme)


if __name__ == "__main__":
    unittest.main()
