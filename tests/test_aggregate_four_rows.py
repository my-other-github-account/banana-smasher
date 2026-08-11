#!/usr/bin/env python3
import json
import math
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ITEMS = REPO / "notes/benchmarks/mmlu-density/mmlu500-v1/items.jsonl"
BASIS = REPO / "notes/benchmarks/mmlu-density/mmlu500-v1/four-row-mission-basis.json"
AGGREGATOR = REPO / "tools/mmlu_density/aggregate_four_rows.py"


class FourRowAggregateTest(unittest.TestCase):
    def setUp(self):
        self.items = [json.loads(line) for line in ITEMS.read_text().splitlines()]
        self.basis = json.loads(BASIS.read_text())

    def write_json(self, path, value):
        path.write_text(json.dumps(value, sort_keys=True) + "\n")

    def write_rows(self, path, wrong_every):
        rows = []
        for i, item in enumerate(self.items):
            gold = item["answer_index"]
            pred = (gold + 1) % 4 if wrong_every and i % wrong_every == 0 else gold
            logits = [-3.0, -2.0, -4.0, -5.0]
            logits[pred] = 3.0
            m = max(logits)
            lse = m + math.log(sum(math.exp(x - m) for x in logits))
            lps = [x - lse for x in logits]
            ordered = sorted(logits, reverse=True)
            rows.append({
                "schema": "banana-smasher.mmlu500-qrow.v1",
                "sample_ordinal": i,
                "source_row_index": item["source_row_index"],
                "row_sha256": item["row_sha256"],
                "subject": item["subject"],
                "gold_index": gold,
                "gold": item["answer_letter"],
                "choice_token_ids": [10, 11, 12, 13],
                "choice_logits": logits,
                "choice_logprobs": lps,
                "prediction_index": pred,
                "prediction": "ABCD"[pred],
                "correct": pred == gold,
                "top2_margin": ordered[0] - ordered[1],
                "elapsed_seconds": 0.01,
            })
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))

    def invoke(self, manifest, output):
        return subprocess.run([
            sys.executable, str(AGGREGATOR),
            "--basis", str(BASIS), "--items", str(ITEMS),
            "--results-manifest", str(manifest), "--output", str(output),
        ], capture_output=True, text=True)

    def test_exact_four_rows_and_iq4_relative_density(self):
        variants = ["UD-IQ4_XS", "UD-IQ3_XXS", "UD-IQ2_XXS", "DwarfStar-Q2-0731"]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            entries = []
            for basis_row, wrong_every in zip(self.basis["rows"], [0, 10, 5, 4]):
                variant = basis_row["variant"]
                qrows = root / f"{variant}.jsonl"
                identity = root / f"{variant}.identity.json"
                tokenizer = root / f"{variant}.tokenizer.json"
                self.write_rows(qrows, wrong_every)
                identity_payload = {
                    "schema": "banana-smasher.mmlu500-public-model-identity.v1",
                    "status": "PASS",
                    "variant": variant,
                    "complete_bytes": basis_row["complete_bytes"],
                    "files": [{"name": f"{variant}.gguf", "bytes": basis_row["complete_bytes"], "sha256": "1" * 64}],
                }
                if variant == "DwarfStar-Q2-0731":
                    identity_payload.update({
                        "base_repository": "antirez/deepseek-v4-gguf",
                        "base_revision": "1cd7b564460821938add0475a60b942c409295e0",
                        "base_sha256": "ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0",
                        "drafter_repository": "bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF",
                        "drafter_revision": "81c6fdd38f9582da45ba27f0ed7b63bcd3ea3b62",
                        "drafter_sha256": "8fa269560dc76fd73e4233ad9b1938b5f65dd363381fd9b1a5c6183f7d12d686",
                    })
                else:
                    identity_payload.update({
                        "repository": "unsloth/DeepSeek-V4-Flash-0731-GGUF",
                        "revision": "fbbb5b93fb787c21338159b0af3318bb3f4d9768",
                    })
                self.write_json(identity, identity_payload)
                self.write_json(tokenizer, {"schema": "banana-smasher.mmlu500-tokenizer.v1", "status": "PASS", "prompt_count": 500, "choices": [{"literal": c, "token_ids": [10+i]} for i, c in enumerate("ABCD")]})
                entries.append({"variant": variant, "qrows": str(qrows), "model_identity": str(identity), "tokenizer_receipt": str(tokenizer)})
            manifest = root / "results.json"
            output = root / "summary.json"
            self.write_json(manifest, {"schema": "banana-smasher.mmlu500-four-results.v1", "rows": entries})
            proc = self.invoke(manifest, output)
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            result = json.loads(output.read_text())
            self.assertEqual(result["source_scoring_basis_sha256"], self.basis["source_scoring_basis_sha256"])
            self.assertEqual([row["variant"] for row in result["rows"]], variants)
            self.assertEqual(result["rows"][0]["correct"], 500)
            self.assertEqual(result["rows"][0]["relative_density"], 1.0)
            self.assertEqual(result["rows"][3]["complete_artifact_bytes"], 93691352992)
            self.assertEqual(result["rows"][3]["complete_decimal_gb"], 93.691352992)
            self.assertEqual(
                [row["base_equivalent_bpw"] for row in result["rows"]],
                ["3.8451166272834685", "2.931978308348837", "2.556445745541928", "2.6360875868777476"],
            )
            self.assertTrue(all(row["n"] == 500 for row in result["rows"]))
            self.assertEqual(result["independent_recomputation"], "PASS")
            self.assertNotIn("task_id", result)

    def test_public_basis_excludes_private_infrastructure(self):
        text = BASIS.read_text()
        self.assertNotIn("task_id", self.basis)
        self.assertEqual(
            self.basis["source_scoring_basis_sha256"],
            "83ace3f25a4f77325479690a47e7b86f7dee5ef44513996b551f24145ff88f8e",
        )
        for forbidden in (
            "/home/", "/Users/", "dnola", "macmini",
        ):
            self.assertNotIn(forbidden.lower(), text.lower())
        self.assertIsNone(re.search(r"(?<![A-Za-z])spark-[1-8](?![0-9])", text, re.I))
        self.assertIn("DeepSeek-V4-Flash-DSpark-drafter-GGUF", text)

    def test_rejects_identity_complete_byte_mismatch(self):
        variants = [row["variant"] for row in self.basis["rows"]]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            entries = []
            for basis_row in self.basis["rows"]:
                variant = basis_row["variant"]
                qrows = root / f"{variant}.jsonl"
                identity = root / f"{variant}.identity.json"
                tokenizer = root / f"{variant}.tokenizer.json"
                self.write_rows(qrows, 0)
                identity_payload = {
                    "schema": "banana-smasher.mmlu500-public-model-identity.v1", "status": "PASS", "variant": variant,
                    "complete_bytes": basis_row["complete_bytes"],
                    "files": [{"name": f"{variant}.gguf", "bytes": basis_row["complete_bytes"], "sha256": "1" * 64}],
                }
                if variant == "DwarfStar-Q2-0731":
                    identity_payload.update({key: basis_row[key] for key in ("base_repository", "base_revision", "base_sha256", "drafter_repository", "drafter_revision", "drafter_sha256")})
                else:
                    identity_payload.update({key: basis_row[key] for key in ("repository", "revision")})
                self.write_json(identity, identity_payload)
                self.write_json(tokenizer, {"schema": "banana-smasher.mmlu500-tokenizer.v1", "status": "PASS", "prompt_count": 500, "choices": [{"literal": c, "token_ids": [10+i]} for i, c in enumerate("ABCD")]})
                entries.append({"variant": variant, "qrows": str(qrows), "model_identity": str(identity), "tokenizer_receipt": str(tokenizer)})
            bad_identity = Path(entries[0]["model_identity"])
            payload = json.loads(bad_identity.read_text())
            payload["complete_bytes"] -= 1
            self.write_json(bad_identity, payload)
            manifest = root / "results.json"
            output = root / "summary.json"
            self.write_json(manifest, {"schema": "banana-smasher.mmlu500-four-results.v1", "rows": entries})
            proc = self.invoke(manifest, output)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("complete bytes mismatch", proc.stderr + proc.stdout)

    def test_rejects_missing_or_extra_model_row(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "results.json"
            output = root / "summary.json"
            self.write_json(manifest, {"schema": "banana-smasher.mmlu500-four-results.v1", "rows": []})
            proc = self.invoke(manifest, output)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("exactly four", proc.stderr + proc.stdout)


if __name__ == "__main__":
    unittest.main()
