#!/usr/bin/env python3
"""Independently aggregate the two Kimi-K3 IQ1S MMLU-500 rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from decimal import Decimal, getcontext
from pathlib import Path

LABELS = "ABCD"
ROWS = 500
EXPECTED_LEDGER_SHA256 = "7cdb6a0a93a3d613212ac9960666ee9a26256de86709aaa9fa5765cd1c91e8b4"
EXPECTED_ITEMS_SHA256 = "df6704c4d02550b9155e106bc9a9e1bfe1164a663d509e41a76736bb60d01ded"
EXPECTED_BANK_MANIFEST_SHA256 = "2325d58687a0b5def7b48979a5886a9f7c5089c294445e885e0867101b07482d"
EXPECTED_CANDIDATE_IDS = {"A": 32, "B": 33, "C": 34, "D": 35}
BASE_PARAMETER_COUNT = 2_779_931_837_184
BASE_PARAMETER_AUTHORITY = {
    "repository": "moonshotai/Kimi-K3",
    "revision": "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
    "field": "safetensors.total",
}
SOURCE_SPECS = (
    {
        "label": "Kimi-K3 Neuron IQ1S",
        "repository": "vcruz305/Kimi-K3-Neuron-IQ1S-GGUF",
        "revision": "a2d6283870dd97d2f177c69d94fb18120e79fe65",
        "variant": "Neuron-IQ1S",
        "complete_artifact_bytes": 330_167_807_328,
        "qrows_sha256": "bd3e7ee3006dc2120ec1e5cee09aff52c9995dcd0c333e8a0f7d572453ed5258",
        "id_format": "qrow-{ordinal:03d}",
    },
    {
        "label": "Kimi-K3 Unsloth UD-IQ1_S",
        "repository": "unsloth/Kimi-K3-GGUF",
        "revision": "a0836360ce58dfec088d966a97f2ddc8a606279b",
        "variant": "UD-IQ1_S",
        "complete_artifact_bytes": 594_040_923_616,
        "qrows_sha256": "8d2514d7ee5f71a6c551d280b39e95a1bd4ae99afc42093b69cfcfde99391124",
        "id_format": "{ordinal:04d}",
    },
)
IQ4_BYTES = 136_662_446_656
IQ4_QROWS_SHA256 = "376279254d98d0efdfaeba1303099c65a9a3ba4599117616cb107444c083eb16"
IQ4_TOKEN_IDS = [35, 36, 37, 38]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
    return rows


def softmax(logits: list[float]) -> list[float]:
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    total = sum(weights)
    return [value / total for value in weights]


def validate_ledger(ledger: list[dict], manifest: dict) -> None:
    if len(ledger) != ROWS or [row.get("sample_ordinal") for row in ledger] != list(range(ROWS)):
        raise ValueError("ledger must contain ordered ordinals 0..499 exactly once")
    if manifest.get("rows") != ROWS or manifest.get("candidate_ids") != EXPECTED_CANDIDATE_IDS:
        raise ValueError("ledger manifest row count or exact Kimi candidate IDs mismatch")
    binding = manifest.get("binding_bank", {})
    if binding.get("items_sha256") != EXPECTED_ITEMS_SHA256 or binding.get("manifest_sha256") != EXPECTED_BANK_MANIFEST_SHA256:
        raise ValueError("ledger manifest frozen bank identity mismatch")
    for ordinal, row in enumerate(ledger):
        answer = row.get("answer")
        if not isinstance(answer, str) or answer not in LABELS or row.get("answer_index") != LABELS.index(answer):
            raise ValueError(f"ledger:{ordinal}: answer mismatch")
        for key in ("row_sha256", "selection_sha256", "prompt_sha256", "packed_uint32_le_sha256"):
            value = row.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"ledger:{ordinal}: missing {key}")


def validate_kimi_rows(rows: list[dict], ledger: list[dict], id_format: str) -> None:
    if len(rows) != ROWS:
        raise ValueError(f"expected 500 Kimi qrows, got {len(rows)}")
    expected_ids = [id_format.format(ordinal=i) for i in range(ROWS)]
    actual_ids = [row.get("id") for row in rows]
    if actual_ids != expected_ids or len(set(actual_ids)) != ROWS:
        raise ValueError("Kimi qrow IDs must bind ordinals 0..499 exactly once in order")
    for ordinal, (row, item) in enumerate(zip(rows, ledger)):
        logits_map = row.get("candidate_logits")
        probabilities_map = row.get("candidate_probabilities")
        if not isinstance(logits_map, dict) or list(logits_map) != list(LABELS):
            raise ValueError(f"Kimi:{ordinal}: exact A/B/C/D logits missing")
        if not isinstance(probabilities_map, dict) or list(probabilities_map) != list(LABELS):
            raise ValueError(f"Kimi:{ordinal}: exact A/B/C/D probabilities missing")
        logits = [logits_map[label] for label in LABELS]
        probabilities = [probabilities_map[label] for label in LABELS]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in logits + probabilities):
            raise ValueError(f"Kimi:{ordinal}: non-finite candidate value")
        expected_probabilities = softmax(logits)
        if max(abs(actual - expected) for actual, expected in zip(probabilities, expected_probabilities)) > 3e-8:
            raise ValueError(f"Kimi:{ordinal}: candidate probabilities do not normalize logits")
        prediction_index = max(range(4), key=logits.__getitem__)
        if row.get("prediction") != LABELS[prediction_index]:
            raise ValueError(f"Kimi:{ordinal}: prediction mismatch")
        if item.get("sample_ordinal") != ordinal:
            raise ValueError(f"Kimi:{ordinal}: ledger order mismatch")


def aggregate_kimi(rows: list[dict], ledger: list[dict], spec: dict) -> dict:
    gold_log2 = []
    correct = 0
    for row, item in zip(rows, ledger):
        logits = [float(row["candidate_logits"][label]) for label in LABELS]
        probabilities = softmax(logits)
        gold_index = item["answer_index"]
        gold_log2.append(-math.log2(probabilities[gold_index]))
        correct += row["prediction"] == item["answer"]
    mmlu_percent = correct / 5.0
    decimal_gb = spec["complete_artifact_bytes"] / 1e9
    density = (mmlu_percent - 25.0) / decimal_gb
    getcontext().prec = 80
    bpw = Decimal(8) * Decimal(spec["complete_artifact_bytes"]) / Decimal(BASE_PARAMETER_COUNT)
    return {
        "model": spec["label"],
        "repository": spec["repository"],
        "revision": spec["revision"],
        "variant": spec["variant"],
        "scope": "base model only / no drafter-MTP claim",
        "n": ROWS,
        "correct": correct,
        "mmlu_percent": mmlu_percent,
        "gold_cross_entropy_bits": statistics.fmean(gold_log2),
        "complete_artifact_bytes": spec["complete_artifact_bytes"],
        "complete_decimal_gb": decimal_gb,
        "base_parameter_count": BASE_PARAMETER_COUNT,
        "base_equivalent_bpw": str(bpw),
        "complete_size_intelligence_density": density,
        "qrows_sha256": spec["qrows_sha256"],
    }


def validate_iq4(rows: list[dict], ledger: list[dict]) -> dict:
    if len(rows) != ROWS or [row.get("sample_ordinal") for row in rows] != list(range(ROWS)):
        raise ValueError("IQ4 reference must contain ordered ordinals 0..499 exactly once")
    correct = 0
    gold_bits = []
    for ordinal, (row, item) in enumerate(zip(rows, ledger)):
        if row.get("row_sha256") != item["row_sha256"] or row.get("source_row_index") != item["source_row_index"]:
            raise ValueError(f"IQ4:{ordinal}: frozen bank identity mismatch")
        if row.get("gold") != item["answer"] or row.get("gold_index") != item["answer_index"]:
            raise ValueError(f"IQ4:{ordinal}: gold mismatch")
        if row.get("choice_token_ids") != IQ4_TOKEN_IDS:
            raise ValueError(f"IQ4:{ordinal}: candidate token IDs mismatch")
        logits = row.get("choice_logits")
        logprobs = row.get("choice_logprobs")
        if not isinstance(logits, list) or not isinstance(logprobs, list) or len(logits) != 4 or len(logprobs) != 4:
            raise ValueError(f"IQ4:{ordinal}: four-choice values missing")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in logits + logprobs):
            raise ValueError(f"IQ4:{ordinal}: non-finite candidate value")
        prediction = max(range(4), key=logits.__getitem__)
        if row.get("prediction_index") != prediction or row.get("prediction") != LABELS[prediction]:
            raise ValueError(f"IQ4:{ordinal}: prediction mismatch")
        correct += prediction == item["answer_index"]
        gold_bits.append(-float(logprobs[item["answer_index"]]) / math.log(2.0))
    percentage = correct / 5.0
    density = (percentage - 25.0) / (IQ4_BYTES / 1e9)
    return {
        "variant": "UD-IQ4_XS",
        "correct": correct,
        "n": ROWS,
        "mmlu_percent": percentage,
        "gold_cross_entropy_bits": statistics.fmean(gold_bits),
        "complete_artifact_bytes": IQ4_BYTES,
        "complete_decimal_gb": IQ4_BYTES / 1e9,
        "complete_size_intelligence_density": density,
        "relative_density": 1.0,
        "qrows_sha256": IQ4_QROWS_SHA256,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--ledger-manifest", type=Path, required=True)
    parser.add_argument("--neuron-qrows", type=Path, required=True)
    parser.add_argument("--unsloth-qrows", type=Path, required=True)
    parser.add_argument("--iq4-qrows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sha256(args.ledger) != EXPECTED_LEDGER_SHA256:
        raise SystemExit("BASIS GATE REFUSAL: Kimi ledger SHA mismatch")
    for path, spec in zip((args.neuron_qrows, args.unsloth_qrows), SOURCE_SPECS):
        if sha256(path) != spec["qrows_sha256"]:
            raise SystemExit(f"BASIS GATE REFUSAL: {spec['variant']} qrows SHA mismatch")
    if sha256(args.iq4_qrows) != IQ4_QROWS_SHA256:
        raise SystemExit("BASIS GATE REFUSAL: IQ4 qrows SHA mismatch")

    ledger = load_jsonl(args.ledger)
    manifest = json.loads(args.ledger_manifest.read_text(encoding="utf-8"))
    validate_ledger(ledger, manifest)
    summaries = []
    for path, spec in zip((args.neuron_qrows, args.unsloth_qrows), SOURCE_SPECS):
        rows = load_jsonl(path)
        validate_kimi_rows(rows, ledger, spec["id_format"])
        summaries.append(aggregate_kimi(rows, ledger, spec))
    reference = validate_iq4(load_jsonl(args.iq4_qrows), ledger)
    for row in summaries:
        row["relative_to_unsloth_iq4_density"] = row["complete_size_intelligence_density"] / reference["complete_size_intelligence_density"]

    output = {
        "schema": "banana-smasher.kimi-iq1s-mmlu500-density.v1",
        "status": "PASS",
        "independent_recomputation": "PASS",
        "bank": {
            "items_sha256": EXPECTED_ITEMS_SHA256,
            "manifest_sha256": EXPECTED_BANK_MANIFEST_SHA256,
            "ledger_sha256": EXPECTED_LEDGER_SHA256,
            "rows": ROWS,
            "candidate_ids": EXPECTED_CANDIDATE_IDS,
        },
        "base_parameter_authority": {**BASE_PARAMETER_AUTHORITY, "base_parameter_count": BASE_PARAMETER_COUNT},
        "density_formula": "(MMLU percentage - 25) / complete decimal GB",
        "density_unit": "MMLU percentage points above chance per complete decimal GB",
        "relative_density_reference": "UD-IQ4_XS",
        "reference_context": reference,
        "rows": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output_sha256": sha256(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
