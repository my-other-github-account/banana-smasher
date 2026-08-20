#!/usr/bin/env python3
"""Fail-closed independent aggregation for the exact four-row MMLU-500 Evals table."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from decimal import Decimal, getcontext
from pathlib import Path

EXPECTED_BASIS_SHA256 = "c5d933b7b1de3b9d22c6f78a042ce44f5be6a7249284a3016342857b92a65423"
EXPECTED_ITEMS_SHA256 = "df6704c4d02550b9155e106bc9a9e1bfe1164a663d509e41a76736bb60d01ded"
EXPECTED_VARIANTS = ["UD-IQ4_XS", "UD-IQ3_XXS", "UD-IQ2_XXS", "DwarfStar-Q2-0731"]
LABELS = "ABCD"
getcontext().prec = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{number}: invalid JSON: {exc}") from exc
    return rows


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def validate_identity(identity: dict, basis_row: dict) -> None:
    variant = basis_row["variant"]
    if identity.get("schema") not in {
        "banana-smasher.mmlu500-model-identity.v1",
        "banana-smasher.mmlu500-public-model-identity.v1",
    }:
        raise SystemExit(f"{variant}: model identity schema mismatch")
    if identity.get("status") != "PASS" or identity.get("variant") != variant:
        raise SystemExit(f"{variant}: model identity status/variant mismatch")
    if identity.get("complete_bytes") != basis_row["complete_bytes"]:
        raise SystemExit(f"{variant}: model identity complete bytes mismatch")
    files = identity.get("files", identity.get("members"))
    if not isinstance(files, list) or not files:
        raise SystemExit(f"{variant}: model identity physical files missing")
    if sum(row.get("bytes", -1) for row in files) != basis_row["complete_bytes"]:
        raise SystemExit(f"{variant}: model identity physical file bytes mismatch")
    for row in files:
        name = row.get("name", row.get("filename"))
        digest = row.get("sha256")
        if not isinstance(name, str) or not name or not isinstance(digest, str) or len(digest) != 64:
            raise SystemExit(f"{variant}: model identity physical file identity incomplete")
    if variant == "DwarfStar-Q2-0731":
        for key in (
            "base_repository", "base_revision", "base_sha256",
            "drafter_repository", "drafter_revision", "drafter_sha256",
        ):
            if identity.get(key) != basis_row[key]:
                raise SystemExit(f"{variant}: model identity {key} mismatch")
    else:
        for key in ("repository", "revision"):
            if identity.get(key) != basis_row[key]:
                raise SystemExit(f"{variant}: model identity {key} mismatch")


def validate_tokenizer(tokenizer: dict, variant: str) -> list[int]:
    if tokenizer.get("schema") != "banana-smasher.mmlu500-tokenizer.v1":
        raise SystemExit(f"{variant}: tokenizer schema mismatch")
    if tokenizer.get("status") != "PASS" or tokenizer.get("prompt_count") != 500:
        raise SystemExit(f"{variant}: tokenizer status/count mismatch")
    choices = tokenizer.get("choices")
    if not isinstance(choices, list) or len(choices) != 4:
        raise SystemExit(f"{variant}: tokenizer choices missing")
    token_ids = []
    for expected, choice in zip(LABELS, choices):
        literal = choice.get("literal", choice.get("choice"))
        ids = choice.get("token_ids")
        if ids is None and isinstance(choice.get("token_id"), int):
            ids = [choice["token_id"]]
        if literal != expected or not isinstance(ids, list) or len(ids) != 1 or not isinstance(ids[0], int):
            raise SystemExit(f"{variant}: whitespace-sensitive literal {expected} token extraction mismatch")
        token_ids.append(ids[0])
    return token_ids


def validate_rows(rows: list[dict], items: list[dict], variant: str, token_ids: list[int]) -> None:
    if len(rows) != 500:
        raise SystemExit(f"{variant}: expected exact n=500, got {len(rows)}")
    for ordinal, (row, item) in enumerate(zip(rows, items)):
        if row.get("schema") != "banana-smasher.mmlu500-qrow.v1":
            raise SystemExit(f"{variant}:{ordinal}: qrow schema mismatch")
        if row.get("sample_ordinal") != ordinal or row.get("row_sha256") != item["row_sha256"]:
            raise SystemExit(f"{variant}:{ordinal}: ordered bank identity mismatch")
        if row.get("source_row_index") != item["source_row_index"]:
            raise SystemExit(f"{variant}:{ordinal}: source row mismatch")
        gold = item["answer_index"]
        if row.get("gold_index") != gold or row.get("gold") != item["answer_letter"]:
            raise SystemExit(f"{variant}:{ordinal}: gold mismatch")
        if row.get("choice_token_ids") != token_ids:
            raise SystemExit(f"{variant}:{ordinal}: choice token IDs drift from tokenizer receipt")
        logits = row.get("choice_logits")
        logprobs = row.get("choice_logprobs")
        if not (isinstance(logits, list) and isinstance(logprobs, list) and len(logits) == len(logprobs) == 4):
            raise SystemExit(f"{variant}:{ordinal}: four-choice logits/logprobs missing")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in logits + logprobs):
            raise SystemExit(f"{variant}:{ordinal}: non-finite choice value")
        maximum = max(logits)
        logsumexp = maximum + math.log(sum(math.exp(value - maximum) for value in logits))
        expected = [value - logsumexp for value in logits]
        if max(abs(a - b) for a, b in zip(logprobs, expected)) > 2e-6:
            raise SystemExit(f"{variant}:{ordinal}: normalized logprob mismatch")
        prediction = max(range(4), key=logits.__getitem__)
        if row.get("prediction_index") != prediction or row.get("prediction") != LABELS[prediction]:
            raise SystemExit(f"{variant}:{ordinal}: prediction mismatch")
        if row.get("correct") is not (prediction == gold):
            raise SystemExit(f"{variant}:{ordinal}: correctness mismatch")
        ordered = sorted(logits, reverse=True)
        if abs(float(row.get("top2_margin")) - (ordered[0] - ordered[1])) > 2e-6:
            raise SystemExit(f"{variant}:{ordinal}: choice margin mismatch")


def aggregate(rows: list[dict], basis_row: dict) -> dict:
    correct = sum(bool(row["correct"]) for row in rows)
    percentage = correct / 5.0
    complete_bytes = int(basis_row["complete_bytes"])
    decimal_gb = complete_bytes / 1e9
    density = (percentage - 25.0) / decimal_gb
    mmlu_per_gb = (Decimal(str(percentage)) - Decimal(25)) / (
        Decimal(complete_bytes) / Decimal(1_000_000_000)
    )
    gold_logprobs = [float(row["choice_logprobs"][row["gold_index"]]) for row in rows]
    return {
        "label": basis_row["label"],
        "variant": basis_row["variant"],
        "n": 500,
        "correct": correct,
        "mmlu_percent": percentage,
        "gold_cross_entropy_bits": -statistics.fmean(gold_logprobs) / math.log(2.0),
        "complete_artifact_bytes": complete_bytes,
        "complete_decimal_gb": decimal_gb,
        "base_equivalent_bpw": basis_row["base_equivalent_bpw"],
        "mmlu_capability_density": density,
        "mmlu_per_gb": str(mmlu_per_gb),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--results-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sha256(args.basis) != EXPECTED_BASIS_SHA256:
        raise SystemExit("BASIS GATE REFUSAL: four-row mission basis SHA mismatch")
    if sha256(args.items) != EXPECTED_ITEMS_SHA256:
        raise SystemExit("BASIS GATE REFUSAL: items SHA mismatch")
    basis = json.loads(args.basis.read_text())
    items = load_jsonl(args.items)
    if len(items) != 500 or [item.get("sample_ordinal") for item in items] != list(range(500)):
        raise SystemExit("BASIS GATE REFUSAL: bank count/order mismatch")
    basis_rows = basis.get("rows")
    if not isinstance(basis_rows, list) or [row.get("variant") for row in basis_rows] != EXPECTED_VARIANTS:
        raise SystemExit("BASIS GATE REFUSAL: exact four model rows/order mismatch")

    manifest = json.loads(args.results_manifest.read_text())
    entries = manifest.get("rows")
    if not isinstance(entries, list) or len(entries) != 4:
        raise SystemExit("results manifest must contain exactly four model rows")
    if [entry.get("variant") for entry in entries] != EXPECTED_VARIANTS:
        raise SystemExit("results manifest must contain exactly four scoped model rows in basis order")

    summaries = []
    for basis_row, entry in zip(basis_rows, entries):
        variant = basis_row["variant"]
        qrows_path = Path(entry["qrows"])
        identity_path = Path(entry["model_identity"])
        tokenizer_path = Path(entry["tokenizer_receipt"])
        identity = json.loads(identity_path.read_text())
        tokenizer = json.loads(tokenizer_path.read_text())
        validate_identity(identity, basis_row)
        token_ids = validate_tokenizer(tokenizer, variant)
        qrows = load_jsonl(qrows_path)
        validate_rows(qrows, items, variant, token_ids)
        summary = aggregate(qrows, basis_row)
        summary.update({
            "qrows_sha256": sha256(qrows_path),
            "model_identity_sha256": sha256(identity_path),
            "tokenizer_receipt_sha256": sha256(tokenizer_path),
            "choice_token_ids": token_ids,
        })
        summaries.append(summary)

    reference_density = summaries[0]["mmlu_capability_density"]
    if not math.isfinite(reference_density) or reference_density == 0.0:
        raise SystemExit("UD-IQ4_XS reference density is zero/non-finite")
    for summary in summaries:
        summary["relative_density"] = summary["mmlu_capability_density"] / reference_density
    summaries[0]["relative_density"] = 1.0

    output = {
        "schema": "banana-smasher.mmlu500-four-row-density-terminal.v1",
        "status": "PASS",
        "basis_sha256": EXPECTED_BASIS_SHA256,
        "source_scoring_basis_sha256": basis["source_scoring_basis_sha256"],
        "public_copy_transform": basis["public_copy_transform"],
        "items_sha256": EXPECTED_ITEMS_SHA256,
        "results_manifest_sha256": sha256(args.results_manifest),
        "independent_recomputation": "PASS",
        "relative_density_reference": "UD-IQ4_XS",
        "rows": summaries,
    }
    atomic_json(args.output, output)
    print(json.dumps({"status": "PASS", "output": str(args.output), "output_sha256": sha256(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
