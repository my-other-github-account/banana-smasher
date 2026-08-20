#!/usr/bin/env python3
"""Build the immutable Banana Smasher zero-shot MMLU-500 bank.

Selection is implementation-independent: rank every pinned MMLU test row by
SHA256(seed + NUL + canonical-row-SHA256), then take the lowest 500 ranks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

DATASET_REPOSITORY = "cais/mmlu"
DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
DATASET_CONFIG = "all"
DATASET_SPLIT = "test"
DATASET_FILE = "all/test-00000-of-00001.parquet"
DATASET_URL = (
    "https://huggingface.co/datasets/cais/mmlu/resolve/"
    f"{DATASET_REVISION}/{DATASET_FILE}"
)
SEED = "banana-smasher-mmlu500-v1-2026-08-10"
SAMPLE_SIZE = 500
CHOICE_LABELS = ("A", "B", "C", "D")
REFERENCE_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
PROMPT_TEMPLATE = (
    "The following are multiple choice questions (with answers) about {subject}.\n\n"
    "{question}\n"
    "A. {choice_a}\n"
    "B. {choice_b}\n"
    "C. {choice_c}\n"
    "D. {choice_d}\n"
    "Answer:"
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_rows(parquet_path: Path) -> list[dict[str, Any]]:
    try:
        import duckdb
    except ImportError as exc:
        raise SystemExit(
            "duckdb is required; run in a venv with `pip install duckdb==1.4.3`"
        ) from exc

    connection = duckdb.connect(database=":memory:")
    try:
        cursor = connection.execute(
            "SELECT question, subject, choices, answer FROM read_parquet(?)",
            [str(parquet_path)],
        )
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        connection.close()


def build_item(source_row_index: int, source: dict[str, Any]) -> dict[str, Any]:
    choices = [str(choice) for choice in source["choices"]]
    if len(choices) != 4:
        raise RuntimeError(
            f"source row {source_row_index} has {len(choices)} choices, expected 4"
        )
    answer_index = int(source["answer"])
    if answer_index not in range(4):
        raise RuntimeError(
            f"source row {source_row_index} answer {answer_index} is outside 0..3"
        )
    subject = str(source["subject"])
    subject_display = subject.replace("_", " ")
    row_identity = {
        "source_row_index": source_row_index,
        "subject": subject,
        "question": str(source["question"]),
        "choices": choices,
        "answer_index": answer_index,
    }
    row_sha256 = sha256_bytes(canonical_bytes(row_identity))
    selection_sha256 = sha256_bytes(f"{SEED}\0{row_sha256}".encode())
    prompt = PROMPT_TEMPLATE.format(
        subject=subject_display,
        question=row_identity["question"],
        choice_a=choices[0],
        choice_b=choices[1],
        choice_c=choices[2],
        choice_d=choices[3],
    )
    return {
        **row_identity,
        "subject_display": subject_display,
        "answer_letter": CHOICE_LABELS[answer_index],
        "prompt": prompt,
        "row_sha256": row_sha256,
        "selection_sha256": selection_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.parquet.is_file():
        raise FileNotFoundError(args.parquet)
    source_rows = read_rows(args.parquet)
    if len(source_rows) < SAMPLE_SIZE:
        raise RuntimeError(
            f"population {len(source_rows)} is smaller than sample {SAMPLE_SIZE}"
        )

    population = [build_item(index, row) for index, row in enumerate(source_rows)]
    selected = sorted(
        population, key=lambda row: (row["selection_sha256"], row["row_sha256"])
    )[:SAMPLE_SIZE]
    for sample_ordinal, row in enumerate(selected):
        row["sample_ordinal"] = sample_ordinal

    item_lines = [canonical_bytes(row) for row in selected]
    items_payload = b"\n".join(item_lines) + b"\n"
    items_sha256 = sha256_bytes(items_payload)
    subjects = Counter(row["subject"] for row in selected)
    answers = Counter(row["answer_letter"] for row in selected)

    manifest = {
        "schema": "banana-smasher.mmlu500-bank.v1",
        "status": "FROZEN",
        "benchmark": "MMLU",
        "protocol": {
            "shots": 0,
            "generation": False,
            "scoring": "one prompt forward; argmax over A/B/C/D logits at final prompt position",
            "choice_labels": list(CHOICE_LABELS),
            "prompt_template": PROMPT_TEMPLATE,
        },
        "dataset": {
            "repository": DATASET_REPOSITORY,
            "revision": DATASET_REVISION,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "file": DATASET_FILE,
            "url": DATASET_URL,
            "parquet_sha256": sha256_file(args.parquet),
            "population_rows": len(population),
        },
        "sample": {
            "seed": SEED,
            "method": "sha256-rank-v1",
            "method_definition": "rank each row by sha256(seed + NUL + canonical_row_sha256); take lowest N",
            "ordered_by": ["selection_sha256", "row_sha256"],
            "rows": SAMPLE_SIZE,
            "items_file": "items.jsonl",
            "items_sha256": items_sha256,
            "first_selection_sha256": selected[0]["selection_sha256"],
            "last_selection_sha256": selected[-1]["selection_sha256"],
            "subject_counts": dict(sorted(subjects.items())),
            "answer_counts": dict(sorted(answers.items())),
        },
        "density": {
            "chance_accuracy": 0.25,
            "complete_size_unit": "decimal GB (complete model artifact bytes / 1e9)",
            "mmlu_density_formula": "(100 * accuracy - 25) / complete_decimal_gb",
            "mmlu_density_unit": "MMLU percentage points above chance per complete decimal GB",
            "relative_density_formula": "candidate_mmlu_density / reference_mmlu_density",
            "reference_model": REFERENCE_MODEL,
        },
    }
    manifest_payload = json.dumps(
        manifest, indent=2, sort_keys=True, ensure_ascii=False
    ).encode("utf-8") + b"\n"

    atomic_write(args.output_dir / "items.jsonl", items_payload)
    atomic_write(args.output_dir / "manifest.json", manifest_payload)
    print(
        json.dumps(
            {
                "status": "PASS",
                "population_rows": len(population),
                "sample_rows": SAMPLE_SIZE,
                "parquet_sha256": manifest["dataset"]["parquet_sha256"],
                "items_sha256": items_sha256,
                "manifest_sha256": sha256_bytes(manifest_payload),
                "subject_count": len(subjects),
                "answer_counts": dict(sorted(answers.items())),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
