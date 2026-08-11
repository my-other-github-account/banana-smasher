#!/usr/bin/env python3
"""Render the sealed four-row MMLU-500 density terminal as public Markdown."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

EXPECTED_VARIANTS = ["UD-IQ4_XS", "UD-IQ3_XXS", "UD-IQ2_XXS", "DwarfStar-Q2-0731"]


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def render(terminal: dict) -> str:
    rows = terminal.get("rows")
    if (
        terminal.get("schema") != "banana-smasher.mmlu500-four-row-density-terminal.v1"
        or terminal.get("status") != "PASS"
        or terminal.get("independent_recomputation") != "PASS"
        or terminal.get("relative_density_reference") != "UD-IQ4_XS"
        or not isinstance(rows, list)
        or [row.get("variant") for row in rows] != EXPECTED_VARIANTS
    ):
        raise ValueError("refusing to render an unsealed or out-of-scope terminal")
    lines = [
        "# MMLU-500 Evals capability density",
        "",
        "All four rows use the immutable `mmlu500-v1` bank: 500 ordered zero-shot literal prompts, no chat template or answer generation, and final-position A/B/C/D logits normalized over the four choices. Aggregates below were independently recomputed from the published per-question rows.",
        "",
        "| Evals row | MMLU | MMLU % | Gold CE (bits) | Complete bytes | Decimal GB | Base-eq BPW | Capability density | Density vs Unsloth IQ4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {correct}/500 | {mmlu:.2f}% | {ce:.6f} | {bytes} | {gb:.9f} | {bpw} | {density} | {relative:.4f}x |".format(
                label=row["label"], correct=row["correct"], mmlu=row["mmlu_percent"],
                ce=row["gold_cross_entropy_bits"], bytes=row["complete_artifact_bytes"],
                gb=row["complete_decimal_gb"], bpw=row["base_equivalent_bpw"],
                density=row["mmlu_capability_density"], relative=row["relative_density"],
            )
        )
    lines += [
        "",
        "Capability density is `(MMLU percentage - 25) / complete decimal GB`. Relative density uses `Unsloth IQ4` as the fixed 1.0x reference. DwarfStar's denominator is the complete base-plus-drafter Evals payload even though the measured next-token logits come from the target/base model.",
        "",
        "Machine-readable aggregates and evidence hashes are in [`results.json`](results.json). The exact model basis is [`four-row-mission-basis.json`](four-row-mission-basis.json), and the frozen prompts are [`items.jsonl`](items.jsonl).",
        "",
        "Public-safe per-question records, tokenizer receipts, and complete physical model identities are in [`evidence/`](evidence/).",
        "",
        f"The public basis records source-scoring basis `{terminal['source_scoring_basis_sha256']}` and its non-scientific public-copy transform; bank, model, and scoring fields are unchanged.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    terminal = json.loads(args.terminal.read_text(encoding="utf-8"))
    atomic_text(args.output, render(terminal))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
