"""Exact U12 specialized-kernel selection shared by dispatch and warmup."""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any


MATRIX_PATH = Path(__file__).with_name("specialized_kernel_matrix.json")
DECODE_VARIANTS = {1: "decode_c1", 2: "decode_c2", 4: "decode_c4", 8: "decode_c8", 16: "decode_c16"}
VARIANT_IDS = {
    "decode_c1": 0,
    "decode_c2": 1,
    "decode_c4": 2,
    "decode_c8": 3,
    "decode_c16": 4,
    "prefill_bm16": 5,
    "prefill_large": 6,
    "prefill_exact_2k": 7,
}


@functools.cache
def _rows() -> dict[tuple[str, str, str], dict[str, Any]]:
    document = json.loads(MATRIX_PATH.read_text())
    if document.get("schema") != "banana-smasher-specialized-kernel-matrix-v1":
        raise RuntimeError("specialized kernel matrix schema drift")
    rows = document.get("rows")
    if not isinstance(rows, list) or len(rows) != 96:
        raise RuntimeError("specialized kernel matrix must contain exactly 96 rows")
    indexed = {
        (str(row["tier"]), str(row["projection"]), str(row["variant"])): row
        for row in rows
    }
    if len(indexed) != len(rows):
        raise RuntimeError("specialized kernel matrix contains duplicate rows")
    return indexed


def variant_for_tokens(tokens: int) -> str:
    if not isinstance(tokens, int) or tokens <= 0 or tokens > 8192:
        raise ValueError(f"tokens must be in [1, 8192], got {tokens!r}")
    if tokens in DECODE_VARIANTS:
        return DECODE_VARIANTS[tokens]
    if tokens == 2048:
        return "prefill_exact_2k"
    if tokens < 64:
        return "prefill_bm16"
    return "prefill_large"


def specialization_for(tier: str, projection: str, tokens: int) -> dict[str, Any]:
    variant = variant_for_tokens(tokens)
    try:
        return _rows()[(tier, projection, variant)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported specialized kernel row: {tier}/{projection}/{variant}"
        ) from exc


def required_warmup_tokens() -> tuple[int, ...]:
    return (1, 2, 4, 8, 16, 32, 64, 2048, 8192)
