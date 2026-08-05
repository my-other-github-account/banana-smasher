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
    "prefill_large_8192": 8,
}
FORBIDDEN_COUNTERS = {
    "mixed_exact_gemv": 24,
    "p1016_generic": 25,
    "triton_fallback": 26,
}


@functools.cache
def _rows() -> dict[tuple[str, str, str], dict[str, Any]]:
    document = json.loads(MATRIX_PATH.read_text())
    if document.get("schema") != "banana-smasher-specialized-kernel-matrix-v1":
        raise RuntimeError("specialized kernel matrix schema drift")
    rows = document.get("rows")
    if not isinstance(rows, list) or len(rows) != 108:
        raise RuntimeError("specialized kernel matrix must contain exactly 108 rows")
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
    if tokens == 8192:
        return "prefill_large_8192"
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


@functools.cache
def required_warmup_tokens() -> tuple[int, ...]:
    document = json.loads(MATRIX_PATH.read_text())
    geometries = document.get("warmup_geometries")
    tokens = geometries.get("tokens") if isinstance(geometries, dict) else None
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(not isinstance(token, int) or token <= 0 for token in tokens)
    ):
        raise RuntimeError("specialized kernel matrix warmup geometries are malformed")
    return tuple(tokens)


def physical_proof(counter_snapshots: list[Any]) -> dict[str, Any]:
    """Aggregate explicit runtime counter snapshots against the exact matrix."""
    document = json.loads(MATRIX_PATH.read_text())
    counter_size = int(document["counter_layout"]["size"])
    totals = [0] * counter_size
    for snapshot in counter_snapshots:
        values = snapshot.detach().cpu().tolist() if hasattr(snapshot, "detach") else list(snapshot)
        if len(values) < len(totals):
            raise ValueError(
                f"physical counter snapshot must contain at least {counter_size} values, got {len(values)}"
            )
        for index, value in enumerate(values[: len(totals)]):
            totals[index] += int(value)

    rows = []
    missing_rows = []
    for row in _rows().values():
        counter = row["counter"]
        count = totals[int(counter["index"])]
        receipt = {
            "tier": row["tier"],
            "projection": row["projection"],
            "variant": row["variant"],
            "counter_index": counter["index"],
            "counter_name": counter["name"],
            "count": count,
        }
        rows.append(receipt)
        if count <= 0:
            missing_rows.append(counter["name"])

    forbidden = {
        name: totals[index] for name, index in FORBIDDEN_COUNTERS.items()
    }
    return {
        "schema": "banana-smasher-specialized-physical-proof-v1",
        "status": (
            "PASS"
            if not missing_rows and all(value == 0 for value in forbidden.values())
            else "FAIL"
        ),
        "snapshot_count": len(counter_snapshots),
        "rows": rows,
        "missing_rows": missing_rows,
        "forbidden_counters": forbidden,
    }
