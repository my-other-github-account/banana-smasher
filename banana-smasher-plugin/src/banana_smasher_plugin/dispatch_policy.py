from __future__ import annotations

from typing import Final


TOP_K: Final = 6
PREFILL_TOKENS: Final = 32


def shape_policy(route_rows: int) -> dict[str, int | str | bool]:
    """Return the immutable V4-equivalent kernel policy for one routed projection.

    Decode uses graph-stable token shapes. Eight- and sixteen-token batches are
    deliberately expressed as independent four-token chunks so an eight-row
    allocator block can never alias two live request groups. Prefill starts at
    the documented 32-token warm shape and keeps packed weights resident.
    """
    if not isinstance(route_rows, int) or route_rows <= 0:
        raise ValueError(f"route_rows must be a positive integer, got {route_rows!r}")
    if route_rows % TOP_K:
        raise ValueError(
            f"route_rows={route_rows} is not divisible by the required top_k={TOP_K}"
        )
    tokens = route_rows // TOP_K
    if tokens == 1:
        kernel, chunk_tokens, chunks, valid_m, mblock = "decode_c1", 1, 1, 1, 1
        physical_symbol = "specialized_kernel_matrix.decode_c1"
    elif tokens == 2:
        kernel, chunk_tokens, chunks, valid_m, mblock = "decode_c2", 2, 1, 2, 2
        physical_symbol = "specialized_kernel_matrix.decode_c2"
    elif tokens <= 4:
        kernel, chunk_tokens, chunks, valid_m, mblock = "decode_c4", 4, 1, 4, 4
        physical_symbol = "specialized_kernel_matrix.decode_c4"
    elif tokens <= 8:
        kernel, chunk_tokens, chunks, valid_m, mblock = (
            "decode_c8", 4, (tokens + 3) // 4, 4, 4
        )
        physical_symbol = "specialized_kernel_matrix.decode_c8"
    elif tokens <= 16:
        kernel, chunk_tokens, chunks, valid_m, mblock = (
            "decode_c16", 4, (tokens + 3) // 4, 4, 4
        )
        physical_symbol = "specialized_kernel_matrix.decode_c16"
    elif tokens <= 8192:
        kernel, chunk_tokens, chunks, valid_m, mblock = (
            ("prefill_exact_2k" if tokens == 2048 else
             ("prefill_bm16" if tokens < 64 else "prefill_large")),
            16, (tokens + 15) // 16, 16, 16
        )
        physical_symbol = f"specialized_kernel_matrix.{kernel}"
    else:
        raise NotImplementedError(
            "direct-packed prefill supports at most 8192 tokens"
        )

    return {
        "kernel": kernel,
        "physical_symbol": physical_symbol,
        "route_rows": route_rows,
        "tokens": tokens,
        "chunk_tokens": chunk_tokens,
        "chunks": chunks,
        "valid_m": valid_m,
        "mblock": mblock,
        "zero_dequant": True,
        "graph_reuse": tokens in {1, 2, 4, 8, 16},
        "activation": "active",
    }


def required_warmup_route_rows() -> tuple[int, ...]:
    """All decode and prompt-matrix shapes required before timed acceptance."""
    return (6, 12, 24, 48, 96, 192, 378, 384, 390, 3072, 12000, 12288, 49152)
