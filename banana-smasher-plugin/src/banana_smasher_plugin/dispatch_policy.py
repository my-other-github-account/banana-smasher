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
        kernel, chunk_tokens, chunks = "singleton_scalar", 1, 1
    elif tokens == 2:
        kernel, chunk_tokens, chunks = "small_m_pair", 2, 1
    elif tokens <= 4:
        kernel, chunk_tokens, chunks = "vector_m4", 4, 1
    elif tokens <= 8:
        kernel, chunk_tokens, chunks = "vector_m8_dealiased", 4, (tokens + 3) // 4
    elif tokens < PREFILL_TOKENS:
        kernel, chunk_tokens, chunks = "vector_m16_chunked", 4, (tokens + 3) // 4
    else:
        kernel, chunk_tokens, chunks = "dense_all_prefill", tokens, 1

    return {
        "kernel": kernel,
        "route_rows": route_rows,
        "tokens": tokens,
        "chunk_tokens": chunk_tokens,
        "chunks": chunks,
        "zero_dequant": True,
        "graph_reuse": True,
    }


def required_warmup_route_rows() -> tuple[int, ...]:
    """All decode and prompt-matrix shapes required before timed acceptance."""
    return (6, 12, 24, 48, 96, 192, 3072, 12000, 49152)
