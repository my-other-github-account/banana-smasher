"""Prebuilt CUDA extension boundary for the V5 specialized mixed-plane runtime."""
from __future__ import annotations

import functools
import importlib
from types import ModuleType
from typing import Any


_VARIANTS = (
    "decode_c1",
    "decode_c2",
    "decode_c4",
    "decode_c8",
    "decode_c16",
    "prefill_bm16",
    "prefill_large",
    "prefill_exact_2k",
)
_REQUIRED_QTIP_EXPORTS = tuple(
    f"qtip{family}_k{width}_{variant}"
    for family in (2, 3)
    for width in (4096, 2048)
    for variant in _VARIANTS
)
_REQUIRED_TORCH_OPERATORS = (
    "compact_routes",
    "qtip_pre_transform",
    "qtip_post_transform",
    "finalize_output",
    "d4_specialized",
    "mxfp4_specialized",
)


@functools.cache
def _module() -> ModuleType:
    try:
        return importlib.import_module("banana_smasher_plugin._v4_moe")
    except ImportError as exc:
        raise RuntimeError(
            "required banana-smasher specialized CUDA extension is unavailable"
        ) from exc


def preflight_native_extensions() -> None:
    """Load the one required platform extension and refuse incomplete images."""
    import torch

    module = _module()
    operators = torch.ops.banana_smasher_v4
    missing = [
        name for name in _REQUIRED_QTIP_EXPORTS if not callable(getattr(module, name, None))
    ]
    missing.extend(
        f"banana_smasher_v4::{name}"
        for name in _REQUIRED_TORCH_OPERATORS
        if not callable(getattr(operators, name, None))
    )
    if missing:
        raise RuntimeError(
            "specialized CUDA extension is missing required exports: "
            + ", ".join(missing)
        )


def specialized_qtip_gemv(
    transformed_x: Any,
    pointer_tables: dict[str, Any],
    codebook: Any,
    out: Any,
    compact: dict[str, Any],
    physical_counters: Any,
    *,
    family: int,
    specialization: dict[str, Any],
) -> Any:
    import torch

    module = _module()
    rows, width = transformed_x.shape
    if width not in (2048, 4096) or tuple(out.shape) != (rows, 4096):
        raise ValueError((transformed_x.shape, out.shape))
    if family not in (0, 1):
        raise ValueError(f"QTIP family must be 0 or 1, got {family}")
    symbol = str(specialization["source_symbol"])
    if (
        specialization["input_k"] != width
        or specialization["family"] != f"qtip{family + 2}"
    ):
        raise ValueError(f"QTIP specialization mismatch: {specialization}")
    x_half = compact["qtip_input"]
    torch.ops.banana_smasher_v4.qtip_pre_transform(
        transformed_x.to(torch.bfloat16).contiguous(),
        pointer_tables["su"],
        x_half,
        compact["family_block_counts"][family : family + 1],
        compact["block_experts"][family],
        compact["block_valid_m"][family],
        compact["block_route_rows"][family],
    )
    getattr(module, symbol)(
        out,
        pointer_tables["qtip_sources"],
        compact["family_block_counts"][family : family + 1],
        compact["block_experts"][family],
        compact["block_valid_m"][family],
        compact["block_route_rows"][family],
        x_half,
        codebook,
        physical_counters,
        int(specialization["counter"]["index"]),
    )
    torch.ops.banana_smasher_v4.qtip_post_transform(
        out,
        pointer_tables["wscale"],
        pointer_tables["sv"],
        compact["family_block_counts"][family : family + 1],
        compact["block_experts"][family],
        compact["block_valid_m"][family],
        compact["block_route_rows"][family],
    )
    return out


def specialized_mxfp4_gemm(
    x: Any,
    pointer_tables: dict[str, Any],
    out: Any,
    compact: dict[str, Any],
    physical_counters: Any,
    *,
    family: int,
    specialization: dict[str, Any],
) -> Any:
    _module()
    if family != 3:
        raise ValueError(f"MXFP4 family must be 3, got {family}")
    import torch

    return torch.ops.banana_smasher_v4.mxfp4_specialized(
        x.to(torch.bfloat16).contiguous(),
        pointer_tables["native_packed"],
        pointer_tables["native_scales"],
        out,
        compact["family_block_counts"][family : family + 1],
        compact["block_experts"][family],
        compact["block_valid_m"][family],
        compact["block_route_rows"][family],
        physical_counters,
        family,
        int(specialization["variant_id"]),
        int(specialization["counter"]["index"]),
    )


def specialized_d4_gemm(
    a: Any,
    out: Any,
    compact: dict[str, Any],
    pointer_tables: dict[str, Any],
    state: dict[str, Any],
    physical_counters: Any,
    *,
    n: int,
    k: int,
    family: int,
    projection: str,
    tokens: int,
) -> None:
    import torch

    _module()
    if family != 2:
        raise ValueError(f"D4 family must be 2, got {family}")
    from .specialized_variants import specialization_for

    rows = [
        specialization_for(tier, projection, tokens)
        for tier in ("d4_k1024", "d4_k2048", "d4_k4096")
    ]
    torch.ops.banana_smasher_v4.d4_specialized(
        a,
        out,
        compact["family_block_counts"][family : family + 1],
        compact["block_experts"][family],
        compact["block_valid_m"][family],
        compact["block_route_rows"][family],
        pointer_tables["d4_codes"],
        pointer_tables["d4_scales"],
        pointer_tables["d4_codebooks"],
        state["code_row_bytes"],
        state["dimension"],
        state["bits"],
        physical_counters,
        n,
        k,
        family,
        int(rows[0]["variant_id"]),
        int(rows[0]["counter"]["index"]),
        int(rows[1]["counter"]["index"]),
        int(rows[2]["counter"]["index"]),
    )
