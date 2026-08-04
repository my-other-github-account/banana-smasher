"""Prebuilt CUDA extension boundary for the V4 mixed-plane runtime."""
from __future__ import annotations

import functools
import importlib
from types import ModuleType
from typing import Any


@functools.cache
def _module() -> ModuleType:
    try:
        return importlib.import_module("banana_smasher_plugin._v4_moe")
    except ImportError as exc:
        raise RuntimeError(
            "required banana-smasher mixed-QTIP CUDA extension is unavailable"
        ) from exc


def preflight_native_extensions() -> None:
    """Load the one required platform extension and refuse incomplete images."""
    _module()


def dynamic_qtip_gemv(
    transformed_x: Any,
    pointer_tables: dict[str, Any],
    codebook: Any,
    out: Any,
    compact: dict[str, Any],
    physical_counters: Any,
    *,
    family: int,
) -> Any:
    import torch

    module = _module()
    rows, width = transformed_x.shape
    if width not in (2048, 4096) or tuple(out.shape) != (rows, 4096):
        raise ValueError((transformed_x.shape, out.shape))
    if family not in (0, 1):
        raise ValueError(f"QTIP family must be 0 or 1, got {family}")
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
    getattr(module, f"decompress_matvec_compact_{family + 2}_{width}")(
        out,
        pointer_tables["qtip_sources"],
        compact["family_block_counts"][family : family + 1],
        compact["block_experts"][family],
        compact["block_valid_m"][family],
        compact["block_route_rows"][family],
        x_half,
        codebook,
        physical_counters,
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


def native_mxfp4_gemv(
    x: Any,
    pointer_tables: dict[str, Any],
    out: Any,
    compact: dict[str, Any],
    physical_counters: Any,
    *,
    family: int,
) -> Any:
    _module()
    if family != 3:
        raise ValueError(f"MXFP4 family must be 3, got {family}")
    import torch

    return torch.ops.banana_smasher_v4.mxfp4_compact(
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
    )


def vq_gemm(
    a: Any,
    out: Any,
    compact: dict[str, Any],
    state: dict[str, Any],
    physical_counters: Any,
    *,
    n: int,
    k: int,
    family: int,
) -> None:
    import torch

    _module()
    if family != 2:
        raise ValueError(f"D4 family must be 2, got {family}")
    torch.ops.banana_smasher_v4.vq_compact(
        a,
        out,
        compact["family_block_counts"][family : family + 1],
        compact["block_experts"][family],
        compact["block_valid_m"][family],
        compact["block_route_rows"][family],
        state["codes"],
        state["scales"],
        state["codebooks"],
        state["code_offset"],
        state["scale_offset"],
        state["code_row_bytes"],
        state["dimension"],
        state["bits"],
        state["cb_offset"],
        physical_counters,
        n,
        k,
        family,
    )
