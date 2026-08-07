"""Graph-stable state and physical dispatch for the mixed-QTIP runtime."""
from __future__ import annotations

from typing import Any

import torch


class IncompleteAccelerationPortError(RuntimeError):
    """The mandatory native route cannot be activated from admitted evidence."""


def _max_compact_blocks(*, rows: int, experts: int, block_rows: int) -> int:
    """Bound per-family blocks without launching one padded block per route row."""
    return min(
        rows,
        (rows + block_rows - 1) // block_rows + min(experts, rows) - 1,
    )


def allocate_compaction_state(
    *,
    rows: int,
    experts: int,
    input_width: int,
    output_width: int,
    block_rows: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Allocate one immutable descriptor set for a graph-captured route shape."""
    if rows <= 0 or experts <= 0 or input_width <= 0 or output_width <= 0:
        raise ValueError((rows, experts, input_width, output_width))
    if block_rows not in (1, 2, 4, 16):
        raise ValueError(f"unsupported compact block size: {block_rows}")
    max_blocks = _max_compact_blocks(
        rows=rows,
        experts=experts,
        block_rows=block_rows,
    )
    return {
        "out": torch.empty((rows, output_width), dtype=torch.float32, device=device),
        "result": torch.empty(
            (rows, output_width), dtype=torch.bfloat16, device=device
        ),
        "qtip_input": torch.empty(
            (rows, input_width), dtype=torch.float16, device=device
        ),
        "family_block_counts": torch.empty(7, dtype=torch.int32, device=device),
        "block_experts": torch.empty((7, max_blocks), dtype=torch.int32, device=device),
        "block_valid_m": torch.empty((7, max_blocks), dtype=torch.int32, device=device),
        "block_route_rows": torch.empty(
            (7, max_blocks, block_rows), dtype=torch.int32, device=device
        ),
        "expert_route_counts": torch.empty(experts, dtype=torch.int32, device=device),
        "expert_last_block": torch.empty(experts, dtype=torch.int32, device=device),
        # [0:24] aggregate compaction/family receipts, [24:27] forbidden routes,
        # [32:140] exhaustive tier x projection x shape physical counters.
        "physical_counters": torch.zeros(256, dtype=torch.int64, device=device),
    }


def build_device_resident_planes(
    states: list[dict[str, torch.Tensor]],
    families: torch.Tensor,
    d4_bits: list[int],
    *,
    input_width: int,
    output_width: int,
    device: torch.device,
) -> dict[str, Any]:
    """Build graph-stable device metadata without copying packed D4 planes."""
    code_row_bytes: list[int] = []
    dimensions: list[int] = []
    bits_values: list[int] = []
    kind: list[int] = []
    family_values = families.tolist()
    for expert, (state, family) in enumerate(
        zip(states, family_values, strict=True)
    ):
        if family == 2:
            bits = int(d4_bits[expert])
            if input_width in (2048, 4096) and bits not in (10, 11, 12):
                raise ValueError(f"unsupported F521 D4 index width: {bits}")
            row_bytes = int(state["codes"].shape[-1])
            code_row_bytes.append(row_bytes)
            dimensions.append(4)
            bits_values.append(bits)
            kind.append(0)
        else:
            code_row_bytes.append(max(1, input_width // 8))
            dimensions.append(4)
            bits_values.append(8)
            kind.append(1)
    resident: dict[str, Any] = {
        "code_row_bytes": torch.tensor(
            code_row_bytes, dtype=torch.int32, device=device
        ),
        "dimension": torch.tensor(dimensions, dtype=torch.uint8, device=device),
        "bits": torch.tensor(bits_values, dtype=torch.uint8, device=device),
        "kind": torch.tensor(kind, dtype=torch.int32, device=device),
        "input_width": torch.tensor(input_width, dtype=torch.int32, device=device),
        "output_width": torch.tensor(output_width, dtype=torch.int32, device=device),
        "compaction": {},
        "physical_counter_tensors": {},
    }
    for rows, block_rows in (
        (6, 1),
        (12, 2),
        (24, 4),
        (48, 4),
        (96, 4),
    ):
        resident["compaction"][(rows, block_rows)] = allocate_compaction_state(
            rows=rows,
            experts=len(states),
            input_width=input_width,
            output_width=output_width,
            block_rows=block_rows,
            device=device,
        )
    return resident


def mixed_exact_native_gemv(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    family_codes: torch.Tensor,
    pointer_tables: dict[str, torch.Tensor],
    qtip_codebook: torch.Tensor,
    vq_state: dict[str, Any],
    *,
    projection: str,
) -> torch.Tensor:
    """Compact on-device and launch each physical packed family independently."""
    from .native_extensions import (
        specialized_d4_gemm,
        specialized_mxfp4_gemm,
        specialized_native_v4_gemv,
        specialized_qtip_gemv,
    )
    from .specialized_variants import specialization_for

    rows, width = x.shape
    if rows % 6:
        raise ValueError(f"route rows must be divisible by top_k=6, got {rows}")
    tokens = rows // 6
    qtip2_row = specialization_for("qtip2_2.0117", projection, tokens)
    qtip3_row = specialization_for("qtip3_3.0117", projection, tokens)
    native_row = specialization_for("native_mxfp4", projection, tokens)
    block_rows = int(qtip2_row["tile_m"])
    key = (rows, block_rows)
    compaction = vq_state["compaction"]
    compact = compaction.get(key)
    if compact is None:
        compact = allocate_compaction_state(
            rows=rows,
            experts=family_codes.numel(),
            input_width=width,
            output_width=int(vq_state["output_width"]),
            block_rows=block_rows,
            device=x.device,
        )
        if bool(qtip2_row["graph_replay"]):
            compaction[key] = compact
    out = compact["out"]
    physical_counters = compact.get("physical_counters")
    vq_state["physical_counter_tensors"][key] = physical_counters
    torch.ops.banana_smasher_v4.compact_routes(
        expert_ids,
        family_codes,
        out,
        compact["family_block_counts"],
        compact["block_experts"],
        compact["block_valid_m"],
        compact["block_route_rows"],
        compact["expert_route_counts"],
        compact["expert_last_block"],
        physical_counters,
        block_rows,
    )
    specialized_qtip_gemv(
        x,
        pointer_tables,
        qtip_codebook,
        out,
        compact,
        compact["physical_counters"],
        family=0,
        specialization=qtip2_row,
    )
    specialized_qtip_gemv(
        x,
        pointer_tables,
        qtip_codebook,
        out,
        compact,
        compact["physical_counters"],
        family=1,
        specialization=qtip3_row,
    )
    specialized_d4_gemm(
        x.to(torch.bfloat16).contiguous(),
        out,
        compact,
        pointer_tables,
        vq_state,
        compact["physical_counters"],
        n=out.shape[1],
        k=width,
        family=2,
        projection=projection,
        tokens=tokens,
    )
    specialized_mxfp4_gemm(
        x,
        pointer_tables,
        out,
        compact,
        compact["physical_counters"],
        family=3,
        specialization=native_row,
    )
    specialized_native_v4_gemv(
        x,
        pointer_tables,
        qtip_codebook,
        out,
        compact,
        compact["physical_counters"],
    )
    return torch.ops.banana_smasher_v4.finalize_output(
        out,
        expert_ids,
        family_codes.numel(),
        compact["result"],
    )


def physical_counter_tensor(vq_state: dict[str, Any], route_rows: int) -> torch.Tensor:
    """Return the immutable on-device counter tensor for one warmed route shape."""
    from .dispatch_policy import shape_policy

    policy = shape_policy(route_rows)
    policy_mblock = int(policy["mblock"])
    block_rows = policy_mblock if policy_mblock == 16 else int(policy["valid_m"])
    try:
        return vq_state["physical_counter_tensors"][(route_rows, block_rows)]
    except KeyError as exc:
        raise ValueError(f"route shape {route_rows} has not executed") from exc


def native_v4_physical_receipt(
    vq_state: dict[str, Any], route_rows: int
) -> dict[str, Any]:
    """Read the packed B7/B9/B10 physical counters for one warmed shape."""

    values = physical_counter_tensor(vq_state, route_rows).detach().cpu().tolist()
    rates = {}
    for offset, bits in enumerate((7, 9, 10)):
        base = 140 + offset * 4
        rates[f"B{bits}"] = {
            "calls": int(values[base]),
            "rows": int(values[base + 1]),
            "code_bytes": int(values[base + 2]),
        }
    return {
        "rates": rates,
        "per_forward_dequantizations": int(values[152]),
        "forbidden_fallbacks": int(sum(values[24:27])),
    }


def runtime_sentinel() -> dict[str, Any]:
    """Report the graph-stable physical family boundary."""
    return {
        "backend": "_v4_moe",
        "activated": True,
        "zero_dequant": True,
        "graph_reuse": True,
        "family_counters": (
            "qtip2",
            "qtip3",
            "d4",
            "native_mxfp4",
            "qtip_native_v4_b7",
            "qtip_native_v4_b9",
            "qtip_native_v4_b10",
        ),
        "physical_launches": (10, 14),
        "physical_blocks": (14, 18),
        "physical_rows": (18, 22),
        "blocked": [],
    }
