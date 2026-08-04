"""Graph-stable state and physical dispatch for the mixed-QTIP runtime."""
from __future__ import annotations

from typing import Any

import torch


class IncompleteAccelerationPortError(RuntimeError):
    """The mandatory native route cannot be activated from admitted evidence."""


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
    max_blocks = rows
    return {
        "out": torch.empty((rows, output_width), dtype=torch.float32, device=device),
        "result": torch.empty(
            (rows, output_width), dtype=torch.bfloat16, device=device
        ),
        "qtip_input": torch.empty(
            (rows, input_width), dtype=torch.float16, device=device
        ),
        "family_block_counts": torch.empty(4, dtype=torch.int32, device=device),
        "block_experts": torch.empty((4, max_blocks), dtype=torch.int32, device=device),
        "block_valid_m": torch.empty((4, max_blocks), dtype=torch.int32, device=device),
        "block_route_rows": torch.empty(
            (4, max_blocks, block_rows), dtype=torch.int32, device=device
        ),
        "expert_route_counts": torch.empty(experts, dtype=torch.int32, device=device),
        "expert_last_block": torch.empty(experts, dtype=torch.int32, device=device),
        # [0:4] descriptor blocks, [4:8] routed rows, [8] block size,
        # [10:14] physical launches, [14:18] physical descriptor blocks,
        # [18:22] physical rows, [22] compactions; [9, 23] are reserved.
        "physical_counters": torch.zeros(24, dtype=torch.int64, device=device),
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
    """Consolidate admitted packed D4 planes into stable CUDA-owned storage."""
    codes: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    codebooks: list[torch.Tensor] = []
    code_offset: list[int] = []
    scale_offset: list[int] = []
    cb_offset: list[int] = []
    code_row_bytes: list[int] = []
    dimensions: list[int] = []
    bits_values: list[int] = []
    kind: list[int] = []
    code_cursor = scale_cursor = cb_cursor = 0
    family_values = families.tolist()
    for expert, (state, family) in enumerate(
        zip(states, family_values, strict=True)
    ):
        code_offset.append(code_cursor)
        scale_offset.append(scale_cursor)
        cb_offset.append(cb_cursor)
        if family == 2:
            code = state["codes"].contiguous().reshape(-1)
            scale = state["scales"].contiguous().reshape(-1)
            codebook = state["codebook"].to(torch.float16).contiguous().reshape(-1)
            bits = int(d4_bits[expert])
            if input_width in (2048, 4096) and bits not in (10, 11, 12):
                raise ValueError(f"unsupported F521 D4 index width: {bits}")
            row_bytes = int(state["codes"].shape[-1])
            codes.append(code)
            scales.append(scale)
            codebooks.append(codebook)
            code_cursor += code.numel()
            scale_cursor += scale.numel()
            cb_cursor += codebook.numel()
            code_row_bytes.append(row_bytes)
            dimensions.append(4)
            bits_values.append(bits)
            kind.append(0)
        else:
            code_row_bytes.append(max(1, input_width // 8))
            dimensions.append(4)
            bits_values.append(8)
            kind.append(1)
    if not codes:
        codes = [torch.zeros(1, dtype=torch.uint8, device=device)]
        scales = [torch.zeros(1, dtype=torch.uint8, device=device)]
        codebooks = [torch.zeros(4, dtype=torch.float16, device=device)]
    resident: dict[str, Any] = {
        "codes": torch.cat(codes).contiguous(),
        "scales": torch.cat(scales).contiguous(),
        "codebooks": torch.cat(codebooks).contiguous(),
        "code_offset": torch.tensor(code_offset, dtype=torch.int64, device=device),
        "scale_offset": torch.tensor(scale_offset, dtype=torch.int64, device=device),
        "code_row_bytes": torch.tensor(
            code_row_bytes, dtype=torch.int32, device=device
        ),
        "dimension": torch.tensor(dimensions, dtype=torch.uint8, device=device),
        "bits": torch.tensor(bits_values, dtype=torch.uint8, device=device),
        "cb_offset": torch.tensor(cb_offset, dtype=torch.int64, device=device),
        "kind": torch.tensor(kind, dtype=torch.int32, device=device),
        "input_width": torch.tensor(input_width, dtype=torch.int32, device=device),
        "output_width": torch.tensor(output_width, dtype=torch.int32, device=device),
        "compaction": {},
    }
    for rows, block_rows in ((6, 1), (12, 2), (24, 4), (48, 4), (96, 4)):
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
) -> torch.Tensor:
    """Compact on-device and launch each physical packed family independently."""
    from .dispatch_policy import shape_policy
    from .native_extensions import dynamic_qtip_gemv, native_mxfp4_gemv, vq_gemm

    rows, width = x.shape
    policy = shape_policy(rows)
    policy_mblock = int(policy["mblock"])
    block_rows = policy_mblock if policy_mblock == 16 else int(policy["valid_m"])
    key = (rows, block_rows)
    compaction = vq_state["compaction"]
    if key not in compaction:
        compaction[key] = allocate_compaction_state(
            rows=rows,
            experts=family_codes.numel(),
            input_width=width,
            output_width=int(vq_state["output_width"]),
            block_rows=block_rows,
            device=x.device,
        )
    compact = compaction[key]
    out = compact["out"]
    physical_counters = compact.get("physical_counters")
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
    dynamic_qtip_gemv(
        x,
        pointer_tables,
        qtip_codebook,
        out,
        compact,
        compact["physical_counters"],
        family=0,
    )
    dynamic_qtip_gemv(
        x,
        pointer_tables,
        qtip_codebook,
        out,
        compact,
        compact["physical_counters"],
        family=1,
    )
    vq_gemm(
        x.to(torch.bfloat16),
        out,
        compact,
        vq_state,
        compact["physical_counters"],
        n=out.shape[1],
        k=width,
        family=2,
    )
    native_mxfp4_gemv(
        x,
        pointer_tables,
        out,
        compact,
        compact["physical_counters"],
        family=3,
    )
    return torch.ops.banana_smasher_v4.finalize_output(out, compact["result"])


def physical_counter_tensor(vq_state: dict[str, Any], route_rows: int) -> torch.Tensor:
    """Return the immutable on-device counter tensor for one warmed route shape."""
    from .dispatch_policy import shape_policy

    policy = shape_policy(route_rows)
    policy_mblock = int(policy["mblock"])
    block_rows = policy_mblock if policy_mblock == 16 else int(policy["valid_m"])
    try:
        return vq_state["compaction"][(route_rows, block_rows)]["physical_counters"]
    except KeyError as exc:
        raise ValueError(f"route shape {route_rows} has not executed") from exc


def runtime_sentinel() -> dict[str, Any]:
    """Report the graph-stable physical family boundary."""
    return {
        "backend": "_v4_moe",
        "activated": True,
        "zero_dequant": True,
        "graph_reuse": True,
        "family_counters": ("qtip2", "qtip3", "d4", "native_mxfp4"),
        "physical_launches": (10, 14),
        "physical_blocks": (14, 18),
        "physical_rows": (18, 22),
        "blocked": [],
    }
