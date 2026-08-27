"""Grouped packed official-K2 projection primitives.

The CPU functions are an exact, differentiable reference for the CUDA path. The
production CUDA entry point is loaded lazily so claim-free CPU tests do not need
a CUDA toolchain.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import torch

_MUL1 = 0x83DCD12D
WORK_ROWS = 64
_BACKWARD_STREAM_SYNC: Callable[[Any], None] | None = None


def bind_backward_stream_sync(sync: Callable[[Any], None]) -> None:
    """Bind the resident engine's canonical CUDA synchronization primitive."""
    if not callable(sync):
        raise TypeError("backward stream synchronization primitive must be callable")
    global _BACKWARD_STREAM_SYNC
    _BACKWARD_STREAM_SYNC = sync


def _inverse_permutation(device: torch.device) -> torch.Tensor:
    from banana_smasher.q2_codec import tensor_core_permutation

    permutation = torch.as_tensor(
        tensor_core_permutation(), device=device, dtype=torch.int64
    )
    return torch.argsort(permutation)


def _unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    if packed.ndim < 3 or packed.shape[-1] != 32 or packed.dtype != torch.int16:
        raise ValueError("packed must end in int16[tiles_k, tiles_m, 32]")
    words = (packed.to(torch.int32) & 0xFFFF).reshape(*packed.shape[:-1], 16, 2)
    words = words[..., [1, 0]]
    shifts = torch.arange(14, -1, -2, device=packed.device, dtype=torch.int32)
    return ((words.unsqueeze(-1) >> shifts) & 3).reshape(*packed.shape[:-1], 256)


def direct_decode_matrix(packed: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Decode canonical packed K2 wire with no persistent state/dense cache."""
    if packed.ndim < 3 or packed.shape[-1] != 32 or packed.dtype != torch.int16:
        raise ValueError("packed must end in [tiles_k, tiles_m, 32]")
    if lut.numel() != 1024 or lut.ndim != 1:
        raise ValueError("lut must contain 1024 values")
    wire_lut = lut if lut.dtype == torch.float16 else lut.to(torch.float16)
    codes = _unpack_codes(packed)
    # Each 16-bit state is exactly the circular window of eight 2-bit codes
    # ending at the current logical position. Build all 256 states in one
    # tensor expression instead of launching 256 sequential CUDA kernels.
    circular = torch.cat((codes[..., 249:], codes), dim=-1)
    windows = circular.unfold(-1, 8, 1).to(torch.int64)
    shifts = torch.arange(14, -1, -2, device=packed.device, dtype=torch.int64)
    states = (windows << shifts).sum(dim=-1) & 0xFFFF
    products = (states * _MUL1) & 0xFFFFFFFF
    parents = (
        (products & 0xFF)
        + ((products >> 8) & 0xFF)
        + ((products >> 16) & 0xFF)
        + ((products >> 24) & 0xFF)
    )
    decoded = wire_lut[parents].to(torch.float32)
    decoded = decoded[..., _inverse_permutation(packed.device)]
    leading = tuple(int(value) for value in decoded.shape[:-3])
    tiles_k, tiles_m = map(int, decoded.shape[-3:-1])
    tiled = decoded.reshape(*leading, tiles_k, tiles_m, 16, 16)
    lead = len(leading)
    order = list(range(lead)) + [lead, lead + 2, lead + 1, lead + 3]
    return tiled.permute(*order).reshape(*leading, tiles_k * 16, tiles_m * 16)


def sealed_bf16_weight_slab(
    packed: torch.Tensor,
    lut_master: torch.Tensor,
    su: torch.Tensor,
    sv: torch.Tensor,
    output_block: int,
) -> torch.Tensor:
    """Materialize one transient 128-column sealed BF16 weight slab."""
    if packed.ndim != 4 or packed.shape[-1] != 32:
        raise ValueError("packed must be [experts, tiles_k, tiles_m, 32]")
    experts, tiles_k, tiles_m, _ = map(int, packed.shape)
    k, m = tiles_k * 16, tiles_m * 16
    if k % 128 or m % 128 or su.shape != (experts, k) or sv.shape != (experts, m):
        raise ValueError("sealed BF16 slab requires 128-aligned grouped geometry")
    if output_block < 0 or output_block >= m // 128:
        raise ValueError("output_block outside grouped projection")
    first_tile = output_block * 8
    decoded = direct_decode_matrix(
        packed[:, :, first_tile : first_tile + 8, :], lut_master
    ).reshape(experts, k // 128, 128, 128)
    from banana_smasher.qtip_k2 import normalized_hadamard_128

    hadamard = normalized_hadamard_128(packed.device, torch.float32)
    # qtip_k2.inverse_transform applies SU between the two H128 transforms.
    # SU is input-channel scaling and therefore cannot commute across the
    # right-hand transform; doing so changes every reconstructed BF16 weight.
    transformed = torch.matmul(hadamard, decoded)
    transformed.mul_(su.float().reshape(experts, k // 128, 128, 1))
    transformed = torch.matmul(transformed, hadamard)
    transformed.mul_(
        sv[:, output_block * 128 : (output_block + 1) * 128]
        .float()
        .reshape(experts, 1, 1, 128)
    )
    return transformed.reshape(experts, k, 128).to(torch.bfloat16)


def sealed_bf16_full_weight(
    packed: torch.Tensor,
    lut_master: torch.Tensor,
    su: torch.Tensor,
    sv: torch.Tensor,
) -> torch.Tensor:
    """Materialize the exact full-width OfficialQtipK2PhysicalLayer weight."""
    if packed.ndim != 3 or packed.shape[-1] != 32:
        raise ValueError("packed must be [tiles_k, tiles_m, 32]")
    k, m = int(packed.shape[0]) * 16, int(packed.shape[1]) * 16
    if k % 128 or m % 128 or su.shape != (k,) or sv.shape != (m,):
        raise ValueError("sealed BF16 full weight requires 128-aligned geometry")
    decoded = direct_decode_matrix(packed, lut_master)
    from banana_smasher.qtip_k2 import normalized_hadamard_128

    hadamard = normalized_hadamard_128(packed.device, torch.float32)
    transformed = torch.matmul(
        hadamard, decoded.reshape(k // 128, 128, m)
    ).reshape(k, m)
    transformed.mul_(su.float().reshape(k, 1))
    transformed = torch.matmul(
        transformed.reshape(k, m // 128, 128), hadamard
    ).reshape(k, m)
    transformed.mul_(sv.float().reshape(1, m))
    # OfficialQtipK2PhysicalLayer stores inverse_transform(...).T as a
    # contiguous [M,K] BF16 weight and passes weight.T to matmul. Preserve that
    # physical transpose flag rather than returning an equivalent contiguous
    # [K,M] tensor: CUDA BF16 GEMM rounding can differ at this layout seam.
    official_weight = transformed.transpose(0, 1).contiguous().to(torch.bfloat16)
    return official_weight.transpose(0, 1)


def grouped_sealed_gate_up_projection(
    x: torch.Tensor,
    assignments: torch.Tensor,
    packed_gate: Any,
    packed_up: Any,
    lut_master: Any,
    su_gate: Any,
    sv_gate: Any,
    su_up: Any,
    sv_up: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the builder's single contiguous gate_up BF16 F.linear."""
    experts, _k, _m = _validate_grouped(
        x, assignments, packed_gate, lut_master, su_gate, sv_gate
    )
    _validate_grouped(x, assignments, packed_up, lut_master, su_up, sv_up)
    order = torch.argsort(assignments, stable=True)
    inverse_order = torch.argsort(order)
    sorted_assignments = assignments[order]
    counts = torch.bincount(sorted_assignments, minlength=experts).to(torch.int64)
    offsets = torch.cat(
        (torch.zeros(1, device=x.device, dtype=torch.int64), counts.cumsum(0))
    )
    projection_width = int(sv_gate.shape[1])
    sorted_gate_up = torch.empty(
        (x.shape[0], projection_width * 2), device=x.device, dtype=torch.bfloat16
    )
    for expert_idx in torch.nonzero(counts, as_tuple=False).flatten().tolist():
        start = int(offsets[expert_idx].item())
        stop = start + int(counts[expert_idx].item())
        # DeepseekV4Experts creates one expert-local routed allocation with
        # hidden_states[token_index] immediately before F.linear.  A slice of
        # one globally sorted routed slab has the same values and strides but a
        # different storage base/allocation geometry, which can select a
        # different CUDA BF16 GEMM implementation.  Preserve the expert-local
        # gather while retaining the existing stable-order output assembly.
        expert_x = x[assignments == expert_idx].to(torch.bfloat16).contiguous()
        gate_weight = sealed_bf16_full_weight(
            packed_gate[expert_idx], lut_master, su_gate[expert_idx], sv_gate[expert_idx]
        ).transpose(0, 1).contiguous()
        up_weight = sealed_bf16_full_weight(
            packed_up[expert_idx], lut_master, su_up[expert_idx], sv_up[expert_idx]
        ).transpose(0, 1).contiguous()
        gate_up_weight = torch.cat((gate_weight, up_weight), dim=0)
        # Preserve the builder's exact public operator boundary.  The sealed
        # control calls F.linear(routed, gate_up_weight), not an equivalent
        # explicit matmul over weight.T; CUDA may select a different BF16 GEMM
        # implementation at that dispatch seam.
        sorted_gate_up[start:stop] = torch.nn.functional.linear(
            expert_x, gate_up_weight
        )
    gate_up = sorted_gate_up[inverse_order]
    gate, up = gate_up.chunk(2, dim=-1)
    return gate, up


_FWHT_BACKEND = "current"
_FWHT_STATS = {"current_calls": 0, "quack_calls": 0, "fallback_calls": 0}


def set_fwht_backend(name: str) -> None:
    global _FWHT_BACKEND
    if name not in {"current", "quack"}:
        raise ValueError(f"unsupported FWHT backend: {name}")
    _FWHT_BACKEND = name


def fwht_backend_stats(*, reset: bool = False) -> dict[str, int]:
    value = dict(_FWHT_STATS)
    if reset:
        for key in _FWHT_STATS:
            _FWHT_STATS[key] = 0
    return value


def block_hadamard_128(value: torch.Tensor) -> torch.Tensor:
    """Apply exact normalized H128 blocks through the selected required backend."""
    if value.shape[-1] % 128:
        raise ValueError("last dimension must be divisible by 128")
    blocks = value.reshape(-1, 128).contiguous()
    if _FWHT_BACKEND == "quack":
        import math
        from quack.hadamard import hadamard_transform

        if not blocks.is_cuda or blocks.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise RuntimeError("required Quack FWHT needs contiguous CUDA fp16/bf16/fp32 blocks")
        _FWHT_STATS["quack_calls"] += 1
        return hadamard_transform(blocks, scale=1 / math.sqrt(128)).reshape_as(value)
    if _FWHT_BACKEND != "current":
        _FWHT_STATS["fallback_calls"] += 1
        raise RuntimeError(f"unknown required FWHT backend: {_FWHT_BACKEND}")
    from banana_smasher.qtip_k2 import normalized_hadamard_128

    _FWHT_STATS["current_calls"] += 1
    matrix = normalized_hadamard_128(value.device, value.dtype)
    return (blocks @ matrix).reshape_as(value)


def _validate_grouped(
    x: torch.Tensor,
    assignments: torch.Tensor,
    packed: torch.Tensor,
    lut_master: torch.Tensor,
    su: torch.Tensor,
    sv: torch.Tensor,
) -> tuple[int, int, int]:
    if x.ndim != 2 or assignments.ndim != 1 or assignments.numel() != x.shape[0]:
        raise ValueError("assignments must be one-dimensional with one entry per x row")
    if assignments.dtype != torch.int64:
        raise ValueError("assignments must use torch.int64")
    if packed.ndim != 4 or packed.shape[-1] != 32 or packed.dtype != torch.int16:
        raise ValueError("packed must be int16[experts, tiles_k, tiles_m, 32]")
    experts, tiles_k, tiles_m, _ = map(int, packed.shape)
    k, m = tiles_k * 16, tiles_m * 16
    if x.shape[1] != k or su.shape != (experts, k) or sv.shape != (experts, m):
        raise ValueError("grouped K2 projection geometry mismatch")
    if lut_master.shape != (1024,) or not lut_master.is_floating_point():
        raise ValueError("lut_master must be floating-point[1024]")
    if assignments.numel() and (
        int(assignments.min()) < 0 or int(assignments.max()) >= experts
    ):
        raise ValueError("assignments contain an invalid expert index")
    devices = {x.device, assignments.device, packed.device, lut_master.device, su.device, sv.device}
    if len(devices) != 1:
        raise ValueError("all grouped K2 tensors must share one device")
    return experts, k, m


def grouped_packed_projection_reference(
    x: torch.Tensor,
    assignments: torch.Tensor,
    packed: torch.Tensor,
    lut_master: torch.Tensor,
    su: torch.Tensor,
    sv: torch.Tensor,
) -> torch.Tensor:
    """Exact differentiable reference; production uses the grouped CUDA op."""
    _experts, _k, _m = _validate_grouped(x, assignments, packed, lut_master, su, sv)
    if x.shape[0] == 0:
        return x.new_empty((0, sv.shape[1]), dtype=torch.float32)
    x_inner = block_hadamard_128(x.float() * su[assignments].float())
    wire_lut = lut_master.to(torch.float16)
    rows = [
        row @ direct_decode_matrix(packed[int(expert)], wire_lut)
        for row, expert in zip(x_inner, assignments)
    ]
    inner = torch.stack(rows)
    return block_hadamard_128(inner) * sv[assignments].float()


@lru_cache(maxsize=1)
def _cuda_extension() -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("grouped packed K2 CUDA extension requires CUDA")
    prebuilt = os.environ.get("FAST_K2_EXTENSION")
    if prebuilt:
        path = Path(prebuilt).resolve()
        if not path.is_file():
            raise RuntimeError(f"FAST_K2_EXTENSION is not a file: {path}")
        expected = os.environ.get("FAST_K2_EXTENSION_SHA256")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected and observed != expected.lower():
            raise RuntimeError(
                f"FAST_K2_EXTENSION sha drift: {observed} != {expected.lower()}"
            )
        module_name = os.environ.get("FAST_K2_MODULE_NAME", "banana_smasher_fast_k2_grouped")
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parent
    sources = [root / "fast_k2_grouped.cpp", root / "fast_k2_grouped_kernel.cu"]
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in sources)).hexdigest()[:12]
    return load(
        name=f"banana_fast_k2_grouped_{digest}",
        sources=[str(path) for path in sources],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        with_cuda=True,
        verbose=bool(int(os.environ.get("FAST_K2_VERBOSE_BUILD", "0"))),
    )


class _GroupedInnerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        offsets: torch.Tensor,
        work_experts: torch.Tensor,
        work_starts: torch.Tensor,
        active_experts: torch.Tensor,
        packed: torch.Tensor,
        lut_master: torch.Tensor,
    ) -> torch.Tensor:
        extension = _cuda_extension()
        output = extension.grouped_inner_forward(
            x, offsets, work_experts, work_starts, packed, lut_master
        )
        _STATS["forward_kernel_calls"] += 1
        ctx.save_for_backward(
            x, offsets, work_experts, work_starts, active_experts, packed, lut_master
        )
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        (
            x,
            offsets,
            work_experts,
            work_starts,
            active_experts,
            packed,
            lut_master,
        ) = ctx.saved_tensors
        diagnostic_path = os.environ.get("FAST_K2_BACKWARD_DIAGNOSTIC")
        extension = _cuda_extension()
        args = (
            grad_output.contiguous(), x, offsets, work_experts, work_starts,
            active_experts, packed, lut_master,
        )
        if diagnostic_path:
            grad_x, grad_lut, maxima, first_nonfinite = (
                extension.grouped_inner_backward_diagnostic(*args)
            )
            _append_backward_diagnostic(
                Path(diagnostic_path), grad_output, x, grad_lut,
                maxima, first_nonfinite,
            )
        else:
            grad_x, grad_lut = extension.grouped_inner_backward(*args)
        if _BACKWARD_STREAM_SYNC is None:
            raise RuntimeError("backward stream synchronization primitive is not bound")
        _BACKWARD_STREAM_SYNC(torch)
        returned_path = os.environ.get("FAST_K2_RETURNED_GRAD_DIAGNOSTIC")
        if returned_path:
            _append_returned_grad_diagnostic(Path(returned_path), grad_lut)
        _STATS["backward_kernel_calls"] += 1
        return grad_x, None, None, None, None, None, grad_lut


_RETURNED_GRAD_DIAGNOSTIC_CALLS = 0


def _append_returned_grad_diagnostic(path: Path, grad_lut: torch.Tensor) -> None:
    """Record the exact tensor returned from the grouped autograd boundary."""
    global _RETURNED_GRAD_DIAGNOSTIC_CALLS
    _RETURNED_GRAD_DIAGNOSTIC_CALLS += 1
    finite = bool(torch.isfinite(grad_lut).all().item())
    maximum = float(grad_lut.detach().abs().max().item()) if grad_lut.numel() else 0.0
    row = {
        "schema": "banana-smasher-lut-accumulation-diagnostic-v1",
        "stage": "returned_grad_lut",
        "call": _RETURNED_GRAD_DIAGNOSTIC_CALLS,
        "finite": finite,
        "max_abs": maximum,
        "dtype": str(grad_lut.dtype),
        "shape": list(grad_lut.shape),
        "data_ptr": int(grad_lut.data_ptr()),
        "created_unix": time.time(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


_BACKWARD_DIAGNOSTIC_CALLS = 0
_BACKWARD_DIAGNOSTIC_STAGES = {
    0: "all_finite",
    1: "x_input",
    2: "grad_output_input",
    3: "x_times_grad_output_product",
    4: "row_fma_total",
    5: "atomic_prevalue",
    6: "atomic_postvalue",
}


def _append_backward_diagnostic(
    path: Path,
    grad_output: torch.Tensor,
    x: torch.Tensor,
    grad_lut: torch.Tensor,
    maxima: torch.Tensor,
    first_nonfinite: torch.Tensor,
) -> None:
    """Append one fsynced receipt row from the diagnostic-only CUDA tap."""
    global _BACKWARD_DIAGNOSTIC_CALLS
    _BACKWARD_DIAGNOSTIC_CALLS += 1
    values = maxima.detach().cpu().tolist()
    first = [int(value) for value in first_nonfinite.detach().cpu().tolist()]
    stage = 7 - first[0] if first[0] else 0
    nonfinite_flags = first[7:13]
    row = {
        "schema": "banana-smasher-fast-k2-backward-diagnostic-v1",
        "call": _BACKWARD_DIAGNOSTIC_CALLS,
        "created_unix": time.time(),
        "source_boundary": (
            "repair_api/assets/fast_k2_grouped_kernel.cu:184-229;"
            "repair_api/assets/fast_k2_grouped.py:_GroupedInnerFunction.backward"
        ),
        "finite": {
            "x_input": bool(torch.isfinite(x).all().item()),
            "grad_output_input": bool(torch.isfinite(grad_output).all().item()),
            "x_times_grad_output_product": not bool(nonfinite_flags[2]),
            "row_fma_total": not bool(nonfinite_flags[3]),
            "atomic_prevalue": not bool(nonfinite_flags[4]),
            "atomic_postvalue": not bool(nonfinite_flags[5]),
            "grad_lut_output": bool(torch.isfinite(grad_lut).all().item()),
        },
        "max_abs": dict(zip((
            "x_input", "grad_output_input", "x_times_grad_output_product",
            "row_fma_total", "atomic_prevalue", "atomic_postvalue",
        ), values)),
        "first_nonfinite": {
            "stage_code": stage,
            "stage": _BACKWARD_DIAGNOSTIC_STAGES.get(stage, "unknown"),
            "expert": first[1],
            "tile_k": first[2],
            "tile_m": first[3],
            "thread": first[4],
            "row": first[5],
            "parent": first[6],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


_STATS = {
    "projection_calls": 0,
    "forward_kernel_calls": 0,
    "backward_kernel_calls": 0,
    "fallback_calls": 0,
    "reconstruction_calls": 0,
    "cpu_relay_bytes": 0,
    "sort_calls": 0,
    "routed_rows": 0,
    "active_expert_observations": 0,
    "work_blocks": 0,
    "hadamard_calls": 0,
    "sealed_bf16_slab_calls": 0,
    "sealed_bf16_weight_tiles": 0,
}


def grouped_k2_stats() -> dict[str, int]:
    return {
        **_STATS,
        "quack_fwht_calls": int(_FWHT_STATS["quack_calls"]),
        "reference_fwht_calls": int(_FWHT_STATS["current_calls"]),
        "fwht_fallback_calls": int(_FWHT_STATS["fallback_calls"]),
    }


def reset_grouped_k2_stats() -> None:
    for name in _STATS:
        _STATS[name] = 0
    for name in _FWHT_STATS:
        _FWHT_STATS[name] = 0


def grouped_packed_projection(
    x: torch.Tensor,
    assignments: torch.Tensor,
    packed: torch.Tensor,
    lut_master: torch.Tensor,
    su: torch.Tensor,
    sv: torch.Tensor,
) -> torch.Tensor:
    """Execute one all-expert projection through the grouped CUDA operator."""
    experts, _k, _m = _validate_grouped(
        x, assignments, packed, lut_master, su, sv
    )
    if not x.is_cuda:
        raise ValueError("grouped_packed_projection requires CUDA tensors")
    if not all(value.is_contiguous() for value in (x, packed, lut_master, su, sv)):
        raise ValueError("grouped_packed_projection requires contiguous tensors")
    if x.shape[0] == 0:
        return x.new_empty((0, sv.shape[1]), dtype=torch.float32)

    if os.environ.get("FAST_K2_SEALED_FULL_WEIGHT_BF16", "0") == "1":
        order = torch.argsort(assignments, stable=True)
        inverse_order = torch.argsort(order)
        sorted_assignments = assignments[order]
        sorted_x = x[order].to(torch.bfloat16).contiguous()
        counts = torch.bincount(sorted_assignments, minlength=experts).to(torch.int64)
        active = torch.nonzero(counts, as_tuple=False).flatten()
        active_list = [int(value) for value in active.tolist()]
        offsets = torch.cat(
            (torch.zeros(1, device=x.device, dtype=torch.int64), counts.cumsum(0))
        )
        sorted_output = torch.empty(
            (x.shape[0], sv.shape[1]), device=x.device, dtype=torch.bfloat16
        )
        output_blocks = int(sv.shape[1]) // 128
        for expert_idx in active_list:
            start = int(offsets[expert_idx].item())
            stop = start + int(counts[expert_idx].item())
            # Match OfficialQtipK2PhysicalLayer._weight/_forward literally:
            # decode and inverse-transform the complete KxM matrix, round the
            # complete transformed weight to BF16, then issue one full-width
            # BF16 matmul for this expert. Splitting M into 128-column GEMMs is
            # not an identical physical binding even when each weight slab is.
            full_weight = sealed_bf16_full_weight(
                packed[expert_idx], lut_master, su[expert_idx], sv[expert_idx]
            )
            sorted_output[start:stop] = torch.matmul(
                sorted_x[start:stop], full_weight
            )
        _STATS["projection_calls"] += 1
        _STATS["sealed_bf16_slab_calls"] += output_blocks
        _STATS["sealed_bf16_weight_tiles"] += (
            len(active_list) * (int(x.shape[1]) // 128) * output_blocks
        )
        # The sealed OfficialQtipK2PhysicalLayer performs a BF16 GEMM and then
        # exposes the projection result as FP32 to the expert wrapper.
        return sorted_output[inverse_order].float()

    order = torch.argsort(assignments, stable=True)
    inverse_order = torch.argsort(order)
    sorted_assignments = assignments[order]
    sorted_x = x[order].contiguous()
    counts = torch.bincount(sorted_assignments, minlength=experts).to(torch.int32)
    offsets = torch.cat(
        (
            torch.zeros(1, device=x.device, dtype=torch.int32),
            counts.cumsum(0, dtype=torch.int32),
        )
    ).contiguous()
    chunks = torch.div(
        counts + WORK_ROWS - 1, WORK_ROWS, rounding_mode="floor"
    )
    expert_ids = torch.arange(experts, device=x.device, dtype=torch.int32)
    work_experts = torch.repeat_interleave(expert_ids, chunks.to(torch.int64)).contiguous()
    chunk_prefix = chunks.cumsum(0, dtype=torch.int32) - chunks
    work_ordinal = torch.arange(
        work_experts.numel(), device=x.device, dtype=torch.int32
    )
    work_local = work_ordinal - chunk_prefix[work_experts.to(torch.int64)]
    work_starts = (
        offsets[work_experts.to(torch.int64)] + work_local * WORK_ROWS
    ).contiguous()
    active_experts = torch.nonzero(counts, as_tuple=False).flatten().to(torch.int32).contiguous()
    _STATS["sort_calls"] += 1
    _STATS["routed_rows"] += int(x.shape[0])
    _STATS["active_expert_observations"] += int(active_experts.numel())
    _STATS["work_blocks"] += int(work_starts.numel())
    _STATS["hadamard_calls"] += 2

    x_inner = block_hadamard_128(
        sorted_x.float() * su[sorted_assignments].float()
    ).contiguous()
    inner = _GroupedInnerFunction.apply(
        x_inner,
        offsets,
        work_experts,
        work_starts,
        active_experts,
        packed,
        lut_master,
    )
    sorted_output = (
        block_hadamard_128(inner) * sv[sorted_assignments].float()
    )
    _STATS["projection_calls"] += 1
    return sorted_output[inverse_order]


__all__ = [
    "block_hadamard_128",
    "direct_decode_matrix",
    "grouped_k2_stats",
    "grouped_packed_projection",
    "grouped_packed_projection_reference",
    "grouped_sealed_gate_up_projection",
    "reset_grouped_k2_stats",
    "sealed_bf16_full_weight",
    "sealed_bf16_weight_slab",
]
