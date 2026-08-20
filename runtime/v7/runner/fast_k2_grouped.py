"""Grouped packed official-K2 projection primitives.

The CPU functions are an exact, differentiable reference for the CUDA path. The
production CUDA entry point is loaded lazily so claim-free CPU tests do not need
a CUDA toolchain.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

_MUL1 = 0x83DCD12D


def _inverse_permutation(device: torch.device) -> torch.Tensor:
    from banana_smasher.q2_codec import tensor_core_permutation

    permutation = torch.as_tensor(
        tensor_core_permutation(), device=device, dtype=torch.int64
    )
    return torch.argsort(permutation)


def _unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    if packed.ndim != 3 or packed.shape[-1] != 32 or packed.dtype != torch.int16:
        raise ValueError("packed must be int16[tiles_k, tiles_m, 32]")
    words = (packed.to(torch.int32) & 0xFFFF).reshape(*packed.shape[:-1], 16, 2)
    words = words[..., [1, 0]]
    shifts = torch.arange(14, -1, -2, device=packed.device, dtype=torch.int32)
    return ((words.unsqueeze(-1) >> shifts) & 3).reshape(*packed.shape[:-1], 256)


def direct_decode_matrix(packed: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Decode canonical packed K2 wire with no persistent state/dense cache."""
    if lut.numel() != 1024 or lut.ndim != 1:
        raise ValueError("lut must contain 1024 values")
    wire_lut = lut if lut.dtype == torch.float16 else lut.to(torch.float16)
    codes = _unpack_codes(packed)
    state = torch.zeros(codes.shape[:-1], device=packed.device, dtype=torch.int64)
    for position in range(248, 256):
        state = ((state << 2) | codes[..., position]) & 0xFFFF
    states = torch.empty_like(codes, dtype=torch.int64)
    for position in range(256):
        state = ((state << 2) | codes[..., position]) & 0xFFFF
        states[..., position] = state
    products = (states * _MUL1) & 0xFFFFFFFF
    parents = (
        (products & 0xFF)
        + ((products >> 8) & 0xFF)
        + ((products >> 16) & 0xFF)
        + ((products >> 24) & 0xFF)
    )
    decoded = wire_lut[parents].to(torch.float32)
    decoded = decoded[..., _inverse_permutation(packed.device)]
    tiles_k, tiles_m, _ = decoded.shape
    return (
        decoded.reshape(tiles_k, tiles_m, 16, 16)
        .permute(0, 2, 1, 3)
        .reshape(tiles_k * 16, tiles_m * 16)
    )


def block_hadamard_128(value: torch.Tensor) -> torch.Tensor:
    """Apply the exact normalized H128 independently to final-axis blocks."""
    if value.shape[-1] % 128:
        raise ValueError("last dimension must be divisible by 128")
    from banana_smasher.qtip_k2 import normalized_hadamard_128

    matrix = normalized_hadamard_128(value.device, value.dtype)
    return (value.reshape(-1, 128) @ matrix).reshape_as(value)


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
        module_name = path.name.split(".", 1)[0]
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
        grad_x, grad_lut = _cuda_extension().grouped_inner_backward(
            grad_output.contiguous(),
            x,
            offsets,
            work_experts,
            work_starts,
            active_experts,
            packed,
            lut_master,
        )
        return grad_x, None, None, None, None, None, grad_lut


_STATS = {
    "projection_calls": 0,
    "forward_kernel_calls": 0,
    "backward_kernel_calls": 0,
    "fallback_calls": 0,
    "reconstruction_calls": 0,
    "cpu_relay_bytes": 0,
}


def grouped_k2_stats() -> dict[str, int]:
    return dict(_STATS)


def reset_grouped_k2_stats() -> None:
    for name in _STATS:
        _STATS[name] = 0


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
    chunks = torch.div(counts + 15, 16, rounding_mode="floor")
    expert_ids = torch.arange(experts, device=x.device, dtype=torch.int32)
    work_experts = torch.repeat_interleave(expert_ids, chunks.to(torch.int64)).contiguous()
    chunk_prefix = chunks.cumsum(0, dtype=torch.int32) - chunks
    work_ordinal = torch.arange(
        work_experts.numel(), device=x.device, dtype=torch.int32
    )
    work_local = work_ordinal - chunk_prefix[work_experts.to(torch.int64)]
    work_starts = (
        offsets[work_experts.to(torch.int64)] + work_local * 16
    ).contiguous()
    active_experts = torch.nonzero(counts, as_tuple=False).flatten().to(torch.int32).contiguous()

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
    _STATS["forward_kernel_calls"] += 1
    if torch.is_grad_enabled() and (x.requires_grad or lut_master.requires_grad):
        _STATS["backward_kernel_calls"] += 1
    return sorted_output[inverse_order]


__all__ = [
    "block_hadamard_128",
    "direct_decode_matrix",
    "grouped_k2_stats",
    "grouped_packed_projection",
    "grouped_packed_projection_reference",
    "reset_grouped_k2_stats",
]
