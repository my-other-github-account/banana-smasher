"""Canonical full-16-branch exact Viterbi for public QTIP L16/K2/V2 solves.

This is the reusable public adaptation of the accepted prefix-DP producer.  It
retains one cost for each of 4,096 overlap prefixes and evaluates all 16 legal
predecessors in strict ascending order.  Separate stream-ordered Triton launches
make timestep ordering explicit; there is no proposal, pruning, approximation,
warm start, or fallback route.
"""
from __future__ import annotations

from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except Exception as exc:  # fail loudly; a silent slow path is forbidden
    raise RuntimeError(
        "public solve could not load the SHA-pinned full-16 exact QTIP2 "
        "producer; install the solve extra on a supported CUDA platform"
    ) from exc

PREFIXES = 4096
BRANCHES = 16
STATES = 65536
STEPS = 128
MAX_CHUNK = 1024
PREFIX_DP_NUM_WARPS = 8
_PRODUCTION_LINEAGE_SHA256 = (
    "379a24289514ead53de1415fdddc9cf77026d46c7b8d9ffef783fe5632a9319b"
)
_PARITY_REFERENCE_SHA256 = (
    "96b83c837a017c36f6630ac7b6b7a3be16888ea15a03593f3c4709b0675c3a50"
)


@triton.jit
def _init_prefix_costs(
    x_ptr,
    lut_ptr,
    overlap_ptr,
    scratch_ptr,
    best_state_ptr,
    B,
    HAS_OVERLAP: tl.constexpr,
):
    seq = tl.program_id(0)
    j = tl.arange(0, 4096)
    residue = j >> 4
    x0 = tl.load(x_ptr + seq).to(tl.float32)
    x1 = tl.load(x_ptr + B + seq).to(tl.float32)
    best = tl.full((4096,), float("inf"), tl.float32)
    chosen = tl.zeros((4096,), tl.int32)
    if HAS_OVERLAP:
        overlap = tl.load(overlap_ptr + seq).to(tl.int32)
        q = overlap // 256
        state = q * 4096 + j
        lut0 = tl.load(lut_ptr + state * 2).to(tl.float32)
        lut1 = tl.load(lut_ptr + state * 2 + 1).to(tl.float32)
        candidate = (lut0 - x0) * (lut0 - x0) + (lut1 - x1) * (lut1 - x1)
        valid = residue == (overlap & 255)
        best = tl.where(valid, candidate, best)
        chosen = state
    else:
        for q in range(16):
            state = q * 4096 + j
            lut0 = tl.load(lut_ptr + state * 2).to(tl.float32)
            lut1 = tl.load(lut_ptr + state * 2 + 1).to(tl.float32)
            candidate = (lut0 - x0) * (lut0 - x0) + (lut1 - x1) * (lut1 - x1)
            take = candidate < best
            best = tl.where(take, candidate, best)
            chosen = tl.where(take, state, chosen)
    base = seq * 4096
    tl.store(scratch_ptr + base + j, best)
    tl.store(best_state_ptr + base + j, chosen)


@triton.jit
def _advance_prefix_costs(
    x_ptr,
    lut_ptr,
    previous_ptr,
    current_ptr,
    best_state_ptr,
    B,
    step,
):
    seq = tl.program_id(0)
    j = tl.arange(0, 4096)
    residue = j >> 4
    x0 = tl.load(x_ptr + (step * 2) * B + seq).to(tl.float32)
    x1 = tl.load(x_ptr + (step * 2 + 1) * B + seq).to(tl.float32)
    best = tl.full((4096,), float("inf"), tl.float32)
    chosen = tl.zeros((4096,), tl.int32)
    previous_base = seq * 4096
    for q in range(16):
        predecessor_prefix = q * 256 + residue
        predecessor_cost = tl.load(
            previous_ptr + previous_base + predecessor_prefix
        )
        state = q * 4096 + j
        lut0 = tl.load(lut_ptr + state * 2).to(tl.float32)
        lut1 = tl.load(lut_ptr + state * 2 + 1).to(tl.float32)
        candidate = (
            predecessor_cost
            + (lut0 - x0) * (lut0 - x0)
            + (lut1 - x1) * (lut1 - x1)
        )
        take = candidate < best
        best = tl.where(take, candidate, best)
        chosen = tl.where(take, state, chosen)
    base = seq * 4096
    tl.store(current_ptr + base + j, best)
    tl.store(best_state_ptr + step * B * 4096 + base + j, chosen)


@triton.jit
def _backtrack(
    best_state_ptr,
    final_prefix_ptr,
    states_ptr,
    B,
    NSTEPS: tl.constexpr,
):
    seq = tl.program_id(0)
    prefix = tl.load(final_prefix_ptr + seq).to(tl.int32)
    for step in tl.static_range(NSTEPS - 1, -1, -1):
        state = tl.load(
            best_state_ptr + step * B * 4096 + seq * 4096 + prefix
        ).to(tl.int32)
        tl.store(states_ptr + step * B + seq, state)
        prefix = state >> 4


def geometry(cb: Any) -> dict[str, int | str | bool]:
    got = (int(cb.L), int(cb.K), int(cb.V))
    if got != (16, 2, 2):
        raise ValueError(f"trellis-v2 is sealed for L16/K2/V2, got {got}")
    return {
        "implementation": "p821-k2-triton-exact-prefix-dp-pingpong-p691-w8-v1",
        "production_lineage_sha256": _PRODUCTION_LINEAGE_SHA256,
        "parity_reference_sha256": _PARITY_REFERENCE_SHA256,
        "L": 16,
        "K": 2,
        "V": 2,
        "steps": STEPS,
        "full_states": STATES,
        "retained_prefix_costs": PREFIXES,
        "branches_per_prefix": BRANCHES,
        "full_branches_per_prefix": BRANCHES,
        "branch_sampling": "full",
        "strict_tie_order": "ascending-q-strict-less-than",
        "prefix_dp_num_warps": PREFIX_DP_NUM_WARPS,
        "chunk_sequences": MAX_CHUNK,
        "backpointer_dtype": "int32",
        "ordering": "separate-stream-ordered-kernel-launches",
        "production_default": True,
    }


def _exact_chunk(
    cb: Any, x: torch.Tensor, overlap: torch.Tensor | None
) -> torch.Tensor:
    batch = int(x.shape[1])
    steps = int(x.shape[0]) // 2
    lut = cb.lut.T.contiguous()
    scratch_a = torch.empty(
        (batch, PREFIXES), device=x.device, dtype=torch.float32
    )
    scratch_b = torch.empty_like(scratch_a)
    best_state = torch.empty(
        (steps, batch, PREFIXES), device=x.device, dtype=torch.int32
    )
    states = torch.empty((steps, batch), device=x.device, dtype=torch.int32)
    overlap_arg = (
        overlap
        if overlap is not None
        else torch.empty((1,), device=x.device, dtype=torch.int32)
    )
    _init_prefix_costs[(batch,)](
        x,
        lut,
        overlap_arg,
        scratch_a,
        best_state,
        B=batch,
        HAS_OVERLAP=overlap is not None,
        num_warps=PREFIX_DP_NUM_WARPS,
        num_stages=1,
    )
    previous, current = scratch_a, scratch_b
    for step in range(1, steps):
        _advance_prefix_costs[(batch,)](
            x,
            lut,
            previous,
            current,
            best_state,
            B=batch,
            step=step,
            num_warps=PREFIX_DP_NUM_WARPS,
            num_stages=1,
        )
        previous, current = current, previous
    final_prefix = (
        previous.argmin(dim=1).to(torch.int32)
        if overlap is None
        else overlap.to(torch.int32)
    )
    _backtrack[(batch,)](
        best_state,
        final_prefix,
        states,
        B=batch,
        NSTEPS=steps,
        num_warps=1,
        num_stages=1,
    )
    return states


def trellis_v2_exact(
    cb: Any,
    x: torch.Tensor,
    overlap: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return canonical all-16-branch states for CUDA input shaped ``[2*T,B]``."""
    geometry(cb)
    if (
        not x.is_cuda
        or x.ndim != 2
        or x.shape[0] % 2
        or x.shape[0] < 8
    ):
        raise ValueError(f"x must be CUDA [2*T,B] with T>=4, got {tuple(x.shape)} {x.device}")
    if x.dtype not in {torch.float16, torch.float32}:
        raise ValueError(
            f"public smash solve must produce float16/float32 trellis input; got {x.dtype}"
        )
    batch = int(x.shape[1])
    if batch < 1 or batch > 8192:
        raise ValueError(f"exact public QTIP2 path requires B in 1..8192, got {batch}")
    if not cb.lut.is_cuda or tuple(cb.lut.shape) != (2, STATES):
        raise ValueError(
            "canonical LUT must be CUDA [2,65536], got "
            f"{tuple(cb.lut.shape)} {cb.lut.device}"
        )
    if cb.lut.device != x.device:
        raise ValueError(
            f"canonical LUT device differs from x: {cb.lut.device} != {x.device}"
        )
    if overlap is not None:
        if (
            not overlap.is_cuda
            or overlap.device != x.device
            or overlap.ndim != 1
            or overlap.numel() != batch
            or overlap.dtype not in {torch.int32, torch.int64}
        ):
            raise ValueError("overlap must be integral CUDA [B] on the input device")
        if bool(((overlap < 0) | (overlap >= PREFIXES)).any()):
            raise ValueError(f"overlap prefixes must be in [0, {PREFIXES})")
    x = x.contiguous()
    outputs = []
    for start in range(0, batch, MAX_CHUNK):
        end = min(batch, start + MAX_CHUNK)
        overlap_arg = (
            None
            if overlap is None
            else overlap[start:end].to(dtype=torch.int32).contiguous()
        )
        outputs.append(
            _exact_chunk(cb, x[:, start:end].contiguous(), overlap_arg)
        )
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)
