"""Manifest-geometry prefix-compressed Triton Viterbi for QTIP rings.

The production kernel assigns one independent sequence to each CTA and keeps all
128 dynamic-programming steps in one launch. The independent parity oracle lives
under ``tests/`` and is not shipped as a fallback.
"""
from __future__ import annotations

from collections.abc import Mapping
import types
from typing import TYPE_CHECKING, Any

import torch

from .qtip_rings import (
    PERSISTENT_BACKENDS,
    PERSISTENT_V32_BACKEND,
    backend_for_geometry,
    effective_cuda_free_bytes,
    plan_qtip_streaming_batches,
    qtip_peak_allocation_bytes,
    require_qtip_memory_capacity,
)

_TRITON_IMPORT_ERROR: ModuleNotFoundError | None = None
if TYPE_CHECKING:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]
else:
    try:
        import triton
        import triton.language as tl
    except ModuleNotFoundError as exc:  # CPU/package inspection stays importable.
        if exc.name not in {"triton", "triton.language"}:
            raise

        class _TritonUnavailable:
            @staticmethod
            def jit(function):
                return function

        triton = _TritonUnavailable()
        tl = types.SimpleNamespace()
        _TRITON_IMPORT_ERROR = exc
    else:
        _TRITON_IMPORT_ERROR = None


def _require_triton() -> None:
    if _TRITON_IMPORT_ERROR is not None:
        raise RuntimeError(
            "this exact GPU path requires the solve extra on a supported platform"
        ) from _TRITON_IMPORT_ERROR


def _validate_overlap_prefixes(
    overlap: torch.Tensor,
    *,
    x: torch.Tensor,
    batch: int,
    prefixes: int,
) -> None:
    if (
        not overlap.is_cuda
        or overlap.device != x.device
        or overlap.ndim != 1
        or overlap.numel() != batch
        or overlap.dtype not in {torch.int32, torch.int64}
    ):
        raise ValueError(
            "overlap must be an integral CUDA tensor on the input device "
            "with one prefix per sequence"
        )
    if bool(((overlap < 0) | (overlap >= prefixes)).any()):
        raise ValueError(f"overlap prefixes must be in [0, {prefixes})")


@triton.jit
def _persistent_prefix_viterbi(
    x_ptr,
    lut_ptr,
    overlap_ptr,
    scratch_ptr,
    best_state_ptr,
    states_ptr,
    B,
    HAS_OVERLAP: tl.constexpr,
):
    """Solve one independent sequence per CTA with all timesteps resident.

    The two 1,024-entry cost rows ping-pong in task-local global scratch.  A CTA
    barrier replaces the 127 host launches while preserving the original q=0..63
    strict-< update order and exact int32 backpointer table.
    """
    seq = tl.program_id(0)
    j = tl.arange(0, 1024)
    residue4 = j >> 6
    x0 = tl.load(x_ptr + seq).to(tl.float32)
    x1 = tl.load(x_ptr + B + seq).to(tl.float32)
    best = tl.full((1024,), float("inf"), tl.float32)
    chosen = tl.zeros((1024,), tl.int32)

    if HAS_OVERLAP:
        overlap = tl.load(overlap_ptr + seq).to(tl.int32)
        q = overlap >> 4
        state = q * 1024 + j
        lut0 = tl.load(lut_ptr + state).to(tl.float32)
        lut1 = tl.load(lut_ptr + 65536 + state).to(tl.float32)
        candidate = (lut0 - x0) * (lut0 - x0) + (lut1 - x1) * (lut1 - x1)
        valid = residue4 == (overlap & 15)
        best = tl.where(valid, candidate, best)
        chosen = state
    else:
        for q in range(64):
            state = q * 1024 + j
            lut0 = tl.load(lut_ptr + state).to(tl.float32)
            lut1 = tl.load(lut_ptr + 65536 + state).to(tl.float32)
            candidate = (lut0 - x0) * (lut0 - x0) + (lut1 - x1) * (lut1 - x1)
            take = candidate < best
            best = tl.where(take, candidate, best)
            chosen = tl.where(take, state, chosen)

    base = seq * 1024
    tl.store(scratch_ptr + base + j, best)
    tl.store(best_state_ptr + base + j, chosen)
    tl.debug_barrier()

    step = 1
    while step < 128:
        previous_base = ((step - 1) & 1) * B * 1024 + base
        current_base = (step & 1) * B * 1024 + base
        x0 = tl.load(x_ptr + (step * 2) * B + seq).to(tl.float32)
        x1 = tl.load(x_ptr + (step * 2 + 1) * B + seq).to(tl.float32)
        best = tl.full((1024,), float("inf"), tl.float32)
        chosen = tl.zeros((1024,), tl.int32)
        for q in range(64):
            predecessor_prefix = q * 16 + residue4
            predecessor_cost = tl.load(scratch_ptr + previous_base + predecessor_prefix)
            state = q * 1024 + j
            lut0 = tl.load(lut_ptr + state).to(tl.float32)
            lut1 = tl.load(lut_ptr + 65536 + state).to(tl.float32)
            candidate = predecessor_cost + (lut0 - x0) * (lut0 - x0) + (lut1 - x1) * (lut1 - x1)
            take = candidate < best
            best = tl.where(take, candidate, best)
            chosen = tl.where(take, state, chosen)
        tl.store(scratch_ptr + current_base + j, best)
        tl.store(best_state_ptr + step * B * 1024 + base + j, chosen)
        tl.debug_barrier()
        step += 1

    if HAS_OVERLAP:
        prefix = tl.load(overlap_ptr + seq).to(tl.int32)
    else:
        prefix = tl.argmin(best, axis=0).to(tl.int32)
    for back_step in tl.static_range(127, -1, -1):
        state = tl.load(best_state_ptr + back_step * B * 1024 + base + prefix).to(tl.int32)
        tl.store(states_ptr + back_step * B + seq, state)
        prefix = state >> 6


@triton.jit
def _persistent_prefix_viterbi_generic(
    x_ptr,
    lut_ptr,
    overlap_ptr,
    scratch_ptr,
    best_state_ptr,
    states_ptr,
    B,
    STATES: tl.constexpr,
    PREFIXES: tl.constexpr,
    BRANCHES: tl.constexpr,
    SHIFT: tl.constexpr,
    Q_FACTOR: tl.constexpr,
    V: tl.constexpr,
    STEPS: tl.constexpr,
    HAS_OVERLAP: tl.constexpr,
):
    """One exact persistent program per sequence, specialized by AOT geometry."""
    seq = tl.program_id(0)
    j = tl.arange(0, PREFIXES)
    residue = j >> SHIFT
    best = tl.full((PREFIXES,), float("inf"), tl.float32)
    chosen = tl.zeros((PREFIXES,), tl.int32)

    if HAS_OVERLAP:
        overlap = tl.load(overlap_ptr + seq).to(tl.int32)
        q = overlap // Q_FACTOR
        state = q * PREFIXES + j
        candidate = tl.zeros((PREFIXES,), tl.float32)
        for lane in tl.static_range(0, V):
            xv = tl.load(x_ptr + lane * B + seq).to(tl.float32)
            lv = tl.load(lut_ptr + lane * STATES + state).to(tl.float32)
            candidate += (lv - xv) * (lv - xv)
        valid = residue == (overlap & (Q_FACTOR - 1))
        best = tl.where(valid, candidate, best)
        chosen = state
    else:
        for q in tl.range(0, BRANCHES):
            state = q * PREFIXES + j
            candidate = tl.zeros((PREFIXES,), tl.float32)
            for lane in tl.static_range(0, V):
                xv = tl.load(x_ptr + lane * B + seq).to(tl.float32)
                lv = tl.load(lut_ptr + lane * STATES + state).to(tl.float32)
                candidate += (lv - xv) * (lv - xv)
            take = candidate < best
            best = tl.where(take, candidate, best)
            chosen = tl.where(take, state, chosen)

    base = seq * PREFIXES
    tl.store(scratch_ptr + base + j, best)
    tl.store(best_state_ptr + base + j, chosen)
    tl.debug_barrier()

    step = 1
    while step < STEPS:
        previous_base = (step & 1 ^ 1) * B * PREFIXES + base
        current_base = (step & 1) * B * PREFIXES + base
        best = tl.full((PREFIXES,), float("inf"), tl.float32)
        chosen = tl.zeros((PREFIXES,), tl.int32)
        for q in tl.range(0, BRANCHES):
            predecessor_prefix = q * Q_FACTOR + residue
            predecessor_cost = tl.load(
                scratch_ptr + previous_base + predecessor_prefix
            )
            state = q * PREFIXES + j
            candidate = predecessor_cost
            for lane in tl.static_range(0, V):
                xv = tl.load(
                    x_ptr + (step * V + lane) * B + seq
                ).to(tl.float32)
                lv = tl.load(lut_ptr + lane * STATES + state).to(tl.float32)
                candidate += (lv - xv) * (lv - xv)
            take = candidate < best
            best = tl.where(take, candidate, best)
            chosen = tl.where(take, state, chosen)
        tl.store(scratch_ptr + current_base + j, best)
        tl.store(
            best_state_ptr + step * B * PREFIXES + base + j,
            chosen,
        )
        tl.debug_barrier()
        step += 1

    if HAS_OVERLAP:
        prefix = tl.load(overlap_ptr + seq).to(tl.int32)
    else:
        prefix = tl.argmin(best, axis=0).to(tl.int32)
    for back_step in tl.static_range(STEPS - 1, -1, -1):
        state = tl.load(
            best_state_ptr + back_step * B * PREFIXES + base + prefix
        ).to(tl.int32)
        tl.store(states_ptr + back_step * B + seq, state)
        prefix = state >> SHIFT


def geometry(cb: Any, *, steps: int = 128) -> dict[str, int | str | float]:
    L, K, V = int(cb.L), int(cb.K), int(cb.V)
    branch_bits = K * V
    sealed = (L, K, V)
    try:
        backend = backend_for_geometry(sealed)
    except ValueError as exc:
        raise ValueError(
            f"geometry L{L}/K{K}/V{V} not in compiled set — run "
            "`smash kernels build --tier qtip --bpw <bpw>`"
        ) from exc
    if backend not in PERSISTENT_BACKENDS:
        raise ValueError(
            f"geometry L{L}/K{K}/V{V} uses backend {backend!r}, not the persistent "
            "prefix backend declared in qtip_rings.json"
        )
    if L < 2 * branch_bits:
        raise ValueError(f"geometry L{L}/K{K}/V{V} cannot retain exact prefixes")
    states = 1 << L
    prefixes = 1 << (L - branch_bits)
    return {
        "implementation": backend,
        "L": L,
        "K": K,
        "V": V,
        "full_states": states,
        "retained_prefix_costs": prefixes,
        "branches_per_prefix": 1 << branch_bits,
        "branch_sampling": "full",
        "steps": steps,
        "min_exact_quality": 1.0,
        "ordering": "one persistent launch per independent sequence batch",
        "best_state_dtype": "int32",
    }


def exact_prefix_viterbi(
    cb: Any,
    x: torch.Tensor,
    overlap: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return exact full-branch Viterbi states for a compiled QTIP geometry."""
    _require_triton()
    if not x.is_cuda or x.ndim != 2:
        raise ValueError(
            f"exact prefix Viterbi expects CUDA [T,B], got {tuple(x.shape)}"
        )
    metadata = geometry(cb, steps=int(x.shape[0]) // int(cb.V))
    L, K, V = int(cb.L), int(cb.K), int(cb.V)
    if x.shape[0] % V:
        raise ValueError(f"input rows {x.shape[0]} not divisible by V={V}")
    batch = int(x.shape[1])
    if batch < 1 or batch > 8192:
        raise ValueError(f"batch outside 1..8192: {batch}")
    steps = int(metadata["steps"])
    states_count = int(metadata["full_states"])
    prefixes = int(metadata["retained_prefix_costs"])
    if overlap is not None:
        _validate_overlap_prefixes(
            overlap,
            x=x,
            batch=batch,
            prefixes=prefixes,
        )
    branches = int(metadata["branches_per_prefix"])
    shift = K * V
    q_factor = 1 << (L - 2 * shift)
    state_elements = steps * batch
    state_bytes = state_elements * 4
    contract = getattr(cb, "_banana_smasher_memory_contract", None)
    builder_scope = contract is not None
    observed_state_elements = 0
    if builder_scope:
        observed_state_elements = getattr(
            cb, "_banana_smasher_observed_state_elements", None
        )
        expected_keys = {
            "schema",
            "state_elements",
            "state_storage_bytes",
            "retained_output_bytes",
        }
        if (
            not isinstance(contract, Mapping)
            or set(contract) != expected_keys
            or contract.get("schema")
            != "banana-smasher-qtip-builder-memory-v2"
            or any(
                isinstance(contract.get(name), bool)
                or not isinstance(contract.get(name), int)
                or contract[name] < 0
                for name in expected_keys - {"schema"}
            )
            or isinstance(observed_state_elements, bool)
            or not isinstance(observed_state_elements, int)
            or observed_state_elements < 0
            or contract["state_elements"] < state_elements
            or contract["state_storage_bytes"] < state_bytes
            or contract["retained_output_bytes"] < 1
            or observed_state_elements + state_elements
            > contract["state_elements"]
        ):
            raise RuntimeError("invalid QTIP builder memory contract")
        retained_state_storage_bytes = contract["state_storage_bytes"]
        retained_output_bytes = contract["retained_output_bytes"]
    else:
        retained_state_storage_bytes = 0
        retained_output_bytes = 0
    allocator_backend = torch.cuda.get_allocator_backend()
    if allocator_backend != "native":
        raise RuntimeError(
            f"unsupported QTIP CUDA allocator backend for exact preflight: "
            f"{allocator_backend}"
        )
    overlap_copy_bytes = (
        4
        if overlap is None
        else overlap.numel() * overlap.element_size()
        if not overlap.is_contiguous()
        else 0
    )
    peak = qtip_peak_allocation_bytes(
        steps=steps,
        batch=batch,
        prefixes=prefixes,
        x_bytes=x.numel() * x.element_size(),
        lut_bytes=cb.lut.numel() * cb.lut.element_size(),
        x_requires_copy=not x.is_contiguous(),
        lut_requires_copy=not cb.lut.is_contiguous(),
        overlap_copy_bytes=overlap_copy_bytes,
        retained_state_storage_bytes=retained_state_storage_bytes,
        retained_output_bytes=retained_output_bytes,
        final_concatenation_bytes=retained_state_storage_bytes,
    )
    driver_free, _total = torch.cuda.mem_get_info(x.device)
    reserved = torch.cuda.memory_reserved(x.device)
    allocated = torch.cuda.memory_allocated(x.device)
    effective_free = effective_cuda_free_bytes(
        driver_free=driver_free,
        reserved=reserved,
        allocated=allocated,
    )
    reserve = 4 << 30
    total_peak = peak["total_bytes"]
    assert isinstance(total_peak, int)
    if builder_scope or total_peak >= 256 << 20:
        try:
            require_qtip_memory_capacity(
                effective_free=effective_free,
                free_source="torch.cuda.mem_get_info+native-cache",
                reserve=reserve,
                peak=peak,
                geometry=(L, K, V),
            )
        except RuntimeError as capacity_error:
            allocations = peak["allocations"]
            assert isinstance(allocations, dict)
            fixed_kernel_bytes = sum(
                int(allocations[name])
                for name in (
                    "x_contiguous_copy",
                    "lut_contiguous_copy",
                    "overlap_storage",
                )
            )
            available_workspace = effective_free - reserve - fixed_kernel_bytes
            if available_workspace < 1:
                raise capacity_error
            try:
                plan = plan_qtip_streaming_batches(
                    steps=steps,
                    batch=batch,
                    prefixes=prefixes,
                    available_workspace_bytes=available_workspace,
                )
            except RuntimeError:
                raise capacity_error
            slices = plan["batch_slices"]
            if not isinstance(slices, list) or len(slices) <= 1:
                raise
            outputs = []
            for start, end in slices:
                streamed_overlap = (
                    None if overlap is None else overlap[start:end].contiguous()
                )
                outputs.append(
                    exact_prefix_viterbi(
                        cb,
                        x[:, start:end],
                        overlap=streamed_overlap,
                    )
                )
            return torch.cat(outputs, dim=1)
    x = x.contiguous()
    # Keep the canonical codebook as V contiguous state planes. Every transition
    # consumes all V planes for the same prefix tile, preserving coalesced SoA loads.
    lut = cb.lut.contiguous()
    if overlap is not None:
        overlap = overlap.contiguous()
    if lut.numel() != V * states_count:
        raise ValueError(
            f"codebook LUT has {lut.numel()} values, expected {V * states_count}"
        )
    scratch = torch.empty(
        (2, batch, prefixes), device=x.device, dtype=torch.float32
    )
    best_state = torch.empty(
        (steps, batch, prefixes), device=x.device, dtype=torch.int32
    )
    states = torch.empty((steps, batch), device=x.device, dtype=torch.int32)
    overlap_arg = (
        overlap
        if overlap is not None
        else torch.empty((1,), device=x.device, dtype=torch.int32)
    )
    if backend_for_geometry((L, K, V)) == PERSISTENT_V32_BACKEND and steps == 128:
        # Preserve the sealed v32 launch byte-for-byte for qtip@3.00 steady state.
        _persistent_prefix_viterbi[(batch,)](
            x,
            lut,
            overlap_arg,
            scratch,
            best_state,
            states,
            B=batch,
            HAS_OVERLAP=overlap is not None,
            num_warps=16,
            num_stages=1,
        )
    else:
        _persistent_prefix_viterbi_generic[(batch,)](
            x,
            lut,
            overlap_arg,
            scratch,
            best_state,
            states,
            B=batch,
            STATES=states_count,
            PREFIXES=prefixes,
            BRANCHES=branches,
            SHIFT=shift,
            Q_FACTOR=q_factor,
            V=V,
            STEPS=steps,
            HAS_OVERLAP=overlap is not None,
            num_warps=16,
            num_stages=1,
        )
    if builder_scope:
        cb._banana_smasher_observed_state_elements = (
            observed_state_elements + state_elements
        )
    return states


def install_exact_prefix_viterbi(
    cb: Any,
) -> dict[str, int | str | float]:
    """Install the accelerated exact methods, refusing an unavailable backend."""
    _require_triton()

    def viterbi(self: Any, x: torch.Tensor, overlap: torch.Tensor | None = None):
        return exact_prefix_viterbi(self, x, overlap)

    def quantize_seq(self: Any, x: torch.Tensor, overlap: torch.Tensor | None = None, **_: Any):
        return exact_prefix_viterbi(self, x, overlap)

    cb.viterbi = types.MethodType(viterbi, cb)
    cb.quantize_seq = types.MethodType(quantize_seq, cb)
    return geometry(cb)
