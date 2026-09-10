"""Manifest-geometry prefix-compressed Triton Viterbi for QTIP rings.

The production kernel assigns one independent sequence to each CTA and keeps all
128 dynamic-programming steps in one launch. The independent parity oracle lives
under ``tests/`` and is not shipped as a fallback.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import types
from typing import TYPE_CHECKING, Any

import torch

from .qtip_rings import (
    PERSISTENT_BACKENDS,
    PERSISTENT_V32_BACKEND,
    backend_for_geometry,
    qtip_admission_memory,
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
def _strict_branch_group_min(candidate, state):
    """Parallel branch reduction with first-state ties and strict-< NaN semantics."""
    finite_or_inf = tl.where(candidate == candidate, candidate, float("inf"))
    best = tl.min(finite_or_inf, axis=1)
    chosen = tl.min(tl.where(finite_or_inf == best[:, None], state, 2147483647), axis=1)
    return best, chosen


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
    BRANCH_UNROLL: tl.constexpr = 1,
    GROUP_BRANCHES: tl.constexpr = False,
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
    elif GROUP_BRANCHES:
        for group in range(16):
            group_q = group * 4 + tl.arange(0, 4)[None, :]
            group_state = group_q * 1024 + j[:, None]
            group_lut0 = tl.load(lut_ptr + group_state).to(tl.float32)
            group_lut1 = tl.load(lut_ptr + 65536 + group_state).to(tl.float32)
            group_candidate = (group_lut0 - x0) * (group_lut0 - x0) + (group_lut1 - x1) * (group_lut1 - x1)
            group_best, group_chosen = _strict_branch_group_min(group_candidate, group_state)
            take = group_best < best
            best = tl.where(take, group_best, best)
            chosen = tl.where(take, group_chosen, chosen)
    else:
        for q in tl.range(0, 64, loop_unroll_factor=BRANCH_UNROLL):
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
        if GROUP_BRANCHES:
            for group in range(16):
                group_q = group * 4 + tl.arange(0, 4)[None, :]
                group_predecessor_prefix = group_q * 16 + residue4[:, None]
                group_predecessor_cost = tl.load(scratch_ptr + previous_base + group_predecessor_prefix)
                group_state = group_q * 1024 + j[:, None]
                group_lut0 = tl.load(lut_ptr + group_state).to(tl.float32)
                group_lut1 = tl.load(lut_ptr + 65536 + group_state).to(tl.float32)
                group_candidate = group_predecessor_cost + (group_lut0 - x0) * (group_lut0 - x0) + (group_lut1 - x1) * (group_lut1 - x1)
                group_best, group_chosen = _strict_branch_group_min(group_candidate, group_state)
                take = group_best < best
                best = tl.where(take, group_best, best)
                chosen = tl.where(take, group_chosen, chosen)
        else:
            for q in tl.range(0, 64, loop_unroll_factor=BRANCH_UNROLL):
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
def _rematerialized_alphabet_key(state):
    # Recompute the cheap integer map instead of keeping four key vectors live
    # across all timesteps. Volatile assembly prevents loop-invariant hoisting.
    return tl.inline_asm_elementwise(
        "{ .reg .u32 a, b; add.u32 a, $1, 1; mul.lo.u32 b, $1, a; shr.u32 a, b, 6; and.b32 $0, a, 1023; }",
        constraints="=r,r", args=[state], dtype=tl.int32,
        is_pure=False, pack=1,
    )


@triton.jit
def _tiled_pointer_offset(step, seq, j, B, PREFIXES: tl.constexpr, STEPS: tl.constexpr):
    # K1 only: eight temporal rows share full 128-byte uint16 cache lines.
    # Other geometries and nonaligned lengths retain the original layout.
    if PREFIXES == 16384 and STEPS % 8 == 0:
        return ((((step // 8) * B + seq) * (PREFIXES // 64) + j // 64) * 8 + step % 8) * 64 + j % 64
    return (step * B + seq) * PREFIXES + j


@triton.jit
def _persistent_prefix_viterbi_generic(
    x_ptr,
    lut_ptr,
    alphabet_ptr,
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
    REGISTER_COSTS: tl.constexpr,
    BRANCH_UNROLL: tl.constexpr,
    STRUCTURED_GATHER: tl.constexpr,
    BRANCH_POINTERS: tl.constexpr = False,
    LUT_EVICTION: tl.constexpr = "",
    DISTANCE_ALPHABET: tl.constexpr = False,
    CONDITIONED_DISTANCE_SUM: tl.constexpr = False,
    FUSED_SCHEDULE: tl.constexpr = False,
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
            lv = tl.load(lut_ptr + lane * STATES + state, eviction_policy=LUT_EVICTION).to(tl.float32)
            candidate += (lv - xv) * (lv - xv)
        valid = residue == (overlap & (Q_FACTOR - 1))
        best = tl.where(valid, candidate, best)
        chosen = state
    else:
        # Hoist the per-step inputs out of the branch loop: they are invariant
        # over q, and re-loading them BRANCHES times per step was the dominant
        # cost for K=4 (256 branches).  Same fp32 values, same strict-< order.
        tl.static_assert(V == 2, "ring family is V=2")
        xa = tl.load(x_ptr + seq).to(tl.float32)
        xb = tl.load(x_ptr + B + seq).to(tl.float32)
        if FUSED_SCHEDULE:
            for static_branch_0 in tl.static_range(0, 4):
                state = static_branch_0 * PREFIXES + j
                la = tl.load(lut_ptr + state, eviction_policy=LUT_EVICTION).to(tl.float32)
                lb = tl.load(lut_ptr + STATES + state, eviction_policy=LUT_EVICTION).to(tl.float32)
                candidate = (la - xa) * (la - xa) + (lb - xb) * (lb - xb)
                take = candidate < best
                best = tl.where(take, candidate, best)
                chosen = tl.where(take, state, chosen)
        else:
            for q in tl.range(0, BRANCHES, loop_unroll_factor=BRANCH_UNROLL):
                state = q * PREFIXES + j
                la = tl.load(lut_ptr + state, eviction_policy=LUT_EVICTION).to(tl.float32)
                lb = tl.load(lut_ptr + STATES + state, eviction_policy=LUT_EVICTION).to(tl.float32)
                candidate = (la - xa) * (la - xa) + (lb - xb) * (lb - xb)
                take = candidate < best
                best = tl.where(take, candidate, best)
                chosen = tl.where(take, state, chosen)

    base = seq * PREFIXES
    if not REGISTER_COSTS:
        tl.store(scratch_ptr + base + j, best)
    # The prefix is the table column; only the winning branch is needed.
    tl.store(best_state_ptr + _tiled_pointer_offset(0, seq, j, B, PREFIXES, STEPS), chosen // PREFIXES if BRANCH_POINTERS else chosen, cache_modifier=".cs" if PREFIXES == 16384 else "")
    if not REGISTER_COSTS:
        tl.debug_barrier()

    if DISTANCE_ALPHABET:
        alphabet_j = tl.arange(0, 1024)
        alphabet_a = tl.load(alphabet_ptr + alphabet_j).to(tl.float32)
        alphabet_b = tl.load(alphabet_ptr + 1024 + alphabet_j).to(tl.float32)

    step = 1
    while step < STEPS:
        # Costs belong to this CTA. Gather the preceding vector directly,
        # avoiding global scratch reloads and stores at every timestep.
        # Default arithmetic is unchanged. The opt-in rebases common cost
        # offsets before applying compact summed distances, limiting FP32 growth.
        previous_costs = best
        if DISTANCE_ALPHABET and HAS_OVERLAP and CONDITIONED_DISTANCE_SUM:
            minimum_cost = tl.min(previous_costs, axis=0)
            previous_costs = previous_costs - tl.where(minimum_cost < float("inf"), minimum_cost, 0.0)
        previous_base = (step & 1 ^ 1) * B * PREFIXES + base
        current_base = (step & 1) * B * PREFIXES + base
        best = tl.full((PREFIXES,), float("inf"), tl.float32)
        chosen = tl.zeros((PREFIXES,), tl.int32)
        xa = tl.load(x_ptr + (step * V) * B + seq).to(tl.float32)
        xb = tl.load(x_ptr + (step * V + 1) * B + seq).to(tl.float32)
        if DISTANCE_ALPHABET:
            delta_a = alphabet_a - xa
            delta_b = alphabet_b - xb
            # Experimental regrouping: sum once over the compact alphabet.
            # FP32 association changes; quality, not assignment identity, gates it.
            distance_sum = delta_a * delta_a + delta_b * delta_b
        if FUSED_SCHEDULE:
            for static_branch_1 in tl.static_range(0, 4):
                predecessor_prefix = static_branch_1 * Q_FACTOR + residue
                if REGISTER_COSTS:
                    if STRUCTURED_GATHER:
                        # Select one contiguous predecessor row, then broadcast its
                        # entries BRANCHES times (K1: 4x4096; K3: 64x16).
                        # Nonnegative costs add only exact zeros, including +inf.
                        cost_rows = tl.reshape(previous_costs, (BRANCHES, Q_FACTOR))
                        selected = tl.sum(tl.where(
                            tl.arange(0, BRANCHES)[:, None] == static_branch_1, cost_rows, 0.0
                        ), axis=0)
                        predecessor_cost = tl.reshape(tl.broadcast_to(
                            selected[:, None], (Q_FACTOR, BRANCHES)
                        ), (PREFIXES,))
                    else:
                        predecessor_cost = tl.gather(previous_costs, predecessor_prefix, axis=0)
                else:
                    predecessor_cost = tl.load(scratch_ptr + previous_base + predecessor_prefix)
                state = static_branch_1 * PREFIXES + j
                if DISTANCE_ALPHABET:
                    alphabet_key = _rematerialized_alphabet_key(state)
                    if HAS_OVERLAP and CONDITIONED_DISTANCE_SUM:
                        candidate = predecessor_cost + tl.gather(distance_sum, alphabet_key, axis=0)
                    else:
                        # Preserve the original full-context heuristic seed.
                        da = tl.gather(delta_a, alphabet_key, axis=0)
                        db = tl.gather(delta_b, alphabet_key, axis=0)
                        candidate = predecessor_cost + da * da + db * db
                else:
                    la = tl.load(lut_ptr + state, eviction_policy=LUT_EVICTION).to(tl.float32)
                    lb = tl.load(lut_ptr + STATES + state, eviction_policy=LUT_EVICTION).to(tl.float32)
                    candidate = predecessor_cost + (la - xa) * (la - xa) + (lb - xb) * (lb - xb)
                take = candidate < best
                best = tl.where(take, candidate, best)
                # Only static_branch_1 varies between candidates at a fixed prefix column.
                # The conditioned-sum down specialization keeps its incumbent layout.
                chosen = tl.where(take, static_branch_1 if DISTANCE_ALPHABET and not CONDITIONED_DISTANCE_SUM else state, chosen)
        else:
            for q in tl.range(0, BRANCHES, loop_unroll_factor=BRANCH_UNROLL):
                predecessor_prefix = q * Q_FACTOR + residue
                if REGISTER_COSTS:
                    if STRUCTURED_GATHER:
                        # Select one contiguous predecessor row, then broadcast its
                        # entries BRANCHES times (K1: 4x4096; K3: 64x16).
                        # Nonnegative costs add only exact zeros, including +inf.
                        cost_rows = tl.reshape(previous_costs, (BRANCHES, Q_FACTOR))
                        selected = tl.sum(tl.where(
                            tl.arange(0, BRANCHES)[:, None] == q, cost_rows, 0.0
                        ), axis=0)
                        predecessor_cost = tl.reshape(tl.broadcast_to(
                            selected[:, None], (Q_FACTOR, BRANCHES)
                        ), (PREFIXES,))
                    else:
                        predecessor_cost = tl.gather(previous_costs, predecessor_prefix, axis=0)
                else:
                    predecessor_cost = tl.load(scratch_ptr + previous_base + predecessor_prefix)
                state = q * PREFIXES + j
                if DISTANCE_ALPHABET:
                    alphabet_key = _rematerialized_alphabet_key(state)
                    if HAS_OVERLAP and CONDITIONED_DISTANCE_SUM:
                        candidate = predecessor_cost + tl.gather(distance_sum, alphabet_key, axis=0)
                    else:
                        # Preserve the original full-context heuristic seed.
                        da = tl.gather(delta_a, alphabet_key, axis=0)
                        db = tl.gather(delta_b, alphabet_key, axis=0)
                        candidate = predecessor_cost + da * da + db * db
                else:
                    la = tl.load(lut_ptr + state, eviction_policy=LUT_EVICTION).to(tl.float32)
                    lb = tl.load(lut_ptr + STATES + state, eviction_policy=LUT_EVICTION).to(tl.float32)
                    candidate = predecessor_cost + (la - xa) * (la - xa) + (lb - xb) * (lb - xb)
                take = candidate < best
                best = tl.where(take, candidate, best)
                # Only q varies between candidates at a fixed prefix column.
                # The conditioned-sum down specialization keeps its incumbent layout.
                chosen = tl.where(take, q if DISTANCE_ALPHABET and not CONDITIONED_DISTANCE_SUM else state, chosen)
        if not REGISTER_COSTS:
            tl.store(scratch_ptr + current_base + j, best)
        if DISTANCE_ALPHABET and not CONDITIONED_DISTANCE_SUM:
            encoded_chosen = chosen if BRANCH_POINTERS else chosen * PREFIXES + j
            # Preserve the original zero sentinel for unreachable prefixes.
            encoded_chosen = tl.where(best < float("inf"), encoded_chosen, 0)
        else:
            encoded_chosen = chosen // PREFIXES if BRANCH_POINTERS else chosen
        tl.store(
            best_state_ptr + _tiled_pointer_offset(step, seq, j, B, PREFIXES, STEPS),
            encoded_chosen,
            cache_modifier=".cs" if PREFIXES == 16384 else "",
        )
        if not REGISTER_COSTS:
            tl.debug_barrier()
        step += 1

    # Traceback may read another lane's last backpointer.
    tl.debug_barrier()
    if HAS_OVERLAP:
        prefix = tl.load(overlap_ptr + seq).to(tl.int32)
    else:
        prefix = tl.argmin(best, axis=0).to(tl.int32)
    if FUSED_SCHEDULE:
        for back_step in tl.range(STEPS - 1, -1, -1, loop_unroll_factor=(8 if STEPS >= 8 else 1)):
            traceback_state = tl.load(
                best_state_ptr + _tiled_pointer_offset(back_step, seq, prefix, B, PREFIXES, STEPS)
            ).to(tl.int32)
            if BRANCH_POINTERS:
                traceback_state = traceback_state * PREFIXES + prefix
            tl.store(states_ptr + back_step * B + seq, traceback_state)
            prefix = traceback_state >> SHIFT
    else:
        for back_step in tl.static_range(STEPS - 1, -1, -1):
            state = tl.load(
                best_state_ptr + _tiled_pointer_offset(back_step, seq, prefix, B, PREFIXES, STEPS)
            ).to(tl.int32)
            if BRANCH_POINTERS:
                state = state * PREFIXES + prefix
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


def resolve_viterbi_num_warps(geometry: tuple[int, int, int], requested: int | None) -> int:
    """Validate the opt-in K3 launch experiment; preserve the incumbent default."""
    if requested is None:
        return 16
    if geometry not in ((16, 1, 2), (16, 3, 2)) or type(requested) is not int or requested not in (4, 8, 16):
        raise ValueError("viterbi_num_warps requires L16/K1-or-K3/V2 and integer 4, 8, or 16")
    return requested


def resolve_structured_gather(geometry: tuple[int, int, int], requested: bool | None) -> bool:
    """Experimental K1/K3 structured predecessor row selection/broadcast."""
    if requested is None or requested is False:
        return False
    if requested is not True or geometry not in {(16, 1, 2), (16, 3, 2)}:
        raise ValueError("viterbi_structured_gather requires boolean and L16/K1-or-K3/V2")
    return True


def resolve_lut_l1_retention(geometry: tuple[int, int, int], requested: bool | None) -> bool:
    """Opt-in immutable LUT retention; only qualified L16/K1/V2 geometry."""
    if requested is None or requested is False:
        return False
    if requested is not True or geometry != (16, 1, 2):
        raise ValueError("viterbi_lut_l1_retention requires boolean and L16/K1/V2")
    return True


def resolve_branch_unroll(geometry: tuple[int, int, int], requested: bool | None) -> int:
    """Opt-in constant-branch scheduling; no branch removal or arithmetic change."""
    if requested is None or requested is False:
        return 1
    if requested is not True or geometry not in {(16, 1, 2), (16, 3, 2)}:
        raise ValueError("viterbi_branch_unroll requires boolean and L16/K1-or-K3/V2")
    return 4


def resolve_backpointer_dtype(geometry: tuple[int, int, int], requested: str | None) -> str:
    """Opt-in lossless K1 workspace compression; returned state wire stays int32."""
    if requested is None or requested == "int32":
        return "int32"
    if requested not in ("uint16", "uint8") or geometry != (16, 1, 2):
        raise ValueError("viterbi_backpointer_dtype requires L16/K1/V2 and uint8, uint16 or int32")
    return requested


@lru_cache(maxsize=1)
def _signed_alphabet_representatives():
    reps = [-1] * 1024
    for state in range(65536):
        key = ((state * (state + 1)) >> 6) & 1023
        if reps[key] < 0:
            reps[key] = state
    assert all(state >= 0 for state in reps)
    return tuple(reps)


def _distance_alphabet_lut(cb):
    if getattr(cb, "decode_mode", None) != "quantlut_sym" or getattr(cb, "tlut_bits", None) != 9:
        raise ValueError("distance alphabet requires quantlut_sym with 9 bits")
    source = cb.lut
    cached = getattr(cb, "_banana_smasher_distance_alphabet_cache", None)
    version = source._version
    if cached is not None and cached[0] is source and cached[1] == version:
        return cached[2]
    # Admission for the bounded map/reconstruction temporaries precedes allocation.
    if qtip_admission_memory(torch, source.device)["available_bytes"] < (4 << 30) + (4 << 20):
        raise RuntimeError("distance alphabet memory admission refused")
    reps = torch.tensor(_signed_alphabet_representatives(), device=source.device, dtype=torch.int64)
    compact = source[:, reps].contiguous()
    states = torch.arange(65536, device=source.device, dtype=torch.int64)
    keys = ((states * (states + 1)) >> 6) & 1023
    if not torch.equal(compact[:, keys], source):
        raise ValueError("codebook does not match signed distance alphabet")
    cb._banana_smasher_distance_alphabet_cache = (source, version, compact)
    return compact


def resolve_conditioned_distance_sum(geometry, projection, value, alphabet):
    if value is None:
        return False
    if type(value) is not bool or (value and (tuple(geometry) != (16, 1, 2) or projection != "down" or alphabet is not True)):
        raise ValueError("viterbi_conditioned_distance_sum requires boolean, L16/K1/V2 down and distance alphabet")
    return value


def exact_prefix_viterbi(
    cb: Any,
    x: torch.Tensor,
    overlap: torch.Tensor | None = None,
) -> torch.Tensor:
    # Public callers always retain recoverable full value validation.
    return _exact_prefix_viterbi_impl(cb, x, overlap, _bounded_overlap=False)


def _exact_prefix_viterbi_impl(
    cb: Any,
    x: torch.Tensor,
    overlap: torch.Tensor | None = None,
    *,
    _bounded_overlap: bool = False,
) -> torch.Tensor:
    """Return exact full-branch Viterbi states for a compiled QTIP geometry."""
    _require_triton()
    if not x.is_cuda or x.ndim != 2:
        raise ValueError(
            f"exact prefix Viterbi expects CUDA [T,B], got {tuple(x.shape)}"
        )
    metadata = geometry(cb, steps=int(x.shape[0]) // int(cb.V))
    L, K, V = int(cb.L), int(cb.K), int(cb.V)
    launch_warps = resolve_viterbi_num_warps(
        (L, K, V), getattr(cb, "_banana_smasher_viterbi_num_warps", None)
    )
    structured_gather = resolve_structured_gather(
        (L, K, V), getattr(cb, "_banana_smasher_structured_gather", None)
    )
    branch_unroll = resolve_branch_unroll(
        (L, K, V), getattr(cb, "_banana_smasher_branch_unroll", None)
    )
    lut_l1_retention = resolve_lut_l1_retention(
        (L, K, V), getattr(cb, "_banana_smasher_lut_l1_retention", None)
    )
    distance_alphabet = getattr(cb, "_banana_smasher_distance_alphabet", False)
    if type(distance_alphabet) is not bool or (distance_alphabet and (L, K, V) != (16, 1, 2)):
        raise ValueError("viterbi_distance_alphabet requires boolean and L16/K1/V2")
    conditioned_distance_sum = resolve_conditioned_distance_sum(
        (L, K, V), getattr(cb, "_banana_smasher_projection", None),
        getattr(cb, "_banana_smasher_conditioned_distance_sum", False), distance_alphabet
    )
    fused_schedule = resolve_fused_schedule(
        (L, K, V), getattr(cb, "_banana_smasher_projection", None),
        getattr(cb, "_banana_smasher_fused_schedule", False), distance_alphabet,
        conditioned_distance_sum, launch_warps,
    )
    group_branches = getattr(cb, "_banana_smasher_branch_grouped", False)
    if type(group_branches) is not bool or (group_branches and (
        (L, K, V) != (16, 3, 2) or x.shape[0] != 256 or structured_gather or branch_unroll != 1
    )):
        raise ValueError("viterbi_branch_grouped requires K3/128 steps, no structured gather or unroll")
    if x.shape[0] % V:
        raise ValueError(f"input rows {x.shape[0]} not divisible by V={V}")
    batch = int(x.shape[1])
    if batch < 1 or batch > 8192:
        raise ValueError(f"batch outside 1..8192: {batch}")
    steps = int(metadata["steps"])
    states_count = int(metadata["full_states"])
    prefixes = int(metadata["retained_prefix_costs"])
    if overlap is not None:
        if _bounded_overlap:
            # Only quantize_from_exact_states creates this internal operand.
            # State producer bounds plus right shift establish its value range.
            if (not overlap.is_cuda or overlap.device != x.device
                    or overlap.ndim != 1 or overlap.numel() != batch
                    or overlap.dtype not in {torch.int32, torch.int64}):
                raise ValueError("invalid internal overlap metadata")
        else:
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
    memory = qtip_admission_memory(torch, x.device)
    effective_free = memory["available_bytes"]
    reserve = (4 << 30) + ((4 << 20) if distance_alphabet else 0)
    total_peak = peak["total_bytes"]
    assert isinstance(total_peak, int)
    if builder_scope or total_peak >= 256 << 20:
        try:
            require_qtip_memory_capacity(
                effective_free=effective_free,
                free_source=memory["source"],
                reserve=reserve,
                peak=peak,
                geometry=(L, K, V),
            )
        except RuntimeError as capacity_error:
            if distance_alphabet:
                raise
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
                    _exact_prefix_viterbi_impl(
                        cb,
                        x[:, start:end],
                        overlap=streamed_overlap,
                        _bounded_overlap=_bounded_overlap,
                    )
                )
            return torch.cat(outputs, dim=1)
    x = x.contiguous()
    # Keep the canonical codebook as V contiguous state planes. Every transition
    # consumes all V planes for the same prefix tile, preserving coalesced SoA loads.
    lut = cb.lut.contiguous()
    alphabet_lut = _distance_alphabet_lut(cb) if distance_alphabet else lut
    if overlap is not None:
        overlap = overlap.contiguous()
    if lut.numel() != V * states_count:
        raise ValueError(
            f"codebook LUT has {lut.numel()} values, expected {V * states_count}"
        )
    scratch = torch.empty(
        (2, batch, prefixes), device=x.device, dtype=torch.float32
    )
    # Conservative admission above still budgets int32 workspace. Only the
    # internal backpointer storage changes; recurrence and int32 wire do not.
    backpointer_dtype = resolve_backpointer_dtype(
        (L, K, V), getattr(cb, "_banana_smasher_backpointer_dtype", None)
    )
    best_state = torch.empty(
        (steps, batch, prefixes), device=x.device, dtype=getattr(torch, backpointer_dtype)
    )
    states = torch.empty((steps, batch), device=x.device, dtype=torch.int32)
    overlap_arg = (
        overlap
        if overlap is not None
        else torch.empty((1,), device=x.device, dtype=torch.int32)
    )
    if backend_for_geometry((L, K, V)) == PERSISTENT_V32_BACKEND and steps == 128 and not structured_gather:
        # Default stays at 16; smaller schedules are explicit unpromoted experiments.
        _persistent_prefix_viterbi[(batch,)](
            x,
            lut,
            overlap_arg,
            scratch,
            best_state,
            states,
            B=batch,
            HAS_OVERLAP=overlap is not None,
            BRANCH_UNROLL=branch_unroll,
            GROUP_BRANCHES=group_branches,
            num_warps=launch_warps,
            num_stages=1,
        )
    else:
        # 512 threads for a 256-wide (K=4) or 4096-wide (K=1) prefix vector is
        # wrong either way; size warps to the vector.  Scheduling only.
        generic_warps = launch_warps if K == 1 or structured_gather else max(4, min(16, prefixes // 64))
        _persistent_prefix_viterbi_generic[(batch,)](
            x,
            lut,
            alphabet_lut,
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
            REGISTER_COSTS=K == 1 or structured_gather,
            BRANCH_UNROLL=branch_unroll,
            STRUCTURED_GATHER=structured_gather,
            BRANCH_POINTERS=backpointer_dtype == "uint8",
            LUT_EVICTION="evict_last" if lut_l1_retention else "",
            DISTANCE_ALPHABET=distance_alphabet,
            CONDITIONED_DISTANCE_SUM=conditioned_distance_sum,
            FUSED_SCHEDULE=fused_schedule,
            num_warps=generic_warps,
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
    cb._banana_smasher_fused_schedule = False
    native_quantize = getattr(cb, "_banana_smasher_native_quantize", None)
    if native_quantize is not None:
        cb.quantize = native_quantize

    def viterbi(self: Any, x: torch.Tensor, overlap: torch.Tensor | None = None):
        return exact_prefix_viterbi(self, x, overlap)

    def quantize_seq(self: Any, x: torch.Tensor, overlap: torch.Tensor | None = None, **_: Any):
        return exact_prefix_viterbi(self, x, overlap)

    cb.viterbi = types.MethodType(viterbi, cb)
    cb.quantize_seq = types.MethodType(quantize_seq, cb)
    return geometry(cb)

def quantize_from_exact_states(cb: Any, X: torch.Tensor):
    """Native two-pass quantize with an internal producer-bounded overlap.

    No caller overlap or caller-supplied states are accepted here. The exact
    recurrence returns int32 states in [0, 2**L); shifting by K*V therefore
    establishes [0, 2**(L-K*V)) without a device-to-host scalar round trip.
    Recheck geometry at entry; retain all per-solve memory and metadata gates.
    Public exact_prefix_viterbi/quantize_seq still validate arbitrary operands.
    """
    if (int(cb.L), int(cb.K), int(cb.V)) != (16, 1, 2):
        raise ValueError("producer-bounded quantize requires L16/K1/V2")
    if X.ndim != 2 or not X.is_cuda or X.shape[1] != 256 or X.shape[0] < 1:
        raise ValueError("producer-bounded quantize expects CUDA [B,256]")
    geometry(cb, steps=128)
    X = X.T.contiguous().to(torch.float16)
    T = X.shape[0]

    def sequence(x, overlap=None, *, bounded=False):
        # Preserve native quantize_seq's exact wide chunk/pad/order, including
        # its 256-column chunk size when the public 8192 width cap is exceeded.
        if x.shape[1] <= 8192:
            return _exact_prefix_viterbi_impl(cb, x, overlap, _bounded_overlap=bounded)
        import math
        T, NO = x.shape
        bs = min(2**(24 - cb.L), NO)
        pad_amt = math.ceil(NO / bs) * bs - NO
        x = torch.nn.functional.pad(x, (0, pad_amt))
        T, N = x.shape
        x = x.reshape(T, N // bs, bs).transpose(0, 1).contiguous()
        if overlap is not None:
            overlap = torch.nn.functional.pad(overlap, (0, pad_amt))
            overlap = overlap.reshape(N // bs, bs)
        states = torch.zeros(N // bs, T // cb.V, bs,
                             dtype=cb.idx_dtype, device=x.device)
        for i in range(len(x)):
            part = None if overlap is None else overlap[i]
            states[i] = _exact_prefix_viterbi_impl(
                cb, x[i], part, _bounded_overlap=bounded)
        return states.transpose(0, 1).reshape(T // cb.V, N)[:, :NO]

    roll_X = torch.roll(X, T // (2 * cb.V) * cb.V, 0)
    state = sequence(roll_X)
    overlap = state[T // (2 * cb.V)] >> (cb.K * cb.V)
    state = sequence(X, overlap, bounded=True)
    hatX = cb.recons(state).transpose(0, 1).reshape(X.shape)
    return hatX.T.contiguous().to(X.device), state.T.contiguous().to(X.device)


def resolve_bounded_overlap(geometry, value, profile_mode):
    if value is None:
        return False
    if type(value) is not bool or (value and (tuple(geometry) != (16, 1, 2) or profile_mode)):
        raise ValueError("viterbi_bounded_overlap requires boolean, L16/K1/V2 and non-profile mode")
    return value


def resolve_fused_schedule(geometry, projection, value, distance_alphabet, conditioned_sum, warps):
    """Default-off measured K1 fused scheduling; no implicit warp mutation."""
    if value is None:
        return False
    if type(value) is not bool or (value and (
        tuple(geometry) != (16, 1, 2) or projection != "fused13"
        or distance_alphabet is not True or conditioned_sum is not False or warps != 8
    )):
        raise ValueError("viterbi_fused_schedule requires boolean, L16/K1/V2 fused13, distance alphabet, no conditioned sum, and 8 warps")
    return value
