"""Homogeneous native QTIP2.5 with one L16/B10/V4 trellis geometry.

The codec appends ten transition bits and emits four scalar weights per state,
so every payload has the exact rational rate B/V = 10/4 = 5/2.  The compact
shared 512x2 QTIP TLUT is reused for both halves of the V4 state codebook; the
second half applies the same public quantlut_sym mapping to a byte-rotated state.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math
from typing import Any, Sequence

import numpy as np

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None


@dataclass(frozen=True)
class NativeQtip25Geometry:
    L: int = 16
    B: int = 10
    V: int = 4
    tlut_bits: int = 9

    def __post_init__(self) -> None:
        if self.L != 16 or self.V != 4 or self.tlut_bits != 9 or not 4 <= self.B <= 16:
            raise ValueError(
                "native QTIP V4 requires L16/V4/Q9 and transition bits B in 4..16"
            )

    @property
    def states(self) -> int:
        return 1 << self.L

    @property
    def prefixes(self) -> int:
        return 1 << (self.L - self.B)

    @property
    def branches(self) -> int:
        return 1 << self.B

    @property
    def rate_num(self) -> int:
        return Fraction(self.B, self.V).numerator

    @property
    def rate_den(self) -> int:
        return Fraction(self.B, self.V).denominator

    def as_mapping(self) -> dict[str, int | str | bool]:
        return {
            "L": self.L,
            "B": self.B,
            "V": self.V,
            "rate_num": self.rate_num,
            "rate_den": self.rate_den,
            "phase_count": 1,
            "unique_transition_bits_per_payload": 1,
            "alternation": False,
            "member_averaging": False,
            "tlut_bits": self.tlut_bits,
            "decode_mode": "paired_quantlut_sym",
        }


NATIVE_QTIP25_GEOMETRY = NativeQtip25Geometry()
_DEFAULT_SCALE_FACTORS = (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20)


def native_v4_geometry(bpw: object) -> NativeQtip25Geometry:
    """Resolve an exact homogeneous quarter-BPW rate to L16/B/V4 geometry."""

    try:
        rate = Fraction(str(bpw))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("native QTIP V4 bpw must be an exact quarter rate") from exc
    transition_bits = rate * 4
    if transition_bits.denominator != 1 or not 4 <= transition_bits.numerator <= 16:
        raise ValueError("native QTIP V4 bpw must use 0.25 increments from 1.00 through 4.00")
    return NativeQtip25Geometry(B=transition_bits.numerator)


@dataclass(frozen=True)
class EncodedNativeQtip25:
    geometry: NativeQtip25Geometry
    shape: tuple[int, int]
    states: np.ndarray
    packed: np.ndarray
    scales: np.ndarray
    distortion: float

    @property
    def weights(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def code_bits(self) -> int:
        return self.weights * self.geometry.rate_num // self.geometry.rate_den

    @property
    def padding_bits(self) -> int:
        return self.packed.nbytes * 8 - self.code_bits

    @property
    def code_bpw(self) -> float:
        return self.code_bits / self.weights


@dataclass(frozen=True)
class NativeV4MatrixResult:
    decoded: np.ndarray
    states: np.ndarray
    packed: np.ndarray
    scales: np.ndarray
    distortion: float
    scale_factor: float
    scale_factors: tuple[float, ...]
    feedback_nonzero_count: int


def _require_tlut(
    tlut: np.ndarray, *, geometry: NativeQtip25Geometry = NATIVE_QTIP25_GEOMETRY
) -> np.ndarray:
    table = np.asarray(tlut, dtype=np.float32)
    if table.shape != (1 << geometry.tlut_bits, 2):
        raise ValueError("native QTIP V4 requires the shared float32 Q9xV2 TLUT")
    if not bool(np.isfinite(table).all()):
        raise ValueError("native QTIP2.5 TLUT must be finite")
    return np.ascontiguousarray(table)


def _quantlut_sym_pair(
    table: np.ndarray,
    states: np.ndarray,
    *,
    geometry: NativeQtip25Geometry = NATIVE_QTIP25_GEOMETRY,
) -> np.ndarray:
    hashed = (states + np.uint64(1)) * states
    sign = np.float32(1.0) - np.float32(2.0) * (
        (hashed >> np.uint64(15)) & np.uint64(1)
    ).astype(np.float32)
    indexes = (
        hashed >> np.uint64(geometry.L - geometry.tlut_bits - 1)
    ) & np.uint64((1 << geometry.tlut_bits) - 1)
    result = table[indexes.astype(np.int64)].copy()
    result[:, 0] *= sign
    return result


def expand_native_v4_tlut(
    tlut: np.ndarray, *, geometry: NativeQtip25Geometry = NATIVE_QTIP25_GEOMETRY
) -> np.ndarray:
    """Expand the shared Q9xV2 TLUT into the deterministic L16xV4 state LUT."""
    table = _require_tlut(tlut, geometry=geometry)
    states = np.arange(geometry.states, dtype=np.uint64)
    half = geometry.L // 2
    rotated = ((states << np.uint64(half)) | (states >> np.uint64(half))) & np.uint64(
        geometry.states - 1
    )
    return np.ascontiguousarray(
        np.concatenate(
            (
                _quantlut_sym_pair(table, states, geometry=geometry),
                _quantlut_sym_pair(table, rotated, geometry=geometry),
            ),
            axis=1,
        )
    )


def native_v5_phase_widths(
    *, geometry: NativeQtip25Geometry = NATIVE_QTIP25_GEOMETRY
) -> tuple[int, int, int, int]:
    """Split one B-bit transition into four ordered PR31 scalar updates."""

    return tuple(
        ((lane + 1) * geometry.B) // geometry.V
        - (lane * geometry.B) // geometry.V
        for lane in range(geometry.V)
    )  # type: ignore[return-value]


def native_v5_edge_states(
    predecessor: np.ndarray,
    branch: np.ndarray,
    *,
    geometry: NativeQtip25Geometry = NATIVE_QTIP25_GEOMETRY,
) -> np.ndarray:
    """Return the four intermediate states for predecessor/branch edges."""

    previous, transition = np.broadcast_arrays(
        np.asarray(predecessor, dtype=np.int64),
        np.asarray(branch, dtype=np.int64),
    )
    if bool(
        np.any(previous < 0)
        or np.any(previous >= geometry.states)
        or np.any(transition < 0)
        or np.any(transition >= geometry.branches)
    ):
        raise ValueError("native V5 edge predecessor or branch is outside geometry")
    state = previous.copy()
    result = np.empty((*state.shape, geometry.V), dtype=np.int32)
    consumed = 0
    for lane, width in enumerate(native_v5_phase_widths(geometry=geometry)):
        consumed += width
        chunk = (transition >> (geometry.B - consumed)) & ((1 << width) - 1)
        state = ((state << width) | chunk) & (geometry.states - 1)
        result[..., lane] = state
    return np.ascontiguousarray(result)


def _validate_states(
    states: np.ndarray, *, geometry: NativeQtip25Geometry = NATIVE_QTIP25_GEOMETRY
) -> np.ndarray:
    values = np.asarray(states)
    if values.ndim != 2 or values.shape[1] * geometry.B < geometry.L:
        raise ValueError("native QTIP2.5 states must be [rows, sufficient transitions]")
    values = values.astype(np.int32, copy=False)
    if bool(np.any(values < 0) or np.any(values >= geometry.states)):
        raise ValueError("native QTIP2.5 state outside L16 geometry")
    suffix_mask = geometry.prefixes - 1
    if values.shape[1] > 1 and bool(
        np.any((values[:, :-1] & suffix_mask) != (values[:, 1:] >> geometry.B))
    ):
        raise ValueError("native QTIP2.5 state path violates B10 transitions")
    if bool(np.any((values[:, -1] & suffix_mask) != (values[:, 0] >> geometry.B))):
        raise ValueError("native QTIP2.5 state path does not close")
    return np.ascontiguousarray(values)


def pack_native_v4_states(
    states: np.ndarray, *, geometry: NativeQtip25Geometry = NATIVE_QTIP25_GEOMETRY
) -> np.ndarray:
    """Pack the exact circular transition stream MSB-first with byte tail padding only."""
    values = _validate_states(states, geometry=geometry)
    bit_count = values.shape[1] * geometry.B
    bits = np.empty((values.shape[0], bit_count), dtype=np.uint8)
    for row in range(values.shape[0]):
        stream = [
            (int(values[row, 0]) >> offset) & 1
            for offset in range(geometry.L - 1, -1, -1)
        ]
        for state in values[row, 1:]:
            stream.extend(
                (int(state) >> offset) & 1
                for offset in range(geometry.B - 1, -1, -1)
            )
        bits[row] = np.asarray(stream[:bit_count], dtype=np.uint8)
    return np.ascontiguousarray(np.packbits(bits, axis=1, bitorder="big"))


def states_from_native_v4_packed(
    packed: np.ndarray,
    *,
    steps: int,
    geometry: NativeQtip25Geometry = NATIVE_QTIP25_GEOMETRY,
) -> np.ndarray:
    """Unpack a circular transition stream into its exact L16 state sequence."""
    words = np.asarray(packed)
    bit_count = int(steps) * geometry.B
    expected_bytes = math.ceil(bit_count / 8)
    if (
        words.dtype != np.uint8
        or words.ndim != 2
        or steps < 1
        or bit_count < geometry.L
        or words.shape[1] != expected_bytes
    ):
        raise ValueError("native QTIP2.5 packed byte shape does not match B10 steps")
    all_bits = np.unpackbits(words, axis=1, bitorder="big")
    if all_bits.shape[1] > bit_count and bool(np.any(all_bits[:, bit_count:])):
        raise ValueError("native QTIP2.5 has nonzero byte-tail padding")
    bits = all_bits[:, :bit_count]
    stream = np.concatenate((bits, bits[:, : geometry.L - geometry.B]), axis=1)
    states = np.empty((words.shape[0], steps), dtype=np.int32)
    first = np.zeros(words.shape[0], dtype=np.int32)
    for offset in range(geometry.L):
        first = (first << 1) | stream[:, offset].astype(np.int32)
    states[:, 0] = first
    mask = geometry.states - 1
    for step in range(1, steps):
        branch = np.zeros(words.shape[0], dtype=np.int32)
        start = geometry.L + (step - 1) * geometry.B
        for offset in range(geometry.B):
            branch = (branch << 1) | stream[:, start + offset].astype(np.int32)
        states[:, step] = ((states[:, step - 1] << geometry.B) & mask) + branch
    return states


def _viterbi_numpy(
    target: np.ndarray,
    state_lut: np.ndarray,
    *,
    overlap: np.ndarray | None,
    geometry: NativeQtip25Geometry = NATIVE_QTIP25_GEOMETRY,
) -> np.ndarray:
    batch, steps, lanes = target.shape
    if lanes != geometry.V or steps * geometry.B < geometry.L:
        raise ValueError("native QTIP2.5 target has insufficient V4 transitions")
    prefix_ids = np.arange(geometry.prefixes, dtype=np.int32)
    predecessors = prefix_ids[:, None] + (
        np.arange(geometry.branches, dtype=np.int32)[None, :] * geometry.prefixes
    )

    def errors(step: int) -> np.ndarray:
        delta = state_lut[None, :, :] - target[:, step, None, :]
        return np.sum(delta * delta, axis=2, dtype=np.float32)

    cost = errors(0)
    if overlap is not None:
        overlap = np.asarray(overlap, dtype=np.int32)
        if overlap.shape != (batch,) or bool(
            np.any(overlap < 0) or np.any(overlap >= geometry.prefixes)
        ):
            raise ValueError("invalid native QTIP2.5 overlap prefixes")
        allowed = (overlap[:, None] << geometry.B) + np.arange(
            geometry.branches, dtype=np.int32
        )
        masked = np.full_like(cost, np.inf)
        masked[np.arange(batch)[:, None], allowed] = cost[
            np.arange(batch)[:, None], allowed
        ]
        cost = masked

    backpointers = np.empty((steps, batch, geometry.prefixes), dtype=np.uint16)
    backpointers[0] = 0
    for step in range(1, steps):
        options = cost[:, predecessors]
        choice = np.argmin(options, axis=2)
        backpointers[step] = choice.astype(np.uint16)
        best = np.take_along_axis(options, choice[:, :, None], axis=2)[:, :, 0]
        cost = errors(step) + np.repeat(best, geometry.branches, axis=1)

    if overlap is None:
        final = np.argmin(cost, axis=1).astype(np.int32)
    else:
        allowed = overlap[:, None] + (
            np.arange(geometry.branches, dtype=np.int32)[None, :] * geometry.prefixes
        )
        final = allowed[
            np.arange(batch),
            np.argmin(cost[np.arange(batch)[:, None], allowed], axis=1),
        ]
    states = np.empty((batch, steps), dtype=np.int32)
    states[:, -1] = final
    for step in range(steps - 1, 0, -1):
        prefix = states[:, step] >> geometry.B
        branch = backpointers[step, np.arange(batch), prefix].astype(np.int32)
        states[:, step - 1] = branch * geometry.prefixes + prefix
    return states


def _viterbi_native_v5_numpy(
    target: np.ndarray,
    scalar_lut: np.ndarray,
    *,
    overlap: np.ndarray | None,
    trellis_objective: str = "sse",
    geometry: NativeQtip25Geometry = NATIVE_QTIP25_GEOMETRY,
) -> np.ndarray:
    """Exact variable-width scalar recurrence for PR31 V5 vector edges."""

    batch, steps, lanes = target.shape
    if lanes != geometry.V or steps * geometry.B < geometry.L:
        raise ValueError("native V5 target has insufficient L16/V4 transitions")
    if trellis_objective not in {"sse", "lexicographic_l4"}:
        raise ValueError("native V5 trellis objective must be sse or lexicographic_l4")
    widths = native_v5_phase_widths(geometry=geometry)
    values = target.reshape(batch, steps * lanes)
    primary_cost: np.ndarray | None = None
    secondary_cost: np.ndarray | None = None
    backpointers: list[np.ndarray | None] = []
    row_ids = np.arange(batch)

    for phase in range(values.shape[1]):
        width = widths[phase % geometry.V]
        prefixes = 1 << (geometry.L - width)
        branches = 1 << width
        delta = scalar_lut[None, :] - values[:, phase, None]
        errors = delta * delta
        primary_errors = errors if trellis_objective == "sse" else errors * errors
        if phase == 0:
            primary_cost = primary_errors
            secondary_cost = errors if trellis_objective == "lexicographic_l4" else None
            if overlap is not None:
                expected = np.asarray(overlap, dtype=np.int32)
                if expected.shape != (batch,) or bool(
                    np.any(expected < 0) or np.any(expected >= prefixes)
                ):
                    raise ValueError("invalid native V5 cyclic overlap prefixes")
                allowed = (expected[:, None] << width) + np.arange(
                    branches, dtype=np.int32
                )
                masked = np.full_like(primary_cost, np.inf)
                masked[row_ids[:, None], allowed] = primary_cost[
                    row_ids[:, None], allowed
                ]
                primary_cost = masked
                if secondary_cost is not None:
                    masked_secondary = np.full_like(secondary_cost, np.inf)
                    masked_secondary[row_ids[:, None], allowed] = secondary_cost[
                        row_ids[:, None], allowed
                    ]
                    secondary_cost = masked_secondary
            backpointers.append(None)
            continue

        assert primary_cost is not None
        prefix_ids = np.arange(prefixes, dtype=np.int32)
        predecessors = prefix_ids[:, None] + (
            np.arange(branches, dtype=np.int32)[None, :] * prefixes
        )
        primary_options = primary_cost[:, predecessors]
        secondary_options: np.ndarray | None = None
        if secondary_cost is None:
            choice = np.argmin(primary_options, axis=2)
        else:
            secondary_options = secondary_cost[:, predecessors]
            minimum_primary = np.min(primary_options, axis=2, keepdims=True)
            tied_secondary = np.where(
                primary_options == minimum_primary, secondary_options, np.inf
            )
            choice = np.argmin(tied_secondary, axis=2)
        best_primary = np.take_along_axis(
            primary_options, choice[:, :, None], axis=2
        )[:, :, 0]
        primary_cost = primary_errors + np.repeat(best_primary, branches, axis=1)
        if secondary_cost is not None:
            assert secondary_options is not None
            best_secondary = np.take_along_axis(
                secondary_options, choice[:, :, None], axis=2
            )[:, :, 0]
            secondary_cost = errors + np.repeat(best_secondary, branches, axis=1)
        backpointers.append(choice.astype(np.uint8))

    assert primary_cost is not None
    if overlap is None:
        if secondary_cost is None:
            final = np.argmin(primary_cost, axis=1).astype(np.int32)
        else:
            minimum_primary = np.min(primary_cost, axis=1, keepdims=True)
            final = np.argmin(
                np.where(primary_cost == minimum_primary, secondary_cost, np.inf), axis=1
            ).astype(np.int32)
    else:
        first_width = widths[0]
        first_prefixes = 1 << (geometry.L - first_width)
        final_allowed = np.asarray(overlap, dtype=np.int32)[:, None] + (
            np.arange(1 << first_width, dtype=np.int32)[None, :] * first_prefixes
        )
        allowed_primary = primary_cost[row_ids[:, None], final_allowed]
        if secondary_cost is None:
            final_choice = np.argmin(allowed_primary, axis=1)
        else:
            minimum_primary = np.min(allowed_primary, axis=1, keepdims=True)
            allowed_secondary = secondary_cost[row_ids[:, None], final_allowed]
            final_choice = np.argmin(
                np.where(allowed_primary == minimum_primary, allowed_secondary, np.inf),
                axis=1,
            )
        final = final_allowed[row_ids, final_choice]

    phase_states = np.empty((batch, values.shape[1]), dtype=np.int32)
    phase_states[:, -1] = final
    for phase in range(values.shape[1] - 1, 0, -1):
        width = widths[phase % geometry.V]
        prefixes = 1 << (geometry.L - width)
        prefix = phase_states[:, phase] >> width
        choice = backpointers[phase]
        assert choice is not None
        predecessor_branch = choice[row_ids, prefix].astype(np.int32)
        phase_states[:, phase - 1] = predecessor_branch * prefixes + prefix
    return phase_states.reshape(batch, steps, lanes)


def solve_native_v4(
    target: np.ndarray,
    *,
    tlut: np.ndarray | None = None,
    state_lut: np.ndarray | None = None,
    scales: np.ndarray | Sequence[float] | None = None,
    geometry: NativeQtip25Geometry = NATIVE_QTIP25_GEOMETRY,
    cyclic_warmup_cycles: int = 1,
    trellis_objective: str = "sse",
) -> EncodedNativeQtip25:
    """Reference cyclic Viterbi encoder for homogeneous L16/B/V4 payloads."""
    values = np.asarray(target, dtype=np.float32)
    if values.ndim == 2 and values.shape[1] % geometry.V == 0:
        values = values.reshape(values.shape[0], -1, geometry.V)
    if values.ndim != 3 or values.shape[2] != geometry.V or not values.shape[0]:
        raise ValueError("native QTIP2.5 target must be [rows, steps, 4]")
    if not bool(np.isfinite(values).all()):
        raise ValueError("native QTIP2.5 target must be finite")
    if cyclic_warmup_cycles not in {1, 2}:
        raise ValueError("native QTIP cyclic warmup must use one or two cycles")
    if trellis_objective not in {"sse", "lexicographic_l4"}:
        raise ValueError("native QTIP trellis objective must be sse or lexicographic_l4")
    if sum(value is not None for value in (tlut, state_lut)) != 1:
        raise ValueError("native QTIP solve requires exactly one TLUT or vector LUT")
    edge_lut: np.ndarray | None = None
    if tlut is not None:
        table = np.asarray(tlut)
        if table.shape == (1024,):
            from .banana_v1 import expand_banana_v1_codebook

            edge_lut = expand_banana_v1_codebook(table)
            lut = edge_lut
        else:
            lut = expand_native_v4_tlut(table, geometry=geometry)
    else:
        lut = np.asarray(state_lut, dtype=np.float32)
    expected_shape = (geometry.states,) if edge_lut is not None else (
        geometry.states,
        geometry.V,
    )
    if lut.shape != expected_shape or not bool(np.isfinite(lut).all()):
        raise ValueError("native QTIP expanded LUT shape does not match its codec")

    flattened = values.reshape(values.shape[0], -1)
    if scales is None:
        source_rms = np.sqrt(np.mean(flattened * flattened, axis=1, dtype=np.float32))
        lut_rms = np.float32(np.sqrt(np.mean(lut * lut, dtype=np.float32)))
        row_scales = np.where(source_rms == 0, 1.0, source_rms / lut_rms).astype(
            np.float32
        )
    else:
        row_scales = np.asarray(scales, dtype=np.float32)
        if row_scales.ndim == 0:
            row_scales = np.full(values.shape[0], row_scales, dtype=np.float32)
    if row_scales.shape != (values.shape[0],) or bool(
        np.any(~np.isfinite(row_scales)) or np.any(row_scales <= 0)
    ):
        raise ValueError("native QTIP2.5 scales must be finite positive row values")

    normalized = values / row_scales[:, None, None]
    midpoint = values.shape[1] // 2
    rolled = np.roll(normalized, midpoint, axis=1)
    warmup = np.concatenate([rolled] * cyclic_warmup_cycles, axis=1)
    overlap_step = (cyclic_warmup_cycles - 1) * values.shape[1] + midpoint
    if edge_lut is not None:
        first_phases = _viterbi_native_v5_numpy(
            warmup,
            edge_lut,
            overlap=None,
            trellis_objective=trellis_objective,
            geometry=geometry,
        )
        first_width = native_v5_phase_widths(geometry=geometry)[0]
        overlap = first_phases[:, overlap_step, 0] >> first_width
        phase_states = _viterbi_native_v5_numpy(
            normalized,
            edge_lut,
            overlap=overlap,
            trellis_objective=trellis_objective,
            geometry=geometry,
        )
        states = phase_states[:, :, -1]
        decoded = phase_states
    else:
        if trellis_objective != "sse":
            raise ValueError("lexicographic L4 is available only for the PR31 scalar LUT")
        first = _viterbi_numpy(
            warmup, lut, overlap=None, geometry=geometry
        )
        overlap = first[:, overlap_step] >> geometry.B
        states = _viterbi_numpy(normalized, lut, overlap=overlap, geometry=geometry)
        decoded = states
    packed = pack_native_v4_states(states, geometry=geometry)
    decoded_values = lut[decoded].reshape(flattened.shape) * row_scales[:, None]
    distortion = float(np.sum((decoded_values - flattened) ** 2, dtype=np.float64))
    return EncodedNativeQtip25(
        geometry=geometry,
        shape=(int(flattened.shape[0]), int(flattened.shape[1])),
        states=states,
        packed=packed,
        scales=row_scales,
        distortion=distortion,
    )


def native_v4_lower_from_hessian(
    hessian: np.ndarray, *, regularization_sigma: float = 1e-2
) -> np.ndarray:
    """Derive the normalized 16-column feedback matrix with qtip_batch block-LDL."""
    import torch

    from .qtip_batch import block_ldl_batch

    value = np.asarray(hessian, dtype=np.float32)
    if (
        value.ndim != 2
        or value.shape[0] != value.shape[1]
        or value.shape[0] % 16
        or not bool(np.isfinite(value).all())
    ):
        raise ValueError("native V4 Hessian must be finite square with width divisible by 16")
    sigma = float(regularization_sigma)
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("native V4 Hessian regularization must be finite and nonnegative")
    tensor = torch.from_numpy(np.ascontiguousarray(value.copy())).unsqueeze(0)
    diagonal = tensor.diagonal(dim1=-2, dim2=-1)
    diagonal_mean = diagonal.mean(dim=-1)
    if bool(torch.any(diagonal_mean <= 0)):
        raise ValueError("native V4 Hessian diagonal mean must be positive")
    diagonal.add_(diagonal_mean[:, None] * sigma)
    lower = block_ldl_batch(tensor, 16)[0]
    lower.diagonal().zero_()
    return np.ascontiguousarray(lower.numpy(), dtype=np.float32)


def ldlq_native_v4_matrix(
    transformed: np.ndarray,
    lower: np.ndarray,
    *,
    tlut: np.ndarray,
    geometry: NativeQtip25Geometry = NATIVE_QTIP25_GEOMETRY,
    scale_factors: Sequence[float] = _DEFAULT_SCALE_FACTORS,
    scale_semantics: str = "absolute_unit",
) -> NativeV4MatrixResult:
    """Run reverse-16 LDLQ at unit, RMS-ratio, or explicit relative scale."""
    source = np.asarray(transformed, dtype=np.float32)
    feedback = np.asarray(lower, dtype=np.float32)
    if (
        source.ndim != 2
        or source.shape[0] % 16
        or source.shape[1] % 16
        or feedback.shape != (source.shape[1], source.shape[1])
        or not bool(np.isfinite(source).all() and np.isfinite(feedback).all())
    ):
        raise ValueError(
            "native V4 LDLQ requires a finite 16-tiled matrix and matching square lower"
        )
    if bool(np.any(np.triu(feedback, 1) != 0)):
        raise ValueError("native V4 LDLQ feedback must be lower triangular")
    factors = tuple(float(value) for value in scale_factors)
    if not factors or any(not math.isfinite(value) or value <= 0 for value in factors):
        raise ValueError("native V4 LDLQ scale factors must be finite and positive")
    if scale_semantics not in {"relative_search", "absolute_unit", "rms_ratio"}:
        raise ValueError(
            "native V4 LDLQ scale semantics must be relative_search, absolute_unit, or rms_ratio"
        )
    effective_factors = (
        factors if scale_semantics == "relative_search" else (1.0,)
    )
    table = np.asarray(tlut)
    edge_lut: np.ndarray | None = None
    if table.shape == (1024,):
        from .banana_v1 import expand_banana_v1_codebook

        edge_lut = expand_banana_v1_codebook(table)
        lut = edge_lut
    else:
        lut = expand_native_v4_tlut(table, geometry=geometry)
    source_rms = float(np.sqrt(np.mean(source.astype(np.float64) ** 2)))
    lut_rms = float(np.sqrt(np.mean(lut.astype(np.float64) ** 2)))
    base_scale = 1.0 if source_rms == 0 else source_rms / lut_rms
    row_blocks = source.shape[0] // 16
    column_blocks = source.shape[1] // 16
    packed_bytes = 8 * geometry.B
    best: tuple[float, float, np.ndarray, np.ndarray, np.ndarray] | None = None
    for factor in effective_factors:
        scale = np.float32(1.0 if scale_semantics == "absolute_unit" else base_scale * factor)
        decoded = np.zeros_like(source)
        state_grid = np.empty((row_blocks, column_blocks, 64), dtype=np.int32)
        packed_grid = np.empty(
            (row_blocks, column_blocks, packed_bytes), dtype=np.uint8
        )
        for column_block in range(column_blocks - 1, -1, -1):
            start = column_block * 16
            end = start + 16
            corrected = source[:, start:end].copy()
            if end < source.shape[1]:
                error_right = source[:, end:] - decoded[:, end:]
                corrected += (feedback[end:, start:end].T @ error_right.T).T.astype(
                    np.float32
                )
            tiles = corrected.reshape(row_blocks, 64, geometry.V)
            encoded = solve_native_v4(
                tiles,
                tlut=table,
                scales=np.full(row_blocks, scale, dtype=np.float32),
                geometry=geometry,
            )
            quantized_order = decode_native_v4(
                encoded.packed,
                encoded.scales,
                positions=256,
                tlut=tlut,
                geometry=geometry,
            ).reshape(row_blocks, 256)
            quantized = quantized_order.reshape(source.shape[0], 16)
            decoded[:, start:end] = quantized
            state_grid[:, column_block] = encoded.states
            packed_grid[:, column_block] = encoded.packed
        distortion = float(
            np.sum((decoded.astype(np.float64) - source.astype(np.float64)) ** 2)
        )
        if best is None or distortion < best[0]:
            best = (
                distortion,
                factor,
                decoded.copy(),
                state_grid.copy(),
                packed_grid.copy(),
            )
    assert best is not None
    distortion, selected_factor, decoded, state_grid, packed_grid = best
    tile_count = row_blocks * column_blocks
    return NativeV4MatrixResult(
        decoded=np.ascontiguousarray(decoded),
        states=np.ascontiguousarray(state_grid.reshape(tile_count, 64)),
        packed=np.ascontiguousarray(packed_grid.reshape(tile_count, packed_bytes)),
        scales=np.full(
            tile_count,
            1.0 if scale_semantics == "absolute_unit" else base_scale * selected_factor,
            dtype=np.float32,
        ),
        distortion=distortion,
        scale_factor=1.0 if scale_semantics == "absolute_unit" else base_scale * selected_factor,
        scale_factors=effective_factors,
        feedback_nonzero_count=int(np.count_nonzero(feedback)),
    )


def decode_native_v4(
    packed: np.ndarray,
    scales: np.ndarray,
    *,
    positions: int,
    tlut: np.ndarray,
    geometry: NativeQtip25Geometry = NATIVE_QTIP25_GEOMETRY,
) -> np.ndarray:
    if positions < geometry.V or positions % geometry.V:
        raise ValueError("native QTIP2.5 positions must be positive and divisible by V4")
    words = np.asarray(packed)
    row_scales = np.asarray(scales, dtype=np.float32)
    if row_scales.shape != (words.shape[0],):
        raise ValueError("native QTIP2.5 requires one scale per packed row")
    states = states_from_native_v4_packed(
        words, steps=positions // geometry.V, geometry=geometry
    )
    table = np.asarray(tlut)
    if table.shape == (1024,):
        from .banana_v1 import expand_banana_v1_codebook

        edge_states = native_v5_edge_states(
            np.roll(states, 1, axis=1),
            states & (geometry.branches - 1),
            geometry=geometry,
        )
        decoded = expand_banana_v1_codebook(table)[edge_states].reshape(
            words.shape[0], positions
        )
    else:
        decoded = expand_native_v4_tlut(table, geometry=geometry)[states].reshape(
            words.shape[0], positions
        )
    return decoded * row_scales[:, None]


def decode_native_v4_torch(
    packed: Any,
    scales: Any,
    *,
    positions: int,
    tlut: Any,
    geometry: NativeQtip25Geometry = NATIVE_QTIP25_GEOMETRY,
) -> Any:
    """Decode on the packed tensor's Torch device; CUDA never falls back to CPU."""
    import torch

    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError("Torch native QTIP2.5 packed payload must be uint8 [rows,bytes]")
    if positions < geometry.V or positions % geometry.V:
        raise ValueError("Torch native QTIP2.5 positions must be divisible by V4")
    steps = positions // geometry.V
    bit_count = steps * geometry.B
    if packed.shape[1] != math.ceil(bit_count / 8):
        raise ValueError("Torch native QTIP2.5 packed byte shape drift")
    if scales.shape != (packed.shape[0],) or scales.device != packed.device:
        raise ValueError("Torch native QTIP2.5 scales must be one row value on-device")
    table_shape = tuple(tlut.shape)
    if table_shape not in {
        (1 << geometry.tlut_bits, 2),
        (1024,),
    } or tlut.device != packed.device:
        raise ValueError(
            "Torch native QTIP LUT must be V4 Q9xV2 or compact PR31 [1024] on-device"
        )

    bit_starts = torch.arange(steps, device=packed.device, dtype=torch.int64) * geometry.B
    byte_starts = bit_starts >> 3
    bit_offsets = bit_starts & 7
    cyclic = torch.cat((packed, packed[:, :2]), dim=1).to(torch.int32)
    word = (
        (cyclic[:, byte_starts] << 16)
        | (cyclic[:, byte_starts + 1] << 8)
        | cyclic[:, byte_starts + 2]
    )
    state_tensor = (word >> (8 - bit_offsets)) & (geometry.states - 1)

    if table_shape == (1024,):
        from .banana_v1 import BANANA_V1_MULTIPLIER, BANANA_V1_OFFSET

        predecessor = torch.roll(state_tensor, 1, dims=1)
        branch = state_tensor & (geometry.branches - 1)
        edge_state = predecessor
        consumed = 0
        lanes = []
        for width in native_v5_phase_widths(geometry=geometry):
            consumed += width
            chunk = (branch >> (geometry.B - consumed)) & ((1 << width) - 1)
            edge_state = ((edge_state << width) | chunk) & (geometry.states - 1)
            level = (
                (edge_state * BANANA_V1_MULTIPLIER + BANANA_V1_OFFSET) & 0xFFFF
            ) >> 6
            lanes.append(tlut.index_select(0, level.reshape(-1)).reshape_as(level))
        decoded = torch.stack(lanes, dim=2).reshape(packed.shape[0], positions)
    else:
        base = torch.arange(geometry.states, device=packed.device, dtype=torch.int64)

        def pair(index: Any) -> Any:
            hashed = (index + 1) * index
            sign = 1 - 2 * ((hashed >> 15) & 1)
            lookup = (hashed >> (16 - geometry.tlut_bits - 1)) & (
                (1 << geometry.tlut_bits) - 1
            )
            result = tlut.index_select(0, lookup).clone()
            result[:, 0] *= sign
            return result

        half = geometry.L // 2
        rotated = ((base << half) | (base >> half)) & (geometry.states - 1)
        expanded = torch.cat((pair(base), pair(rotated)), dim=1)
        decoded = expanded.index_select(0, state_tensor.reshape(-1)).reshape(
            packed.shape[0], positions
        )
    return decoded * scales.reshape(-1, 1)


if triton is not None:

    @triton.jit
    def _native_v4_prefix_viterbi(
        x_ptr,
        lut_ptr,
        overlap_ptr,
        scratch_ptr,
        best_state_ptr,
        states_ptr,
        batch,
        steps: tl.constexpr,
        has_overlap: tl.constexpr,
        prefixes: tl.constexpr,
        branches: tl.constexpr,
        transition_bits: tl.constexpr,
        lanes: tl.constexpr,
        state_count: tl.constexpr,
        overlap_only: tl.constexpr,
        capture_step: tl.constexpr,
    ):
        """One exact homogeneous L16/B/V4 sequence per CTA.

        Branches are reduced in 512xprefix tiles instead of a scalar 4,096-branch
        loop.  For QTIP3 B12 this exposes 8,192 independent state errors per
        iteration while preserving the original strict-<, first-branch tie
        order and exact traceback bytes.
        """
        sequence = tl.program_id(0)
        prefix = tl.arange(0, prefixes)
        branch_offset = tl.arange(0, 512)[:, None]
        prefix_grid = prefix[None, :]
        best = tl.full((prefixes,), float("inf"), tl.float32)
        chosen = tl.zeros((prefixes,), tl.int32)
        expected_overlap = tl.load(overlap_ptr + sequence).to(tl.int32)

        for branch_base in tl.range(0, branches, 512):
            branch_grid = branch_base + branch_offset
            state = branch_grid * prefixes + prefix_grid
            candidate = tl.zeros((512, prefixes), tl.float32)
            for lane in tl.static_range(0, lanes):
                value = tl.load(x_ptr + lane * batch + sequence).to(tl.float32)
                code = tl.load(lut_ptr + lane * state_count + state).to(tl.float32)
                candidate += (code - value) * (code - value)
            if has_overlap:
                candidate = tl.where(
                    (state >> transition_bits) == expected_overlap,
                    candidate,
                    float("inf"),
                )
            tile_best = tl.min(candidate, axis=0)
            tile_branch = tl.argmin(candidate, axis=0).to(tl.int32) + branch_base
            take = tile_best < best
            best = tl.where(take, tile_best, best)
            chosen = tl.where(take, tile_branch * prefixes + prefix, chosen)
        base = sequence * prefixes
        tl.store(scratch_ptr + base + prefix, best)
        if not overlap_only:
            tl.store(best_state_ptr + base + prefix, chosen)
        tl.debug_barrier()

        step = 1
        while step < steps:
            previous = ((step - 1) & 1) * batch * prefixes + base
            current = (step & 1) * batch * prefixes + base
            best = tl.full((prefixes,), float("inf"), tl.float32)
            chosen = tl.zeros((prefixes,), tl.int32)
            for branch_base in tl.range(0, branches, 512):
                branch_grid = branch_base + branch_offset
                state = branch_grid * prefixes + prefix_grid
                candidate = tl.load(
                    scratch_ptr + previous + (state >> transition_bits)
                )
                for lane in tl.static_range(0, lanes):
                    value = tl.load(
                        x_ptr + (step * lanes + lane) * batch + sequence
                    ).to(tl.float32)
                    code = tl.load(
                        lut_ptr + lane * state_count + state
                    ).to(tl.float32)
                    candidate += (code - value) * (code - value)
                tile_best = tl.min(candidate, axis=0)
                tile_branch = tl.argmin(candidate, axis=0).to(tl.int32) + branch_base
                take = tile_best < best
                best = tl.where(take, tile_best, best)
                chosen = tl.where(take, tile_branch * prefixes + prefix, chosen)
            tl.store(scratch_ptr + current + prefix, best)
            if not overlap_only:
                tl.store(
                    best_state_ptr + step * batch * prefixes + base + prefix,
                    chosen,
                )
            elif step >= capture_step:
                tl.store(
                    best_state_ptr
                    + (step - capture_step) * batch * prefixes
                    + base
                    + prefix,
                    chosen,
                )
            tl.debug_barrier()
            step += 1

        traceback_prefix = (
            expected_overlap if has_overlap else tl.argmin(best, axis=0).to(tl.int32)
        )
        if overlap_only:
            for back_step in tl.static_range(steps - 1, capture_step - 1, -1):
                state = tl.load(
                    best_state_ptr
                    + (back_step - capture_step) * batch * prefixes
                    + base
                    + traceback_prefix
                ).to(tl.int32)
                traceback_prefix = state >> transition_bits
            tl.store(states_ptr + sequence, traceback_prefix)
        else:
            for back_step in tl.static_range(steps - 1, -1, -1):
                state = tl.load(
                    best_state_ptr
                    + back_step * batch * prefixes
                    + base
                    + traceback_prefix
                ).to(tl.int32)
                tl.store(states_ptr + back_step * batch + sequence, state)
                traceback_prefix = state >> transition_bits


_NATIVE_V4_CUDA_CPP = r"""
#include <torch/extension.h>
torch::Tensor qtip3_viterbi_cuda(
    torch::Tensor x, torch::Tensor lut, torch::Tensor overlap,
    bool has_overlap, int64_t capture_step, bool overlap_only);
"""

_NATIVE_V4_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

namespace {
constexpr int PREFIXES = 16;
constexpr int STATES = 65536;
constexpr int BRANCH_BITS = 12;
constexpr int LANES = 4;
constexpr int THREADS = 256;

__global__ void qtip3_viterbi_kernel(
    const float* __restrict__ x,
    const float* __restrict__ lut,
    const int* __restrict__ overlap,
    int* __restrict__ choices,
    int* __restrict__ output,
    int batch,
    int steps,
    int capture_step,
    bool has_overlap,
    bool overlap_only) {
  const int sequence = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float thread_cost[THREADS];
  __shared__ int thread_state[THREADS];
  __shared__ float costs[2][PREFIXES];

  // Q1 retains 4,096 prefixes, so one block cannot assign one thread per
  // prefix.  Preserve the same state order and strict-< tie rule while each
  // thread owns a strided prefix subset; Q3/Q4 keep the proven reduction below.
  if (PREFIXES > THREADS) {
    for (int step = 0; step < steps; ++step) {
      const float value0 = x[(step * LANES + 0) * batch + sequence];
      const float value1 = x[(step * LANES + 1) * batch + sequence];
      const float value2 = x[(step * LANES + 2) * batch + sequence];
      const float value3 = x[(step * LANES + 3) * batch + sequence];
      for (int prefix = tid; prefix < PREFIXES; prefix += THREADS) {
        float best = __int_as_float(0x7f800000);
        int best_state = 0;
        for (int branch = 0; branch < (1 << BRANCH_BITS); ++branch) {
          const int state = branch * PREFIXES + prefix;
          const int predecessor = state >> BRANCH_BITS;
          if (has_overlap && step == 0 && predecessor != overlap[sequence]) {
            continue;
          }
          float candidate =
              step == 0 ? 0.0f : costs[(step - 1) & 1][predecessor];
          const float4 code = reinterpret_cast<const float4*>(lut)[state];
          const float delta0 = code.x - value0;
          const float delta1 = code.y - value1;
          const float delta2 = code.z - value2;
          const float delta3 = code.w - value3;
          candidate += delta0 * delta0 + delta1 * delta1 +
                       delta2 * delta2 + delta3 * delta3;
          if (candidate < best ||
              (candidate == best && state < best_state)) {
            best = candidate;
            best_state = state;
          }
        }
        costs[step & 1][prefix] = best;
        if (step >= capture_step) {
          choices[((step - capture_step) * batch + sequence) * PREFIXES +
                  prefix] = best_state;
        }
      }
      __syncthreads();
    }
    if (tid == 0) {
      int prefix = has_overlap ? overlap[sequence] : 0;
      if (!has_overlap) {
        float best = costs[(steps - 1) & 1][0];
        for (int candidate_prefix = 1; candidate_prefix < PREFIXES;
             ++candidate_prefix) {
          const float candidate = costs[(steps - 1) & 1][candidate_prefix];
          if (candidate < best) {
            best = candidate;
            prefix = candidate_prefix;
          }
        }
      }
      for (int step = steps - 1; step >= capture_step; --step) {
        const int state =
            choices[((step - capture_step) * batch + sequence) * PREFIXES +
                    prefix];
        if (!overlap_only) {
          output[step * batch + sequence] = state;
        }
        prefix = state >> BRANCH_BITS;
      }
      if (overlap_only) {
        output[sequence] = prefix;
      }
    }
    return;
  }

  for (int step = 0; step < steps; ++step) {
    const float value0 = x[(step * LANES + 0) * batch + sequence];
    const float value1 = x[(step * LANES + 1) * batch + sequence];
    const float value2 = x[(step * LANES + 2) * batch + sequence];
    const float value3 = x[(step * LANES + 3) * batch + sequence];
    float local_cost = __int_as_float(0x7f800000);
    int local_state = 0;
    for (int state = tid; state < STATES; state += THREADS) {
      const int predecessor = state >> BRANCH_BITS;
      if (has_overlap && step == 0 && predecessor != overlap[sequence]) {
        continue;
      }
      float candidate = step == 0 ? 0.0f : costs[(step - 1) & 1][predecessor];
      const float4 code = reinterpret_cast<const float4*>(lut)[state];
      const float delta0 = code.x - value0;
      const float delta1 = code.y - value1;
      const float delta2 = code.z - value2;
      const float delta3 = code.w - value3;
      candidate += delta0 * delta0;
      candidate += delta1 * delta1;
      candidate += delta2 * delta2;
      candidate += delta3 * delta3;
      if (candidate < local_cost ||
          (candidate == local_cost && state < local_state)) {
        local_cost = candidate;
        local_state = state;
      }
    }
    thread_cost[tid] = local_cost;
    thread_state[tid] = local_state;
    __syncthreads();
    if (tid < PREFIXES) {
      float best = __int_as_float(0x7f800000);
      int best_state = 0;
#pragma unroll
      for (int group = 0; group < THREADS / PREFIXES; ++group) {
        const int index = tid + group * PREFIXES;
        const float candidate = thread_cost[index];
        const int state = thread_state[index];
        if (candidate < best || (candidate == best && state < best_state)) {
          best = candidate;
          best_state = state;
        }
      }
      costs[step & 1][tid] = best;
      if (step >= capture_step) {
        choices[((step - capture_step) * batch + sequence) * PREFIXES + tid] =
            best_state;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    int prefix = has_overlap ? overlap[sequence] : 0;
    if (!has_overlap) {
      float best = costs[(steps - 1) & 1][0];
      for (int candidate_prefix = 1; candidate_prefix < PREFIXES;
           ++candidate_prefix) {
        const float candidate = costs[(steps - 1) & 1][candidate_prefix];
        if (candidate < best) {
          best = candidate;
          prefix = candidate_prefix;
        }
      }
    }
    for (int step = steps - 1; step >= capture_step; --step) {
      const int state =
          choices[((step - capture_step) * batch + sequence) * PREFIXES + prefix];
      if (!overlap_only) {
        output[step * batch + sequence] = state;
      }
      prefix = state >> BRANCH_BITS;
    }
    if (overlap_only) {
      output[sequence] = prefix;
    }
  }
}
} // namespace

torch::Tensor qtip3_viterbi_cuda(
    torch::Tensor x, torch::Tensor lut, torch::Tensor overlap,
    bool has_overlap, int64_t capture_step, bool overlap_only) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.dim() == 2,
              "qtip3 x must be CUDA float32 [steps*4,batch]");
  TORCH_CHECK(lut.is_cuda() && lut.scalar_type() == torch::kFloat32 &&
              lut.sizes() == torch::IntArrayRef({STATES, LANES}),
              "qtip3 lut must be CUDA float32 [65536,4]");
  TORCH_CHECK(overlap.is_cuda() && overlap.scalar_type() == torch::kInt32 &&
              overlap.dim() == 1 && overlap.size(0) == x.size(1),
              "qtip3 overlap must be CUDA int32 [batch]");
  TORCH_CHECK(x.size(0) % LANES == 0, "qtip3 step geometry drift");
  const int steps = static_cast<int>(x.size(0) / LANES);
  const int batch = static_cast<int>(x.size(1));
  TORCH_CHECK(capture_step >= 0 && capture_step < steps, "qtip3 capture step drift");
  auto options = torch::TensorOptions().device(x.device()).dtype(torch::kInt32);
  auto choices = torch::empty({steps - capture_step, batch, PREFIXES}, options);
  auto output = overlap_only ? torch::empty({batch}, options)
                             : torch::empty({steps, batch}, options);
  c10::cuda::CUDAGuard guard(x.device());
  qtip3_viterbi_kernel<<<batch, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), lut.data_ptr<float>(), overlap.data_ptr<int>(),
      choices.data_ptr<int>(), output.data_ptr<int>(), batch, steps,
      static_cast<int>(capture_step), has_overlap, overlap_only);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
"""


def _native_v4_cuda_source(geometry: NativeQtip25Geometry) -> str:
    """Specialize the proven V7 recurrence for one exact ladder geometry."""

    return _NATIVE_V4_CUDA_SOURCE.replace(
        "constexpr int PREFIXES = 16;",
        f"constexpr int PREFIXES = {geometry.prefixes};",
    ).replace(
        "constexpr int BRANCH_BITS = 12;",
        f"constexpr int BRANCH_BITS = {geometry.B};",
    )


@lru_cache(maxsize=None)
def _load_native_v4_cuda_extension(
    geometry: NativeQtip25Geometry = NATIVE_QTIP25_GEOMETRY,
) -> Any:
    """Build the exact register-bounded CUDA recurrence once per tier geometry."""
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name=f"banana_smasher_qtip_v7_viterbi_b{geometry.B}_a26",
        cpp_sources=_NATIVE_V4_CUDA_CPP,
        cuda_sources=_native_v4_cuda_source(geometry),
        functions=["qtip3_viterbi_cuda"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        with_cuda=True,
        verbose=False,
    )


def _native_v4_cuda_pass(
    x: Any,
    state_lut: Any,
    overlap: Any | None,
    *,
    geometry: NativeQtip25Geometry = NATIVE_QTIP25_GEOMETRY,
) -> Any:
    import torch

    if triton is None:
        raise RuntimeError("native QTIP2.5 CUDA solve requires the solve extra")
    if (
        not x.is_cuda
        or x.dtype != torch.float32
        or x.ndim != 2
        or x.shape[0] % geometry.V
    ):
        raise ValueError("native QTIP V4 CUDA input must be float32 [steps*4,batch]")
    if (
        not state_lut.is_cuda
        or state_lut.device != x.device
        or state_lut.dtype != torch.float32
        or tuple(state_lut.shape) != (geometry.states, geometry.V)
    ):
        raise ValueError("native QTIP2.5 CUDA state LUT must be float32 [65536,4]")
    steps = int(x.shape[0]) // geometry.V
    batch = int(x.shape[1])
    if steps * geometry.B < geometry.L or batch < 1:
        raise ValueError("native QTIP2.5 CUDA solve requires at least two transitions")
    if overlap is not None and (
        not overlap.is_cuda
        or overlap.device != x.device
        or overlap.dtype not in {torch.int32, torch.int64}
        or tuple(overlap.shape) != (batch,)
        or bool(((overlap < 0) | (overlap >= geometry.prefixes)).any())
    ):
        raise ValueError("native QTIP2.5 CUDA overlap must be one prefix in [0,64) per row")
    source = x.contiguous()
    lut = state_lut.contiguous()
    overlap_arg = (
        overlap.to(torch.int32).contiguous()
        if overlap is not None
        else torch.zeros(batch, device=x.device, dtype=torch.int32)
    )
    extension = _load_native_v4_cuda_extension(geometry)
    return extension.qtip3_viterbi_cuda(
        source,
        lut,
        overlap_arg,
        overlap is not None,
        0,
        False,
    )


def _native_v4_cuda_warmup_overlap(
    x: Any,
    state_lut: Any,
    capture_step: int,
    *,
    geometry: NativeQtip25Geometry = NATIVE_QTIP25_GEOMETRY,
) -> Any:
    """Return the exact warmup prefix while retaining only its traceback suffix."""

    import torch

    if triton is None:
        raise RuntimeError("native QTIP2.5 CUDA solve requires the solve extra")
    steps = int(x.shape[0]) // geometry.V
    batch = int(x.shape[1])
    if (
        not x.is_cuda
        or x.dtype != torch.float32
        or x.ndim != 2
        or x.shape[0] % geometry.V
        or tuple(state_lut.shape) != (geometry.states, geometry.V)
        or capture_step < 0
        or capture_step >= steps - 1
    ):
        raise ValueError("native QTIP compact warmup geometry is invalid")
    source = x.contiguous()
    lut = state_lut.contiguous()
    overlap_arg = torch.zeros(batch, device=x.device, dtype=torch.int32)
    extension = _load_native_v4_cuda_extension(geometry)
    return extension.qtip3_viterbi_cuda(
        source,
        lut,
        overlap_arg,
        False,
        capture_step,
        True,
    )


def _native_v5_cuda_pass(
    target: Any,
    scalar_lut: Any,
    overlap: Any | None,
    *,
    trellis_objective: str = "sse",
    geometry: NativeQtip25Geometry = NATIVE_QTIP25_GEOMETRY,
) -> Any:
    """Run the exact PR31 recurrence with variable-width on-device phases."""

    import torch

    batch, steps, lanes = target.shape
    if trellis_objective not in {"sse", "lexicographic_l4"}:
        raise ValueError("native V5 trellis objective must be sse or lexicographic_l4")
    widths = native_v5_phase_widths(geometry=geometry)
    values = target.reshape(batch, steps * lanes)
    primary_cost = None
    secondary_cost = None
    backpointers = []
    row_ids = torch.arange(batch, device=target.device)
    for phase in range(values.shape[1]):
        width = widths[phase % geometry.V]
        prefixes = 1 << (geometry.L - width)
        branches = 1 << width
        errors = (scalar_lut.unsqueeze(0) - values[:, phase, None]).square()
        primary_errors = errors if trellis_objective == "sse" else errors.square()
        if phase == 0:
            primary_cost = primary_errors
            secondary_cost = errors if trellis_objective == "lexicographic_l4" else None
            if overlap is not None:
                allowed = (overlap[:, None].to(torch.int64) << width) + torch.arange(
                    branches, device=target.device
                )
                masked = torch.full_like(primary_cost, torch.inf)
                masked.scatter_(1, allowed, primary_cost.gather(1, allowed))
                primary_cost = masked
                if secondary_cost is not None:
                    masked_secondary = torch.full_like(secondary_cost, torch.inf)
                    masked_secondary.scatter_(
                        1, allowed, secondary_cost.gather(1, allowed)
                    )
                    secondary_cost = masked_secondary
            backpointers.append(None)
            continue

        assert primary_cost is not None
        primary_options = primary_cost.reshape(batch, branches, prefixes)
        best_secondary = None
        if secondary_cost is None:
            best_primary, choice = primary_options.min(dim=1)
        else:
            secondary_options = secondary_cost.reshape(batch, branches, prefixes)
            minimum_primary = primary_options.min(dim=1, keepdim=True).values
            tied_secondary = torch.where(
                primary_options == minimum_primary,
                secondary_options,
                torch.inf,
            )
            choice = tied_secondary.argmin(dim=1)
            gather_choice = choice.unsqueeze(1)
            best_primary = primary_options.gather(1, gather_choice)[:, 0, :]
            best_secondary = secondary_options.gather(1, gather_choice)[:, 0, :]
        primary_cost = primary_errors + best_primary.repeat_interleave(
            branches, dim=1
        )
        if secondary_cost is not None:
            assert best_secondary is not None
            secondary_cost = errors + best_secondary.repeat_interleave(
                branches, dim=1
            )
        backpointers.append(choice.to(torch.uint8))

    assert primary_cost is not None
    if overlap is None:
        if secondary_cost is None:
            final = primary_cost.argmin(dim=1).to(torch.int32)
        else:
            minimum_primary = primary_cost.min(dim=1, keepdim=True).values
            final = torch.where(
                primary_cost == minimum_primary, secondary_cost, torch.inf
            ).argmin(dim=1).to(torch.int32)
    else:
        first_width = widths[0]
        first_prefixes = 1 << (geometry.L - first_width)
        final_allowed = overlap[:, None].to(torch.int64) + torch.arange(
            1 << first_width, device=target.device
        )[None, :] * first_prefixes
        allowed_primary = primary_cost.gather(1, final_allowed)
        if secondary_cost is None:
            final_choice = allowed_primary.argmin(dim=1)
        else:
            minimum_primary = allowed_primary.min(dim=1, keepdim=True).values
            allowed_secondary = secondary_cost.gather(1, final_allowed)
            final_choice = torch.where(
                allowed_primary == minimum_primary, allowed_secondary, torch.inf
            ).argmin(dim=1)
        final = final_allowed[row_ids, final_choice].to(torch.int32)

    phase_states = torch.empty(
        (batch, values.shape[1]), device=target.device, dtype=torch.int32
    )
    phase_states[:, -1] = final
    for phase in range(values.shape[1] - 1, 0, -1):
        width = widths[phase % geometry.V]
        prefixes = 1 << (geometry.L - width)
        prefix = phase_states[:, phase] >> width
        choice = backpointers[phase]
        assert choice is not None
        predecessor_branch = choice.gather(1, prefix[:, None].to(torch.int64))[:, 0].to(
            torch.int32
        )
        phase_states[:, phase - 1] = predecessor_branch * prefixes + prefix
    return phase_states.reshape(batch, steps, lanes)


def _solve_native_v5_cuda(
    target: Any,
    scalar_lut: Any,
    *,
    geometry: NativeQtip25Geometry = NATIVE_QTIP25_GEOMETRY,
    cyclic_warmup_cycles: int = 1,
    trellis_objective: str = "sse",
) -> Any:
    import torch

    midpoint = int(target.shape[1]) // 2
    rolled = target.roll(midpoint, dims=1)
    warmup = torch.cat([rolled] * cyclic_warmup_cycles, dim=1)
    first = _native_v5_cuda_pass(
        warmup,
        scalar_lut,
        None,
        trellis_objective=trellis_objective,
        geometry=geometry,
    )
    first_width = native_v5_phase_widths(geometry=geometry)[0]
    overlap_step = (cyclic_warmup_cycles - 1) * int(target.shape[1]) + midpoint
    overlap = first[:, overlap_step, 0] >> first_width
    phases = _native_v5_cuda_pass(
        target,
        scalar_lut,
        overlap,
        trellis_objective=trellis_objective,
        geometry=geometry,
    )
    return phases[:, :, -1].contiguous()


_CYCLIC_FIXED_POINT_COUNTERS = {"attempts": 0, "accepted": 0, "fallbacks": 0}


def native_v4_cyclic_fast_path_counters() -> dict[str, int]:
    """Return process counters for exact compact warmup traceback engagement."""

    return dict(_CYCLIC_FIXED_POINT_COUNTERS)


def solve_native_v4_cuda(
    target: Any,
    *,
    state_lut: Any,
    geometry: NativeQtip25Geometry = NATIVE_QTIP25_GEOMETRY,
    cyclic_warmup_cycles: int = 1,
    trellis_objective: str = "sse",
    cyclic_fixed_point_fast_path: bool = False,
) -> Any:
    """Exact full-branch CUDA solve for ``[rows,steps,4]`` transformed targets."""
    import torch

    if (
        not target.is_cuda
        or target.dtype != torch.float32
        or target.ndim != 3
        or target.shape[2] != geometry.V
    ):
        raise ValueError("native QTIP2.5 CUDA target must be float32 [rows,steps,4]")
    if cyclic_warmup_cycles not in {1, 2}:
        raise ValueError("native QTIP CUDA cyclic warmup must use one or two cycles")
    if trellis_objective not in {"sse", "lexicographic_l4"}:
        raise ValueError("native QTIP trellis objective must be sse or lexicographic_l4")
    if (
        state_lut.is_cuda
        and state_lut.device == target.device
        and state_lut.dtype == torch.float32
        and tuple(state_lut.shape) == (geometry.states,)
    ):
        return _solve_native_v5_cuda(
            target,
            state_lut,
            geometry=geometry,
            cyclic_warmup_cycles=cyclic_warmup_cycles,
            trellis_objective=trellis_objective,
        )
    if trellis_objective != "sse":
        raise ValueError("lexicographic L4 is available only for the PR31 scalar LUT")
    midpoint = int(target.shape[1]) // 2
    rolled = torch.roll(target, midpoint, dims=1)
    warmup = torch.cat([rolled] * cyclic_warmup_cycles, dim=1)
    overlap_step = (cyclic_warmup_cycles - 1) * int(target.shape[1]) + midpoint
    if cyclic_fixed_point_fast_path and cyclic_warmup_cycles == 2:
        _CYCLIC_FIXED_POINT_COUNTERS["attempts"] += 1
        overlap = _native_v4_cuda_warmup_overlap(
            warmup.permute(1, 2, 0).reshape(-1, target.shape[0]),
            state_lut,
            overlap_step,
            geometry=geometry,
        )
        _CYCLIC_FIXED_POINT_COUNTERS["accepted"] += 1
        return _native_v4_cuda_pass(
            target.permute(1, 2, 0).reshape(-1, target.shape[0]),
            state_lut,
            overlap,
            geometry=geometry,
        ).transpose(0, 1).contiguous()
    first = _native_v4_cuda_pass(
        warmup.permute(1, 2, 0).reshape(-1, target.shape[0]),
        state_lut,
        None,
        geometry=geometry,
    )
    overlap = first[overlap_step] >> geometry.B
    return _native_v4_cuda_pass(
        target.permute(1, 2, 0).reshape(-1, target.shape[0]),
        state_lut,
        overlap,
        geometry=geometry,
    ).transpose(0, 1).contiguous()


def native_v4_wire_accounting(
    *,
    position_count: int,
    sequence_count: int = 1,
    transform_bytes: int = 0,
    scale_bytes: int = 0,
    shared_tlut_bytes: int = 0,
    routing_bytes: int = 0,
    alignment_bytes: int = 0,
    geometry: NativeQtip25Geometry = NATIVE_QTIP25_GEOMETRY,
) -> dict[str, int | float | str | bool]:
    """Account exact integer code bits separately from every physical overhead."""
    count = int(position_count)
    if count < 0 or count % geometry.V:
        raise ValueError("native QTIP2.5 position count must be divisible by V4")
    sequences = int(sequence_count)
    transitions = count // geometry.V
    if sequences < 1 or transitions % sequences:
        raise ValueError("native QTIP2.5 transitions must divide evenly across sequences")
    transform = int(transform_bytes)
    scale = int(scale_bytes)
    shared = int(shared_tlut_bytes)
    routing = int(routing_bytes)
    alignment = int(alignment_bytes)
    if min(transform, scale, shared, routing, alignment) < 0:
        raise ValueError("native QTIP2.5 byte counts must be nonnegative")
    code_bits_numerator = count * geometry.B
    if code_bits_numerator % geometry.V:
        raise ValueError("native QTIP V4 code bits must be integral")
    code_bits = code_bits_numerator // geometry.V
    code_bytes = sequences * math.ceil(
        ((transitions // sequences) * geometry.B) / 8
    )
    auxiliary = transform + scale
    logical = code_bytes + auxiliary + routing + alignment

    def bpw(byte_count: int) -> float:
        return 0.0 if count == 0 else byte_count * 8.0 / count

    return {
        "codec_form": f"native_l{geometry.L}_b{geometry.B}_v{geometry.V}",
        "rate_num": geometry.rate_num,
        "rate_den": geometry.rate_den,
        "L": geometry.L,
        "B": geometry.B,
        "V": geometry.V,
        "phase_count": 1,
        "unique_transition_bits_per_payload": 1,
        "alternation": False,
        "member_averaging": False,
        "position_count": count,
        "sequence_count": sequences,
        "code_bits": code_bits,
        "code_payload_bytes": code_bytes,
        "code_padding_bits": code_bytes * 8 - code_bits,
        "code_alignment_padding_bytes": alignment,
        "selected_indices_bytes": code_bytes,
        "assignment_map_bytes": 0,
        "routing_bytes": routing,
        "transform_bytes": transform,
        "scale_bytes": scale,
        "auxiliary_bytes": auxiliary,
        "logical_expert_plane_bytes": logical,
        "logical_expert_plane_wire_bytes": logical,
        "deduplicated_shared_tlut_bytes": shared,
        "full_wire_bytes": logical + shared,
        "code_bpw": geometry.rate_num / geometry.rate_den if count else 0.0,
        "auxiliary_bpw": bpw(auxiliary + routing + alignment),
        "logical_expert_plane_bpw": bpw(logical),
    }


__all__ = [
    "EncodedNativeQtip25",
    "NATIVE_QTIP25_GEOMETRY",
    "NativeV4MatrixResult",
    "NativeQtip25Geometry",
    "decode_native_v4",
    "decode_native_v4_torch",
    "expand_native_v4_tlut",
    "ldlq_native_v4_matrix",
    "native_v4_lower_from_hessian",
    "native_v4_wire_accounting",
    "native_v4_geometry",
    "native_v4_cyclic_fast_path_counters",
    "native_v5_edge_states",
    "native_v5_phase_widths",
    "pack_native_v4_states",
    "solve_native_v4",
    "solve_native_v4_cuda",
    "states_from_native_v4_packed",
]
