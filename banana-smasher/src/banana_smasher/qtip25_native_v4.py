"""Homogeneous native QTIP2.5 with one L16/B10/V4 trellis geometry.

The codec appends ten transition bits and emits four scalar weights per state,
so every payload has the exact rational rate B/V = 10/4 = 5/2.  The compact
shared 512x2 QTIP TLUT is reused for both halves of the V4 state codebook; the
second half applies the same public quantlut_sym mapping to a byte-rotated state.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
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


def solve_native_v4(
    target: np.ndarray,
    *,
    tlut: np.ndarray | None = None,
    state_lut: np.ndarray | None = None,
    scales: np.ndarray | Sequence[float] | None = None,
    geometry: NativeQtip25Geometry = NATIVE_QTIP25_GEOMETRY,
) -> EncodedNativeQtip25:
    """Reference cyclic Viterbi encoder for homogeneous L16/B/V4 payloads."""
    values = np.asarray(target, dtype=np.float32)
    if values.ndim == 2 and values.shape[1] % geometry.V == 0:
        values = values.reshape(values.shape[0], -1, geometry.V)
    if values.ndim != 3 or values.shape[2] != geometry.V or not values.shape[0]:
        raise ValueError("native QTIP2.5 target must be [rows, steps, 4]")
    if not bool(np.isfinite(values).all()):
        raise ValueError("native QTIP2.5 target must be finite")
    if (tlut is None) == (state_lut is None):
        raise ValueError("native QTIP2.5 solve requires exactly one TLUT or expanded LUT")
    lut = (
        expand_native_v4_tlut(np.asarray(tlut), geometry=geometry)
        if tlut is not None
        else np.asarray(state_lut, dtype=np.float32)
    )
    if lut.shape != (geometry.states, geometry.V) or not bool(np.isfinite(lut).all()):
        raise ValueError("native QTIP2.5 expanded LUT must be finite [65536,4]")

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
    first = _viterbi_numpy(
        np.roll(normalized, midpoint, axis=1), lut, overlap=None, geometry=geometry
    )
    overlap = first[:, midpoint] >> geometry.B
    states = _viterbi_numpy(normalized, lut, overlap=overlap, geometry=geometry)
    packed = pack_native_v4_states(states, geometry=geometry)
    decoded = lut[states].reshape(flattened.shape) * row_scales[:, None]
    distortion = float(np.sum((decoded - flattened) ** 2, dtype=np.float64))
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
) -> NativeV4MatrixResult:
    """Run bounded global-scale search with reverse 16-column LDLQ feedback."""
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
    lut = expand_native_v4_tlut(tlut, geometry=geometry)
    source_rms = float(np.sqrt(np.mean(source.astype(np.float64) ** 2)))
    lut_rms = float(np.sqrt(np.mean(lut.astype(np.float64) ** 2)))
    base_scale = 1.0 if source_rms == 0 else source_rms / lut_rms
    row_blocks = source.shape[0] // 16
    column_blocks = source.shape[1] // 16
    packed_bytes = 8 * geometry.B
    best: tuple[float, float, np.ndarray, np.ndarray, np.ndarray] | None = None
    for factor in factors:
        scale = np.float32(base_scale * factor)
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
            tiles = corrected.reshape(row_blocks, 16, 16).reshape(
                row_blocks, 64, geometry.V
            )
            encoded = solve_native_v4(
                tiles,
                state_lut=lut,
                scales=np.full(row_blocks, scale, dtype=np.float32),
                geometry=geometry,
            )
            quantized = decode_native_v4(
                encoded.packed,
                encoded.scales,
                positions=256,
                tlut=tlut,
                geometry=geometry,
            ).reshape(source.shape[0], 16)
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
            tile_count, base_scale * selected_factor, dtype=np.float32
        ),
        distortion=distortion,
        scale_factor=base_scale * selected_factor,
        scale_factors=factors,
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
    decoded = expand_native_v4_tlut(tlut, geometry=geometry)[states].reshape(
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
    if tlut.shape != (1 << geometry.tlut_bits, 2) or tlut.device != packed.device:
        raise ValueError("Torch native QTIP2.5 TLUT must be Q9xV2 on-device")

    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.int64)
    all_bits = ((packed.to(torch.int64).unsqueeze(-1) >> shifts) & 1).reshape(
        packed.shape[0], -1
    )
    bits = all_bits[:, :bit_count]
    stream = torch.cat((bits, bits[:, : geometry.L - geometry.B]), dim=1)
    powers_l = 1 << torch.arange(
        geometry.L - 1, -1, -1, device=packed.device, dtype=torch.int64
    )
    first = torch.sum(stream[:, : geometry.L] * powers_l, dim=1)
    states = [first]
    powers_b = 1 << torch.arange(
        geometry.B - 1, -1, -1, device=packed.device, dtype=torch.int64
    )
    mask = geometry.states - 1
    for step in range(1, steps):
        start = geometry.L + (step - 1) * geometry.B
        branch = torch.sum(stream[:, start : start + geometry.B] * powers_b, dim=1)
        states.append(((states[-1] << geometry.B) & mask) + branch)
    state_tensor = torch.stack(states, dim=1)

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
    ):
        """One exact homogeneous L16/B/V4 sequence per CTA."""
        sequence = tl.program_id(0)
        prefix = tl.arange(0, prefixes)
        best = tl.full((prefixes,), float("inf"), tl.float32)
        chosen = tl.zeros((prefixes,), tl.int32)
        expected_overlap = tl.load(overlap_ptr + sequence).to(tl.int32)

        for branch in tl.range(0, branches):
            state = branch * prefixes + prefix
            candidate = tl.zeros((prefixes,), tl.float32)
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
            take = candidate < best
            best = tl.where(take, candidate, best)
            chosen = tl.where(take, state, chosen)
        base = sequence * prefixes
        tl.store(scratch_ptr + base + prefix, best)
        tl.store(best_state_ptr + base + prefix, chosen)
        tl.debug_barrier()

        step = 1
        while step < steps:
            previous = ((step - 1) & 1) * batch * prefixes + base
            current = (step & 1) * batch * prefixes + base
            best = tl.full((prefixes,), float("inf"), tl.float32)
            chosen = tl.zeros((prefixes,), tl.int32)
            for previous_prefix in tl.range(0, prefixes):
                previous_cost = tl.load(
                    scratch_ptr + previous + previous_prefix
                )
                for extension in tl.static_range(0, branches // prefixes):
                    branch = previous_prefix * (branches // prefixes) + extension
                    state = branch * prefixes + prefix
                    candidate = previous_cost
                    for lane in tl.static_range(0, lanes):
                        value = tl.load(
                            x_ptr + (step * lanes + lane) * batch + sequence
                        ).to(tl.float32)
                        code = tl.load(
                            lut_ptr + lane * state_count + state
                        ).to(tl.float32)
                        candidate += (code - value) * (code - value)
                    take = candidate < best
                    best = tl.where(take, candidate, best)
                    chosen = tl.where(take, state, chosen)
            tl.store(scratch_ptr + current + prefix, best)
            tl.store(
                best_state_ptr + step * batch * prefixes + base + prefix,
                chosen,
            )
            tl.debug_barrier()
            step += 1

        traceback_prefix = (
            expected_overlap if has_overlap else tl.argmin(best, axis=0).to(tl.int32)
        )
        for back_step in tl.static_range(steps - 1, -1, -1):
            state = tl.load(
                best_state_ptr + back_step * batch * prefixes + base + traceback_prefix
            ).to(tl.int32)
            tl.store(states_ptr + back_step * batch + sequence, state)
            traceback_prefix = state >> transition_bits


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
    lut = state_lut.T.contiguous()
    overlap_arg = (
        overlap.to(torch.int32).contiguous()
        if overlap is not None
        else torch.zeros(batch, device=x.device, dtype=torch.int32)
    )
    scratch = torch.empty(
        (2, batch, geometry.prefixes), device=x.device, dtype=torch.float32
    )
    best_state = torch.empty(
        (steps, batch, geometry.prefixes), device=x.device, dtype=torch.int32
    )
    states = torch.empty((steps, batch), device=x.device, dtype=torch.int32)
    _native_v4_prefix_viterbi[(batch,)](
        source,
        lut,
        overlap_arg,
        scratch,
        best_state,
        states,
        batch,
        steps=steps,
        has_overlap=overlap is not None,
        prefixes=geometry.prefixes,
        branches=geometry.branches,
        transition_bits=geometry.B,
        lanes=geometry.V,
        state_count=geometry.states,
        num_warps=(
            1
            if geometry.prefixes <= 64
            else 8 if geometry.prefixes >= 256 else 4
        ),
        num_stages=1,
    )
    return states


def solve_native_v4_cuda(
    target: Any,
    *,
    state_lut: Any,
    geometry: NativeQtip25Geometry = NATIVE_QTIP25_GEOMETRY,
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
    midpoint = int(target.shape[1]) // 2
    rolled = torch.roll(target, midpoint, dims=1)
    first = _native_v4_cuda_pass(
        rolled.permute(1, 2, 0).reshape(-1, target.shape[0]),
        state_lut,
        None,
        geometry=geometry,
    )
    overlap = first[midpoint] >> geometry.B
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
    "pack_native_v4_states",
    "solve_native_v4",
    "solve_native_v4_cuda",
    "states_from_native_v4_packed",
]
