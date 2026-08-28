"""Banana-native scalar V1 trellis codec.

This is the minimal portable path for the L16/B2/V1 anchor: a balanced affine
state map, compact Gaussian codebook, global scale search, cyclic Viterbi,
optional full randomized Hadamard transforms, exact 2-bpw packing, and a simple
reverse-block LDLQ feedback loop.  The NumPy solver is the correctness/PoC path;
the Torch decoder keeps packed runtime execution on the caller's device.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np

BANANA_V1_MULTIPLIER = 48917
BANANA_V1_OFFSET = 50631
_DEFAULT_SCALE_FACTORS = (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20)


@dataclass(frozen=True)
class BananaV1Geometry:
    L: int = 16
    B: int = 2
    V: int = 1
    codebook_levels: int = 1024

    def __post_init__(self) -> None:
        if (self.L, self.B, self.V, self.codebook_levels) != (16, 2, 1, 1024):
            raise ValueError("Banana V1 requires exactly L16/B2/V1 with 1,024 levels")

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
        return 2

    @property
    def rate_den(self) -> int:
        return 1

    def as_mapping(self) -> dict[str, int | str]:
        return {
            "L": self.L,
            "B": self.B,
            "V": self.V,
            "rate_num": self.rate_num,
            "rate_den": self.rate_den,
            "codebook_levels": self.codebook_levels,
            "decode_mode": "banana_affine_gaussian",
        }


BANANA_V1_GEOMETRY = BananaV1Geometry()


@dataclass(frozen=True)
class EncodedBananaV1:
    geometry: BananaV1Geometry
    shape: tuple[int, int]
    states: np.ndarray
    packed: np.ndarray
    scales: np.ndarray
    distortion: float
    scale_factor: float
    scale_factors: tuple[float, ...]

    @property
    def weights(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def code_bits(self) -> int:
        return self.weights * self.geometry.B

    @property
    def code_bpw(self) -> float:
        return self.code_bits / self.weights


@dataclass(frozen=True)
class BananaV1MatrixResult:
    decoded: np.ndarray
    states: np.ndarray
    packed: np.ndarray
    scales: np.ndarray
    distortion: float
    scale_factor: float
    scale_factors: tuple[float, ...]


@dataclass(frozen=True)
class BananaV1BuildResult:
    source_shape: tuple[int, int]
    decoded: np.ndarray
    states: np.ndarray
    packed: np.ndarray
    scales: np.ndarray
    codebook: np.ndarray
    su: np.ndarray
    sv: np.ndarray
    distortion: float
    scale_factor: float
    scale_factors: tuple[float, ...]
    accounting: dict[str, int | float]


def banana_v1_state_levels() -> np.ndarray:
    """Map all L16 states into 1,024 perfectly balanced native levels."""
    states = np.arange(BANANA_V1_GEOMETRY.states, dtype=np.uint64)
    levels = (
        (states * np.uint64(BANANA_V1_MULTIPLIER) + np.uint64(BANANA_V1_OFFSET))
        & np.uint64(0xFFFF)
    ) >> np.uint64(6)
    return levels.astype(np.int32)


def banana_v1_gaussian_codebook() -> np.ndarray:
    """Return the deterministic 2 KiB FP16 Gaussian-quantile codebook."""
    normal = NormalDist()
    values = np.asarray(
        [normal.inv_cdf((index + 0.5) / 1024.0) for index in range(1024)],
        dtype=np.float64,
    )
    values /= np.sqrt(np.mean(values * values, dtype=np.float64))
    return np.ascontiguousarray(values.astype(np.float16))


def fit_banana_v1_codebook_from_statistics(
    original: np.ndarray,
    level_counts: np.ndarray,
    target_sums: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Apply one deterministic TRAIN-only centroid update to the shared LUT.

    Empty levels retain their input value. The result preserves the input
    floating dtype so the production FP16[1024] wire identity remains exact.
    """
    codebook = np.asarray(original)
    counts = np.asarray(level_counts)
    sums = np.asarray(target_sums, dtype=np.float64)
    blend = float(alpha)
    if (
        codebook.shape != (BANANA_V1_GEOMETRY.codebook_levels,)
        or codebook.dtype.kind != "f"
        or not bool(np.isfinite(codebook).all())
    ):
        raise ValueError("Banana V1 codebook fit requires finite floating [1024]")
    if counts.shape != codebook.shape or counts.dtype.kind not in "iu" or bool(np.any(counts < 0)):
        raise ValueError("Banana V1 level counts must be nonnegative integer [1024]")
    if sums.shape != codebook.shape or not bool(np.isfinite(sums).all()):
        raise ValueError("Banana V1 target sums must be finite [1024]")
    if not math.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise ValueError("Banana V1 codebook alpha must be finite in [0,1]")
    centroids = codebook.astype(np.float64)
    assigned = counts > 0
    centroids[assigned] = sums[assigned] / counts[assigned]
    fitted = (1.0 - blend) * codebook.astype(np.float64) + blend * centroids
    return np.ascontiguousarray(fitted.astype(codebook.dtype))


def fit_banana_v1_codebook(
    original: np.ndarray,
    level_ids: np.ndarray,
    normalized_targets: np.ndarray,
    *,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the shared LUT from authentic assignments and normalized targets."""
    levels = np.asarray(level_ids)
    targets = np.asarray(normalized_targets, dtype=np.float64)
    if levels.shape != targets.shape or levels.ndim != 1:
        raise ValueError("Banana V1 fit IDs and targets must be matching vectors")
    if levels.dtype.kind not in "iu" or bool(np.any(levels < 0) or np.any(levels >= 1024)):
        raise ValueError("Banana V1 fit level ID outside [0,1024)")
    if not bool(np.isfinite(targets).all()):
        raise ValueError("Banana V1 fit targets must be finite")
    counts = np.bincount(levels.astype(np.int64), minlength=1024).astype(np.int64)
    sums = np.bincount(levels.astype(np.int64), weights=targets, minlength=1024)
    return fit_banana_v1_codebook_from_statistics(
        original, counts, sums, alpha=alpha
    ), counts


def expand_banana_v1_codebook(codebook: np.ndarray | None = None) -> np.ndarray:
    compact = (
        banana_v1_gaussian_codebook() if codebook is None else np.asarray(codebook)
    )
    if (
        compact.shape != (1024,)
        or compact.dtype.kind != "f"
        or not bool(np.isfinite(compact).all())
    ):
        raise ValueError("Banana V1 codebook must be finite [1024] floating values")
    return np.ascontiguousarray(compact.astype(np.float32)[banana_v1_state_levels()])


def _fwht(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float32)
    count = result.shape[-1]
    if count <= 0 or count & (count - 1):
        raise ValueError("Banana V1 Hadamard axes must be positive powers of two")
    width = 1
    while width < count:
        grouped = result.reshape(*result.shape[:-1], count // (2 * width), 2, width)
        left = grouped[..., 0, :].copy()
        right = grouped[..., 1, :].copy()
        result = np.concatenate((left + right, left - right), axis=-1).reshape(
            result.shape
        )
        width *= 2
    return np.ascontiguousarray(result / np.float32(math.sqrt(count)), dtype=np.float32)


def banana_v1_transform(
    source: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply deterministic two-sided randomized normalized Hadamard transforms."""
    weights = np.asarray(source, dtype=np.float32)
    if weights.ndim != 2 or not weights.size or not bool(np.isfinite(weights).all()):
        raise ValueError("Banana V1 transform requires a finite non-empty matrix")
    rows, columns = weights.shape
    if rows & (rows - 1) or columns & (columns - 1):
        raise ValueError("Banana V1 transform dimensions must be powers of two")
    rng = np.random.default_rng(int(seed))
    su = np.where(rng.integers(0, 2, size=columns, dtype=np.int8), 1.0, -1.0).astype(
        np.float32
    )
    sv = np.where(rng.integers(0, 2, size=rows, dtype=np.int8), 1.0, -1.0).astype(
        np.float32
    )
    transformed = _fwht(weights * su)
    transformed = _fwht((transformed * sv[:, None]).T).T
    return np.ascontiguousarray(transformed), su, sv


def banana_v1_inverse_transform(
    transformed: np.ndarray, *, su: np.ndarray, sv: np.ndarray
) -> np.ndarray:
    values = np.asarray(transformed, dtype=np.float32)
    sign_u = np.asarray(su, dtype=np.float32)
    sign_v = np.asarray(sv, dtype=np.float32)
    if (
        values.ndim != 2
        or sign_u.shape != (values.shape[1],)
        or sign_v.shape != (values.shape[0],)
    ):
        raise ValueError(
            "Banana V1 inverse transform signs do not match matrix geometry"
        )
    physical = _fwht(values.T).T * sign_v[:, None]
    return np.ascontiguousarray(_fwht(physical) * sign_u)


def _validate_state_lut(
    state_lut: np.ndarray | None, codebook: np.ndarray | None
) -> np.ndarray:
    if state_lut is not None and codebook is not None:
        raise ValueError(
            "provide either a compact codebook or expanded state LUT, not both"
        )
    lut = (
        expand_banana_v1_codebook(codebook)
        if state_lut is None
        else np.asarray(state_lut, dtype=np.float32)
    )
    if lut.shape != (BANANA_V1_GEOMETRY.states,) or not bool(np.isfinite(lut).all()):
        raise ValueError("Banana V1 expanded state LUT must be finite [65536]")
    return np.ascontiguousarray(lut, dtype=np.float32)


def _viterbi_numpy(
    target: np.ndarray,
    state_lut: np.ndarray,
    *,
    overlap: np.ndarray | None,
) -> np.ndarray:
    geometry = BANANA_V1_GEOMETRY
    values = np.asarray(target, dtype=np.float32)
    batch, steps = values.shape
    if steps * geometry.B < geometry.L:
        raise ValueError("Banana V1 sequence is too short to close the trellis")
    prefix_ids = np.arange(geometry.prefixes, dtype=np.int32)
    predecessors = prefix_ids[:, None] + (
        np.arange(geometry.branches, dtype=np.int32)[None, :] * geometry.prefixes
    )

    def errors(step: int) -> np.ndarray:
        delta = state_lut[None, :] - values[:, step, None]
        return delta * delta

    cost = errors(0)
    if overlap is not None:
        expected = np.asarray(overlap, dtype=np.int32)
        if expected.shape != (batch,) or bool(
            np.any(expected < 0) or np.any(expected >= geometry.prefixes)
        ):
            raise ValueError("invalid Banana V1 cyclic overlap prefixes")
        allowed = (expected[:, None] << geometry.B) + np.arange(
            geometry.branches, dtype=np.int32
        )
        masked = np.full_like(cost, np.inf)
        masked[np.arange(batch)[:, None], allowed] = cost[
            np.arange(batch)[:, None], allowed
        ]
        cost = masked

    backpointers = np.empty((steps, batch, geometry.prefixes), dtype=np.uint8)
    backpointers[0] = 0
    for step in range(1, steps):
        options = cost[:, predecessors]
        choice = np.argmin(options, axis=2)
        backpointers[step] = choice.astype(np.uint8)
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


def _cyclic_states(normalized: np.ndarray, state_lut: np.ndarray) -> np.ndarray:
    midpoint = normalized.shape[1] // 2
    first = _viterbi_numpy(
        np.roll(normalized, midpoint, axis=1), state_lut, overlap=None
    )
    overlap = first[:, midpoint] >> BANANA_V1_GEOMETRY.B
    return _viterbi_numpy(normalized, state_lut, overlap=overlap)


def _solve_with_scales(
    values: np.ndarray,
    state_lut: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    normalized = values / scales[:, None]
    states = _cyclic_states(normalized, state_lut)
    decoded = state_lut[states] * scales[:, None]
    distortion = float(
        np.sum((decoded.astype(np.float64) - values.astype(np.float64)) ** 2)
    )
    return states, decoded.astype(np.float32), distortion


def solve_banana_v1(
    target: np.ndarray,
    *,
    codebook: np.ndarray | None = None,
    state_lut: np.ndarray | None = None,
    scales: np.ndarray | Sequence[float] | float | None = None,
    scale_factors: Sequence[float] = _DEFAULT_SCALE_FACTORS,
) -> EncodedBananaV1:
    """Run global-scale search and two-pass cyclic Viterbi over scalar sequences."""
    values = np.asarray(target, dtype=np.float32)
    if values.ndim != 2 or not values.shape[0] or not bool(np.isfinite(values).all()):
        raise ValueError("Banana V1 target must be finite [sequences,positions]")
    lut = _validate_state_lut(state_lut, codebook)
    if scales is not None:
        row_scales = np.asarray(scales, dtype=np.float32)
        if row_scales.ndim == 0:
            row_scales = np.full(values.shape[0], row_scales, dtype=np.float32)
        if row_scales.shape != (values.shape[0],) or bool(
            np.any(~np.isfinite(row_scales)) or np.any(row_scales <= 0)
        ):
            raise ValueError("Banana V1 scales must be finite positive row values")
        states, _decoded, distortion = _solve_with_scales(values, lut, row_scales)
        factors = (1.0,)
        selected_factor = 1.0
    else:
        factors = tuple(float(value) for value in scale_factors)
        if not factors or any(
            not math.isfinite(value) or value <= 0 for value in factors
        ):
            raise ValueError("Banana V1 scale factors must be finite and positive")
        source_rms = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
        lut_rms = float(np.sqrt(np.mean(lut.astype(np.float64) ** 2)))
        base_scale = 1.0 if source_rms == 0 else source_rms / lut_rms
        best: tuple[float, float, np.ndarray, np.ndarray] | None = None
        for factor in factors:
            candidate_scales = np.full(
                values.shape[0], base_scale * factor, dtype=np.float32
            )
            candidate_states, candidate_decoded, candidate_distortion = (
                _solve_with_scales(values, lut, candidate_scales)
            )
            if best is None or candidate_distortion < best[0]:
                best = (
                    candidate_distortion,
                    factor,
                    candidate_states,
                    candidate_decoded,
                )
        assert best is not None
        distortion, selected_factor, states, _decoded = best
        row_scales = np.full(
            values.shape[0], base_scale * selected_factor, dtype=np.float32
        )
    packed = pack_banana_v1_states(states)
    return EncodedBananaV1(
        geometry=BANANA_V1_GEOMETRY,
        shape=(int(values.shape[0]), int(values.shape[1])),
        states=states,
        packed=packed,
        scales=row_scales,
        distortion=distortion,
        scale_factor=selected_factor,
        scale_factors=factors,
    )


def _validate_states(states: np.ndarray) -> np.ndarray:
    geometry = BANANA_V1_GEOMETRY
    values = np.asarray(states)
    if values.ndim != 2 or values.shape[1] * geometry.B < geometry.L:
        raise ValueError("Banana V1 states must be [sequences,sufficient positions]")
    values = values.astype(np.int32, copy=False)
    if bool(np.any(values < 0) or np.any(values >= geometry.states)):
        raise ValueError("Banana V1 state outside L16 geometry")
    suffix_mask = geometry.prefixes - 1
    if values.shape[1] > 1 and bool(
        np.any((values[:, :-1] & suffix_mask) != (values[:, 1:] >> geometry.B))
    ):
        raise ValueError("Banana V1 state path violates B2 transitions")
    if bool(np.any((values[:, -1] & suffix_mask) != (values[:, 0] >> geometry.B))):
        raise ValueError("Banana V1 state path does not close")
    return np.ascontiguousarray(values)


def pack_banana_v1_states(states: np.ndarray) -> np.ndarray:
    """Pack the exact circular B2 stream MSB-first."""
    geometry = BANANA_V1_GEOMETRY
    values = _validate_states(states)
    bit_count = values.shape[1] * geometry.B
    bits = np.empty((values.shape[0], bit_count), dtype=np.uint8)
    for row in range(values.shape[0]):
        stream = [
            (int(values[row, 0]) >> offset) & 1
            for offset in range(geometry.L - 1, -1, -1)
        ]
        for state in values[row, 1:]:
            stream.extend(
                (int(state) >> offset) & 1 for offset in range(geometry.B - 1, -1, -1)
            )
        bits[row] = np.asarray(stream[:bit_count], dtype=np.uint8)
    return np.ascontiguousarray(np.packbits(bits, axis=1, bitorder="big"))


def states_from_banana_v1_packed(packed: np.ndarray, *, steps: int) -> np.ndarray:
    geometry = BANANA_V1_GEOMETRY
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
        raise ValueError("Banana V1 packed byte shape does not match B2 positions")
    all_bits = np.unpackbits(words, axis=1, bitorder="big")
    if all_bits.shape[1] > bit_count and bool(np.any(all_bits[:, bit_count:])):
        raise ValueError("Banana V1 has nonzero byte-tail padding")
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


def decode_banana_v1(
    packed: np.ndarray,
    scales: np.ndarray,
    *,
    positions: int,
    codebook: np.ndarray | None = None,
    state_lut: np.ndarray | None = None,
) -> np.ndarray:
    lut = _validate_state_lut(state_lut, codebook)
    states = states_from_banana_v1_packed(np.asarray(packed), steps=positions)
    row_scales = np.asarray(scales, dtype=np.float32)
    if row_scales.shape != (states.shape[0],):
        raise ValueError("Banana V1 decode requires one scale per sequence")
    return np.ascontiguousarray(lut[states] * row_scales[:, None])


def decode_banana_v1_torch(
    packed: Any,
    scales: Any,
    *,
    positions: int,
    codebook: Any | None = None,
    state_lut: Any | None = None,
) -> Any:
    """Decode exact packed B2 states on the packed tensor's Torch device."""
    import torch

    geometry = BANANA_V1_GEOMETRY
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError(
            "Torch Banana V1 packed payload must be uint8 [sequences,bytes]"
        )
    bit_count = int(positions) * geometry.B
    if (
        positions < 1
        or bit_count < geometry.L
        or packed.shape[1] != math.ceil(bit_count / 8)
    ):
        raise ValueError("Torch Banana V1 packed byte shape drift")
    if scales.shape != (packed.shape[0],) or scales.device != packed.device:
        raise ValueError(
            "Torch Banana V1 scales must be one value per sequence on-device"
        )
    if state_lut is not None and codebook is not None:
        raise ValueError("provide either compact codebook or expanded state LUT")
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
    for step in range(1, positions):
        start = geometry.L + (step - 1) * geometry.B
        branch = torch.sum(stream[:, start : start + geometry.B] * powers_b, dim=1)
        states.append(((states[-1] << geometry.B) & mask) + branch)
    state_tensor = torch.stack(states, dim=1)
    if state_lut is None:
        compact = (
            torch.as_tensor(banana_v1_gaussian_codebook(), device=packed.device)
            if codebook is None
            else codebook
        )
        if compact.device != packed.device or tuple(compact.shape) != (1024,):
            raise ValueError("Torch Banana V1 codebook must be [1024] on-device")
        levels = (
            (state_tensor * BANANA_V1_MULTIPLIER + BANANA_V1_OFFSET) & 0xFFFF
        ) >> 6
        decoded = compact.index_select(0, levels.reshape(-1)).reshape_as(levels)
    else:
        if state_lut.device != packed.device or tuple(state_lut.shape) != (65536,):
            raise ValueError("Torch Banana V1 expanded LUT must be [65536] on-device")
        decoded = state_lut.index_select(0, state_tensor.reshape(-1)).reshape_as(
            state_tensor
        )
    return decoded * scales.reshape(-1, 1)


def ldlq_banana_v1_matrix(
    transformed: np.ndarray,
    lower: np.ndarray,
    *,
    codebook: np.ndarray | None = None,
    scale_factors: Sequence[float] = _DEFAULT_SCALE_FACTORS,
) -> BananaV1MatrixResult:
    """Reverse 16-column LDLQ feedback with one global matrix scale search."""
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
            "Banana V1 LDLQ requires finite matrix multiples of 16 and square lower"
        )
    factors = tuple(float(value) for value in scale_factors)
    if not factors or any(not math.isfinite(value) or value <= 0 for value in factors):
        raise ValueError("Banana V1 LDLQ scale factors must be finite and positive")
    lut = expand_banana_v1_codebook(codebook)
    source_rms = float(np.sqrt(np.mean(source.astype(np.float64) ** 2)))
    lut_rms = float(np.sqrt(np.mean(lut.astype(np.float64) ** 2)))
    base_scale = 1.0 if source_rms == 0 else source_rms / lut_rms
    row_blocks = source.shape[0] // 16
    column_blocks = source.shape[1] // 16
    best: tuple[float, float, np.ndarray, np.ndarray, np.ndarray] | None = None
    for factor in factors:
        scale = np.float32(base_scale * factor)
        decoded = np.zeros_like(source)
        state_grid = np.empty((row_blocks, column_blocks, 256), dtype=np.int32)
        packed_grid = np.empty((row_blocks, column_blocks, 64), dtype=np.uint8)
        for column_block in range(column_blocks - 1, -1, -1):
            start = column_block * 16
            end = start + 16
            corrected = source[:, start:end].copy()
            if end < source.shape[1]:
                error_right = source[:, end:] - decoded[:, end:]
                corrected += (feedback[end:, start:end].T @ error_right.T).T.astype(
                    np.float32
                )
            tiles = corrected.reshape(row_blocks, 16, 16).reshape(row_blocks, 256)
            encoded = solve_banana_v1(
                tiles,
                state_lut=lut,
                scales=np.full(row_blocks, scale, dtype=np.float32),
            )
            quantized = decode_banana_v1(
                encoded.packed,
                encoded.scales,
                positions=256,
                state_lut=lut,
            ).reshape(source.shape[0], 16)
            decoded[:, start:end] = quantized
            state_grid[:, column_block] = encoded.states
            packed_grid[:, column_block] = encoded.packed
        distortion = float(
            np.sum((decoded.astype(np.float64) - source.astype(np.float64)) ** 2)
        )
        if best is None or distortion < best[0]:
            best = (distortion, factor, decoded, state_grid.copy(), packed_grid.copy())
    assert best is not None
    distortion, selected_factor, decoded, state_grid, packed_grid = best
    tile_count = row_blocks * column_blocks
    return BananaV1MatrixResult(
        decoded=np.ascontiguousarray(decoded),
        states=np.ascontiguousarray(state_grid.reshape(tile_count, 256)),
        packed=np.ascontiguousarray(packed_grid.reshape(tile_count, 64)),
        scales=np.full(tile_count, base_scale * selected_factor, dtype=np.float32),
        distortion=distortion,
        scale_factor=selected_factor,
        scale_factors=factors,
    )


def banana_v1_wire_accounting(
    *,
    position_count: int,
    sequence_count: int,
    scale_bytes: int = 0,
    transform_bytes: int = 0,
    shared_codebook_bytes: int = 2048,
) -> dict[str, int | float]:
    count = int(position_count)
    sequences = int(sequence_count)
    if count < 0 or sequences < 1 or count % sequences:
        raise ValueError("Banana V1 positions must divide evenly across sequences")
    positions_per_sequence = count // sequences
    if positions_per_sequence * BANANA_V1_GEOMETRY.B < BANANA_V1_GEOMETRY.L:
        raise ValueError("Banana V1 sequences are too short for L16 packing")
    scale = int(scale_bytes)
    transform = int(transform_bytes)
    shared = int(shared_codebook_bytes)
    if min(scale, transform, shared) < 0:
        raise ValueError("Banana V1 byte counts must be nonnegative")
    code_bits = count * BANANA_V1_GEOMETRY.B
    code_bytes = sequences * math.ceil(
        positions_per_sequence * BANANA_V1_GEOMETRY.B / 8
    )
    return {
        "weights": count,
        "code_bits": code_bits,
        "code_payload_bytes": code_bytes,
        "code_bpw": 2.0,
        "scale_bytes": scale,
        "transform_bytes": transform,
        "shared_codebook_bytes": shared,
        "cell_payload_bytes": code_bytes + scale + transform,
        "full_wire_bytes": code_bytes + scale + transform + shared,
        "full_wire_bpw": 0.0
        if count == 0
        else (code_bytes + scale + transform + shared) * 8.0 / count,
    }


def build_banana_v1(
    source: np.ndarray,
    *,
    lower: np.ndarray | None = None,
    seed: int = 0,
    codebook: np.ndarray | None = None,
    scale_factors: Sequence[float] = _DEFAULT_SCALE_FACTORS,
) -> BananaV1BuildResult:
    """Build one physical matrix through transform, LDLQ, pack, and inverse transform."""
    weights = np.asarray(source, dtype=np.float32)
    if weights.ndim != 2 or weights.shape[0] % 16 or weights.shape[1] % 16:
        raise ValueError("Banana V1 build requires matrix dimensions divisible by 16")
    compact = (
        banana_v1_gaussian_codebook() if codebook is None else np.asarray(codebook)
    )
    transformed, su, sv = banana_v1_transform(weights, seed=seed)
    feedback = (
        np.zeros((weights.shape[1], weights.shape[1]), dtype=np.float32)
        if lower is None
        else np.asarray(lower, dtype=np.float32)
    )
    quantized = ldlq_banana_v1_matrix(
        transformed,
        feedback,
        codebook=compact,
        scale_factors=scale_factors,
    )
    decoded = banana_v1_inverse_transform(quantized.decoded, su=su, sv=sv)
    distortion = float(
        np.sum((decoded.astype(np.float64) - weights.astype(np.float64)) ** 2)
    )
    transform_bytes = (su.size + sv.size) * np.dtype(np.float16).itemsize
    accounting = banana_v1_wire_accounting(
        position_count=int(weights.size),
        sequence_count=int(quantized.packed.shape[0]),
        scale_bytes=int(quantized.scales.nbytes),
        transform_bytes=transform_bytes,
        shared_codebook_bytes=int(compact.astype(np.float16).nbytes),
    )
    return BananaV1BuildResult(
        source_shape=(int(weights.shape[0]), int(weights.shape[1])),
        decoded=np.ascontiguousarray(decoded, dtype=np.float32),
        states=quantized.states,
        packed=quantized.packed,
        scales=quantized.scales,
        codebook=np.ascontiguousarray(compact, dtype=np.float16),
        su=np.ascontiguousarray(su, dtype=np.float32),
        sv=np.ascontiguousarray(sv, dtype=np.float32),
        distortion=distortion,
        scale_factor=quantized.scale_factor,
        scale_factors=quantized.scale_factors,
        accounting=accounting,
    )


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_banana_v1_candidate(
    output: str | Path,
    result: BananaV1BuildResult,
) -> dict[str, Any]:
    """Write one exact-byte candidate that the generic Backpack pricer can consume."""
    root = Path(output).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"output already exists: {root}")
    root.mkdir(parents=True)
    arrays = {
        "codes.npy": np.ascontiguousarray(result.packed, dtype=np.uint8),
        "scales.npy": np.ascontiguousarray(result.scales, dtype=np.float32),
        "SU.npy": np.ascontiguousarray(result.su, dtype=np.float16),
        "SV.npy": np.ascontiguousarray(result.sv, dtype=np.float16),
        "decoded.npy": np.ascontiguousarray(result.decoded, dtype=np.float32),
    }
    for name, array in arrays.items():
        np.save(root / name, array, allow_pickle=False)
    compact = np.ascontiguousarray(result.codebook, dtype=np.float16)
    codebook_path = root / "codebook.fp16"
    codebook_path.write_bytes(compact.tobytes())
    receipt: dict[str, Any] = {
        "schema": "banana-smasher-banana-v1-candidate-v1",
        "status": "PASS",
        "provider_id": "banana_v1",
        "runtime_family": "banana_v1",
        "codec_form": "banana_l16_b2_v1_affine_gaussian",
        "geometry": BANANA_V1_GEOMETRY.as_mapping(),
        "source_shape": list(result.source_shape),
        "sequence_count": int(result.packed.shape[0]),
        "position_count": int(np.prod(result.source_shape)),
        "scale_factor": result.scale_factor,
        "scale_factors": list(result.scale_factors),
        "distortion": result.distortion,
        **result.accounting,
        "activation_artifacts": [
            {
                "id": "banana-v1-gaussian-lut-v1",
                "bytes": int(codebook_path.stat().st_size),
                "path": str(codebook_path),
                "sha256": hashlib.sha256(codebook_path.read_bytes()).hexdigest(),
            }
        ],
        "artifacts": {
            **{
                name[:-4]: {
                    "file": name,
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "data_bytes": int(array.nbytes),
                    "data_sha256": _array_sha(array),
                }
                for name, array in arrays.items()
            },
            "codebook": {
                "file": codebook_path.name,
                "dtype": str(compact.dtype),
                "shape": [1024],
                "data_bytes": int(compact.nbytes),
                "data_sha256": _array_sha(compact),
            },
        },
    }
    (root / "BANANA_V1_RECEIPT.json").write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    return receipt


def verify_banana_v1_candidate(value: str | Path) -> bool:
    try:
        root = Path(value).expanduser().resolve()
        receipt = json.loads((root / "BANANA_V1_RECEIPT.json").read_text())
        if (
            receipt.get("schema") != "banana-smasher-banana-v1-candidate-v1"
            or receipt.get("status") != "PASS"
            or receipt.get("provider_id") != "banana_v1"
            or receipt.get("geometry") != BANANA_V1_GEOMETRY.as_mapping()
        ):
            return False
        rows, columns = (int(item) for item in receipt["source_shape"])
        tile_count = rows * columns // 256
        expected = {
            "codes": (np.dtype(np.uint8), (tile_count, 64)),
            "scales": (np.dtype(np.float32), (tile_count,)),
            "SU": (np.dtype(np.float16), (columns,)),
            "SV": (np.dtype(np.float16), (rows,)),
            "decoded": (np.dtype(np.float32), (rows, columns)),
        }
        for name, (dtype, shape) in expected.items():
            array = np.load(root / f"{name}.npy", allow_pickle=False)
            spec = receipt["artifacts"][name]
            if (
                array.dtype != dtype
                or array.shape != shape
                or int(array.nbytes) != int(spec["data_bytes"])
                or _array_sha(array) != spec["data_sha256"]
            ):
                return False
        compact = np.fromfile(root / "codebook.fp16", dtype=np.float16)
        compact_spec = receipt["artifacts"]["codebook"]
        if (
            compact.shape != (1024,)
            or int(compact.nbytes) != int(compact_spec["data_bytes"])
            or _array_sha(compact) != compact_spec["data_sha256"]
        ):
            return False
        return True
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def materialize_banana_v1_candidate(value: str | Path) -> dict[str, Any]:
    root = Path(value).expanduser().resolve()
    if not verify_banana_v1_candidate(root):
        raise ValueError(f"invalid Banana V1 candidate: {root}")
    receipt = json.loads((root / "BANANA_V1_RECEIPT.json").read_text())
    return {
        "receipt": receipt,
        "codebook": np.fromfile(root / "codebook.fp16", dtype=np.float16),
        **{
            name: np.asarray(np.load(root / f"{name}.npy", allow_pickle=False))
            for name in ("codes", "scales", "SU", "SV", "decoded")
        },
    }


def predict_banana_v1_candidate(value: str | Path) -> np.ndarray:
    materialized = materialize_banana_v1_candidate(value)
    rows, columns = (int(item) for item in materialized["receipt"]["source_shape"])
    blocks = decode_banana_v1(
        materialized["codes"],
        materialized["scales"],
        positions=256,
        codebook=materialized["codebook"],
    )
    transformed = (
        blocks.reshape(rows // 16, columns // 16, 16, 16)
        .transpose(0, 2, 1, 3)
        .reshape(rows, columns)
    )
    return banana_v1_inverse_transform(
        transformed,
        su=materialized["SU"].astype(np.float32),
        sv=materialized["SV"].astype(np.float32),
    )


def banana_v1_provider() -> Any:
    """Return a public provider object compatible with Backpack's provider seam."""
    from .backpack_providers import BackpackFamilyProvider, price_backpack_candidate

    return BackpackFamilyProvider(
        provider_id="banana_v1",
        kind="banana_v1",
        runtime_family="banana_v1",
        generate=write_banana_v1_candidate,
        materialize=materialize_banana_v1_candidate,
        price=price_backpack_candidate,
        predict=predict_banana_v1_candidate,
        verify=verify_banana_v1_candidate,
    )


__all__ = [
    "BANANA_V1_GEOMETRY",
    "BANANA_V1_MULTIPLIER",
    "BANANA_V1_OFFSET",
    "BananaV1BuildResult",
    "BananaV1Geometry",
    "BananaV1MatrixResult",
    "EncodedBananaV1",
    "banana_v1_gaussian_codebook",
    "fit_banana_v1_codebook",
    "fit_banana_v1_codebook_from_statistics",
    "banana_v1_inverse_transform",
    "banana_v1_provider",
    "banana_v1_state_levels",
    "banana_v1_transform",
    "banana_v1_wire_accounting",
    "build_banana_v1",
    "decode_banana_v1",
    "decode_banana_v1_torch",
    "expand_banana_v1_codebook",
    "ldlq_banana_v1_matrix",
    "materialize_banana_v1_candidate",
    "pack_banana_v1_states",
    "predict_banana_v1_candidate",
    "solve_banana_v1",
    "states_from_banana_v1_packed",
    "verify_banana_v1_candidate",
    "write_banana_v1_candidate",
]
