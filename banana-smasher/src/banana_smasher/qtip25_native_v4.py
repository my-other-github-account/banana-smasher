"""Homogeneous native QTIP2.5 with one L16/B10/V4 trellis geometry.

The codec appends ten transition bits and emits four scalar weights per state,
so every payload has the exact rational rate B/V = 10/4 = 5/2.  The compact
shared 512x2 QTIP TLUT is reused for both halves of the V4 state codebook; the
second half applies the same public quantlut_sym mapping to a byte-rotated state.
"""
from __future__ import annotations

from dataclasses import dataclass
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
        if (self.L, self.B, self.V, self.tlut_bits) != (16, 10, 4, 9):
            raise ValueError("native QTIP2.5 PoC requires exactly L16/B10/V4/Q9")

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
        return 5

    @property
    def rate_den(self) -> int:
        return 2

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


def _require_tlut(tlut: np.ndarray) -> np.ndarray:
    table = np.asarray(tlut, dtype=np.float32)
    if table.shape != (1 << NATIVE_QTIP25_GEOMETRY.tlut_bits, 2):
        raise ValueError("native QTIP2.5 requires the shared float32 Q9xV2 TLUT")
    if not bool(np.isfinite(table).all()):
        raise ValueError("native QTIP2.5 TLUT must be finite")
    return np.ascontiguousarray(table)


def _quantlut_sym_pair(table: np.ndarray, states: np.ndarray) -> np.ndarray:
    hashed = (states + np.uint64(1)) * states
    sign = np.float32(1.0) - np.float32(2.0) * (
        (hashed >> np.uint64(15)) & np.uint64(1)
    ).astype(np.float32)
    indexes = (
        hashed >> np.uint64(16 - NATIVE_QTIP25_GEOMETRY.tlut_bits - 1)
    ) & np.uint64((1 << NATIVE_QTIP25_GEOMETRY.tlut_bits) - 1)
    result = table[indexes.astype(np.int64)].copy()
    result[:, 0] *= sign
    return result


def expand_native_v4_tlut(tlut: np.ndarray) -> np.ndarray:
    """Expand the shared Q9xV2 TLUT into the deterministic L16xV4 state LUT."""
    table = _require_tlut(tlut)
    states = np.arange(NATIVE_QTIP25_GEOMETRY.states, dtype=np.uint64)
    rotated = ((states << np.uint64(8)) | (states >> np.uint64(8))) & np.uint64(
        0xFFFF
    )
    return np.ascontiguousarray(
        np.concatenate(
            (_quantlut_sym_pair(table, states), _quantlut_sym_pair(table, rotated)),
            axis=1,
        )
    )


def _validate_states(states: np.ndarray) -> np.ndarray:
    geometry = NATIVE_QTIP25_GEOMETRY
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


def pack_native_v4_states(states: np.ndarray) -> np.ndarray:
    """Pack the exact circular B10 stream MSB-first with byte tail padding only."""
    geometry = NATIVE_QTIP25_GEOMETRY
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
                (int(state) >> offset) & 1
                for offset in range(geometry.B - 1, -1, -1)
            )
        bits[row] = np.asarray(stream[:bit_count], dtype=np.uint8)
    return np.ascontiguousarray(np.packbits(bits, axis=1, bitorder="big"))


def states_from_native_v4_packed(packed: np.ndarray, *, steps: int) -> np.ndarray:
    """Unpack a circular B10 stream into its exact L16 state sequence."""
    geometry = NATIVE_QTIP25_GEOMETRY
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
) -> np.ndarray:
    geometry = NATIVE_QTIP25_GEOMETRY
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
) -> EncodedNativeQtip25:
    """Reference cyclic Viterbi encoder for homogeneous L16/B10/V4 payloads."""
    geometry = NATIVE_QTIP25_GEOMETRY
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
        expand_native_v4_tlut(np.asarray(tlut))
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
    first = _viterbi_numpy(np.roll(normalized, midpoint, axis=1), lut, overlap=None)
    overlap = first[:, midpoint] >> geometry.B
    states = _viterbi_numpy(normalized, lut, overlap=overlap)
    packed = pack_native_v4_states(states)
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


def decode_native_v4(
    packed: np.ndarray,
    scales: np.ndarray,
    *,
    positions: int,
    tlut: np.ndarray,
) -> np.ndarray:
    geometry = NATIVE_QTIP25_GEOMETRY
    if positions < geometry.V or positions % geometry.V:
        raise ValueError("native QTIP2.5 positions must be positive and divisible by V4")
    words = np.asarray(packed)
    row_scales = np.asarray(scales, dtype=np.float32)
    if row_scales.shape != (words.shape[0],):
        raise ValueError("native QTIP2.5 requires one scale per packed row")
    states = states_from_native_v4_packed(words, steps=positions // geometry.V)
    decoded = expand_native_v4_tlut(tlut)[states].reshape(words.shape[0], positions)
    return decoded * row_scales[:, None]


def decode_native_v4_torch(
    packed: Any,
    scales: Any,
    *,
    positions: int,
    tlut: Any,
) -> Any:
    """Decode on the packed tensor's Torch device; CUDA never falls back to CPU."""
    import torch

    geometry = NATIVE_QTIP25_GEOMETRY
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

    rotated = ((base << 8) | (base >> 8)) & 0xFFFF
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
    ):
        """One exact L16/B10/V4 sequence per CTA with 64 retained costs."""
        sequence = tl.program_id(0)
        prefix = tl.arange(0, 64)
        best = tl.full((64,), float("inf"), tl.float32)
        chosen = tl.zeros((64,), tl.int32)
        expected_overlap = tl.load(overlap_ptr + sequence).to(tl.int32)

        for branch in tl.range(0, 1024):
            state = branch * 64 + prefix
            candidate = tl.zeros((64,), tl.float32)
            for lane in tl.static_range(0, 4):
                value = tl.load(x_ptr + lane * batch + sequence).to(tl.float32)
                code = tl.load(lut_ptr + lane * 65536 + state).to(tl.float32)
                candidate += (code - value) * (code - value)
            if has_overlap:
                candidate = tl.where((state >> 10) == expected_overlap, candidate, float("inf"))
            take = candidate < best
            best = tl.where(take, candidate, best)
            chosen = tl.where(take, state, chosen)
        base = sequence * 64
        tl.store(scratch_ptr + base + prefix, best)
        tl.store(best_state_ptr + base + prefix, chosen)
        tl.debug_barrier()

        step = 1
        while step < steps:
            previous = ((step - 1) & 1) * batch * 64 + base
            current = (step & 1) * batch * 64 + base
            best = tl.full((64,), float("inf"), tl.float32)
            chosen = tl.zeros((64,), tl.int32)
            for branch in tl.range(0, 1024):
                state = branch * 64 + prefix
                candidate = tl.load(scratch_ptr + previous + (state >> 10))
                for lane in tl.static_range(0, 4):
                    value = tl.load(
                        x_ptr + (step * 4 + lane) * batch + sequence
                    ).to(tl.float32)
                    code = tl.load(lut_ptr + lane * 65536 + state).to(tl.float32)
                    candidate += (code - value) * (code - value)
                take = candidate < best
                best = tl.where(take, candidate, best)
                chosen = tl.where(take, state, chosen)
            tl.store(scratch_ptr + current + prefix, best)
            tl.store(
                best_state_ptr + step * batch * 64 + base + prefix,
                chosen,
            )
            tl.debug_barrier()
            step += 1

        traceback_prefix = (
            expected_overlap if has_overlap else tl.argmin(best, axis=0).to(tl.int32)
        )
        for back_step in tl.static_range(steps - 1, -1, -1):
            state = tl.load(
                best_state_ptr + back_step * batch * 64 + base + traceback_prefix
            ).to(tl.int32)
            tl.store(states_ptr + back_step * batch + sequence, state)
            traceback_prefix = state >> 10


def _native_v4_cuda_pass(x: Any, state_lut: Any, overlap: Any | None) -> Any:
    import torch

    if triton is None:
        raise RuntimeError("native QTIP2.5 CUDA solve requires the solve extra")
    if not x.is_cuda or x.dtype != torch.float32 or x.ndim != 2 or x.shape[0] % 4:
        raise ValueError("native QTIP2.5 CUDA input must be float32 [steps*4,batch]")
    if (
        not state_lut.is_cuda
        or state_lut.device != x.device
        or state_lut.dtype != torch.float32
        or tuple(state_lut.shape) != (65536, 4)
    ):
        raise ValueError("native QTIP2.5 CUDA state LUT must be float32 [65536,4]")
    steps = int(x.shape[0]) // 4
    batch = int(x.shape[1])
    if steps < 2 or batch < 1:
        raise ValueError("native QTIP2.5 CUDA solve requires at least two transitions")
    if overlap is not None and (
        not overlap.is_cuda
        or overlap.device != x.device
        or overlap.dtype not in {torch.int32, torch.int64}
        or tuple(overlap.shape) != (batch,)
        or bool(((overlap < 0) | (overlap >= 64)).any())
    ):
        raise ValueError("native QTIP2.5 CUDA overlap must be one prefix in [0,64) per row")
    source = x.contiguous()
    lut = state_lut.T.contiguous()
    overlap_arg = (
        overlap.to(torch.int32).contiguous()
        if overlap is not None
        else torch.zeros(batch, device=x.device, dtype=torch.int32)
    )
    scratch = torch.empty((2, batch, 64), device=x.device, dtype=torch.float32)
    best_state = torch.empty((steps, batch, 64), device=x.device, dtype=torch.int32)
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
        num_warps=4,
        num_stages=1,
    )
    return states


def solve_native_v4_cuda(target: Any, *, state_lut: Any) -> Any:
    """Exact full-branch CUDA solve for ``[rows,steps,4]`` transformed targets."""
    import torch

    if (
        not target.is_cuda
        or target.dtype != torch.float32
        or target.ndim != 3
        or target.shape[2] != 4
    ):
        raise ValueError("native QTIP2.5 CUDA target must be float32 [rows,steps,4]")
    midpoint = int(target.shape[1]) // 2
    rolled = torch.roll(target, midpoint, dims=1)
    first = _native_v4_cuda_pass(
        rolled.permute(1, 2, 0).reshape(-1, target.shape[0]), state_lut, None
    )
    overlap = first[midpoint] >> 10
    return _native_v4_cuda_pass(
        target.permute(1, 2, 0).reshape(-1, target.shape[0]),
        state_lut,
        overlap,
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
) -> dict[str, int | float | str | bool]:
    """Account exact integer code bits separately from every physical overhead."""
    count = int(position_count)
    if count < 0 or count % NATIVE_QTIP25_GEOMETRY.V:
        raise ValueError("native QTIP2.5 position count must be divisible by V4")
    sequences = int(sequence_count)
    transitions = count // NATIVE_QTIP25_GEOMETRY.V
    if sequences < 1 or transitions % sequences:
        raise ValueError("native QTIP2.5 transitions must divide evenly across sequences")
    transform = int(transform_bytes)
    scale = int(scale_bytes)
    shared = int(shared_tlut_bytes)
    routing = int(routing_bytes)
    alignment = int(alignment_bytes)
    if min(transform, scale, shared, routing, alignment) < 0:
        raise ValueError("native QTIP2.5 byte counts must be nonnegative")
    code_bits = count * 5 // 2
    code_bytes = sequences * math.ceil((code_bits // sequences) / 8)
    auxiliary = transform + scale
    logical = code_bytes + auxiliary + routing + alignment

    def bpw(byte_count: int) -> float:
        return 0.0 if count == 0 else byte_count * 8.0 / count

    return {
        "codec_form": "native_l16_b10_v4",
        "rate_num": 5,
        "rate_den": 2,
        "L": 16,
        "B": 10,
        "V": 4,
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
        "code_bpw": 2.5 if count else 0.0,
        "auxiliary_bpw": bpw(auxiliary + routing + alignment),
        "logical_expert_plane_bpw": bpw(logical),
    }


__all__ = [
    "EncodedNativeQtip25",
    "NATIVE_QTIP25_GEOMETRY",
    "NativeQtip25Geometry",
    "decode_native_v4",
    "decode_native_v4_torch",
    "expand_native_v4_tlut",
    "native_v4_wire_accounting",
    "pack_native_v4_states",
    "solve_native_v4",
    "solve_native_v4_cuda",
    "states_from_native_v4_packed",
]
