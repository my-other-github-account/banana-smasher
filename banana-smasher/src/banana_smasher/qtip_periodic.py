from __future__ import annotations

from typing import Any

import numpy as np

PERIODIC_QTIP25_FORMAT: dict[str, Any] = {
    "codec_form": "periodic_2_3",
    "rate_num": 5,
    "rate_den": 2,
    "transition_k": [2, 3],
    "values_per_transition": 2,
    "transition_bits": [4, 6],
    "bit_order": "msb-first",
}


def _position_count(value: int) -> int:
    count = int(value)
    if count < 0 or count % 4:
        raise ValueError(
            "QTIP2.5-PERIODIC requires a position count divisible by four, "
            f"got {count}"
        )
    return count


def _transition_count(value: int) -> int:
    count = int(value)
    if count < 0 or count % 2:
        raise ValueError(
            f"QTIP2.5-PERIODIC requires an even transition count, got {count}"
        )
    return count


def _symbols(value: np.ndarray) -> np.ndarray:
    symbols = np.asarray(value)
    if symbols.ndim != 1 or symbols.dtype.kind not in "iu":
        raise ValueError("QTIP2.5-PERIODIC symbols must be a one-dimensional integer array")
    _transition_count(symbols.size)
    symbols = symbols.astype(np.uint8, copy=False)
    if symbols.size:
        even = symbols[0::2]
        odd = symbols[1::2]
        if bool(np.any(even >= 16)):
            raise ValueError("QTIP2.5-PERIODIC 4-bit transition is outside [0, 16)")
        if bool(np.any(odd >= 64)):
            raise ValueError("QTIP2.5-PERIODIC 6-bit transition is outside [0, 64)")
    return np.ascontiguousarray(symbols)


def _logical_bits(symbols: np.ndarray) -> np.ndarray:
    bits = np.empty(symbols.size * 5, dtype=np.uint8)
    cursor = 0
    for transition, symbol in enumerate(symbols.tolist()):
        width = 4 if transition % 2 == 0 else 6
        shifts = np.arange(width - 1, -1, -1, dtype=np.uint8)
        bits[cursor : cursor + width] = (symbol >> shifts) & 1
        cursor += width
    return bits


def pack_symbols(symbols: np.ndarray) -> np.ndarray:
    """Pack alternating K2/K3 V=2 transitions into one MSB-first stream."""
    logical = _logical_bits(_symbols(symbols))
    return np.packbits(logical, bitorder="big")


def unpack_symbols(packed: np.ndarray, transition_count: int) -> np.ndarray:
    """Unpack one periodic stream without materializing a family assignment map."""
    count = _transition_count(transition_count)
    source = np.asarray(packed)
    if source.ndim != 1 or source.dtype != np.uint8:
        raise ValueError("QTIP2.5-PERIODIC packed payload must be one-dimensional uint8")
    code_bits = count * 5
    expected_bytes = (code_bits + 7) // 8
    if source.size != expected_bytes:
        raise ValueError(
            f"QTIP2.5-PERIODIC payload has {source.size} bytes; expected {expected_bytes}"
        )
    bits = np.unpackbits(source, bitorder="big")
    if bits.size > code_bits and bool(np.any(bits[code_bits:])):
        raise ValueError("QTIP2.5-PERIODIC nonzero code padding bits")
    result = np.empty(count, dtype=np.uint8)
    cursor = 0
    for transition in range(count):
        width = 4 if transition % 2 == 0 else 6
        value = 0
        for bit in bits[cursor : cursor + width]:
            value = (value << 1) | int(bit)
        result[transition] = value
        cursor += width
    return result


def _decode_bits(bits: np.ndarray, transition_count: int, lut: np.ndarray) -> np.ndarray:
    table = np.asarray(lut)
    if table.shape != (1 << 16, 2):
        raise ValueError(f"QTIP LUT must have shape (65536, 2), got {table.shape}")
    if transition_count == 0:
        return np.empty((0, 2), dtype=table.dtype)
    states = np.empty(transition_count, dtype=np.uint16)
    cursor = 0
    bit_count = bits.size
    for transition in range(transition_count):
        state = 0
        for offset in range(16):
            state = (state << 1) | int(bits[(cursor + offset) % bit_count])
        states[transition] = state
        cursor += 4 if transition % 2 == 0 else 6
    return np.ascontiguousarray(table[states])


def decode_symbols(symbols: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Reference decode of periodic transition symbols through 16-bit QTIP states."""
    validated = _symbols(symbols)
    return _decode_bits(_logical_bits(validated), validated.size, lut)


def decode_packed(
    packed: np.ndarray, transition_count: int, lut: np.ndarray
) -> np.ndarray:
    """Decode the physical periodic byte stream directly to QTIP LUT vectors."""
    symbols = unpack_symbols(packed, transition_count)
    code_bits = _logical_bits(symbols)
    return _decode_bits(code_bits, symbols.size, lut)


def states_from_symbols(symbols: np.ndarray) -> np.ndarray:
    """Return the circular 16-bit QTIP state at every periodic transition."""
    validated = _symbols(symbols)
    if validated.size == 0:
        return np.empty(0, dtype=np.uint16)
    bits = _logical_bits(validated)
    states = np.empty(validated.size, dtype=np.uint16)
    cursor = 0
    for transition in range(validated.size):
        state = 0
        for offset in range(16):
            state = (state << 1) | int(bits[(cursor + offset) % bits.size])
        states[transition] = state
        cursor += 4 if transition % 2 == 0 else 6
    return states


def _viterbi_pass(
    target: np.ndarray,
    lut: np.ndarray,
    *,
    closing_prefix: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve one open path or one exact cyclic-prefix-conditioned path."""
    transitions = target.shape[0]
    widths = np.resize(np.array([4, 6], dtype=np.int64), transitions)
    state_ids = np.arange(1 << 16, dtype=np.int64)
    backpointers: list[np.ndarray] = []
    reduced_costs: np.ndarray | None = None
    for step, width_value in enumerate(widths.tolist()):
        difference = lut - target[step]
        distortion = np.sum(difference * difference, axis=1, dtype=np.float32)
        if step == 0:
            candidate = distortion
            if closing_prefix is not None:
                valid = (state_ids >> int(widths[-1])) == closing_prefix
                candidate = np.where(valid, candidate, np.float32(np.inf))
        else:
            assert reduced_costs is not None
            candidate = reduced_costs[state_ids >> int(widths[step - 1])] + distortion
        prefix_count = 1 << (16 - width_value)
        by_branch = candidate.reshape(1 << width_value, prefix_count)
        branch = np.argmin(by_branch, axis=0)
        prefix = np.arange(prefix_count)
        reduced_costs = by_branch[branch, prefix]
        backpointers.append((branch * prefix_count + prefix).astype(np.uint16))
    assert reduced_costs is not None
    if closing_prefix is None:
        final_prefix = int(np.argmin(reduced_costs))
    else:
        final_prefix = int(closing_prefix)
    states = np.empty(transitions, dtype=np.uint16)
    for step in range(transitions - 1, -1, -1):
        state = int(backpointers[step][final_prefix])
        states[step] = state
        if step:
            final_prefix = state >> int(widths[step - 1])
    return states, reduced_costs


def solve_periodic(
    target: np.ndarray,
    lut: np.ndarray,
    *,
    overlap_candidates: int = 8,
) -> dict[str, Any]:
    """Encode a target sequence with cyclic alternating K2/K3 Viterbi.

    The open pass ranks closing prefixes.  Each retained prefix is then solved
    with the exact last-to-first overlap constraint, and the least-distortion
    cyclic wire is returned.  ``overlap_candidates`` controls this explicit
    quality/speed tradeoff; it is part of the result identity.
    """
    values = np.asarray(target, dtype=np.float32)
    table = np.asarray(lut, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] % 2:
        raise ValueError("periodic target must have an even number of V=2 rows")
    if not values.shape[0]:
        raise ValueError("periodic target must contain at least one transition")
    if table.shape != (1 << 16, 2):
        raise ValueError(f"QTIP LUT must have shape (65536, 2), got {table.shape}")
    if not bool(np.isfinite(values).all() and np.isfinite(table).all()):
        raise ValueError("periodic target and QTIP LUT must be finite")
    candidate_count = int(overlap_candidates)
    final_prefix_count = 1 << (16 - 6)
    if not 1 <= candidate_count <= final_prefix_count:
        raise ValueError(
            f"overlap_candidates must be in [1, {final_prefix_count}]"
        )
    _, open_costs = _viterbi_pass(values, table, closing_prefix=None)
    overlaps = np.argsort(open_costs, kind="stable")[:candidate_count]
    best: dict[str, Any] | None = None
    for overlap_value in overlaps.tolist():
        states, _ = _viterbi_pass(
            values, table, closing_prefix=int(overlap_value)
        )
        symbols = np.empty(states.size, dtype=np.uint8)
        for transition, state in enumerate(states.tolist()):
            width = 4 if transition % 2 == 0 else 6
            symbols[transition] = state >> (16 - width)
        packed = pack_symbols(symbols)
        observed_states = states_from_symbols(symbols)
        if not np.array_equal(observed_states, states):
            raise RuntimeError("periodic cyclic Viterbi traceback does not match packed wire")
        difference = table[states] - values
        distortion = float(np.sum(difference * difference, dtype=np.float64))
        result = {
            "codec_form": "periodic_2_3",
            "rate_num": 5,
            "rate_den": 2,
            "overlap_candidates": candidate_count,
            "closing_prefix": int(overlap_value),
            "distortion": distortion,
            "states": states,
            "symbols": symbols,
            "packed": packed,
        }
        if best is None or distortion < best["distortion"]:
            best = result
    assert best is not None
    return best


def periodic_wire_accounting(
    *,
    position_count: int,
    transform_bytes: int = 0,
    scale_bytes: int = 0,
    shared_tlut_bytes: int = 0,
) -> dict[str, int | float | str]:
    """Return exact logical wire accounting with explicit zero routing overhead."""
    count = _position_count(position_count)
    transform = int(transform_bytes)
    scale = int(scale_bytes)
    shared = int(shared_tlut_bytes)
    if min(transform, scale, shared) < 0:
        raise ValueError("QTIP2.5-PERIODIC byte counts must be nonnegative")
    code_bits = count * 5 // 2
    code_bytes = (code_bits + 7) // 8
    auxiliary = transform + scale
    logical = code_bytes + auxiliary

    def bpw(byte_count: int) -> float:
        return 0.0 if count == 0 else byte_count * 8.0 / count

    return {
        "codec_form": "periodic_2_3",
        "rate_num": 5,
        "rate_den": 2,
        "position_count": count,
        "code_bits": code_bits,
        "code_payload_bytes": code_bytes,
        "code_padding_bits": code_bytes * 8 - code_bits,
        "code_alignment_padding_bytes": 0,
        "selected_indices_bytes": 0,
        "assignment_map_bytes": 0,
        "routing_bytes": 0,
        "transform_bytes": transform,
        "scale_bytes": scale,
        "auxiliary_bytes": auxiliary,
        "logical_expert_plane_bytes": logical,
        "logical_expert_plane_wire_bytes": logical,
        "deduplicated_shared_tlut_bytes": shared,
        "full_wire_bytes": logical + shared,
        "code_bpw": 2.5 if count else 0.0,
        "auxiliary_bpw": bpw(auxiliary),
        "logical_expert_plane_bpw": bpw(logical),
    }
