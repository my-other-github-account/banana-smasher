from __future__ import annotations

from typing import Any

import numpy as np

PERIODIC_QTIP25_FORMAT: dict[str, Any] = {
    "codec_form": "qtip25_periodic_23",
    "rate_num": 5,
    "rate_den": 2,
    "transition_bits": [2, 3],
    "bit_order": "msb-first",
}


def _position_count(value: int) -> int:
    count = int(value)
    if count < 0 or count % 2:
        raise ValueError(
            f"QTIP2.5-PERIODIC requires an even position count, got {count}"
        )
    return count


def _symbols(value: np.ndarray) -> np.ndarray:
    symbols = np.asarray(value)
    if symbols.ndim != 1 or symbols.dtype.kind not in "iu":
        raise ValueError("QTIP2.5-PERIODIC symbols must be a one-dimensional integer array")
    _position_count(symbols.size)
    symbols = symbols.astype(np.uint8, copy=False)
    if symbols.size:
        even = symbols[0::2]
        odd = symbols[1::2]
        if bool(np.any(even >= 4)):
            raise ValueError("QTIP2.5-PERIODIC 2-bit transition is outside [0, 4)")
        if bool(np.any(odd >= 8)):
            raise ValueError("QTIP2.5-PERIODIC 3-bit transition is outside [0, 8)")
    return np.ascontiguousarray(symbols)


def _logical_bits(symbols: np.ndarray) -> np.ndarray:
    bits = np.empty(symbols.size * 5 // 2, dtype=np.uint8)
    cursor = 0
    for position, symbol in enumerate(symbols.tolist()):
        width = 2 if position % 2 == 0 else 3
        shifts = np.arange(width - 1, -1, -1, dtype=np.uint8)
        bits[cursor : cursor + width] = (symbol >> shifts) & 1
        cursor += width
    return bits


def pack_symbols(symbols: np.ndarray) -> np.ndarray:
    """Pack alternating 2/3-bit transitions into one MSB-first byte stream."""
    logical = _logical_bits(_symbols(symbols))
    return np.packbits(logical, bitorder="big")


def unpack_symbols(packed: np.ndarray, position_count: int) -> np.ndarray:
    """Unpack one periodic stream without materializing a family assignment map."""
    count = _position_count(position_count)
    source = np.asarray(packed)
    if source.ndim != 1 or source.dtype != np.uint8:
        raise ValueError("QTIP2.5-PERIODIC packed payload must be one-dimensional uint8")
    code_bits = count * 5 // 2
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
    for position in range(count):
        width = 2 if position % 2 == 0 else 3
        value = 0
        for bit in bits[cursor : cursor + width]:
            value = (value << 1) | int(bit)
        result[position] = value
        cursor += width
    return result


def _decode_bits(bits: np.ndarray, position_count: int, lut: np.ndarray) -> np.ndarray:
    table = np.asarray(lut)
    if table.shape != (1 << 16, 2):
        raise ValueError(f"QTIP LUT must have shape (65536, 2), got {table.shape}")
    if position_count == 0:
        return np.empty((0, 2), dtype=table.dtype)
    states = np.empty(position_count, dtype=np.uint16)
    cursor = 0
    bit_count = bits.size
    for position in range(position_count):
        state = 0
        for offset in range(16):
            state = (state << 1) | int(bits[(cursor + offset) % bit_count])
        states[position] = state
        cursor += 2 if position % 2 == 0 else 3
    return np.ascontiguousarray(table[states])


def decode_symbols(symbols: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Reference decode of periodic transition symbols through 16-bit QTIP states."""
    validated = _symbols(symbols)
    return _decode_bits(_logical_bits(validated), validated.size, lut)


def decode_packed(
    packed: np.ndarray, position_count: int, lut: np.ndarray
) -> np.ndarray:
    """Decode the physical periodic byte stream directly to QTIP LUT vectors."""
    symbols = unpack_symbols(packed, position_count)
    code_bits = _logical_bits(symbols)
    return _decode_bits(code_bits, symbols.size, lut)


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
        "codec_form": "qtip25_periodic_23",
        "rate_num": 5,
        "rate_den": 2,
        "position_count": count,
        "code_bits": code_bits,
        "code_payload_bytes": code_bytes,
        "code_padding_bits": code_bytes * 8 - code_bits,
        "selected_indices_bytes": 0,
        "assignment_map_bytes": 0,
        "routing_bytes": 0,
        "transform_bytes": transform,
        "scale_bytes": scale,
        "auxiliary_bytes": auxiliary,
        "logical_expert_plane_bytes": logical,
        "deduplicated_shared_tlut_bytes": shared,
        "code_bpw": 2.5 if count else 0.0,
        "auxiliary_bpw": bpw(auxiliary),
        "logical_expert_plane_bpw": bpw(logical),
    }
