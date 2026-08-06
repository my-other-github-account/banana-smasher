from __future__ import annotations

from typing import Any

PERIODIC_QTIP25_RUNTIME_FAMILY = "qtip25_periodic"
_TRANSITIONS_PER_BLOCK = 128
_CODE_BITS_PER_BLOCK = 640
_CODE_BYTES_PER_BLOCK = 80


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("periodic QTIP runtime requires torch") from exc
    return torch


def dequantize_periodic_blocks(packed: Any, lut: Any) -> Any:
    """Decode MSB-first periodic QTIP2.5 blocks on the input torch device.

    Each 80-byte block is one circular 640-bit stream.  Alternating 4/6-bit
    transitions index 128 overlapping 16-bit QTIP states and emit V=2 values,
    returned as a row-major 16x16 block.
    """
    torch = _torch()
    if not isinstance(packed, torch.Tensor) or packed.dtype != torch.uint8:
        raise ValueError("periodic QTIP codes must be a torch.uint8 tensor")
    if packed.ndim < 1 or packed.shape[-1] != _CODE_BYTES_PER_BLOCK:
        raise ValueError(
            "periodic QTIP codes must end in exactly 80 bytes per 16x16 block"
        )
    if (
        not isinstance(lut, torch.Tensor)
        or tuple(lut.shape) != (1 << 16, 2)
        or not lut.is_floating_point()
    ):
        raise ValueError("periodic QTIP LUT must be a floating tensor of shape (65536, 2)")
    if packed.device != lut.device:
        raise ValueError("periodic QTIP codes and LUT must be on the same device")

    leading_shape = tuple(packed.shape[:-1])
    flat = packed.reshape(-1, _CODE_BYTES_PER_BLOCK)
    byte_shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.int64)
    bits = ((flat.unsqueeze(-1) >> byte_shifts) & 1).reshape(
        -1, _CODE_BITS_PER_BLOCK
    )
    transition_ids = torch.arange(
        _TRANSITIONS_PER_BLOCK, device=packed.device, dtype=torch.int64
    )
    widths = torch.where((transition_ids & 1) == 0, 4, 6)
    starts = torch.cumsum(widths, dim=0) - widths
    bit_offsets = torch.arange(16, device=packed.device, dtype=torch.int64)
    indexes = (starts.unsqueeze(1) + bit_offsets.unsqueeze(0)) % _CODE_BITS_PER_BLOCK
    state_weights = 1 << torch.arange(
        15, -1, -1, device=packed.device, dtype=torch.int64
    )
    states = torch.sum(bits[:, indexes] * state_weights, dim=-1, dtype=torch.int64)
    decoded = lut[states].reshape(*leading_shape, 16, 16)
    return decoded
