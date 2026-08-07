from __future__ import annotations

from typing import Any

from banana_smasher.qtip25_native_v4 import decode_native_v4_torch, native_v4_geometry

NATIVE_QTIP_V4_RUNTIME_FAMILY = "qtip_native_v4"
NATIVE_QTIP25_V4_RUNTIME_FAMILY = "qtip25_native_v4"
_DECODE_CALLS = 0
_DECODE_BLOCKS = 0
_DECODE_CODE_BYTES = 0
_DECODE_CUDA_CALLS = 0
_FALLBACK_CALLS = 0


def native_v4_decode_counters() -> dict[str, int]:
    return {
        "decode_calls": _DECODE_CALLS,
        "decode_blocks": _DECODE_BLOCKS,
        "decode_code_bytes": _DECODE_CODE_BYTES,
        "cuda_decode_calls": _DECODE_CUDA_CALLS,
        "fallback_calls": _FALLBACK_CALLS,
    }


def reset_native_v4_decode_counters() -> None:
    global _DECODE_CALLS, _DECODE_BLOCKS, _DECODE_CODE_BYTES, _DECODE_CUDA_CALLS
    global _FALLBACK_CALLS
    _DECODE_CALLS = 0
    _DECODE_BLOCKS = 0
    _DECODE_CODE_BYTES = 0
    _DECODE_CUDA_CALLS = 0
    _FALLBACK_CALLS = 0


def dequantize_native_v4_blocks(packed: Any, tlut: Any, *, bpw: object = 2.5) -> Any:
    """Decode homogeneous L16/B/V4 blocks on the input Torch device."""
    import torch

    geometry = native_v4_geometry(bpw)
    block_bytes = 8 * geometry.B
    global _DECODE_CALLS, _DECODE_BLOCKS, _DECODE_CODE_BYTES, _DECODE_CUDA_CALLS
    if not isinstance(packed, torch.Tensor) or packed.dtype != torch.uint8:
        raise ValueError("native QTIP2.5 V4 codes must be a torch.uint8 tensor")
    if packed.ndim < 1 or packed.shape[-1] != block_bytes:
        raise ValueError(
            f"native QTIP V4 codes must end in {block_bytes} bytes per 16x16 block"
        )
    if not isinstance(tlut, torch.Tensor) or tuple(tlut.shape) != (512, 2):
        raise ValueError("native QTIP2.5 V4 TLUT must have shape (512,2)")
    if packed.device != tlut.device or not tlut.is_floating_point():
        raise ValueError("native QTIP2.5 V4 codes/TLUT must be floating-compatible on one device")
    leading = tuple(packed.shape[:-1])
    flat = packed.reshape(-1, block_bytes)
    scales = torch.ones(flat.shape[0], device=packed.device, dtype=tlut.dtype)
    decoded = decode_native_v4_torch(
        flat, scales, positions=256, tlut=tlut, geometry=geometry
    )
    _DECODE_CALLS += 1
    _DECODE_BLOCKS += int(flat.shape[0])
    _DECODE_CODE_BYTES += int(flat.numel())
    if packed.is_cuda:
        _DECODE_CUDA_CALLS += 1
    return decoded.reshape(*leading, 16, 16)


__all__ = [
    "NATIVE_QTIP_V4_RUNTIME_FAMILY",
    "NATIVE_QTIP25_V4_RUNTIME_FAMILY",
    "dequantize_native_v4_blocks",
    "native_v4_decode_counters",
    "reset_native_v4_decode_counters",
]
