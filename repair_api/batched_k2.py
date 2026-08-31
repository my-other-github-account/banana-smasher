"""Vectorized, byte-equivalent official K2 decode primitives."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch

_MUL1 = 0x83DCD12D


@lru_cache(maxsize=None)
def _inverse_permutation(device_type: str, device_index: int | None) -> torch.Tensor:
    from banana_smasher.q2_codec import tensor_core_permutation

    device = torch.device(device_type, device_index)
    permutation = torch.as_tensor(
        tensor_core_permutation(), device=device, dtype=torch.int64
    )
    return torch.argsort(permutation)


def decode_k2_matrix_batched(
    packed: torch.Tensor, parent_lut: torch.Tensor
) -> torch.Tensor:
    """Decode int16[..., tiles_k, tiles_m, 32] in one tensor program."""
    if packed.ndim < 4 or packed.shape[-1] != 32 or packed.dtype != torch.int16:
        raise ValueError("packed must be int16[..., tiles_k, tiles_m, 32]")
    if parent_lut.dtype != torch.float16 or parent_lut.shape != (1024,):
        raise ValueError("parent_lut must be float16[1024]")
    if packed.device != parent_lut.device:
        raise ValueError("packed and parent_lut must share one device")

    words = (packed.to(torch.int32) & 0xFFFF).reshape(*packed.shape[:-1], 16, 2)
    words = words[..., [1, 0]]
    shifts = torch.arange(14, -1, -2, device=packed.device, dtype=torch.int32)
    codes = ((words.unsqueeze(-1) >> shifts) & 3).reshape(*packed.shape[:-1], 256)

    # The official decoder forms each cyclic eight-code state serially. Unfold
    # expresses the same windows in one kernel while preserving integer math.
    circular = torch.cat((codes[..., 249:], codes), dim=-1)
    windows = circular.unfold(-1, 8, 1).to(torch.int64)
    state_shifts = torch.arange(14, -1, -2, device=packed.device, dtype=torch.int64)
    states = (windows << state_shifts).sum(dim=-1) & 0xFFFF
    products = (states * _MUL1) & 0xFFFFFFFF
    parents = (
        (products & 0xFF)
        + ((products >> 8) & 0xFF)
        + ((products >> 16) & 0xFF)
        + ((products >> 24) & 0xFF)
    )
    decoded = parent_lut[parents].to(torch.float32)
    inverse = _inverse_permutation(packed.device.type, packed.device.index)
    decoded = decoded[..., inverse]

    leading = tuple(int(value) for value in decoded.shape[:-3])
    tiles_k, tiles_m = map(int, decoded.shape[-3:-1])
    tiled = decoded.reshape(*leading, tiles_k, tiles_m, 16, 16)
    lead = len(leading)
    order = list(range(lead)) + [lead, lead + 2, lead + 1, lead + 3]
    return tiled.permute(*order).reshape(
        *leading, tiles_k * 16, tiles_m * 16
    )


def inverse_transform_batched(
    weight: torch.Tensor, su: torch.Tensor, sv: torch.Tensor
) -> torch.Tensor:
    """Apply official H128/scales to a batch of decoded float32 matrices."""
    if weight.ndim < 3 or weight.dtype != torch.float32:
        raise ValueError("weight must be float32[..., input, output]")
    leading = tuple(int(value) for value in weight.shape[:-2])
    k, m = map(int, weight.shape[-2:])
    if su.shape != (*leading, k) or sv.shape != (*leading, m):
        raise ValueError("batched inverse-transform scale geometry mismatch")

    from banana_smasher.qtip_k2 import normalized_hadamard_128

    hadamard = normalized_hadamard_128(weight.device, weight.dtype)
    matrix = torch.matmul(
        hadamard, weight.reshape(*leading, -1, 128, m)
    ).reshape_as(weight)
    matrix *= su.to(device=matrix.device, dtype=matrix.dtype).unsqueeze(-1)
    matrix = torch.matmul(
        matrix.reshape(*leading, k, -1, 128), hadamard
    ).reshape_as(matrix)
    matrix *= sv.to(device=matrix.device, dtype=matrix.dtype).unsqueeze(-2)
    return matrix
