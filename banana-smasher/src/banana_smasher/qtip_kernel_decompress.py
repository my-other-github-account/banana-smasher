"""Canonical QTIP packed tensor decoder used by builder conformance.

Byte and word assembly uses integer operations instead of size-changing dtype
views, whose contiguity requirements can be lost during Inductor lowering.
Keep this implementation in the same git pin as the builder.
"""
import torch


@torch.compile
def decode_compressed(L, S, R, V, m, k, compressed, expanded_lut):
    if compressed.dtype != torch.uint16:
        compressed = compressed.view(torch.uint16)
    assert compressed.shape == (R * m * k // 16,)
    block_size = 16 * 16
    bits_per_block = R * block_size
    compressed = (
        compressed.view(torch.uint8)
        .reshape(m // 16 // 2, k // 16 // 2, block_size // 8, 2, 2, R)
        .permute(0, -2, 1, -3, 2, -1)
        .flip((-1,))
        .reshape(m // 16, k // 16, bits_per_block // 16, 2)
        .flip((-1,))
    )
    # Assemble little-endian words explicitly. Inductor may choose a non-unit
    # last stride even for contiguous() before a size-changing dtype-view.
    # Integer assembly preserves wire bits without imposing a physical layout.
    compressed = compressed.to(torch.int32)
    compressed = compressed[..., 0] | (compressed[..., 1] << 8)
    assert L <= 16
    blocked = compressed.reshape(R * m * k // bits_per_block, bits_per_block // 16)
    blocked_roll = torch.roll(blocked, -1, -1)
    blocked32 = blocked_roll | (blocked << 16)
    expanded32 = blocked32.reshape(*blocked32.shape, 1).expand(
        *blocked32.shape, 16
    )
    shifts = torch.arange(0, 16, dtype=torch.int32, device=blocked.device).reshape(
        1, 1, -1
    ).expand(expanded32.shape)
    shifted = expanded32 >> (16 - shifts)
    indices = torch.bitwise_and(
        shifted.reshape(shifted.shape[0], -1)[:, 16 - L::R << V], (1 << L) - 1
    )
    mma_swizzled = expanded_lut[indices]
    return mma_swizzled.reshape(m // 16, k // 16, 16, 16).reshape(
        m // 16, k // 16, 8, 4, 2, 2, 2
    ).permute(0, -2, 2, 1, -3, 3, -1).reshape(m, k)
