"""Canonical QTIP packed tensor decoder used by builder conformance.

The byte-unswizzle must be contiguous before reinterpreting uint16, including
under torch.compile. Keep this implementation in the same git pin as the builder.
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
    torch._dynamo.graph_break()
    compressed = compressed.contiguous().view(torch.uint16).reshape(
        m // 16, k // 16, bits_per_block // 16
    )
    assert L <= 16
    blocked = compressed.reshape(R * m * k // bits_per_block, bits_per_block // 16, 1)
    blocked_roll = torch.roll(blocked.to(torch.int32), -1, -2).to(blocked.dtype)
    blocked32 = torch.cat((blocked_roll, blocked), dim=-1).reshape(
        blocked.shape[0], -1
    ).contiguous().view(torch.uint32)
    expanded32 = blocked32.reshape(*blocked32.shape, 1).expand(
        *blocked32.shape, 16
    ).view(torch.int32)
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
