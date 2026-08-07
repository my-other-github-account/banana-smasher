from __future__ import annotations

import numpy as np


def unpack_d4_codes(
    packed: np.ndarray | bytes | bytearray | memoryview, *, bits: int, count: int
) -> np.ndarray:
    source = (
        np.frombuffer(packed, dtype=np.uint8)
        if isinstance(packed, (bytes, bytearray, memoryview))
        else np.asarray(packed, dtype=np.uint8)
    )
    bit_rows = np.unpackbits(
        source,
        count=count * bits,
        bitorder="little",
    ).reshape(count, bits)
    weights = np.left_shift(np.uint16(1), np.arange(bits, dtype=np.uint16))
    return np.asarray(bit_rows @ weights, dtype=np.uint16)


def decode_d4_expert(
    packed_codes: np.ndarray,
    packed_scales: np.ndarray,
    codebook: np.ndarray,
    *,
    bits: int,
    rows: int,
    columns: int,
    torch: object,
    device: str,
) -> object:
    code_count = rows * columns // 4
    codes = unpack_d4_codes(packed_codes, bits=bits, count=code_count)
    indices = torch.from_numpy(codes.astype(np.int64, copy=False)).to(device)
    vectors = torch.from_numpy(
        np.asarray(codebook, dtype=np.float16).copy()
    ).to(device=device, dtype=torch.float32)
    values = vectors[indices].reshape(rows, columns)
    scales = torch.from_numpy(
        np.asarray(packed_scales, dtype=np.uint8).copy().reshape(rows, -1)
    ).to(device=device, dtype=torch.float32)
    scales = torch.exp2(scales - 127.0).repeat_interleave(32, dim=1)
    return (values * scales[:, :columns]).to(torch.bfloat16)
