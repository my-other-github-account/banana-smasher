from __future__ import annotations

import numpy as np
import torch

from banana_smasher.qtip_periodic import decode_packed, pack_symbols
from banana_smasher_plugin.periodic_qtip import (
    PERIODIC_QTIP25_RUNTIME_FAMILY,
    dequantize_periodic_blocks,
)


def test_periodic_torch_consumer_matches_direct_codec_for_two_blocks() -> None:
    rng = np.random.default_rng(7002)
    symbols = np.empty((2, 128), dtype=np.uint8)
    symbols[:, 0::2] = rng.integers(0, 16, size=(2, 64), dtype=np.uint8)
    symbols[:, 1::2] = rng.integers(0, 64, size=(2, 64), dtype=np.uint8)
    packed = np.stack([pack_symbols(row) for row in symbols])
    lut = np.arange(1 << 17, dtype=np.float32).reshape(1 << 16, 2)

    observed = dequantize_periodic_blocks(
        torch.from_numpy(packed).reshape(1, 2, 80),
        torch.from_numpy(lut),
    )
    expected = np.stack(
        [decode_packed(row, 128, lut).reshape(16, 16) for row in packed]
    ).reshape(1, 2, 16, 16)

    assert PERIODIC_QTIP25_RUNTIME_FAMILY == "qtip25_periodic"
    assert observed.shape == (1, 2, 16, 16)
    assert np.array_equal(observed.numpy(), expected)
