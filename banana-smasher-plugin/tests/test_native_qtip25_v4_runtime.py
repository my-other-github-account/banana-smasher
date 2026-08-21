from __future__ import annotations

import numpy as np
import torch

from banana_smasher.qtip1 import gaussian_tlut
from banana_smasher.qtip25_native_v4 import (
    decode_native_v4,
    native_v4_geometry,
    pack_native_v4_states,
    states_from_native_v4_packed,
)
from banana_smasher_plugin.native_qtip25_v4 import (
    NATIVE_QTIP25_V4_RUNTIME_FAMILY,
    dequantize_native_v4_blocks,
    native_v4_decode_counters,
    reset_native_v4_decode_counters,
)


def test_native_v4_installed_consumer_matches_reference_and_counts_no_fallback() -> None:
    rng = np.random.default_rng(57101415)
    raw = rng.integers(0, 2, size=(2, 640), dtype=np.uint8)
    packed = np.packbits(raw, axis=1, bitorder="big")
    states = states_from_native_v4_packed(packed, steps=64)
    packed = pack_native_v4_states(states)
    tlut = gaussian_tlut(bits=9, columns=2)

    reset_native_v4_decode_counters()
    observed = dequantize_native_v4_blocks(
        torch.from_numpy(packed).reshape(1, 2, 80), torch.from_numpy(tlut)
    )
    expected = decode_native_v4(
        packed,
        np.ones(2, dtype=np.float32),
        positions=256,
        tlut=tlut,
    ).reshape(1, 2, 16, 16)

    assert NATIVE_QTIP25_V4_RUNTIME_FAMILY == "qtip25_native_v4"
    assert torch.equal(observed, torch.from_numpy(expected))
    assert native_v4_decode_counters() == {
        "decode_calls": 1,
        "decode_blocks": 2,
        "decode_code_bytes": 160,
        "cuda_decode_calls": 0,
        "fallback_calls": 0,
    }


def test_native_v4_installed_consumer_decodes_exact_three_bpw_geometry() -> None:
    geometry = native_v4_geometry(3.0)
    rng = np.random.default_rng(4899)
    raw = rng.integers(0, 2, size=(2, 64 * geometry.B), dtype=np.uint8)
    packed = np.packbits(raw, axis=1, bitorder="big")
    states = states_from_native_v4_packed(packed, steps=64, geometry=geometry)
    packed = pack_native_v4_states(states, geometry=geometry)
    tlut = gaussian_tlut(bits=9, columns=2)

    observed = dequantize_native_v4_blocks(
        torch.from_numpy(packed).reshape(1, 2, 96),
        torch.from_numpy(tlut),
        bpw=3.0,
    )
    expected = decode_native_v4(
        packed,
        np.ones(2, dtype=np.float32),
        positions=256,
        tlut=tlut,
        geometry=geometry,
    ).reshape(1, 2, 16, 16)
    assert torch.equal(observed, torch.from_numpy(expected))
