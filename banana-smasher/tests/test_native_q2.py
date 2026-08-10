from __future__ import annotations

import numpy as np

from banana_smasher.native_q2 import (
    DUPLICATE_CHILD_PARENT_PAIRS,
    PARENT_LUT_DATA_SHA256,
    SEEDED_LUT_DATA_SHA256,
    canonical_parent_lut,
    decode_states,
    pack_trellis,
    seeded_parent_lut,
    sha256_bytes,
    state_levels,
    unpack_trellis,
)


def test_native_q2_lut_is_exact_and_procedural() -> None:
    parent = canonical_parent_lut()
    seeded = seeded_parent_lut()

    assert parent.shape == (1024,)
    assert parent.dtype == np.float16
    assert np.unique(parent).size == 1024
    assert sha256_bytes(parent.tobytes()) == PARENT_LUT_DATA_SHA256
    assert seeded.shape == (1024,)
    assert seeded.dtype == np.float16
    assert np.unique(seeded).size == 913
    assert len(DUPLICATE_CHILD_PARENT_PAIRS) == 111
    assert sha256_bytes(seeded.tobytes()) == SEEDED_LUT_DATA_SHA256
    for child, parent_index in DUPLICATE_CHILD_PARENT_PAIRS:
        assert seeded[child].tobytes() == parent[parent_index].tobytes()


def test_native_q2_state_decode_needs_no_state_map() -> None:
    states = np.arange(65536, dtype=np.uint16)
    levels = state_levels(states)
    decoded = decode_states(states)
    seeded = seeded_parent_lut()

    assert levels.shape == states.shape
    assert np.unique(levels).size == 913
    assert decoded.dtype == np.float16
    assert np.array_equal(decoded, seeded[levels])
    assert set(levels.tolist()).isdisjoint(
        child for child, _ in DUPLICATE_CHILD_PARENT_PAIRS
    )


def test_native_q2_pack_round_trip_preserves_transition_bits() -> None:
    random = np.random.default_rng(36)
    encoded = random.integers(
        np.iinfo(np.int16).min,
        np.iinfo(np.int16).max + 1,
        size=(3, 5, 256),
        dtype=np.int16,
    )

    packed = pack_trellis(encoded)
    unpacked = unpack_trellis(packed)

    assert packed.shape == (3, 5, 32)
    assert packed.dtype == np.int16
    assert np.array_equal(unpacked, encoded.view(np.uint16) & 3)
