from __future__ import annotations

import hashlib

import numpy as np

from banana_smasher.q2_codec import (
    K2_CHILD_PARENT_PAIRS,
    cyclic_states_from_codes,
    decode_k2_states,
    k2_lut_fp16,
    k2_parent_lut_fp16,
    mul1_lut_indices,
    pack_k2_trellis,
    tensor_core_permutation,
    unpack_k2_trellis,
)


def test_k2_native_lut_has_exact_declared_parent_aliases() -> None:
    parents = k2_parent_lut_fp16()
    lut = k2_lut_fp16()
    children = {child for child, _ in K2_CHILD_PARENT_PAIRS}

    assert parents.shape == lut.shape == (1024,)
    assert parents.dtype == lut.dtype == np.float16
    assert len(K2_CHILD_PARENT_PAIRS) == 111
    assert len(children) == 111
    assert np.unique(parents).size == 1024
    assert np.unique(lut).size == 913
    assert hashlib.sha256(parents.tobytes()).hexdigest() == (
        "cfe39c9eee6226ee2a8694172c5cf9e6c69b2afdb10d5e6d0e14f6b9ec4377e1"
    )
    assert hashlib.sha256(lut.tobytes()).hexdigest() == (
        "1fcb3546038bc65ab7847ef4473a2d1a8c66631315655c1b3d9f989325572a3c"
    )
    for child, parent in K2_CHILD_PARENT_PAIRS:
        assert lut[child].view(np.uint16) == parents[parent].view(np.uint16)


def test_k2_procedural_indices_reach_only_parent_slots() -> None:
    states = np.arange(1 << 16, dtype=np.uint16)
    indices = mul1_lut_indices(states)
    children = {child for child, _ in K2_CHILD_PARENT_PAIRS}

    assert indices.min() == 0
    assert indices.max() == 1005
    assert np.unique(indices).size == 913
    assert children.isdisjoint(set(np.unique(indices).tolist()))
    assert decode_k2_states(states).shape == states.shape


def test_k2_pack_roundtrip_preserves_cyclic_states() -> None:
    codes = np.tile(np.arange(4, dtype=np.uint16), 64).reshape(1, 1, 256)
    states = cyclic_states_from_codes(codes)
    packed = pack_k2_trellis(states)

    assert packed.shape == (1, 1, 32)
    assert packed.dtype == np.uint16
    np.testing.assert_array_equal(unpack_k2_trellis(packed), states)


def test_tensor_core_permutation_is_a_bijection() -> None:
    permutation = tensor_core_permutation()

    assert permutation.shape == (256,)
    np.testing.assert_array_equal(np.sort(permutation), np.arange(256))
    np.testing.assert_array_equal(
        np.arange(256)[permutation][np.argsort(permutation)], np.arange(256)
    )
