from __future__ import annotations

import numpy as np
import pytest

import banana_smasher
from banana_smasher.banana_v1 import (
    BANANA_V1_GEOMETRY,
    BANANA_V1_MULTIPLIER,
    BANANA_V1_OFFSET,
    banana_v1_gaussian_codebook,
    banana_v1_inverse_transform,
    banana_v1_state_levels,
    banana_v1_transform,
    banana_v1_wire_accounting,
    banana_v1_provider,
    build_banana_v1,
    decode_banana_v1,
    decode_banana_v1_torch,
    expand_banana_v1_codebook,
    ldlq_banana_v1_matrix,
    predict_banana_v1_candidate,
    solve_banana_v1,
    states_from_banana_v1_packed,
    verify_banana_v1_candidate,
    write_banana_v1_candidate,
)
from banana_smasher.backpack_providers import price_backpack_candidate


def _closed_states(symbols: np.ndarray) -> np.ndarray:
    branch_bits = np.unpackbits(symbols.astype(np.uint8), bitorder="big").reshape(
        -1, 8
    )[:, 6:]
    stream = branch_bits.reshape(-1)
    circular = np.concatenate((stream, stream[:16]))
    states = np.empty(len(symbols), dtype=np.int32)
    for step in range(len(symbols)):
        value = 0
        for bit in circular[step * 2 : step * 2 + 16]:
            value = (value << 1) | int(bit)
        states[step] = value
    return states


def test_banana_v1_native_codebook_is_balanced_and_two_kib() -> None:
    levels = banana_v1_state_levels()
    counts = np.bincount(levels, minlength=1024)
    codebook = banana_v1_gaussian_codebook()

    assert BANANA_V1_GEOMETRY.as_mapping() == {
        "L": 16,
        "B": 2,
        "V": 1,
        "rate_num": 2,
        "rate_den": 1,
        "codebook_levels": 1024,
        "decode_mode": "banana_affine_gaussian",
    }
    assert BANANA_V1_MULTIPLIER == 48917
    assert BANANA_V1_OFFSET == 50631
    assert BANANA_V1_MULTIPLIER & 1
    assert levels.shape == (65536,)
    assert np.array_equal(counts, np.full(1024, 64))
    assert codebook.dtype == np.float16
    assert codebook.shape == (1024,)
    assert codebook.nbytes == 2048
    assert np.isclose(np.mean(codebook.astype(np.float64) ** 2), 1.0, rtol=2e-3)


def test_banana_v1_cyclic_solve_pack_and_device_decode() -> None:
    torch = pytest.importorskip("torch")
    symbols = np.asarray([0, 3, 1, 2, 3, 0, 2, 1], dtype=np.uint8)
    states = _closed_states(symbols)
    state_lut = np.full(65536, 1000.0, dtype=np.float32)
    target = np.arange(len(states), dtype=np.float32) - 3.5
    state_lut[states] = target

    solved = solve_banana_v1(
        target[None, :],
        state_lut=state_lut,
        scales=np.ones(1, dtype=np.float32),
    )
    assert solved.distortion == 0.0
    assert np.array_equal(solved.states, states[None, :])
    assert np.array_equal(
        states_from_banana_v1_packed(solved.packed, steps=len(states)), solved.states
    )
    assert np.array_equal(
        decode_banana_v1(
            solved.packed,
            solved.scales,
            positions=len(states),
            state_lut=state_lut,
        ),
        target[None, :],
    )
    observed_torch = decode_banana_v1_torch(
        torch.from_numpy(solved.packed),
        torch.from_numpy(solved.scales),
        positions=len(states),
        state_lut=torch.from_numpy(state_lut),
    )
    assert torch.equal(observed_torch, torch.from_numpy(target[None, :]))


def test_banana_v1_global_scale_search_is_never_worse_than_rms_only() -> None:
    rng = np.random.default_rng(41)
    target = rng.normal(size=(1, 32)).astype(np.float32)
    rms_only = solve_banana_v1(target, scale_factors=(1.0,))
    searched = solve_banana_v1(target)

    assert searched.distortion <= rms_only.distortion
    assert searched.scale_factor in searched.scale_factors
    assert len(searched.scale_factors) > 1


def test_banana_v1_full_hadamard_transform_roundtrips() -> None:
    source = np.arange(256, dtype=np.float32).reshape(16, 16) / 127.0 - 1.0
    transformed, su, sv = banana_v1_transform(source, seed=73)
    restored = banana_v1_inverse_transform(transformed, su=su, sv=sv)

    assert transformed.shape == source.shape
    assert set(np.unique(su)) <= {-1.0, 1.0}
    assert set(np.unique(sv)) <= {-1.0, 1.0}
    assert np.allclose(restored, source, rtol=2e-6, atol=2e-6)


def test_banana_v1_ldlq_path_closes_exact_wire_and_accounting() -> None:
    rng = np.random.default_rng(89)
    transformed = rng.normal(size=(16, 16)).astype(np.float32)
    lower = np.zeros((16, 16), dtype=np.float32)
    result = ldlq_banana_v1_matrix(
        transformed,
        lower,
        scale_factors=(1.0,),
    )

    assert result.decoded.shape == transformed.shape
    assert result.packed.shape == (1, 64)
    assert np.isfinite(result.decoded).all()
    assert np.array_equal(
        states_from_banana_v1_packed(result.packed, steps=256), result.states
    )
    accounting = banana_v1_wire_accounting(
        position_count=256,
        sequence_count=1,
        scale_bytes=result.scales.nbytes,
        transform_bytes=64,
        shared_codebook_bytes=2048,
    )
    assert accounting["code_bits"] == 512
    assert accounting["code_payload_bytes"] == 64
    assert accounting["code_bpw"] == 2.0
    assert accounting["full_wire_bytes"] == 64 + result.scales.nbytes + 64 + 2048


def test_banana_v1_is_exported_from_public_api() -> None:
    assert banana_smasher.solve_banana_v1 is solve_banana_v1
    assert banana_smasher.ldlq_banana_v1_matrix is ldlq_banana_v1_matrix
    assert banana_smasher.banana_v1_gaussian_codebook is banana_v1_gaussian_codebook
    assert expand_banana_v1_codebook().shape == (65536,)


def test_banana_v1_build_artifact_is_priceable_and_predictable(tmp_path) -> None:
    rng = np.random.default_rng(137)
    source = rng.normal(size=(16, 16)).astype(np.float32)
    result = build_banana_v1(
        source,
        lower=np.zeros((16, 16), dtype=np.float32),
        seed=101,
        scale_factors=(1.0,),
    )
    root = tmp_path / "banana-v1"
    receipt = write_banana_v1_candidate(root, result)

    assert receipt["provider_id"] == "banana_v1"
    assert receipt["geometry"] == BANANA_V1_GEOMETRY.as_mapping()
    assert verify_banana_v1_candidate(root)
    price = price_backpack_candidate(root / "BANANA_V1_RECEIPT.json")
    assert price.cell_payload_bytes == 64 + 4 + 64
    assert price.activation_bytes == 2048
    assert np.allclose(predict_banana_v1_candidate(root), result.decoded)
    provider = banana_v1_provider()
    assert provider.provider_id == "banana_v1"
    assert provider.kind == "banana_v1"
    assert provider.runtime_family == "banana_v1"
    assert provider.verify(root)
