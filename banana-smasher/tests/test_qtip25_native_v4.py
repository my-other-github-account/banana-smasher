from __future__ import annotations

import numpy as np
import torch

from banana_smasher.qtip25_native_v4 import (
    NATIVE_QTIP25_GEOMETRY,
    decode_native_v4,
    decode_native_v4_torch,
    expand_native_v4_tlut,
    native_v4_geometry,
    _native_v4_cuda_source,
    native_v4_wire_accounting,
    pack_native_v4_states,
    solve_native_v4,
    states_from_native_v4_packed,
)
from banana_smasher.qtip1 import gaussian_tlut


def _closed_states(symbols: np.ndarray) -> np.ndarray:
    bits = np.unpackbits(symbols.astype(">u2").view(np.uint8), bitorder="big")
    bit_count = symbols.size * 10
    bits = bits.reshape(-1, 16)[:, 6:].reshape(-1)[:bit_count]
    stream = np.concatenate((bits, bits[:16]))
    states = np.empty(symbols.size, dtype=np.int32)
    for step in range(symbols.size):
        value = 0
        for bit in stream[step * 10 : step * 10 + 16]:
            value = (value << 1) | int(bit)
        states[step] = value
    return states


def test_native_v4_roundtrip_decode_and_exact_code_rate() -> None:
    symbols = np.array([0, 1023, 17, 513, 7, 992, 341, 682], dtype=np.uint16)
    states = _closed_states(symbols)[None, :]
    packed = pack_native_v4_states(states)
    observed_states = states_from_native_v4_packed(packed, steps=symbols.size)

    assert NATIVE_QTIP25_GEOMETRY.as_mapping() == {
        "L": 16,
        "B": 10,
        "V": 4,
        "rate_num": 5,
        "rate_den": 2,
        "phase_count": 1,
        "unique_transition_bits_per_payload": 1,
        "alternation": False,
        "member_averaging": False,
        "tlut_bits": 9,
        "decode_mode": "paired_quantlut_sym",
    }
    assert np.array_equal(observed_states, states)
    assert packed.shape == (1, 10)

    tlut = gaussian_tlut(bits=9, columns=2)
    scales = np.array([0.75], dtype=np.float32)
    expected = expand_native_v4_tlut(tlut)[states].reshape(1, -1) * scales[:, None]
    assert np.array_equal(
        decode_native_v4(packed, scales, positions=32, tlut=tlut),
        expected,
    )
    observed_torch = decode_native_v4_torch(
        torch.from_numpy(packed),
        torch.from_numpy(scales),
        positions=32,
        tlut=torch.from_numpy(tlut),
    )
    assert torch.equal(observed_torch, torch.from_numpy(expected))

    accounting = native_v4_wire_accounting(
        position_count=32,
        transform_bytes=20,
        scale_bytes=4,
        shared_tlut_bytes=4096,
    )
    assert accounting["code_bits"] == 80
    assert accounting["code_payload_bytes"] == 10
    assert accounting["code_padding_bits"] == 0
    assert accounting["code_bpw"] == 2.5
    assert accounting["phase_count"] == 1
    assert accounting["unique_transition_bits_per_payload"] == 1
    assert accounting["assignment_map_bytes"] == 0
    assert accounting["routing_bytes"] == 0



def test_qtip1_cuda_source_strides_all_prefixes_beyond_one_block() -> None:
    source = _native_v4_cuda_source(native_v4_geometry(1.0))

    assert "for (int prefix = tid; prefix < PREFIXES; prefix += THREADS)" in source
    assert "for (int branch = 0; branch < (1 << BRANCH_BITS); ++branch)" in source
    assert source.count("static_cast<int64_t>(step - capture_step)") == 3


def test_native_v4_torch_extracts_state_windows_without_reductions(monkeypatch) -> None:
    symbols = np.arange(32, dtype=np.uint16)
    states = _closed_states(symbols)[None, :]
    packed = pack_native_v4_states(states)
    tlut = gaussian_tlut(bits=9, columns=2)
    calls = 0
    original_sum = torch.sum

    def counted_sum(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_sum(*args, **kwargs)

    monkeypatch.setattr(torch, "sum", counted_sum)

    observed = decode_native_v4_torch(
        torch.from_numpy(packed),
        torch.ones(1),
        positions=128,
        tlut=torch.from_numpy(tlut),
    )

    expected = decode_native_v4(
        packed,
        np.ones(1, dtype=np.float32),
        positions=128,
        tlut=tlut,
    )
    assert torch.equal(observed, torch.from_numpy(expected))
    assert calls == 0


def test_native_v4_torch_direct_byte_windows_match_qtip3_b12_reference() -> None:
    geometry = native_v4_geometry(3.0)
    rng = np.random.default_rng(7)
    packed = rng.integers(0, 256, size=(2, 24), dtype=np.uint8)
    scales = np.asarray([0.75, 1.25], dtype=np.float32)
    tlut = gaussian_tlut(bits=9, columns=2)

    expected = decode_native_v4(
        packed,
        scales,
        positions=64,
        tlut=tlut,
        geometry=geometry,
    )
    observed = decode_native_v4_torch(
        torch.from_numpy(packed),
        torch.from_numpy(scales),
        positions=64,
        tlut=torch.from_numpy(tlut),
        geometry=geometry,
    )

    assert torch.equal(observed, torch.from_numpy(expected))


def test_native_v4_reference_solve_recovers_zero_distortion_closed_path() -> None:
    symbols = np.array([0, 1023, 17, 513, 7, 992, 341, 682], dtype=np.uint16)
    states = _closed_states(symbols)
    lut = np.full((1 << 16, 4), 1000.0, dtype=np.float32)
    target = np.empty((symbols.size, 4), dtype=np.float32)
    for step, state in enumerate(states):
        target[step] = (step, step + 0.25, step + 0.5, step + 0.75)
        lut[state] = target[step]

    solved = solve_native_v4(
        target[None, :, :], state_lut=lut, scales=np.ones(1, dtype=np.float32)
    )

    assert solved.distortion == 0.0
    assert np.array_equal(
        states_from_native_v4_packed(solved.packed, steps=symbols.size),
        solved.states,
    )
    assert np.array_equal(
        lut[solved.states].reshape(1, -1),
        target.reshape(1, -1),
    )


