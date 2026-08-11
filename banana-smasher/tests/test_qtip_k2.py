from __future__ import annotations

from pathlib import Path
import runpy

import numpy as np
import pytest
import torch

from banana_smasher.q2_codec import (
    cyclic_states_from_codes,
    k2_lut_fp16,
    pack_k2_trellis,
)
from banana_smasher.qtip_k2 import (
    _block_ldl_lower,
    _block_trace,
    _finalize_raw_hessian_cpu,
    _finalize_raw_hessian_on_device,
    _require_same_device,
    _source_transform,
    _transformed_hessian,
    _transformed_raw_hessian,
    decode_k2_matrix,
    pack_k2,
    unpack_k2,
)


def test_torch_k2_pack_matches_canonical_numpy_wire_order() -> None:
    codes = np.tile(np.arange(4, dtype=np.uint16), 64).reshape(1, 1, 256)
    states = cyclic_states_from_codes(codes)
    encoded = torch.from_numpy(states.view(np.int16))

    packed = pack_k2(encoded)

    np.testing.assert_array_equal(
        packed.numpy().view(np.uint16), pack_k2_trellis(states)
    )
    assert torch.equal(unpack_k2(packed), encoded)


def test_packed_decode_uses_only_native_parent_lut() -> None:
    codes = np.tile(np.array([3, 0, 2, 1], dtype=np.uint16), 64).reshape(1, 1, 256)
    states = cyclic_states_from_codes(codes)
    packed = pack_k2(torch.from_numpy(states.view(np.int16)))
    lut = torch.from_numpy(k2_lut_fp16())

    matrix = decode_k2_matrix(packed, lut)

    assert matrix.shape == (16, 16)
    assert matrix.dtype == torch.float32
    assert torch.isfinite(matrix).all()
    assert torch.unique(lut).numel() == 913


def test_k2_boundaries_reject_mixed_devices() -> None:
    cpu = torch.empty(1)
    meta = torch.empty(1, device="meta")

    with np.testing.assert_raises_regex(ValueError, "share one device"):
        _require_same_device(cpu, meta)


def test_block_ldl_lower_reuses_the_input_storage_before_normalization() -> None:
    torch.manual_seed(4)
    value = torch.randn(32, 32)
    hessian = value @ value.T + 4 * torch.eye(32)
    original = hessian.clone()

    lower = _block_ldl_lower(hessian)

    assert not torch.equal(hessian, original)
    assert torch.count_nonzero(torch.triu(lower, diagonal=1)) == 0
    assert torch.count_nonzero(lower.diagonal()) == 0


def test_block_trace_matches_full_hessian_quadratic() -> None:
    error = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 8
    basis = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16
    hessian = basis @ basis.T + torch.eye(4)

    actual = _block_trace(error, hessian, block_size=2)
    expected = torch.einsum("ik,ij,jk->", error, hessian, error).item()

    assert actual == expected


def test_source_transform_seed_zero_is_repeatable() -> None:
    source = torch.linspace(-1, 1, 128 * 128).reshape(128, 128)
    lut = torch.from_numpy(k2_lut_fp16())

    def quantize(tiles: torch.Tensor, unused_lut: torch.Tensor):
        del unused_lut
        return torch.round(tiles), torch.zeros_like(tiles, dtype=torch.int16)

    first = _source_transform(source, lut, seed=0, quantize_tiles_fn=quantize)
    second = _source_transform(source, lut, seed=0, quantize_tiles_fn=quantize)

    for name in ("target_inner", "su", "sv", "input_signs", "output_signs"):
        assert torch.equal(first[name], second[name])
    assert first["su"].dtype == torch.float32
    assert first["sv"].dtype == torch.float32
    assert torch.equal(first["suh"], first["su"].half())
    assert torch.equal(first["svh"], first["sv"].half())


def test_raw_hessian_sum_matches_ordered_capture_accumulation() -> None:
    torch.manual_seed(8)
    captures = torch.randn(16, 128)
    signs = torch.randn(128).sign().unsqueeze(1)
    raw_sum = captures.T @ captures
    raw_copy = raw_sum.clone()

    from_captures, *_ = _transformed_hessian(
        captures,
        signs,
        sample_weights=None,
        regularization_sigma=0.025,
    )
    from_sum = _transformed_raw_hessian(
        raw_sum,
        len(captures),
        signs,
        regularization_sigma=0.025,
    )

    assert torch.equal(raw_sum, raw_copy)
    assert torch.equal(from_sum, from_captures)


def test_raw_hessian_finalization_is_cpu_only() -> None:
    raw_sum = torch.eye(32, dtype=torch.float32) * 16

    finalized = _finalize_raw_hessian_cpu(
        raw_sum,
        16,
        regularization_sigma=0.025,
    )

    assert finalized.device.type == "cpu"
    assert torch.equal(finalized.diagonal(), torch.full((32,), 1.025))
    if torch.cuda.is_available():
        with np.testing.assert_raises_regex(ValueError, "finalized on CPU"):
            _finalize_raw_hessian_cpu(
                raw_sum.cuda(),
                16,
                regularization_sigma=0.025,
            )


def test_executable_raw_hessian_finalization_uses_target_device() -> None:
    raw_sum = torch.eye(32, dtype=torch.float32) * 16

    finalized = _finalize_raw_hessian_on_device(
        raw_sum,
        16,
        regularization_sigma=0.025,
        device=raw_sum.device,
    )

    assert finalized.device == raw_sum.device
    assert torch.equal(finalized.diagonal(), torch.full((32,), 1.025))


def test_source_runner_resume_gate_accepts_only_owned_partial_artifacts(
    tmp_path: Path,
) -> None:
    runner = runpy.run_path(str(Path(__file__).with_name("run_q2_k2_source_e000.py")))
    validate = runner["validate_resume_output"]

    output = tmp_path / "run"
    (output / "checkpoints").mkdir(parents=True)
    (output / "members").mkdir()
    (output / "PROGRESS.json").write_text("{}")
    (output / "checkpoints" / "w1.pt").write_bytes(b"checkpoint")
    (output / "members" / "w1.states.npy").write_bytes(b"states")
    validate(output)

    (output / "TERMINAL.json").write_text("{}")
    with pytest.raises(RuntimeError, match="sealed or foreign"):
        validate(output)
