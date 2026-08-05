from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from banana_smasher.qtip_matrix_lifetime import build_qtip_bounded
from banana_smasher.qtip_runner import pack_kernel_layout


class _Runner:
    def __init__(self) -> None:
        self.ldl_inputs: tuple[weakref.ReferenceType[torch.Tensor], weakref.ReferenceType[torch.Tensor]] | None = None
        self.pack_observed_released = False
        self.decode_calls = 0

    @staticmethod
    def fwht(value: torch.Tensor) -> torch.Tensor:
        return value.clone()

    @staticmethod
    def build_hessian(windows, signs, device):
        assert windows == ["fit"]
        width = signs.numel()
        return torch.eye(width, device=device), 1, 1.0

    def pack_kernel_layout(self, cb, states, m, k):
        del cb, m, k
        gc.collect()
        assert self.ldl_inputs is not None
        self.pack_observed_released = all(ref() is None for ref in self.ldl_inputs)
        return states.to(torch.uint16), {"canonical_pack_roundtrip_exact": True}

    def decode_packed(self, candidate, kernel_decode, device):
        del kernel_decode
        self.decode_calls += 1
        decoded = candidate["reconstructed_weight"].to(
            device=device, dtype=torch.float32
        )
        return decoded, {
            "fp16_bit_exact": True,
            "packed_sha256": "test-wire",
        }


class _Math:
    @staticmethod
    def block_LDL(hessian, block):
        assert block == 16
        return hessian.clone(), torch.ones(1, device=hessian.device)


class _LDLQ:
    def __init__(self, runner: _Runner) -> None:
        self.runner = runner

    def LDLQ(self, transformed, lower, cb, args, *, buf_cols, for_kernel):
        del cb
        assert args.td_x == args.td_y == 16 and args.V == 2
        assert buf_cols == 128 and for_kernel is True
        self.runner.ldl_inputs = (weakref.ref(transformed), weakref.ref(lower))
        states = torch.zeros(
            (transformed.shape[0], transformed.shape[1] // 2),
            dtype=torch.int16,
            device=transformed.device,
        )
        return transformed.clone(), states


class _Codebook:
    def __init__(self) -> None:
        self.lut = torch.tensor([-1.0, 1.0])
        self.tlut = torch.arange(8, dtype=torch.float16).reshape(4, 2)


def test_bounded_builder_releases_ldl_inputs_before_pack_and_preserves_weight() -> None:
    runner = _Runner()
    source = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64
    candidate, receipt = build_qtip_bounded(
        runner,
        source,
        ["fit"],
        _Codebook(),
        _LDLQ(runner),
        _Math(),
        kernel_decode=None,
        device=torch.device("cpu"),
        rht_seed=7,
    )

    assert runner.pack_observed_released is True
    assert runner.decode_calls == 1
    assert torch.equal(candidate["reconstructed_weight"], source.half())
    lifetime = receipt["matrix_lifetime"]
    assert lifetime["schema"] == "banana-smasher-qtip-matrix-lifetime-v1"
    assert lifetime["max_live_fp32_matrix_equivalents"] <= 2
    assert lifetime["released_before_pack"] == ["lower", "transformed"]
    assert lifetime["reconstructed_weight_device"] == "cpu"
    assert receipt["packed_decode"]["runtime_check_performed"] is True
    assert receipt["packed_decode"]["fp16_bit_exact"] is True
    assert receipt["phase_seconds"]["packed_decode_conformance"] >= 0.0


def test_bounded_builder_uses_manifest_bound_batch_pack_when_top_level_is_absent() -> None:
    runner = _Runner()
    setattr(runner, "pack_kernel_layout", None)
    batch_calls = []

    def pack_kernel_layout_batch(cb, states, m, k):
        del cb, m, k
        gc.collect()
        assert runner.ldl_inputs is not None
        runner.pack_observed_released = all(ref() is None for ref in runner.ldl_inputs)
        batch_calls.append(tuple(states.shape))
        return states.to(torch.uint16), [
            {
                "canonical_pack_roundtrip_exact": True,
                "canonical_packed_sha256": "test-batch-wire",
            }
        ]

    setattr(
        runner,
        "_rate",
        SimpleNamespace(pack_kernel_layout_batch=pack_kernel_layout_batch),
    )
    candidate, receipt = build_qtip_bounded(
        runner,
        torch.eye(8, dtype=torch.float32),
        ["fit"],
        _Codebook(),
        _LDLQ(runner),
        _Math(),
        kernel_decode=None,
        device=torch.device("cpu"),
        rht_seed=19,
    )

    assert batch_calls == [(1, 8, 4)]
    assert runner.pack_observed_released is True
    assert candidate["trellis"].shape == (8, 4)
    assert receipt["canonical_pack"]["canonical_packed_sha256"] == "test-batch-wire"


def test_bounded_builder_decodes_k2_with_the_geometry_bound_kernel_path() -> None:
    runner = _Runner()
    quantized_for_decode = None

    class K2Codebook(_Codebook):
        L = 16
        K = 2
        V = 2
        tlut_bits = 9

        def __init__(self) -> None:
            super().__init__()
            self.lut = self.lut.reshape(1, 2)

    class CapturingLDLQ(_LDLQ):
        def LDLQ(self, *args, **kwargs):
            nonlocal quantized_for_decode
            quantized, states = super().LDLQ(*args, **kwargs)
            quantized_for_decode = quantized.clone()
            return quantized, states

    class KernelDecode:
        calls = []

        def decode_compressed(self, L, S, R, V, m, k, packed, expanded_lut):
            self.calls.append((L, S, R, V, m, k, tuple(packed.shape)))
            assert torch.equal(expanded_lut, K2Codebook().lut.T.contiguous())
            assert quantized_for_decode is not None
            return quantized_for_decode.clone()

    def reject_k3_parent_decode(*_args, **_kwargs):
        raise AssertionError("K2 bounded build must not use the inherited K3 decoder")

    runner.decode_packed = reject_k3_parent_decode
    kernel_decode = KernelDecode()
    candidate, receipt = build_qtip_bounded(
        runner,
        torch.eye(8, dtype=torch.float32),
        ["fit"],
        K2Codebook(),
        CapturingLDLQ(runner),
        _Math(),
        kernel_decode=kernel_decode,
        device=torch.device("cpu"),
        rht_seed=23,
    )

    assert kernel_decode.calls == [(16, 9, 2, 1, 8, 8, (32,))]
    assert torch.equal(candidate["reconstructed_weight"], torch.eye(8).half())
    assert receipt["packed_decode"]["geometry_bound_k"] == 2
    assert receipt["packed_decode"]["runtime_check_performed"] is True
    assert receipt["packed_decode"]["fp16_bit_exact"] is True


def test_bounded_builder_rejects_non_exact_canonical_pack_roundtrip() -> None:
    class NonExactRunner(_Runner):
        def pack_kernel_layout(self, cb, states, m, k):
            packed, _receipt = super().pack_kernel_layout(cb, states, m, k)
            return packed, {"canonical_pack_roundtrip_exact": False}

    runner = NonExactRunner()
    with pytest.raises(RuntimeError, match="canonical pack roundtrip is not exact"):
        build_qtip_bounded(
            runner,
            torch.eye(8, dtype=torch.float32),
            ["fit"],
            _Codebook(),
            _LDLQ(runner),
            _Math(),
            kernel_decode=None,
            device=torch.device("cpu"),
            rht_seed=13,
        )


def test_bounded_builder_rejects_packed_decode_that_differs_from_reconstruction() -> None:
    class MismatchingDecodeRunner(_Runner):
        def decode_packed(self, candidate, kernel_decode, device):
            decoded, receipt = super().decode_packed(
                candidate, kernel_decode, device
            )
            return decoded.add(1), receipt

    runner = MismatchingDecodeRunner()
    with pytest.raises(RuntimeError, match="differs from the stored reconstruction"):
        build_qtip_bounded(
            runner,
            torch.eye(8, dtype=torch.float32),
            ["fit"],
            _Codebook(),
            _LDLQ(runner),
            _Math(),
            kernel_decode=None,
            device=torch.device("cpu"),
            rht_seed=17,
        )


def test_bounded_builder_regularizes_hessian_without_second_full_matrix() -> None:
    runner = _Runner()
    source = torch.eye(8, dtype=torch.float32)
    _, receipt = build_qtip_bounded(
        runner,
        source,
        ["fit"],
        _Codebook(),
        _LDLQ(runner),
        _Math(),
        kernel_decode=SimpleNamespace(),
        device=torch.device("cpu"),
        rht_seed=11,
    )

    events = receipt["matrix_lifetime"]["events"]
    regularized = next(event for event in events if event["phase"] == "hessian_regularized")
    assert regularized["live_fp32_matrices"] == ["hessian"]
    assert regularized["unique_storage_count"] == 1


def test_manifest_packed_wire_is_inverse_of_production_decoder_layout() -> None:
    m = k = 32
    codebook_k = 2
    states = torch.arange(m * k // 2, dtype=torch.int32).reshape(m, k // 2)
    canonical = (
        torch.arange(m * k * codebook_k // 16, dtype=torch.int32)
        .mul(257)
        .add(0x1234)
        .bitwise_and(0xFFFF)
        .to(torch.uint16)
        .reshape(m * k // 256, 16 * codebook_k)
    )

    class Codebook:
        L = 16
        K = codebook_k
        V = 2
        _banana_smasher_public_runner_pack_contract = {
            "schema": "banana-smasher-public-runner-pack-contract-v1",
            "geometry": (16, codebook_k, 2),
            "matrix_shape": (m, k),
            "input_tile": (16, 16),
            "dtype": "uint16",
            "packed_words_per_tile_per_k": 16,
            "output_rows": "input_tile_grid",
            "expected_shape": tuple(canonical.shape),
        }

        def pack_trellis(self, tiled):
            self.tiled = tiled
            return canonical.clone()

        def unpack_trellis(self, packed, tile_size):
            assert torch.equal(packed, canonical)
            assert tile_size == 256
            return self.tiled.clone()

    packed_wire, receipt = pack_kernel_layout(Codebook(), states, m, k)
    decoder_canonical = (
        packed_wire.view(torch.uint8)
        .reshape(m // 32, k // 32, 32, 2, 2, codebook_k)
        .permute(0, 4, 1, 3, 2, 5)
        .flip((-1,))
        .reshape(m // 16, k // 16, 16 * codebook_k, 2)
        .flip((-1,))
        .contiguous()
        .view(torch.uint16)
        .reshape(canonical.shape)
    )

    assert receipt["canonical_pack_roundtrip_exact"] is True
    assert receipt["kernel_swizzle"] == (
        "reshape(m//32,2,k//32,2,32,K).permute(0,2,4,3,1,5)"
    )
    assert torch.equal(decoder_canonical, canonical)
