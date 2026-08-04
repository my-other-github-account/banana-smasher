"""Memory-bounded QTIP matrix builder.

The reference QTIP builder is mathematically correct but historically retained
Hessian, transform, LDL, quantized, and reconstruction matrices across phase
boundaries.  This module keeps the same objective and wire payload while making
those phase boundaries explicit and releasing full-size CUDA storage as soon as
its last consumer returns.
"""
from __future__ import annotations

import time
import types
from typing import Any, cast

import torch


class _MatrixLifetime:
    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape = shape
        self.events: list[dict[str, Any]] = []
        self.max_live = 0

    def observe(self, phase: str, **values: Any) -> None:
        matrices: list[str] = []
        storages: set[tuple[str, int | None]] = set()
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.dtype != torch.float32 or tuple(value.shape) != self.shape:
                continue
            matrices.append(name)
            pointer = value.untyped_storage().data_ptr() if value.numel() else None
            storages.add((value.device.type, pointer))
        self.max_live = max(self.max_live, len(storages))
        self.events.append(
            {
                "phase": phase,
                "live_fp32_matrices": sorted(matrices),
                "unique_storage_count": len(storages),
            }
        )

    def receipt(self, reconstructed: torch.Tensor) -> dict[str, Any]:
        return {
            "schema": "banana-smasher-qtip-matrix-lifetime-v1",
            "matrix_shape": list(self.shape),
            "max_live_fp32_matrix_equivalents": self.max_live,
            "released_before_pack": ["lower", "transformed"],
            "reconstructed_weight_device": reconstructed.device.type,
            "events": self.events,
        }


def _regularize_hessian_inplace(hessian: torch.Tensor, sigma: float) -> torch.Tensor:
    """Match QTIP's regularize_H arithmetic without its final full copy."""
    diagonal_mean = hessian.diagonal().mean()
    hessian.div_(diagonal_mean)
    hessian.diagonal().add_(sigma)
    hessian.mul_(diagonal_mean)
    return hessian


def _codebook_geometry(cb: Any) -> dict[str, Any]:
    return {
        "L": int(getattr(cb, "L", 16)),
        "K": int(getattr(cb, "K", 3)),
        "V": int(getattr(cb, "V", 2)),
        "tlut_bits": int(getattr(cb, "tlut_bits", 9)),
        "decode_mode": str(getattr(cb, "decode_mode", "quantlut_sym")),
        "td_x": int(getattr(cb, "td_x", 16)),
        "td_y": int(getattr(cb, "td_y", 16)),
    }


def build_qtip_bounded(
    runner: Any,
    source_weight: torch.Tensor,
    fit_windows: list[Any],
    cb: Any,
    ldlq: Any,
    math_utils: Any,
    kernel_decode: Any,
    device: torch.device,
    rht_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one exact QTIP unit with bounded full-matrix phase overlap.

    ``runner`` supplies the pinned QTIP functions (FWHT, Hessian construction,
    canonical packing, and packed decode).  The packed bytes are decoded once
    before return so every produced unit is runtime-checked against the stored
    reconstruction rather than relying only on test-time conformance.
    """
    m, k = (int(source_weight.shape[0]), int(source_weight.shape[1]))
    lifetime = _MatrixLifetime((m, k))
    phase_seconds: dict[str, float] = {}

    torch.manual_seed(rht_seed)
    su = (torch.randn(k, device=device).sign() + 1e-5).sign().float()
    sv = (torch.randn(m, device=device).sign() + 1e-5).sign().float()

    with torch.no_grad():
        started = time.perf_counter()
        hessian, fit_rows, fit_mass = runner.build_hessian(fit_windows, su, device)
        phase_seconds["hessian_fwht_accumulation"] = time.perf_counter() - started
        del fit_windows
        lifetime.observe("hessian_ready", hessian=hessian)

        started = time.perf_counter()
        _regularize_hessian_inplace(hessian, 1e-2)
        phase_seconds["hessian_regularization"] = time.perf_counter() - started
        lifetime.observe("hessian_regularized", hessian=hessian)

        started = time.perf_counter()
        weight = source_weight.to(device=device, dtype=torch.float32)
        del source_weight
        transformed = runner.fwht(runner.fwht(weight.T * sv).T * su)
        del weight
        phase_seconds["forward_fwht"] = time.perf_counter() - started
        lifetime.observe(
            "forward_transform_ready", hessian=hessian, transformed=transformed
        )

        wscale = transformed.square().mean().sqrt() / (
            cb.lut.double().square().mean().sqrt().float() * 0.9
        )
        transformed.div_(wscale)

        started = time.perf_counter()
        factorization = math_utils.block_LDL(hessian, 16)
        if not isinstance(factorization, tuple) or len(factorization) != 2:
            raise RuntimeError("QTIP block LDL factorization failed")
        lower, diagonal_blocks = factorization
        del diagonal_blocks, factorization, hessian
        lower.diagonal().zero_()
        phase_seconds["block_ldl"] = time.perf_counter() - started
        lifetime.observe("ldl_ready", transformed=transformed, lower=lower)

        geometry = _codebook_geometry(cb)
        args = types.SimpleNamespace(
            td_x=geometry["td_x"], td_y=geometry["td_y"], V=geometry["V"]
        )
        started = time.perf_counter()
        quantized, states = ldlq.LDLQ(
            transformed, lower, cb, args, buf_cols=128, for_kernel=True
        )
        phase_seconds["ldlq"] = time.perf_counter() - started
        # LDLQ has consumed both full matrices. Release them before packing,
        # rather than carrying them into the wire and reconstruction phases.
        del lower, transformed
        lifetime.observe("ldlq_ready", quantized=quantized)

        started = time.perf_counter()
        pack = getattr(runner, "pack_kernel_layout", None)
        if callable(pack):
            packed, pack_conformance = cast(Any, pack)(cb, states, m, k)
        else:
            rate = getattr(runner, "_rate", None)
            pack_batch = getattr(rate, "pack_kernel_layout_batch", None)
            if not callable(pack_batch):
                raise RuntimeError("QTIP runner lacks a canonical pack path")
            packed_batch, pack_receipts = cast(Any, pack_batch)(
                cb, states.unsqueeze(0), m, k
            )
            if (
                not isinstance(packed_batch, torch.Tensor)
                or packed_batch.shape[0] != 1
                or not isinstance(pack_receipts, list)
                or len(pack_receipts) != 1
                or not isinstance(pack_receipts[0], dict)
            ):
                raise RuntimeError("QTIP runner canonical batch pack result is invalid")
            packed = packed_batch[0]
            pack_conformance = pack_receipts[0]
        del states
        if pack_conformance.get("canonical_pack_roundtrip_exact") is not True:
            raise RuntimeError("canonical pack roundtrip is not exact")
        phase_seconds["canonical_pack"] = time.perf_counter() - started
        lifetime.observe("packed", quantized=quantized)

        started = time.perf_counter()
        quantized.mul_(wscale)
        rotated = runner.fwht(quantized.T).T
        del quantized
        rotated.mul_(sv[:, None])
        reconstructed = runner.fwht(rotated)
        del rotated
        reconstructed.mul_(su)
        reconstructed_fp16 = reconstructed.to(device="cpu", dtype=torch.float16)
        del reconstructed
        phase_seconds["inverse_fwht_reconstruction"] = (
            time.perf_counter() - started
        )
        lifetime.observe("reconstruction_sealed")

    candidate = {
        "schema": "banana-smasher-qtip-unit-v1",
        "shape": [m, k],
        "trellis": packed.cpu(),
        "SU": su.half().cpu(),
        "SV": sv.half().cpu(),
        "Wscale": wscale.cpu(),
        "tlut": cb.tlut.cpu(),
        "reconstructed_weight": reconstructed_fp16,
        "geometry": geometry,
    }
    packed_decode_started = time.perf_counter()
    decoded, packed_decode = runner.decode_packed(candidate, kernel_decode, device)
    if packed_decode.get("fp16_bit_exact") is not True:
        raise RuntimeError(f"packed decode conformance failed {m}x{k}: {packed_decode}")
    decoded_fp16 = decoded.to(device="cpu", dtype=torch.float16)
    if not torch.equal(decoded_fp16, reconstructed_fp16):
        raise RuntimeError(
            f"packed decode differs from the stored reconstruction {m}x{k}"
        )
    lifetime.observe("packed_decode_verified", decoded=decoded)
    del decoded, decoded_fp16
    phase_seconds["packed_decode_conformance"] = (
        time.perf_counter() - packed_decode_started
    )
    packed_decode = {**packed_decode, "runtime_check_performed": True}
    build_receipt: dict[str, Any] = {
        "rht_seed": rht_seed,
        "quant_seconds": phase_seconds["ldlq"],
        "fit_rows": fit_rows,
        "fit_route_mass": fit_mass,
        "canonical_pack": pack_conformance,
        "packed_decode": packed_decode,
        "phase_seconds": phase_seconds,
        "matrix_lifetime": lifetime.receipt(reconstructed_fp16),
    }
    return candidate, build_receipt


__all__ = ["build_qtip_bounded"]
