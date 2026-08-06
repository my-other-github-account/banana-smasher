"""Exact cross-unit batching for current QTIP2 builds.

Independent unit state is preserved on the leading axis.  The expensive LDLQ
codebook calls flatten that axis into one larger sequence batch so the current
K2/L16/V2 full-16 recurrence is invoked once per matrix tile instead of once
per unit.  Packing and packed-wire reconstruction remain per-unit and use the
runner's existing canonical consumer path.
"""
from __future__ import annotations

import time
import types
from typing import Any, cast

import torch


_PERMUTE = torch.arange(256).reshape(2, 8, 2, 4, 2).permute(1, 3, 2, 0, 4).flatten()
_INV_PERMUTE = torch.empty(256, dtype=torch.int64)
_INV_PERMUTE[_PERMUTE] = torch.arange(256)


class _BatchMatrixLifetime:
    """Record batch storage in source-FP32-matrix equivalents."""

    def __init__(self, shape: tuple[int, int, int]) -> None:
        self.units, self.rows, self.width = shape
        if min(shape) < 1:
            raise ValueError(f"invalid QTIP batch matrix shape: {shape}")
        self._matrix_bytes = self.rows * self.width * torch.empty(
            (), dtype=torch.float32
        ).element_size()
        self.events: list[dict[str, Any]] = []
        self.max_live = 0.0

    def observe(self, phase: str, **values: Any) -> None:
        matrices: list[str] = []
        storages: dict[tuple[str, int | None], int] = {}
        for name, value in values.items():
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
                continue
            matrices.append(name)
            storage = value.untyped_storage()
            pointer = storage.data_ptr() if value.numel() else None
            key = (value.device.type, pointer)
            storages[key] = max(storages.get(key, 0), storage.nbytes())
        equivalents = sum(storages.values()) / self._matrix_bytes
        self.max_live = max(self.max_live, equivalents)
        self.events.append(
            {
                "phase": phase,
                "live_fp32_matrices": sorted(matrices),
                "unique_storage_count": len(storages),
                "live_fp32_matrix_equivalents": equivalents,
            }
        )

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "banana-smasher-qtip-batch-matrix-lifetime-v1",
            "batch_units": self.units,
            "unit_matrix_shape": [self.rows, self.width],
            "max_live_fp32_matrix_equivalents": self.max_live,
            "released_after_ldlq": ["lower", "transformed"],
            "released_before_pack": ["lower", "transformed", "quantized"],
            "reconstruction_source": "canonical-packed-bytes",
            "events": self.events,
        }


def block_ldl_batch(hessian: torch.Tensor, block: int) -> torch.Tensor:
    """Return normalized block-LDL lower factors for ``[units,n,n]`` input."""
    if hessian.ndim != 3 or hessian.shape[-1] != hessian.shape[-2]:
        raise ValueError("batched block LDL expects [units,n,n]")
    units, width, _ = hessian.shape
    if units < 1 or block < 1 or width % block:
        raise ValueError(
            f"invalid batched block LDL geometry: units={units} width={width} block={block}"
        )
    blocks = width // block
    lower = torch.linalg.cholesky(hessian)
    view = lower.reshape(units, blocks, block, blocks, block)
    index = torch.arange(blocks, device=hessian.device)
    diagonal = view.permute(0, 1, 3, 2, 4)[:, index, index]
    inverse = torch.linalg.inv(diagonal)
    normalized = torch.einsum(
        "unib,uibc->unic",
        lower.view(units, width, blocks, block),
        inverse,
    ).reshape(units, width, width).contiguous()
    block_view = normalized.view(
        units, blocks, block, blocks, block
    ).permute(0, 1, 3, 2, 4)
    block_view[:, index, index] = torch.eye(
        block, dtype=hessian.dtype, device=hessian.device
    )
    return normalized


def ldlq_batch(
    weights: torch.Tensor,
    lower: torch.Tensor,
    codebook: Any,
    args: Any,
    *,
    buf_cols: int = 128,
    for_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run exact LDLQ while preserving independent state on the unit axis."""
    if weights.ndim != 3 or lower.ndim != 3:
        raise ValueError("batched LDLQ expects weights [units,m,n] and lower [units,n,n]")
    units, rows, width = weights.shape
    if lower.shape != (units, width, width):
        raise ValueError(
            f"batch/shape mismatch: weights={tuple(weights.shape)} lower={tuple(lower.shape)}"
        )
    if for_kernel and (args.td_x, args.td_y) != (16, 16):
        raise ValueError("kernel layout requires td_x=td_y=16")
    buf_cols = max(buf_cols, args.td_y)
    trellis_size = args.td_x * args.td_y
    if (
        buf_cols % args.td_y
        or width % buf_cols
        or args.td_y % args.V
        or trellis_size != 256
    ):
        raise ValueError("incompatible batched LDLQ tile geometry")
    buf_size = buf_cols // args.td_y

    hat_t = torch.zeros((units, width, rows), dtype=lower.dtype, device=lower.device)
    indices_t = torch.zeros(
        (units, width // args.V, rows),
        dtype=codebook.idx_dtype,
        device=lower.device,
    )
    weight_t = weights.transpose(1, 2).contiguous().to(lower.device)
    product = torch.zeros_like(weight_t)
    permute = _PERMUTE.to(lower.device)
    inverse_permute = _INV_PERMUTE.to(lower.device)

    for cur_col in range(width // args.td_y, 0, -buf_size):
        lo = args.td_y * (cur_col - buf_size)
        hi = args.td_y * cur_col
        buffered_weight = weight_t[:, lo:hi]
        buffered_hat = hat_t[:, lo:hi]
        buffered_lower = lower[:, lo:hi].contiguous()
        buffered_product = product[:, lo:hi]
        for index in reversed(range(buf_size)):
            inner_lo = args.td_y * index
            inner_hi = args.td_y * (index + 1)
            correction = torch.bmm(
                buffered_lower[:, inner_hi:, lo + inner_lo : lo + inner_hi].transpose(1, 2),
                buffered_weight[:, inner_hi:] - buffered_hat[:, inner_hi:],
            )
            target = (
                buffered_weight[:, inner_lo:inner_hi]
                + correction
                + buffered_product[:, inner_lo:inner_hi]
            )
            sequences = target.transpose(1, 2).reshape(units, -1, trellis_size)
            if for_kernel:
                sequences = sequences[..., permute]
            sequence_rows = sequences.shape[1]
            quantized, state_indices = codebook.quantize(
                sequences.reshape(units * sequence_rows, trellis_size)
            )
            if for_kernel:
                quantized = quantized[..., inverse_permute]
            quantized = quantized.reshape(units, rows, args.td_y)
            buffered_hat[:, inner_lo:inner_hi] = quantized.transpose(1, 2)
            index_rows = args.td_y // args.V
            state_indices = state_indices.reshape(units, rows, index_rows)
            state_lo = lo // args.V + index_rows * index
            state_hi = state_lo + index_rows
            indices_t[:, state_lo:state_hi] = state_indices.transpose(1, 2)
        product.add_(
            torch.bmm(
                buffered_lower.transpose(1, 2),
                buffered_weight - buffered_hat,
            )
        )
        hat_t[:, lo:hi] = buffered_hat

    return (
        hat_t.transpose(1, 2).contiguous(),
        indices_t.transpose(1, 2).contiguous(),
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _regularize_hessian_batch(hessians: torch.Tensor, sigma: float) -> None:
    diagonal = torch.diagonal(hessians, dim1=-2, dim2=-1)
    mean = diagonal.mean(dim=-1)
    hessians.div_(mean[:, None, None])
    diagonal = torch.diagonal(hessians, dim1=-2, dim2=-1)
    diagonal.add_(sigma)
    hessians.mul_(mean[:, None, None])


def _decode_candidate(
    runner: Any,
    candidate: dict[str, Any],
    codebook: Any,
    kernel_decode: Any,
    device: torch.device,
) -> dict[str, Any]:
    geometry = candidate["geometry"]
    rows, width = (int(candidate["shape"][0]), int(candidate["shape"][1]))
    raw = kernel_decode.decode_compressed(
        int(geometry["L"]),
        int(geometry["tlut_bits"]),
        int(geometry["K"]),
        int(geometry["V"]) - 1,
        rows,
        width,
        candidate["trellis"].to(device).reshape(-1),
        codebook.lut.T.contiguous(),
    )
    decoded = raw * candidate["Wscale"].to(device)
    decoded = runner.fwht(decoded.T).T * candidate["SV"].float().to(device)[:, None]
    decoded = runner.fwht(decoded) * candidate["SU"].float().to(device)
    stored = candidate["reconstructed_weight"]
    decoded_fp16 = decoded.to(device="cpu", dtype=torch.float16)
    equal = decoded_fp16.view(torch.int16).eq(stored.view(torch.int16))
    receipt = {
        "path": "existing geometry-bound canonical packed-wire consumer",
        "shape": [rows, width],
        "geometry": geometry,
        "fp16_bit_equal_fraction": float(equal.float().mean()),
        "fp16_bit_exact": bool(equal.all()),
        "max_abs_fp32_vs_stored_fp16": float(
            (decoded - stored.to(device).float()).abs().max()
        ),
        "runtime_check_performed": True,
    }
    del raw, decoded, decoded_fp16
    if receipt["fp16_bit_exact"] is not True:
        raise RuntimeError(
            f"batched QTIP packed decode conformance failed {rows}x{width}: {receipt}"
        )
    return receipt


def build_qtip_batch(
    runner: Any,
    source_weights: list[torch.Tensor],
    fit_windows_batch: list[list[Any]],
    codebook: Any,
    kernel_decode: Any,
    device: torch.device,
    rht_seeds: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build same-shape independent K2/L16/V2 units in one exact GPU batch."""
    units = len(source_weights)
    if not units or len(fit_windows_batch) != units or len(rht_seeds) != units:
        raise ValueError("QTIP batch requires aligned non-empty inputs")
    if (int(codebook.L), int(codebook.K), int(codebook.V)) != (16, 2, 2):
        raise ValueError("cross-unit batch is sealed for current K2/L16/V2")
    shapes = {tuple(weight.shape) for weight in source_weights}
    if len(shapes) != 1:
        raise ValueError(f"QTIP batch requires one matrix shape, got {sorted(shapes)}")
    rows, width = cast(tuple[int, int], next(iter(shapes)))
    if rows % 16 or width % 128:
        raise ValueError(f"QTIP batch matrix shape is not tile-compatible: {(rows, width)}")

    phase_seconds: dict[str, float] = {}
    lifetime = _BatchMatrixLifetime((units, rows, width))
    _synchronize(device)
    batch_started = time.perf_counter()
    hessians = []
    transformed_rows = []
    sus = []
    svs = []
    wscales = []
    fit_rows_values = []
    fit_mass_values = []
    lut_rms = codebook.lut.double().square().mean().sqrt().float() * 0.9

    started = time.perf_counter()
    with torch.no_grad():
        for source_weight, fit_windows, seed in zip(
            source_weights, fit_windows_batch, rht_seeds, strict=True
        ):
            torch.manual_seed(seed)
            su = (torch.randn(width, device=device).sign() + 1e-5).sign().float()
            sv = (torch.randn(rows, device=device).sign() + 1e-5).sign().float()
            hessian, fit_rows, fit_mass = runner.build_hessian(
                fit_windows, su, device
            )
            weight = source_weight.to(device=device, dtype=torch.float32)
            transformed = runner.fwht(
                runner.fwht(weight.T * sv).T * su
            )
            del weight
            wscale = transformed.square().mean().sqrt() / lut_rms
            transformed.div_(wscale)
            hessians.append(hessian)
            transformed_rows.append(transformed)
            sus.append(su)
            svs.append(sv)
            wscales.append(wscale)
            fit_rows_values.append(fit_rows)
            fit_mass_values.append(fit_mass)
    _synchronize(device)
    phase_seconds["hessian_forward_fwht"] = time.perf_counter() - started

    started = time.perf_counter()
    hessian_batch = torch.stack(hessians)
    _regularize_hessian_batch(hessian_batch, 1e-2)
    lower = block_ldl_batch(hessian_batch, 16)
    lower.diagonal(dim1=-2, dim2=-1).zero_()
    transformed = torch.stack(transformed_rows)
    lifetime.observe(
        "batched_ldl_ready",
        hessian=hessian_batch,
        lower=lower,
        transformed=transformed,
    )
    del hessians, hessian_batch, transformed_rows
    _synchronize(device)
    phase_seconds["batched_block_ldl"] = time.perf_counter() - started

    started = time.perf_counter()
    quantized, states = ldlq_batch(
        transformed,
        lower,
        codebook,
        types.SimpleNamespace(td_x=16, td_y=16, V=2),
        buf_cols=128,
        for_kernel=True,
    )
    lifetime.observe(
        "batched_ldlq_ready",
        transformed=transformed,
        lower=lower,
        quantized=quantized,
    )
    del transformed, lower, quantized
    lifetime.observe("quantized_released_before_pack")
    _synchronize(device)
    phase_seconds["batched_ldlq"] = time.perf_counter() - started

    pack = getattr(runner, "pack_kernel_layout", None)
    rate = getattr(runner, "_rate", None)
    pack_batch = getattr(rate, "pack_kernel_layout_batch", None)
    if not callable(pack) and not callable(pack_batch):
        raise RuntimeError("current QTIP runner lacks its canonical per-unit pack path")
    started = time.perf_counter()
    packed_rows = []
    pack_receipts = []
    for unit in range(units):
        if callable(pack):
            packed, receipt = pack(codebook, states[unit], rows, width)
        else:
            packed_batch, receipts = pack_batch(
                codebook, states[unit].unsqueeze(0), rows, width
            )
            if (
                not isinstance(packed_batch, torch.Tensor)
                or packed_batch.shape[0] != 1
                or not isinstance(receipts, list)
                or len(receipts) != 1
            ):
                raise RuntimeError("canonical batch pack returned invalid unit output")
            packed, receipt = packed_batch[0], receipts[0]
        if receipt.get("canonical_pack_roundtrip_exact") is not True:
            raise RuntimeError(f"canonical pack roundtrip is not exact for batch unit {unit}")
        packed_rows.append(packed)
        pack_receipts.append(receipt)
    del states
    _synchronize(device)
    phase_seconds["canonical_per_unit_pack"] = time.perf_counter() - started

    started = time.perf_counter()
    candidates = []
    packed_decode_receipts = []
    geometry = {
        "L": 16,
        "K": 2,
        "V": 2,
        "tlut_bits": int(codebook.tlut_bits),
        "decode_mode": str(codebook.decode_mode),
        "td_x": 16,
        "td_y": 16,
    }
    for unit in range(units):
        candidate = {
            "schema": "banana-smasher-qtip-unit-v1",
            "shape": [rows, width],
            "trellis": packed_rows[unit].cpu(),
            "SU": sus[unit].half().cpu(),
            "SV": svs[unit].half().cpu(),
            "Wscale": wscales[unit].cpu(),
            "tlut": codebook.tlut.cpu(),
            "geometry": geometry,
        }
        # Reconstruct from canonical packed bytes, never from pre-pack state.
        raw = kernel_decode.decode_compressed(
            16,
            int(codebook.tlut_bits),
            2,
            1,
            rows,
            width,
            packed_rows[unit].reshape(-1),
            codebook.lut.T.contiguous(),
        ) * wscales[unit]
        reconstructed = runner.fwht(raw.T).T * svs[unit][:, None]
        reconstructed = runner.fwht(reconstructed) * sus[unit]
        candidate["reconstructed_weight"] = reconstructed.half().cpu()
        candidates.append(candidate)
        del raw, reconstructed
        packed_decode_receipts.append(
            _decode_candidate(runner, candidate, codebook, kernel_decode, device)
        )
    del packed_rows
    _synchronize(device)
    phase_seconds["packed_decode_conformance"] = time.perf_counter() - started
    batch_wall_seconds = time.perf_counter() - batch_started

    return candidates, {
        "schema": "banana-smasher-qtip-cross-unit-build-v1",
        "status": "PASS",
        "implementation": "current-k2-full16-cross-unit-batched-ldlq-v1",
        "batch_units": units,
        "batch_wall_seconds": batch_wall_seconds,
        "mean_build_wall_seconds": batch_wall_seconds / units,
        "phase_seconds": phase_seconds,
        "mean_phase_seconds": {
            key: seconds / units for key, seconds in phase_seconds.items()
        },
        "fit_rows": fit_rows_values,
        "fit_route_mass": fit_mass_values,
        "canonical_pack": pack_receipts,
        "packed_decode": packed_decode_receipts,
        "matrix_lifetime": lifetime.receipt(),
        "fresh_no_warm_start": True,
        "independent_unit_state": True,
        "block_ldl_unit_axis": "batched",
        "ldlq_unit_axis": "batched-and-flattened-only-at-codebook-call",
        "solver_geometry": {
            "L": 16,
            "K": 2,
            "V": 2,
            "retained_prefix_costs": 4096,
            "branches_per_prefix": 16,
            "branch_sampling": "full",
        },
    }


__all__ = ["block_ldl_batch", "ldlq_batch", "build_qtip_batch"]
