"""Exact QTIP2 V7 batch-10 producer primitives."""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import math
from typing import Any


_V7_PROJECTIONS = ("w1", "w2", "w3")
_V7_EXPERTS_PER_BATCH = 10


def _tensor_sha256(tensor: Any) -> str:
    import torch

    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _transform_regularized_hessian_batched(
    q2: Any, hessian: Any, input_signs: Any
) -> Any:
    """Apply the measured exact H128 transform as batched GEMMs."""
    block = 128
    hadamard = q2.normalized_hadamard_128(hessian.device, hessian.dtype)
    hessian *= input_signs.T
    hessian.copy_(
        (hessian.reshape(hessian.shape[0], -1, block) @ hadamard).reshape_as(
            hessian
        )
    )
    hessian *= input_signs
    hessian.copy_(
        (hadamard @ hessian.reshape(-1, block, hessian.shape[1])).reshape_as(
            hessian
        )
    )
    return hessian


def _source_transform_batched(
    q2: Any,
    source_out_in: Any,
    parent_lut: Any,
    *,
    quantize_tiles_fn: Any,
    input_signs: Any,
    output_signs: Any,
) -> dict[str, Any]:
    """Run the frozen V7 source transform with batch-issued H128 GEMMs."""
    import torch

    weight = source_out_in.transpose(0, 1).float().contiguous()
    size_k, size_n = weight.shape
    output_scales = q2._block_rms(weight, dim=0)
    mean = float(output_scales.mean().item())
    if mean <= 1e-30:
        raise ValueError("K2 source weight has no nonzero output scale")
    output_scales /= mean
    zero_outputs = output_scales.abs() < 1e-30
    output_scales[zero_outputs] = 0.1
    output_transform = (output_signs * output_scales + 1e-10).float()
    weight /= output_transform
    output_transform[zero_outputs] = 0.0
    hadamard = q2.normalized_hadamard_128(weight.device, torch.float32)
    weight.copy_((weight.reshape(size_k, -1, 128) @ hadamard).reshape_as(weight))
    input_scales = q2._block_rms(weight, dim=1)
    input_scales[input_scales.abs() < 1e-30] = 0.1
    input_transform = (
        input_signs * input_scales / (-q2._CODEBOOK_SCALE) + 1e-10
    ).float()
    weight /= input_transform
    weight.copy_(
        (hadamard @ weight.reshape(-1, 128, size_n)).reshape_as(weight)
    )
    global_scale = q2._global_scale(
        q2._sample_scale_tiles(weight), parent_lut, quantize_tiles_fn
    )
    su = (input_transform / global_scale).flatten().contiguous()
    sv = output_transform.flatten().contiguous()
    return {
        "target_inner": (weight * global_scale).contiguous(),
        "su": su,
        "sv": sv,
        "suh": su.half().contiguous(),
        "svh": sv.half().contiguous(),
        "global_scale": global_scale,
        "input_signs": input_signs,
        "output_signs": output_signs,
    }


def prepare_v7_unit(
    q2: Any,
    unit: dict[str, Any],
    parent_lut: Any,
    hessian_cache: dict[tuple[str, int, int], tuple[Any, str, Any, str]],
) -> dict[str, Any]:
    """Prepare one V7 member while reusing an exact shared-Hessian factor."""
    import torch

    source = unit["source"]
    raw_hessian = unit["raw_h"]
    count = int(unit["raw_h_count"])
    counters = {"cuda_calls": 0, "cuda_tiles": 0, "fallback_calls": 0}

    def counted(tiles: Any, lut: Any) -> tuple[Any, Any]:
        quantized, states = q2.quantize_k2_tiles(tiles, lut)
        counters["cuda_calls"] += 1
        counters["cuda_tiles"] += int(tiles.shape[0])
        return quantized, states

    device = source.device
    cache_key = (
        str(unit["input_identity"]["raw_hessian_data_sha256"]),
        count,
        int(source.shape[1]),
    )
    cache_hit = cache_key in hessian_cache
    fork_devices = [] if device.type == "cpu" else [device]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(36)
        input_signs = q2._draw_signs(int(source.shape[1]), device).unsqueeze(1)
        if cache_hit:
            hessian, hessian_sha, lower, lower_sha = hessian_cache[cache_key]
        else:
            regularized = q2._finalize_raw_hessian_on_device(
                raw_hessian,
                count,
                regularization_sigma=0.025,
                device=device,
            )
            hessian = _transform_regularized_hessian_batched(
                q2, regularized, input_signs
            )
            hessian_sha = _tensor_sha256(hessian)
            lower = q2._block_ldl_lower(hessian)
            lower_sha = _tensor_sha256(lower)
            hessian_cache[cache_key] = (hessian, hessian_sha, lower, lower_sha)
        output_signs = q2._draw_signs(int(source.shape[0]), device).unsqueeze(0)
    transformed = _source_transform_batched(
        q2,
        source,
        parent_lut,
        quantize_tiles_fn=counted,
        input_signs=input_signs,
        output_signs=output_signs,
    )
    return {
        **unit,
        "lower": lower,
        "transformed": transformed,
        "prepare_counters": counters,
        "hessian_cache_hit": cache_hit,
        "hessian_sha256": hessian_sha,
        "lower_sha256": lower_sha,
    }


def group_v7_batch10(
    units: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return one deterministic ten-expert group for each projection."""
    if len(units) != len(_V7_PROJECTIONS) * _V7_EXPERTS_PER_BATCH:
        raise ValueError("QTIP2 V7 batch10 requires exactly 30 projection members")
    groups = {
        projection: sorted(
            (unit for unit in units if unit.get("projection") == projection),
            key=lambda unit: int(unit["expert"]),
        )
        for projection in _V7_PROJECTIONS
    }
    expected_experts = [int(unit["expert"]) for unit in groups["w1"]]
    if len(set(expected_experts)) != _V7_EXPERTS_PER_BATCH:
        raise ValueError("QTIP2 V7 batch10 requires ten distinct experts")
    for projection, group in groups.items():
        experts = [int(unit["expert"]) for unit in group]
        if len(group) != _V7_EXPERTS_PER_BATCH or experts != expected_experts:
            raise ValueError(
                f"QTIP2 V7 batch10 projection {projection} does not match the expert roster"
            )
    return groups


def buffered_ldlq_cross_unit(
    q2: Any,
    prepared: list[dict[str, Any]],
    parent_lut: Any,
    *,
    chunk_tiles: int = 4096,
) -> tuple[list[Any], list[Any], dict[str, int]]:
    """Run the measured exact LDLQ recurrence across one projection group."""
    import torch

    if not prepared:
        raise ValueError("empty batch")
    shape = tuple(prepared[0]["transformed"]["target_inner"].shape)
    if any(
        tuple(item["transformed"]["target_inner"].shape) != shape
        for item in prepared
    ):
        raise ValueError("cross-unit group shape mismatch")
    size_k, size_n = shape
    buffer_rows = 128
    if size_k % buffer_rows or size_n % 128:
        raise ValueError("QTIP2 V7 LDLQ requires 128-aligned matrix geometry")
    permutation = torch.from_numpy(q2.tensor_core_permutation()).to(
        device=parent_lut.device, dtype=torch.int64
    )
    inverse = torch.argsort(permutation)
    states = []
    work = []
    for item in prepared:
        weight = item["transformed"]["target_inner"]
        lower = item["lower"]
        work.append(
            {
                "weight": weight,
                "lower": lower,
                "product": torch.zeros_like(weight),
                "quantized": torch.zeros_like(weight),
            }
        )
        states.append(
            torch.zeros(
                (size_k // 16, size_n // 16, 256),
                dtype=torch.int16,
                device=weight.device,
            )
        )
    qfn_calls = 0
    extension_calls = 0
    total_tiles = 0
    for high in range(size_k, 0, -buffer_rows):
        low = high - buffer_rows
        for target_high in range(buffer_rows, 0, -16):
            target_low = target_high - 16
            tiles_per_unit = []
            for entry in work:
                buffer_weight = entry["weight"][low:high]
                buffer_quantized = entry["quantized"][low:high]
                buffer_product = entry["product"][low:high]
                buffer_lower = entry["lower"][low:high]
                error = buffer_weight[target_high:] - buffer_quantized[target_high:]
                lower_slice = buffer_lower[
                    target_high:, low + target_low : low + target_high
                ]
                compensation = buffer_product[target_low:target_high]
                compensation.addmm_(lower_slice.T, error, alpha=1.0, beta=1.0)
                rows = buffer_weight[target_low:target_high] + compensation
                tiles = (
                    rows.reshape(16, size_n // 16, 16)
                    .permute(1, 0, 2)
                    .reshape(-1, 256)
                )
                tiles_per_unit.append(tiles[:, permutation])
            merged = torch.cat(tiles_per_unit, dim=0).contiguous()
            quantized, indices = q2.quantize_k2_tiles(
                merged, parent_lut, chunk_tiles=chunk_tiles
            )
            qfn_calls += 1
            extension_calls += math.ceil(int(merged.shape[0]) / chunk_tiles)
            total_tiles += int(merged.shape[0])
            offset = 0
            for unit_index, entry in enumerate(work):
                count = tiles_per_unit[unit_index].shape[0]
                unit_quantized = quantized[offset : offset + count, inverse]
                unit_indices = indices[offset : offset + count]
                quantized_rows = (
                    unit_quantized.reshape(size_n // 16, 16, 16)
                    .permute(1, 0, 2)
                    .reshape(16, size_n)
                )
                entry["quantized"][
                    low + target_low : low + target_high
                ] = quantized_rows
                states[unit_index][
                    (low + target_low) // 16 : (low + target_high) // 16
                ] = unit_indices.unsqueeze(0)
                offset += count
        for entry in work:
            buffer_weight = entry["weight"][low:high]
            buffer_quantized = entry["quantized"][low:high]
            buffer_lower = entry["lower"][low:high]
            entry["product"].addmm_(
                buffer_lower.T,
                buffer_weight - buffer_quantized,
                alpha=1.0,
                beta=1.0,
            )
    return [entry["quantized"] for entry in work], states, {
        "qfn_calls": qfn_calls,
        "extension_calls": extension_calls,
        "cuda_tiles": total_tiles,
        "fallback_calls": 0,
        "chunk_tiles": chunk_tiles,
    }


def finalize_batch_unit(
    q2: Any,
    item: dict[str, Any],
    quantized_inner: Any,
    states: Any,
) -> dict[str, Any]:
    """Finalize one measured V7 batch member into reusable physical outputs."""
    import torch

    transformed = item["transformed"]
    packed = q2.pack_k2(states)
    physical = q2.inverse_transform(
        quantized_inner.clone(), transformed["su"], transformed["sv"]
    ).T.contiguous()
    physical_bfloat16 = physical.to(torch.bfloat16)
    source_float = item["source"].float()
    source_bfloat16 = item["source"].to(torch.bfloat16)
    physical_sse = float((physical - source_float).double().square().sum().item())
    physical_bfloat16_sse = float(
        (physical_bfloat16.float() - source_bfloat16.float())
        .double()
        .square()
        .sum()
        .item()
    )
    inner_sse = float(
        (quantized_inner - transformed["target_inner"])
        .double()
        .square()
        .sum()
        .item()
    )
    proxy_hessian = item["proxy_hessian"].to(item["source"].device)
    proxy_error = transformed["target_inner"] - quantized_inner
    proxy_numerator = q2._block_trace(proxy_error, proxy_hessian)
    proxy_denominator = q2._block_trace(
        transformed["target_inner"], proxy_hessian
    )
    objective_proxy_error = proxy_numerator / max(proxy_denominator, 1e-8)
    boundaries = {
        "hessian_sha256": item["hessian_sha256"],
        "lower_sha256": item["lower_sha256"],
        "target_inner_sha256": _tensor_sha256(transformed["target_inner"]),
        "states_sha256": _tensor_sha256(states),
        "packed_sha256": _tensor_sha256(packed),
        "suh_sha256": _tensor_sha256(transformed["suh"]),
        "svh_sha256": _tensor_sha256(transformed["svh"]),
        "physical_bfloat16_sha256": _tensor_sha256(physical_bfloat16),
    }
    layer = item.get("layer")
    prefix = f"L{int(layer):03d}/" if layer is not None else ""
    return {
        "member": f"{prefix}E{int(item['expert']):03d}/{item['projection']}",
        "states": states,
        "packed_codes": packed,
        "su": transformed["su"],
        "sv": transformed["sv"],
        "suh": transformed["suh"],
        "svh": transformed["svh"],
        "global_scale": transformed["global_scale"],
        "decoded_inner": quantized_inner,
        "physical_fp32": physical,
        "physical_bfloat16": physical_bfloat16,
        "inner_sse": inner_sse,
        "objective_proxy_error": objective_proxy_error,
        "physical_sse": physical_sse,
        "physical_bfloat16_sse": physical_bfloat16_sse,
        "source_only": True,
        "artifact_seed_inputs": 0,
        "comparator_inputs": 0,
        "external_state_map": False,
        "solver_counters": {"fallback_calls": 0},
        "boundaries": boundaries,
    }


def produce_qtip2_v7_batch10(
    units: Sequence[dict[str, Any]],
    parent_lut: Any,
    *,
    q2: Any | None = None,
    chunk_tiles: int = 4096,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Produce exactly ten experts across w1/w2/w3 with the attempt9 path."""
    if q2 is None:
        from . import qtip_k2 as q2

    group_v7_batch10(units)
    hessian_cache: dict[tuple[str, int, int], tuple[Any, str, Any, str]] = {}
    prepared = []
    for unit in units:
        item = prepare_v7_unit(q2, unit, parent_lut, hessian_cache)
        if "proxy_hessian" not in item:
            cache_key = (
                str(unit["input_identity"]["raw_hessian_data_sha256"]),
                int(unit["raw_h_count"]),
                int(unit["source"].shape[1]),
            )
            item["proxy_hessian"] = hessian_cache[cache_key][0].detach().cpu()
        prepared.append(item)

    groups = group_v7_batch10(prepared)
    counters = {
        "qfn_calls": 0,
        "extension_calls": 0,
        "cuda_tiles": 0,
        "fallback_calls": 0,
        "chunk_tiles": chunk_tiles,
    }
    results = []
    for projection in _V7_PROJECTIONS:
        group = groups[projection]
        quantized, states, group_counters = buffered_ldlq_cross_unit(
            q2, group, parent_lut, chunk_tiles=chunk_tiles
        )
        for name in ("qfn_calls", "extension_calls", "cuda_tiles", "fallback_calls"):
            counters[name] += int(group_counters[name])
        results.extend(
            finalize_batch_unit(q2, item, quantized_inner, state)
            for item, quantized_inner, state in zip(
                group, quantized, states, strict=True
            )
        )

    prepare_fallbacks = sum(
        int(item["prepare_counters"]["fallback_calls"])
        for item in prepared
        if "prepare_counters" in item
    )
    result_fallbacks = sum(
        int(result["solver_counters"]["fallback_calls"]) for result in results
    )
    total_fallbacks = counters["fallback_calls"] + prepare_fallbacks + result_fallbacks
    if total_fallbacks:
        raise RuntimeError(f"QTIP2 V7 batch10 fallback is forbidden: {total_fallbacks}")
    counters.update(
        {
            "factor_cache_hits": sum(
                int(item.get("hessian_cache_hit", False)) for item in prepared
            ),
            "factor_cache_misses": len(hessian_cache),
            "factor_cache_entries": len(hessian_cache),
        }
    )
    return results, {
        "schema": "banana-smasher-qtip2-v7-batch10-producer-v1",
        "status": "PASS",
        "implementation": "qtip2-v7-attempt9-dp4a-half2-per-geometry-batch10",
        "batch_units": len(results),
        "group_sizes": {
            projection: len(groups[projection]) for projection in _V7_PROJECTIONS
        },
        "invocation": [
            "prepare_v7_unit",
            "group_v7_batch10",
            "buffered_ldlq_cross_unit",
            "finalize_batch_unit",
        ],
        "factor_cache_entries": len(hessian_cache),
        "counters": counters,
        "source_only": True,
        "fallback_calls": 0,
        "one_member_per_call": False,
        "deterministic_pack": True,
    }


__all__ = [
    "buffered_ldlq_cross_unit",
    "finalize_batch_unit",
    "group_v7_batch10",
    "prepare_v7_unit",
    "produce_qtip2_v7_batch10",
]
