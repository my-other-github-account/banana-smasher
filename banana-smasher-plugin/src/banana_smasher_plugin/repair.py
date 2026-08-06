from __future__ import annotations

import math
from typing import Any

import torch
from safetensors.torch import load_file

from .contract import RuntimeContract


def _resolve(root: Any, dotted: str) -> Any:
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _runtime_name(name: str) -> str:
    """Map the checkpoint repair namespace onto stock DeepSeek-V4 modules."""

    return (
        name.replace(".input_layernorm", ".attn_norm")
        .replace(".post_attention_layernorm", ".ffn_norm")
        .replace(".self_attn", ".attn")
        .replace(".q_a_norm", ".q_norm")
        .replace(".o_b_proj", ".wo_b")
    )


def apply_dense_norm_repair(module: Any, contract: RuntimeContract) -> tuple[str, ...]:
    state = load_file(str(contract.repair_state), device="cpu")
    applied: list[str] = []
    for key, value in state.items():
        if not key.startswith("norms/"):
            continue
        name = _runtime_name(key[len("norms/"):])
        target = _resolve(module, name)
        weight = getattr(target, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise RuntimeError(f"dense repair target has no tensor weight: {name}")
        weight.data.copy_(value.to(device=weight.device, dtype=weight.dtype))
        applied.append(name + ".weight")
    return tuple(sorted(applied))


def load_output_log_gains(contract: RuntimeContract) -> dict[str, float]:
    state = load_file(str(contract.repair_state), device="cpu")
    suffix = ".output_log_gain"
    return {
        key[len("outputs/"):-len(suffix)]: float(value.item())
        for key, value in state.items()
        if key.startswith("outputs/") and key.endswith(suffix)
    }


_FLOAT8_DTYPES = {
    dtype
    for name in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2")
    if (dtype := getattr(torch, name, None)) is not None
}
_E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", None)


def _requantize_e8m0_block_weight(
    target: Any,
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    name: str,
    factor: float,
) -> None:
    block_size = getattr(target, "weight_block_size", None)
    if (
        not isinstance(block_size, (list, tuple))
        or len(block_size) != 2
        or any(not isinstance(value, int) or value <= 0 for value in block_size)
    ):
        raise RuntimeError(f"FP8 E8M0 output repair target has no block shape: {name}")
    if weight.ndim != 2 or scale.ndim != 2:
        raise RuntimeError(
            f"FP8 E8M0 output repair tensors must be 2-D for {name}: "
            f"weight={tuple(weight.shape)} scale={tuple(scale.shape)}"
        )
    block_n, block_k = block_size
    rows, columns = weight.shape
    expected_scale_shape = (
        (rows + block_n - 1) // block_n,
        (columns + block_k - 1) // block_k,
    )
    if tuple(scale.shape) != expected_scale_shape:
        raise RuntimeError(
            f"FP8 E8M0 output repair scale shape mismatch for {name}: "
            f"actual={tuple(scale.shape)} expected={expected_scale_shape}"
        )

    weight_max = float(torch.finfo(weight.dtype).max)
    scale_info = torch.finfo(scale.dtype)
    minimum_exponent = math.ceil(math.log2(float(scale_info.tiny)))
    maximum_exponent = math.floor(math.log2(float(scale_info.max)))
    with torch.no_grad():
        old_scales = scale.float()
        if not torch.all(torch.isfinite(old_scales) & (old_scales > 0)):
            raise RuntimeError(f"FP8 E8M0 output repair scale is invalid for {name}")
        two = torch.tensor(2.0, device=scale.device, dtype=torch.float32)
        for block_row in range(expected_scale_shape[0]):
            start = block_row * block_n
            stop = min(start + block_n, rows)
            row_count = stop - start
            values = weight[start:stop].float()
            padded_columns = expected_scale_shape[1] * block_k
            if columns != padded_columns:
                padded = values.new_zeros((row_count, padded_columns))
                padded[:, :columns].copy_(values)
                values = padded
            values = values.reshape(row_count, expected_scale_shape[1], block_k)
            old_scale = old_scales[block_row]
            maximum = values.abs().amax(dim=(0, 2))
            ideal_scale = maximum * old_scale * factor / weight_max
            exponent = torch.ceil(torch.log2(ideal_scale)).clamp(
                minimum_exponent, maximum_exponent
            )
            new_scale = torch.pow(two, exponent)
            new_scale = torch.where(maximum > 0, new_scale, old_scale)
            residual = old_scale * factor / new_scale
            quantized = (values * residual[None, :, None]).clamp(
                -weight_max, weight_max
            )
            weight[start:stop].copy_(
                quantized.reshape(row_count, padded_columns)[:, :columns].to(weight.dtype)
            )
            scale[block_row].copy_(new_scale.to(scale.dtype))


def _fold_output_gain(target: Any, *, name: str, log_gain: float) -> None:
    existing = getattr(target, "_banana_smasher_output_log_gain", None)
    if existing is not None:
        if float(existing) != log_gain:
            raise RuntimeError(
                f"output repair gain changed after installation for {name}: "
                f"{existing} != {log_gain}"
            )
        return

    weight = getattr(target, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise RuntimeError(f"output repair target has no tensor weight: {name}")
    factor = math.exp(log_gain)
    if weight.dtype in _FLOAT8_DTYPES:
        scale = getattr(target, "weight_scale_inv", None)
        if not isinstance(scale, torch.Tensor):
            scale = getattr(target, "weight_scale", None)
        if not isinstance(scale, torch.Tensor):
            raise RuntimeError(f"FP8 output repair target has no tensor scale: {name}")
        if _E8M0_DTYPE is not None and scale.dtype == _E8M0_DTYPE:
            _requantize_e8m0_block_weight(
                target, weight, scale, name=name, factor=factor
            )
        else:
            with torch.no_grad():
                scale.mul_(factor)
    else:
        with torch.no_grad():
            weight.mul_(factor)
    setattr(target, "_banana_smasher_output_log_gain", log_gain)


def apply_runtime_repairs(
    module: Any,
    contract: RuntimeContract,
) -> dict[str, tuple[str, ...]]:
    """Apply the exact dense repair state once after stock-vLLM weight loading."""
    norms = apply_dense_norm_repair(module, contract)
    gains = load_output_log_gains(contract)
    output_names = tuple(sorted(gains))
    for name in output_names:
        runtime_name = _runtime_name(name)
        _fold_output_gain(
            _resolve(module, runtime_name), name=runtime_name, log_gain=gains[name]
        )
    return {"norms": norms, "output_log_gains": output_names}
