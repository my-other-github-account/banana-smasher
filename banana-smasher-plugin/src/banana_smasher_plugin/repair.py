from __future__ import annotations

import functools
import math
from typing import Any

import torch
from safetensors.torch import load_file

from .contract import RuntimeContract


def _translate_stock_dsv4_repair_path(dotted: str) -> str:
    """Translate exported Hugging Face repair names to stock-vLLM V4 names."""
    translated = dotted.replace(
        ".self_attn.compressor.indexer.kv_norm",
        ".attn.indexer.compressor.norm",
    )
    translated = translated.replace(
        ".self_attn.compressor.kv_norm",
        ".attn.compressor.norm",
    )
    translated = translated.replace(".self_attn.", ".attn.")
    translated = translated.replace(".input_layernorm", ".attn_norm")
    translated = translated.replace(".post_attention_layernorm", ".ffn_norm")
    translated = translated.replace(".q_a_norm", ".q_norm")
    return translated.replace(".o_b_proj", ".wo_b")


def _resolve_exact(root: Any, dotted: str) -> Any:
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _resolve(root: Any, dotted: str) -> Any:
    try:
        return _resolve_exact(root, dotted)
    except AttributeError:
        translated = _translate_stock_dsv4_repair_path(dotted)
        if translated == dotted:
            raise
        try:
            return _resolve_exact(root, translated)
        except AttributeError as translated_error:
            raise AttributeError(
                "repair target is absent from both exported and stock-vLLM "
                f"DeepSeek-V4 module paths: {dotted!r} -> {translated!r}"
            ) from translated_error


def apply_dense_norm_repair(module: Any, contract: RuntimeContract) -> tuple[str, ...]:
    state = load_file(str(contract.repair_state), device="cpu")
    applied: list[str] = []
    for key, value in state.items():
        if not key.startswith("norms/"):
            continue
        name = key[len("norms/"):]
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


def _install_output_gain(target: Any, *, name: str, log_gain: float) -> None:
    existing = getattr(target, "_banana_smasher_output_log_gain", None)
    if existing is not None:
        if float(existing) != log_gain:
            raise RuntimeError(
                f"output repair gain changed after installation for {name}: "
                f"{existing} != {log_gain}"
            )
        return

    original_forward = target.forward
    factor = math.exp(log_gain)

    @functools.wraps(original_forward)
    def repaired_forward(*args: Any, **kwargs: Any) -> Any:
        output = original_forward(*args, **kwargs)
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(f"output repair target returned non-tensor: {name}")
        return output * factor

    setattr(target, "forward", repaired_forward)
    setattr(target, "_banana_smasher_output_log_gain", log_gain)


def apply_runtime_repairs(
    module: Any,
    contract: RuntimeContract,
) -> dict[str, tuple[str, ...]]:
    """Apply the exact dense repair state once after stock-vLLM weight loading."""
    norms = apply_dense_norm_repair(module, contract)
    output_names = tuple(sorted(load_output_log_gains(contract)))
    gains = load_output_log_gains(contract)
    for name in output_names:
        _install_output_gain(_resolve(module, name), name=name, log_gain=gains[name])
    return {"norms": norms, "output_log_gains": output_names}
