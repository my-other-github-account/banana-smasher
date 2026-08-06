from __future__ import annotations

from typing import Any

import torch
from safetensors.torch import load_file

from .contract import RuntimeContract


def _resolve(root: Any, dotted: str) -> Any:
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


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


def apply_runtime_repairs(
    module: Any,
    contract: RuntimeContract,
) -> dict[str, tuple[str, ...]]:
    """Confirm export-folded repair without changing the steady-state graph."""
    del module
    if (
        contract.repair_application != "export-folded-v1"
        or contract.runtime_output_gain
    ):
        raise RuntimeError("runtime repair requires export-folded-v1 materialization")
    return {"norms": (), "output_log_gains": ()}
