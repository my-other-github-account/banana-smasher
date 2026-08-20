from __future__ import annotations

from typing import Any, Sequence


def measure_local_projection_losses(
    *,
    inputs: Any,
    hessian_weights: Any,
    class_ids: Any,
    class_names: Sequence[str],
    native_fused: Any,
    native_down: Any,
    tier_fused: Any,
    tier_down: Any,
    swiglu_limit: float,
) -> dict[str, dict[str, float]]:
    """Measure additive routed Gauss-Newton loss for one expert's two cells."""

    import torch
    import torch.nn.functional as functional

    if inputs.ndim != 2 or inputs.shape[0] == 0:
        raise ValueError("own-base expert inputs must be one non-empty rank-2 tensor")
    token_count = inputs.shape[0]
    if hessian_weights.shape != (token_count,) or class_ids.shape != (token_count,):
        raise ValueError("own-base route weights and class ids must match token count")
    if not class_names or len(class_names) != len(set(class_names)):
        raise ValueError("own-base class names must be non-empty and unique")
    if native_fused.shape != tier_fused.shape or native_down.shape != tier_down.shape:
        raise ValueError("own-base native/tier projection shapes differ")

    def intermediate(weight: Any) -> Any:
        gate, up = functional.linear(inputs, weight).chunk(2, dim=-1)
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        return functional.silu(gate) * up

    native_intermediate = intermediate(native_fused)
    native_output = functional.linear(native_intermediate, native_down)
    fused_output = functional.linear(intermediate(tier_fused), native_down)
    down_output = functional.linear(native_intermediate, tier_down)
    weights = hessian_weights.to(torch.float64)

    result: dict[str, dict[str, float]] = {"fused13": {}, "down": {}}
    for projection, output in (("fused13", fused_output), ("down", down_output)):
        per_route = 0.5 * weights * (
            output.to(torch.float64) - native_output.to(torch.float64)
        ).square().sum(dim=-1)
        for class_id, name in enumerate(class_names):
            result[projection][name] = float(
                per_route[class_ids == class_id].sum().item()
            )
    return result
