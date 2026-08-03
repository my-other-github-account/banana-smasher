from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .token_sizing import MemoryBudget
from .update import prepare_tensor_segments
from .update_engine import run_segmented_update

_REQUIRED_ACCELERATION_SENTINELS = {
    "kmajor_batch",
    "kmajor_fused",
    "grouped_vjp",
    "layer_graph",
    "fwht",
}


def _parameters(modules: Sequence[Any]) -> list[Any]:
    values: list[Any] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            values.append(parameter)
    return values


def _frozen_surface(modules: Sequence[Any]) -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = []
    seen: set[int] = set()
    for module_index, module in enumerate(modules):
        for kind, named_values in (
            ("parameter", module.named_parameters(recurse=True)),
            ("buffer", module.named_buffers(recurse=True)),
        ):
            for name, value in named_values:
                if id(value) in seen:
                    continue
                seen.add(id(value))
                values.append((f"{module_index}:{kind}:{name}", value))
    return values


def _surface_identity(surface: Sequence[tuple[str, Any]]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            name,
            id(value),
            int(value._version),
            tuple(value.shape),
            str(value.dtype),
            str(value.device),
        )
        for name, value in surface
    )


def _validate_sentinels(
    sentinels: dict[str, Any], *, allow_reference: bool
) -> dict[str, int]:
    missing = _REQUIRED_ACCELERATION_SENTINELS.difference(sentinels)
    if missing:
        raise ValueError(
            f"accelerated production backend lacks sentinels: {sorted(missing)}"
        )
    selected: dict[str, int] = {}
    for name in sorted(_REQUIRED_ACCELERATION_SENTINELS):
        value = sentinels[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"acceleration sentinel {name} must be an integer count")
        selected[name] = value
    inactive = [name for name, value in selected.items() if value <= 0]
    if inactive and not allow_reference:
        raise ValueError(
            "accelerated production backend requires observed path activity; "
            f"explicit reference opt-in was not provided for {inactive}"
        )
    return selected


def run_full_depth_update(
    *,
    layers: Sequence[Any],
    frozen_modules: Sequence[Any],
    input_ids: Any,
    teacher_targets: Any,
    teacher_mask: Any,
    positions: Any,
    requested_tokens: int,
    segments: int,
    batch_size: int,
    memory_budget: MemoryBudget,
    encode: Callable[[dict[str, Any]], Any],
    loss_sum: Callable[[Any, dict[str, Any]], Any],
    output: str | Path,
    identity: dict[str, Any],
    peak_memory_bytes: int | Callable[[], int],
    optimizer_factory: Callable[[list[Any]], Any],
    backend_sentinels: Callable[[], dict[str, Any]],
    receipt: str | Path | None = None,
    resume: bool = True,
    restart: bool = False,
    synchronize: Callable[[], None] | None = None,
    allow_reference: bool = False,
    semantic_claim: str = "causal-segmented-no-equivalence-claim",
    semantic_parity_tested: bool = False,
) -> dict[str, Any]:
    """Run one fresh, full-depth production update through the portable core.

    This helper deliberately owns no model-specific loading or path defaults.
    Callers provide the ordered layer surface and input/loss adapters. Every
    depth is checked against the selected physical token dimension before the
    core performs exactly one optimizer step.
    """
    import torch

    ordered_layers = list(layers)
    ordered_frozen_modules = list(frozen_modules)
    if not ordered_layers:
        raise ValueError("full-depth production update requires at least one layer")
    if allow_reference and semantic_parity_tested:
        raise ValueError("reference production runs cannot claim tested accelerated parity")
    trainable = _parameters(ordered_layers)
    frozen_surface = _frozen_surface(ordered_frozen_modules)
    frozen = [value for _, value in frozen_surface]
    if not trainable:
        raise ValueError("full-depth production update has no trainable parameters")
    overlap = {id(value) for value in trainable}.intersection(id(value) for value in frozen)
    if overlap:
        raise ValueError("trainable and frozen module surfaces overlap")
    invalid_trainable = [
        index
        for index, parameter in enumerate(trainable)
        if not parameter.requires_grad or not bool(torch.isfinite(parameter.detach()).all())
    ]
    if invalid_trainable:
        raise RuntimeError(
            "production trainable surface must be explicit, finite, and trainable: "
            f"invalid={invalid_trainable}"
        )
    invalid_frozen = [
        index
        for index, value in enumerate(frozen)
        if value.requires_grad or not bool(torch.isfinite(value.detach()).all())
    ]
    if invalid_frozen:
        raise RuntimeError(
            f"production frozen surface contains trainable/non-finite tensors: {invalid_frozen}"
        )
    frozen_identity = _surface_identity(frozen_surface)
    if not callable(backend_sentinels):
        raise TypeError("backend_sentinels must be a callable counter probe")
    baseline_sentinels = _validate_sentinels(
        dict(backend_sentinels()), allow_reference=True
    )
    selected_sentinels: dict[str, int] = {}

    work, sizing = prepare_tensor_segments(
        input_ids=input_ids,
        teacher_targets=teacher_targets,
        teacher_mask=teacher_mask,
        positions=positions,
        requested_tokens=requested_tokens,
        segments=segments,
        batch_size=batch_size,
        memory_budget=memory_budget,
    )
    physical_tokens = int(sizing["physical_tokens"])
    optimizer = optimizer_factory(trainable)
    if optimizer.state:
        raise RuntimeError("new production update cannot warm-start optimizer state")
    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group.get("params", [])
    }
    if optimizer_parameters != {id(parameter) for parameter in trainable}:
        raise RuntimeError("optimizer surface does not match explicit trainable modules")

    def validate_frozen_surface() -> None:
        current = _frozen_surface(ordered_frozen_modules)
        if _surface_identity(current) != frozen_identity:
            raise RuntimeError("frozen production tensor surface was mutated")
        non_finite = [
            name
            for name, value in current
            if not bool(torch.isfinite(value.detach()).all())
        ]
        if non_finite:
            raise RuntimeError(f"frozen production tensors became non-finite: {non_finite}")

    def validate_acceleration_activity() -> None:
        observed = _validate_sentinels(
            dict(backend_sentinels()), allow_reference=True
        )
        deltas = {
            name: observed[name] - baseline_sentinels[name]
            for name in _REQUIRED_ACCELERATION_SENTINELS
        }
        selected = _validate_sentinels(deltas, allow_reference=allow_reference)
        selected_sentinels.clear()
        selected_sentinels.update(selected)

    class CheckedOptimizer:
        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped

        def __getattr__(self, name: str) -> Any:
            return getattr(self.wrapped, name)

        def step(self, *args: Any, **kwargs: Any) -> Any:
            validate_frozen_surface()
            validate_acceleration_activity()
            return self.wrapped.step(*args, **kwargs)

    optimizer = CheckedOptimizer(optimizer)
    depth_shapes: list[list[list[int]]] = []

    def production_loss(segment: dict[str, Any]) -> Any:
        hidden = encode(segment)
        current_shapes: list[list[int]] = []
        for depth, layer in enumerate(ordered_layers):
            hidden = layer(hidden)
            shape = [int(value) for value in hidden.shape]
            if len(shape) < 2 or shape[0] != 1 or shape[1] != physical_tokens:
                raise RuntimeError(
                    "physical token geometry drift inside production depth: "
                    f"depth={depth}, shape={shape}, physical_tokens={physical_tokens}"
                )
            current_shapes.append(shape)
        depth_shapes.append(current_shapes)
        value = loss_sum(hidden, segment)
        if getattr(value, "ndim", None) != 0:
            raise ValueError("production loss_sum must return a scalar summed loss")
        return value

    def post_step_validate() -> dict[str, Any]:
        validate_frozen_surface()
        frozen_gradients = [
            index
            for index, parameter in enumerate(frozen)
            if parameter.grad is not None
        ]
        if frozen_gradients:
            raise RuntimeError(
                "frozen production parameters received gradients: "
                f"{frozen_gradients}"
            )
        return {
            "production_frozen_gradients": 0,
            "production_frozen_surface_unchanged": True,
        }

    first = work[0]
    return run_segmented_update(
        parameters=trainable,
        optimizer=optimizer,
        segments=work,
        item_count=lambda segment: int(segment["teacher_mask"].sum().item()),
        loss_sum=production_loss,
        output=output,
        receipt=receipt,
        identity=identity,
        physical_tokens=physical_tokens,
        observed_input_shape=list(first["input_ids"].shape),
        teacher_geometry={
            "target_shape": list(first["teacher_targets"].shape),
            "mask_shape": list(first["teacher_mask"].shape),
            "position_shape": list(first["positions"].shape),
        },
        peak_memory_bytes=peak_memory_bytes,
        backend="accelerated" if not allow_reference else "reference",
        resume=resume,
        restart=restart,
        synchronize=synchronize,
        receipt_fields={
            "requested_physical_tokens": int(requested_tokens),
            "memory_sizing": sizing,
            "production_runtime": {
                "depth": len(ordered_layers),
                "depth_shapes": depth_shapes,
                "trainable_parameter_tensors": len(trainable),
                "frozen_parameter_tensors": sum(
                    ":parameter:" in name for name, _ in frozen_surface
                ),
                "frozen_buffer_tensors": sum(
                    ":buffer:" in name for name, _ in frozen_surface
                ),
                "warm_start_used": False,
                "backend_sentinels": selected_sentinels,
                "reference_opt_in": bool(allow_reference),
            },
        },
        post_step_validate=post_step_validate,
        semantic_claim=semantic_claim,
        semantic_parity_tested=semantic_parity_tested,
    )
