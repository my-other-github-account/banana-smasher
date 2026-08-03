from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .accumulation import backward_logical_mean
from .token_sizing import require_integer
from .update_checkpoint import (
    atomic_json,
    atomic_torch_save,
    canonical_identity,
    commit_segment_checkpoint,
    finalize_checkpoint,
    load_checkpoint,
    verify_completed_files,
)


def _tensor_bytes(value: Any) -> bytes:
    tensor = value.detach().cpu().contiguous()
    try:
        return tensor.numpy().tobytes()
    except TypeError:
        import torch

        return tensor.view(-1).view(torch.uint8).numpy().tobytes()


def _parameter_sha256(parameters: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for parameter in parameters:
        digest.update(_tensor_bytes(parameter))
    return digest.hexdigest()


def _cpu_parameters(parameters: Sequence[Any]) -> list[Any]:
    return [parameter.detach().cpu().clone() for parameter in parameters]


def _cpu_gradients(parameters: Sequence[Any]) -> list[Any | None]:
    return [
        None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for parameter in parameters
    ]


def _restore_parameters(parameters: Sequence[Any], values: Sequence[Any]) -> None:
    import torch

    if len(parameters) != len(values):
        raise RuntimeError("checkpoint trainable parameter count mismatch")
    with torch.no_grad():
        for parameter, value in zip(parameters, values):
            if (
                tuple(parameter.shape) != tuple(value.shape)
                or parameter.dtype != value.dtype
            ):
                raise RuntimeError("checkpoint trainable parameter identity mismatch")
            parameter.copy_(value.to(device=parameter.device))


def _restore_gradients(
    parameters: Sequence[Any], gradients: Sequence[Any | None]
) -> None:
    if len(parameters) != len(gradients):
        raise RuntimeError("checkpoint gradient count mismatch")
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = (
            None
            if gradient is None
            else gradient.to(device=parameter.device, dtype=parameter.dtype).clone()
        )


def _rng_state(torch: Any) -> dict[str, Any]:
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(torch: Any, state: dict[str, Any]) -> None:
    torch.set_rng_state(state["cpu"])
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _memory_value(probe: int | Callable[[], int]) -> int:
    value = probe() if callable(probe) else probe
    value = require_integer("peak_memory_bytes", value)
    if value < 0:
        raise RuntimeError(f"peak memory probe returned a negative value: {value}")
    return value


def _validate_geometry(
    physical_tokens: int,
    observed_input_shape: Sequence[int],
    teacher_geometry: dict[str, Any],
) -> tuple[list[int], dict[str, list[int]]]:
    physical_tokens = require_integer("physical_tokens", physical_tokens)
    if physical_tokens <= 0:
        raise ValueError("physical_tokens must be positive")
    shape = [
        require_integer(f"observed_input_shape[{index}]", value)
        for index, value in enumerate(observed_input_shape)
    ]
    if len(shape) < 2 or shape[0] < 1 or shape[-1] != physical_tokens:
        raise ValueError(
            "observed input shape must include the actual physical token dimension: "
            f"shape={shape}, physical_tokens={physical_tokens}"
        )
    required = ("target_shape", "mask_shape", "position_shape")
    normalized: dict[str, list[int]] = {}
    for name in required:
        value = teacher_geometry.get(name)
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(f"teacher geometry requires non-empty {name}")
        current = [
            require_integer(f"teacher_geometry.{name}[{index}]", item)
            for index, item in enumerate(value)
        ]
        token_axis = 1 if name == "target_shape" and len(current) >= 2 else -1
        if current[token_axis] != physical_tokens:
            raise ValueError(
                f"teacher {name} does not match physical tokens on axis {token_axis}: "
                f"{current} != {physical_tokens}"
            )
        normalized[name] = current
    return shape, normalized


def _checkpoint_payload(
    *,
    run_id: str,
    next_segment_index: int,
    detached_loss_sum: float,
    base_parameters: Sequence[Any],
    parameters: Sequence[Any],
    optimizer: Any,
    optimizer_steps: int,
    phase_rows: list[dict[str, Any]],
    torch: Any,
    state: str,
    started_unix: float,
    peak_memory_bytes: int,
    optimizer_started_unix: float | None,
    optimizer_completed_unix: float | None,
    optimizer_seconds: float,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "state": state,
        "next_segment_index": int(next_segment_index),
        "completed_segments": list(range(int(next_segment_index))),
        "detached_loss_sum": float(detached_loss_sum),
        "base_parameters": list(base_parameters),
        "parameters": _cpu_parameters(parameters),
        "gradients": _cpu_gradients(parameters),
        "optimizer_state": optimizer.state_dict(),
        "optimizer_steps": int(optimizer_steps),
        "rng_state": _rng_state(torch),
        "phase_rows": phase_rows,
        "started_unix": float(started_unix),
        "peak_memory_bytes": int(peak_memory_bytes),
        "optimizer_started_unix": optimizer_started_unix,
        "optimizer_completed_unix": optimizer_completed_unix,
        "optimizer_seconds": float(optimizer_seconds),
    }


def run_segmented_update(
    *,
    parameters: Sequence[Any],
    optimizer: Any,
    segments: Sequence[Any],
    item_count: Callable[[Any], int],
    loss_sum: Callable[[Any], Any],
    output: str | Path,
    identity: dict[str, Any],
    physical_tokens: int,
    observed_input_shape: Sequence[int],
    teacher_geometry: dict[str, Any],
    peak_memory_bytes: int | Callable[[], int],
    receipt: str | Path | None = None,
    backend: str = "accelerated",
    resume: bool = True,
    restart: bool = False,
    synchronize: Callable[[], None] | None = None,
    on_segment_committed: Callable[[int, dict[str, Any]], None] | None = None,
    receipt_fields: dict[str, Any] | None = None,
    post_step_validate: Callable[[], dict[str, Any] | None] | None = None,
    semantic_claim: str = "causal-segmented-no-equivalence-claim",
    semantic_parity_tested: bool = False,
) -> dict[str, Any]:
    """Run one resumable exact logical-mean update with one optimizer step."""
    import torch

    if backend not in {"accelerated", "reference"}:
        raise ValueError(f"unsupported update backend {backend!r}")
    if not isinstance(semantic_parity_tested, bool):
        raise TypeError(
            f"semantic_parity_tested must be a bool, got {semantic_parity_tested!r}"
        )
    if semantic_claim in {"exact", "equal-work"} and not semantic_parity_tested:
        raise ValueError(
            f"semantic claim {semantic_claim!r} requires a passing semantic parity test"
        )
    identity = canonical_identity(identity)
    input_shape, teacher_shapes = _validate_geometry(
        physical_tokens, observed_input_shape, teacher_geometry
    )
    values = list(parameters)
    work = list(segments)
    counts = [
        require_integer(f"segment_items[{index}]", item_count(segment))
        for index, segment in enumerate(work)
    ]
    if not values:
        raise ValueError("update requires at least one trainable parameter")
    if not counts or any(count <= 0 for count in counts):
        raise ValueError(f"accumulation segments must be non-empty, got {counts}")
    logical_items = sum(counts)
    output_path = Path(output).resolve()
    receipt_path = (
        Path(receipt).resolve()
        if receipt is not None
        else output_path.with_name(f"{output_path.name}.receipt.json")
    )
    checkpoint_dir = Path(f"{output_path}.checkpoint")
    synchronize = (lambda: None) if synchronize is None else synchronize
    payload: dict[str, Any] | None = None

    if restart:
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        output_path.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)
    elif checkpoint_dir.is_dir():
        payload, manifest = load_checkpoint(
            checkpoint_dir,
            expected_identity=identity,
            expected_backend=backend,
            expected_segment_plan=counts,
        )
        if manifest["status"] == "COMPLETE":
            completed_output, completed_receipt = verify_completed_files(
                checkpoint_dir, manifest
            )
            if completed_output != output_path:
                raise RuntimeError(
                    "completed update artifact path mismatch after relocation"
                )
            if completed_receipt != receipt_path:
                raise RuntimeError(
                    "completed update receipt path mismatch after relocation"
                )
            return json.loads(completed_receipt.read_text())
        if not resume:
            raise RuntimeError(
                "incomplete update checkpoint exists; use resume or restart"
            )
    elif output_path.exists() or receipt_path.exists():
        raise RuntimeError(
            "update output exists without a completion checkpoint; use restart"
        )

    invocation_started = time.time()
    monotonic_started = time.perf_counter()
    if payload is None:
        run_id = uuid.uuid4().hex
        base_parameters = _cpu_parameters(values)
        start_index = 0
        detached_loss_sum = 0.0
        phase_rows: list[dict[str, Any]] = []
        optimizer_steps = 0
        optimizer_started_unix: float | None = None
        optimizer_completed_unix: float | None = None
        optimizer_seconds = 0.0
        started_unix = invocation_started
        peak_observed = _memory_value(peak_memory_bytes)
        optimizer.zero_grad(set_to_none=True)
    else:
        run_id = str(payload["run_id"])
        base_parameters = payload["base_parameters"]
        _restore_parameters(values, payload["parameters"])
        optimizer.load_state_dict(payload["optimizer_state"])
        _restore_gradients(values, payload["gradients"])
        _restore_rng(torch, payload["rng_state"])
        start_index = int(payload["next_segment_index"])
        detached_loss_sum = float(payload["detached_loss_sum"])
        phase_rows = list(payload["phase_rows"])
        optimizer_steps = int(payload["optimizer_steps"])
        optimizer_started_unix = payload.get("optimizer_started_unix")
        optimizer_completed_unix = payload.get("optimizer_completed_unix")
        optimizer_seconds = float(payload.get("optimizer_seconds", 0.0))
        started_unix = float(payload["started_unix"])
        peak_observed = max(
            int(payload.get("peak_memory_bytes", 0)), _memory_value(peak_memory_bytes)
        )
        if optimizer_steps not in {0, 1}:
            raise RuntimeError(
                f"checkpoint optimizer step count is invalid: {optimizer_steps}"
            )

    for index in range(start_index, len(work)):
        forward_started_unix = time.time()
        forward_started = time.perf_counter()
        current = loss_sum(work[index])
        synchronize()
        forward_completed_unix = time.time()
        forward_seconds = time.perf_counter() - forward_started
        peak_observed = max(peak_observed, _memory_value(peak_memory_bytes))
        if getattr(current, "ndim", None) != 0:
            raise ValueError("loss_sum must return a scalar summed loss")
        if not bool(torch.isfinite(current)):
            raise RuntimeError(f"non-finite update loss at segment {index}")
        detached_loss_sum += float(current.detach())

        backward_started_unix = time.time()
        backward_started = time.perf_counter()
        backward_logical_mean(current, logical_items)
        synchronize()
        backward_completed_unix = time.time()
        backward_seconds = time.perf_counter() - backward_started
        peak_observed = max(peak_observed, _memory_value(peak_memory_bytes))
        phase_rows.append(
            {
                "segment_index": index,
                "items": counts[index],
                "forward_started_unix": forward_started_unix,
                "forward_completed_unix": forward_completed_unix,
                "forward_seconds": forward_seconds,
                "backward_started_unix": backward_started_unix,
                "backward_completed_unix": backward_completed_unix,
                "backward_seconds": backward_seconds,
                "loss_sum": float(current.detach()),
            }
        )
        manifest = commit_segment_checkpoint(
            checkpoint_dir,
            _checkpoint_payload(
                run_id=run_id,
                next_segment_index=index + 1,
                detached_loss_sum=detached_loss_sum,
                base_parameters=base_parameters,
                parameters=values,
                optimizer=optimizer,
                optimizer_steps=optimizer_steps,
                phase_rows=phase_rows,
                torch=torch,
                state="optimizer_pending" if index + 1 == len(work) else "accumulating",
                started_unix=started_unix,
                peak_memory_bytes=peak_observed,
                optimizer_started_unix=optimizer_started_unix,
                optimizer_completed_unix=optimizer_completed_unix,
                optimizer_seconds=optimizer_seconds,
            ),
            identity=identity,
            backend=backend,
            segment_plan=counts,
        )
        if on_segment_committed is not None:
            on_segment_committed(index, manifest)

    missing = [
        index for index, parameter in enumerate(values) if parameter.grad is None
    ]
    non_finite = [
        index
        for index, parameter in enumerate(values)
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
    ]
    if missing or non_finite:
        raise RuntimeError(
            "update produced invalid required trainable gradients: "
            f"missing={missing}, non_finite={non_finite}"
        )

    if optimizer_steps == 0:
        optimizer_started_unix = time.time()
        optimizer_started = time.perf_counter()
        optimizer.step()
        synchronize()
        optimizer_seconds = time.perf_counter() - optimizer_started
        optimizer_completed_unix = time.time()
        peak_observed = max(peak_observed, _memory_value(peak_memory_bytes))
        optimizer_steps = 1
        commit_segment_checkpoint(
            checkpoint_dir,
            _checkpoint_payload(
                run_id=run_id,
                next_segment_index=len(work),
                detached_loss_sum=detached_loss_sum,
                base_parameters=base_parameters,
                parameters=values,
                optimizer=optimizer,
                optimizer_steps=optimizer_steps,
                phase_rows=phase_rows,
                torch=torch,
                state="optimizer_done",
                started_unix=started_unix,
                peak_memory_bytes=peak_observed,
                optimizer_started_unix=optimizer_started_unix,
                optimizer_completed_unix=optimizer_completed_unix,
                optimizer_seconds=optimizer_seconds,
            ),
            identity=identity,
            backend=backend,
            segment_plan=counts,
        )
    if optimizer_steps != 1:
        raise RuntimeError(
            f"update requires exactly one optimizer step, got {optimizer_steps}"
        )

    before_sha = _parameter_sha256(base_parameters)
    after_sha = _parameter_sha256(values)
    max_abs_diff = max(
        float((after.detach().cpu() - before).abs().max())
        for before, after in zip(base_parameters, values)
    )
    if before_sha == after_sha or not math.isfinite(max_abs_diff) or max_abs_diff <= 0:
        raise RuntimeError("optimizer step did not produce a finite parameter mutation")
    validated_fields = post_step_validate() if post_step_validate is not None else None

    artifact = {
        "schema": "banana-smasher-update-artifact-v2",
        "backend": backend,
        "immutable_identity": identity,
        "physical_tokens": int(physical_tokens),
        "logical_tokens": int(physical_tokens) * len(work),
        "segments": len(work),
        "optimizer_steps": optimizer_steps,
        "parameters": _cpu_parameters(values),
        "optimizer_state": optimizer.state_dict(),
    }
    output_record = atomic_torch_save(output_path, artifact)
    completed_unix = time.time()
    result: dict[str, Any] = {
        "schema": "banana-smasher-update-receipt-v4",
        "status": "PASS_UPDATE",
        "command": "update",
        "backend": {"requested": backend, "used": backend},
        "fallback": {"used": False, "reason": None},
        "physical_tokens": int(physical_tokens),
        "logical_tokens": int(physical_tokens) * len(work),
        "logical_items": logical_items,
        "observed_input_shape": input_shape,
        "teacher_geometry": teacher_shapes,
        "segments": len(work),
        "segment_items": counts,
        "completed_segments": len(work),
        "resumed_segments": start_index,
        "forward_count": len(phase_rows),
        "backward_count": len(phase_rows),
        "optimizer_steps": optimizer_steps,
        "logical_mean_loss": detached_loss_sum / logical_items,
        "peak_memory_bytes": peak_observed,
        "immutable_identity": identity,
        "semantic_parity": {
            "claim": semantic_claim,
            "tested": bool(semantic_parity_tested),
        },
        "timing": {
            "started_unix": started_unix,
            "invocation_started_unix": invocation_started,
            "completed_unix": completed_unix,
            "invocation_seconds": time.perf_counter() - monotonic_started,
            "segments": phase_rows,
            "optimizer_started_unix": optimizer_started_unix,
            "optimizer_completed_unix": optimizer_completed_unix,
            "optimizer_seconds": optimizer_seconds,
        },
        "gradient_tensors": len(values),
        "finite_required_trainable_gradients": True,
        "parameter": {
            "sha256_before": before_sha,
            "sha256_after": after_sha,
            "max_abs_diff": max_abs_diff,
        },
        "output_artifact": os.path.relpath(output_path, checkpoint_dir),
        "receipt": os.path.relpath(receipt_path, checkpoint_dir),
        "durable_completion": True,
    }
    reserved = set(result)
    if receipt_fields:
        overlap = reserved.intersection(receipt_fields)
        if overlap:
            raise ValueError(
                f"receipt_fields cannot replace reserved fields: {sorted(overlap)}"
            )
        result.update(receipt_fields)
    if validated_fields:
        overlap = set(result).intersection(validated_fields)
        if overlap:
            raise ValueError(
                f"post-step fields cannot replace receipt fields: {sorted(overlap)}"
            )
        result.update(validated_fields)
    atomic_json(receipt_path, result)
    finalize_checkpoint(
        checkpoint_dir, receipt=receipt_path, output_record=output_record
    )
    durable_completed_unix = time.time()
    result["timing"]["artifact_completed_unix"] = completed_unix
    result["timing"]["completed_unix"] = durable_completed_unix
    result["timing"]["durable_completed_unix"] = durable_completed_unix
    result["timing"]["invocation_seconds"] = time.perf_counter() - monotonic_started
    atomic_json(receipt_path, result)
    finalize_checkpoint(
        checkpoint_dir, receipt=receipt_path, output_record=output_record
    )
    return result
