from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from .token_sizing import MemoryBudget, choose_physical_tokens, require_integer
from .update_engine import run_segmented_update


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in value.shape)


def prepare_tensor_segments(
    *,
    input_ids: Any,
    teacher_targets: Any,
    teacher_mask: Any,
    positions: Any,
    requested_tokens: int,
    segments: int,
    batch_size: int,
    memory_budget: MemoryBudget,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Autosize, validate, and slice batch-1 tensors before model compute."""
    import torch

    segments = require_integer("segments", segments)
    batch_size = require_integer("batch_size", batch_size)
    if batch_size != 1:
        raise ValueError(
            f"the portable update core currently requires batch_size=1, got {batch_size}"
        )
    if segments <= 0:
        raise ValueError("segments must be positive")

    sizing = choose_physical_tokens(
        requested_tokens=requested_tokens,
        batch_size=batch_size,
        budget=memory_budget,
    )
    physical_tokens = int(sizing["physical_tokens"])
    logical_extent = physical_tokens * segments
    tensors = {
        "input_ids": input_ids,
        "teacher_mask": teacher_mask,
        "positions": positions,
    }
    for name, value in tensors.items():
        shape = _shape(value)
        if len(shape) != 2 or shape[0] != batch_size:
            raise ValueError(f"{name} must have batch-1 rank-2 geometry, got {shape}")
        if shape[1] < logical_extent:
            raise ValueError(
                f"{name} has {shape[1]} tokens but selected geometry requires {logical_extent}"
            )
    target_shape = _shape(teacher_targets)
    if len(target_shape) < 2 or target_shape[0] != batch_size:
        raise ValueError(
            "teacher_targets must have batch-1 geometry with token axis 1, "
            f"got {target_shape}"
        )
    if target_shape[1] < logical_extent:
        raise ValueError(
            f"teacher_targets has {target_shape[1]} tokens but selected geometry "
            f"requires {logical_extent}"
        )
    if teacher_mask.dtype != torch.bool:
        raise ValueError("teacher_mask must have boolean dtype")
    selected_positions = positions[:, :logical_extent]
    if selected_positions.numel() > 1 and not bool(
        torch.equal(
            selected_positions[:, 1:] - selected_positions[:, :-1],
            torch.ones_like(selected_positions[:, 1:]),
        )
    ):
        raise ValueError(
            "positions are not contiguous across selected logical geometry"
        )

    work: list[dict[str, Any]] = []
    for index in range(segments):
        start = index * physical_tokens
        stop = start + physical_tokens
        position_slice = positions[:, start:stop]
        work.append(
            {
                "segment_index": index,
                "token_start": start,
                "token_stop": stop,
                "input_ids": input_ids[:, start:stop],
                "teacher_targets": teacher_targets[:, start:stop, ...],
                "teacher_mask": teacher_mask[:, start:stop],
                "positions": position_slice,
            }
        )
    sizing = {
        **sizing,
        "segments": segments,
        "logical_tokens": logical_extent,
    }
    return work, sizing


def run_tensor_update(
    *,
    parameters: Sequence[Any],
    optimizer: Any,
    input_ids: Any,
    teacher_targets: Any,
    teacher_mask: Any,
    positions: Any,
    requested_tokens: int,
    segments: int,
    batch_size: int,
    memory_budget: MemoryBudget,
    loss_sum: Callable[[dict[str, Any]], Any],
    output: str | Path,
    identity: dict[str, Any],
    peak_memory_bytes: int | Callable[[], int],
    receipt: str | Path | None = None,
    backend: str = "accelerated",
    resume: bool = True,
    restart: bool = False,
    synchronize: Callable[[], None] | None = None,
    on_segment_committed: Callable[[int, dict[str, Any]], None] | None = None,
    semantic_claim: str = "causal-segmented-no-equivalence-claim",
    semantic_parity_tested: bool = False,
) -> dict[str, Any]:
    """Run one portable tensor update after memory-derived physical autosizing."""
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
    first = work[0]
    return run_segmented_update(
        parameters=parameters,
        optimizer=optimizer,
        segments=work,
        item_count=lambda segment: int(segment["teacher_mask"].sum().item()),
        loss_sum=loss_sum,
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
        backend=backend,
        resume=resume,
        restart=restart,
        synchronize=synchronize,
        on_segment_committed=on_segment_committed,
        receipt_fields={
            "requested_physical_tokens": int(requested_tokens),
            "memory_sizing": sizing,
        },
        semantic_claim=semantic_claim,
        semantic_parity_tested=semantic_parity_tested,
    )


def _update_entry_point(name: str) -> Any:
    discovered = metadata.entry_points()
    if hasattr(discovered, "select"):
        matches = list(
            discovered.select(group="banana_smasher.update_backends", name=name)
        )
    else:  # pragma: no cover - Python 3.9 compatibility
        group_entries = getattr(discovered, "get")("banana_smasher.update_backends", [])
        matches = [entry for entry in group_entries if entry.name == name]
    if len(matches) != 1:
        raise RuntimeError(
            "requested update backend is not uniquely installed: "
            f"name={name!r}, matches={len(matches)}"
        )
    return matches[0]


def run_registered_update(
    *,
    backend_name: str,
    request: str | Path,
    output: str | Path,
    receipt: str | Path | None,
    identity: dict[str, Any],
    requested_tokens: int,
    segments: int,
    batch_size: int,
    memory_budget: MemoryBudget,
    resume: bool = True,
    restart: bool = False,
) -> dict[str, Any]:
    """Dispatch a fail-closed installed backend after core-owned autosizing.

    Backends are standard ``banana_smasher.update_backends`` entry points. They
    must use the public tensor/update-engine API and return its durable receipt;
    no reference or reduced-work fallback is selected here.
    """
    from .update_checkpoint import canonical_identity

    request_path = Path(request).resolve()
    if not request_path.is_file():
        raise FileNotFoundError(
            f"update backend request does not exist: {request_path}"
        )
    requested_tokens = require_integer("requested_tokens", requested_tokens)
    segments = require_integer("segments", segments)
    batch_size = require_integer("batch_size", batch_size)
    if batch_size != 1:
        raise ValueError(
            f"the portable update core requires batch_size=1, got {batch_size}"
        )
    if segments <= 0:
        raise ValueError("segments must be positive")
    sizing = choose_physical_tokens(
        requested_tokens=requested_tokens,
        batch_size=batch_size,
        budget=memory_budget,
    )
    entry = _update_entry_point(backend_name)
    runner = entry.load()
    if not callable(runner):
        raise RuntimeError(
            f"update backend entry point is not callable: {backend_name!r}"
        )
    result = runner(
        request=request_path,
        output=Path(output).resolve(),
        receipt=None if receipt is None else Path(receipt).resolve(),
        identity=canonical_identity(identity),
        requested_tokens=requested_tokens,
        physical_tokens=int(sizing["physical_tokens"]),
        segments=segments,
        batch_size=batch_size,
        memory_sizing=sizing,
        resume=bool(resume),
        restart=bool(restart),
    )
    if not isinstance(result, dict):
        raise RuntimeError("update backend returned a non-receipt value")
    if result.get("status") != "PASS_UPDATE":
        raise RuntimeError(
            "update backend did not return the exact passing receipt status: "
            f"{result.get('status')!r}"
        )
    required = {
        "physical_tokens": int(sizing["physical_tokens"]),
        "segments": segments,
        "optimizer_steps": 1,
    }
    for name, expected in required.items():
        if result.get(name) != expected:
            raise RuntimeError(
                f"update backend receipt mismatch for {name}: "
                f"{result.get(name)!r} != {expected!r}"
            )
    shape = result.get("observed_input_shape")
    expected_shape = [1, sizing["physical_tokens"]]
    if shape != expected_shape:
        raise RuntimeError(
            "update backend did not observe the batch-1 physical token shape: "
            f"{shape!r} != {expected_shape!r}"
        )
    if result.get("fallback", {"used": False}).get("used"):
        raise RuntimeError("update backend used a forbidden fallback")
    return result
