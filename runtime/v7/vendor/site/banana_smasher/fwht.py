from __future__ import annotations

import importlib
import math
from typing import Any

import torch

_STATS: dict[str, int] = {
    "calls": 0,
    "inplace_calls": 0,
    "autograd_calls": 0,
    "reference_calls": 0,
    "fused_calls": 0,
    "max_tensor_bytes": 0,
    "max_scratch_bytes": 0,
}

_QUACK_MAX_N = 32768


def _validate(x: torch.Tensor) -> int:
    if x.ndim < 1:
        raise ValueError("FWHT expects a tensor with at least one dimension")
    n = int(x.shape[-1])
    if n <= 0 or n & (n - 1):
        raise ValueError(f"FWHT length must be a positive power of two, got {n}")
    if not (x.is_floating_point() or x.is_complex()):
        raise TypeError(f"FWHT expects floating or complex input, got {x.dtype}")
    return n


def _record(
    x: torch.Tensor,
    *,
    inplace: bool,
    autograd: bool,
    backend: str = "reference",
) -> None:
    tensor_bytes = int(x.numel() * x.element_size())
    scratch_bytes = tensor_bytes // 2
    _STATS["calls"] += 1
    _STATS["inplace_calls"] += int(inplace)
    _STATS["autograd_calls"] += int(autograd)
    _STATS[f"{backend}_calls"] += 1
    _STATS["max_tensor_bytes"] = max(_STATS["max_tensor_bytes"], tensor_bytes)
    if backend == "reference":
        _STATS["max_scratch_bytes"] = max(_STATS["max_scratch_bytes"], scratch_bytes)


def _validate_fused_input(x: torch.Tensor, n: int) -> None:
    if not x.is_cuda:
        raise ValueError("Quack FWHT requires CUDA input")
    if not x.is_contiguous():
        raise ValueError("Quack FWHT requires contiguous input")
    if not x.is_floating_point() or x.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise TypeError(f"Quack FWHT requires supported floating input, got {x.dtype}")
    if n < 2 or n > _QUACK_MAX_N:
        raise ValueError(
            f"Quack FWHT length must be a supported power of two in [2, {_QUACK_MAX_N}], got {n}"
        )


def _butterfly_inplace(y: torch.Tensor, *, normalize: bool) -> torch.Tensor:
    """Transform contiguous ``y`` with one half-tensor scratch allocation per stage."""
    n = _validate(y)
    original_shape = y.shape
    h = 1
    while h < n:
        stage = y.reshape(*original_shape[:-1], -1, 2, h)
        left = stage[..., 0, :]
        right = stage[..., 1, :]
        # Keep only the old left half. The old right half remains live in ``right``
        # until both outputs have been written. This avoids a+b/a-b plus cat.
        old_left = left.clone()
        left.add_(right)
        right.copy_(old_left.sub_(right))
        y = stage.reshape(original_shape)
        h *= 2
    if normalize:
        # Preserve the incumbent arithmetic exactly (division, not reciprocal
        # multiplication) so byte-level decoder regressions remain meaningful.
        y.div_(math.sqrt(n))
    return y


class _BoundedFwht(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, normalize: bool) -> torch.Tensor:
        _validate(x)
        ctx.normalize = bool(normalize)
        y = x.clone(memory_format=torch.contiguous_format)
        _record(y, inplace=False, autograd=True)
        return _butterfly_inplace(y, normalize=ctx.normalize)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        grad = grad_output.clone(memory_format=torch.contiguous_format)
        # The normalized Walsh-Hadamard transform is self-adjoint; without
        # normalization its adjoint is the same unnormalized transform.
        return _butterfly_inplace(grad, normalize=ctx.normalize), None


def bounded_fwht(
    x: torch.Tensor,
    *,
    normalize: bool = True,
    inplace: bool = False,
    backend: str = "bounded",
) -> torch.Tensor:
    """Walsh-Hadamard transform with an explicit required backend.

    ``inplace=True`` is intended for frozen decode temporaries. It mutates a
    contiguous input and uses at most one half-tensor scratch allocation. A
    non-contiguous input is first materialized contiguously and that returned
    tensor is transformed. Grad-tracked inputs use the self-adjoint custom
    autograd path and cannot be mutated in place.

    The default ``backend="bounded"`` preserves the incumbent implementation.
    ``backend="quack"`` requires a contiguous CUDA floating-point tensor and
    lazily imports the fused CuTe kernel; an unavailable or invalid fused path
    raises instead of silently falling back.
    """
    n = _validate(x)
    if backend == "quack":
        if not normalize:
            raise ValueError("Quack FWHT requires normalize=True")
        if inplace:
            raise ValueError("Quack FWHT requires inplace=False")
        _validate_fused_input(x, n)
        hadamard_transform = importlib.import_module(
            "quack.hadamard"
        ).hadamard_transform
        result = hadamard_transform(x, scale=1 / math.sqrt(n))
        _record(x, inplace=False, autograd=x.requires_grad, backend="fused")
        return result
    if backend != "bounded":
        raise ValueError(f"unknown FWHT backend {backend!r}")
    if not inplace:
        return _BoundedFwht.apply(x, bool(normalize))
    if x.requires_grad:
        raise ValueError("in-place FWHT is only valid for tensors without gradients")
    y = x if x.is_contiguous() else x.contiguous()
    _record(y, inplace=True, autograd=False)
    return _butterfly_inplace(y, normalize=normalize)


def fwht_stats(*, reset: bool = False) -> dict[str, int]:
    result = dict(_STATS)
    if reset:
        for key in _STATS:
            _STATS[key] = 0
    return result
