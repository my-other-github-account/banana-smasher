"""Exact paired-step CUDA-graph solver for Periodic QTIP3 PR31 phases."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

STEPS = 256
PREFIXES = 8192
STATES = 65536
BRANCHES = 8
BATCH = 256

_EXTENSION: Any | None = None
_COUNTERS = {
    "aot_load_calls": 0,
    "graph_replay_calls": 0,
    "fallback_calls": 0,
}


def geometry() -> dict[str, int | str | bool]:
    return {
        "implementation": "periodic-qtip3-pr31-paired-step-graph-b256-v1",
        "L": 16,
        "K": 3,
        "V": 1,
        "steps": STEPS,
        "full_states": STATES,
        "retained_prefix_costs": PREFIXES,
        "branches_per_prefix": BRANCHES,
        "branch_sampling": "full",
        "strict_tie_order": "ascending-q-strict-less-than",
        "backpointer_dtype": "packed-u3-pair-per-byte",
        "batch_contract": "exact-contiguous-B256",
        "launch_submission": "one-cuda-graph-replay-per-pass",
        "fallback": False,
    }


def counters() -> dict[str, int]:
    return dict(_COUNTERS)


def _extension() -> Any:
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    try:
        path = Path(os.environ["BANANA_SMASHER_PERIODIC_QTIP3_EXTENSION"])
        expected_sha256 = os.environ[
            "BANANA_SMASHER_PERIODIC_QTIP3_EXTENSION_SHA256"
        ]
        from ..qtip_kernel_cache import _load_sha_pinned_extension

        _EXTENSION = _load_sha_pinned_extension(
            "periodic_qtip3_cuda_exact", path, expected_sha256
        )
    except Exception as exc:
        raise RuntimeError(
            "Periodic QTIP3 B256 solve could not load its SHA-pinned exact "
            "paired-step CUDA producer; build and seal the packaged AOT extension"
        ) from exc
    _COUNTERS["aot_load_calls"] += 1
    return _EXTENSION


def solve_periodic_qtip3_exact(
    x: torch.Tensor,
    scalar_lut: torch.Tensor,
    *,
    overlap: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return scalar PR31 phase states for CUDA input shaped ``[256,256]``."""
    if (
        not x.is_cuda
        or x.dtype != torch.float32
        or x.ndim != 2
        or tuple(x.shape) != (STEPS, BATCH)
        or not x.is_contiguous()
    ):
        raise ValueError("Periodic QTIP3 AOT input must be contiguous float32 [256,256] CUDA")
    if (
        not scalar_lut.is_cuda
        or scalar_lut.device != x.device
        or scalar_lut.dtype != torch.float32
        or tuple(scalar_lut.shape) != (STATES,)
        or not scalar_lut.is_contiguous()
    ):
        raise ValueError("Periodic QTIP3 AOT LUT must be contiguous float32 [65536] CUDA")
    overlap_arg = None
    if overlap is not None:
        if (
            not overlap.is_cuda
            or overlap.device != x.device
            or overlap.dtype not in {torch.int32, torch.int64}
            or tuple(overlap.shape) != (BATCH,)
        ):
            raise ValueError("Periodic QTIP3 overlap must be integral CUDA [256]")
        if bool(((overlap < 0) | (overlap >= PREFIXES)).any()):
            raise ValueError("Periodic QTIP3 overlap prefixes must be in [0,8192)")
        overlap_arg = overlap.to(dtype=torch.int32).contiguous()
    states = _extension().viterbi(x, scalar_lut, overlap_arg)[0]
    if states.dtype != torch.int32 or tuple(states.shape) != (STEPS, BATCH):
        raise RuntimeError(
            "Periodic QTIP3 AOT producer returned an invalid state tensor: "
            f"{tuple(states.shape)} {states.dtype}"
        )
    _COUNTERS["graph_replay_calls"] += 1
    return states


__all__ = ["counters", "geometry", "solve_periodic_qtip3_exact"]
