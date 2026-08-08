"""Exact paired-step CUDA-graph solver for Periodic QTIP3 PR31 phases."""
from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
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


def solve_periodic_qtip3_cells_exact(
    targets: Sequence[np.ndarray],
    scalar_lut: torch.Tensor,
) -> tuple[list[np.ndarray], dict[str, int]]:
    """Encode homogeneous ``[blocks,64,4]`` cells through exact AOT B256 passes."""
    if not targets:
        raise ValueError("Periodic QTIP3 AOT cell batch must not be empty")
    checked: list[np.ndarray] = []
    for target in targets:
        value = np.asarray(target)
        if (
            value.dtype != np.float32
            or value.ndim != 3
            or value.shape[1:] != (64, 4)
            or not np.isfinite(value).all()
        ):
            raise ValueError(
                "Periodic QTIP3 AOT cell targets must be finite float32 [N,64,4]"
            )
        if len(value) % BATCH:
            raise ValueError(
                "Periodic QTIP3 AOT cells require a multiple of 256 blocks"
            )
        checked.append(np.ascontiguousarray(value))

    from ..qtip25_native_v4 import native_v4_geometry
    from ..qtip25_native_v4_cuda_cell import _pack_cuda_states_v4

    before = counters()
    packed_cells: list[np.ndarray] = []
    chunk_calls = 0
    for target in checked:
        packed_parts = []
        for start in range(0, len(target), BATCH):
            host = torch.from_numpy(target[start : start + BATCH])
            if scalar_lut.is_cuda:
                host = host.pin_memory()
            values = host.to(
                scalar_lut.device, non_blocking=scalar_lut.is_cuda
            ).contiguous()
            phases = values.reshape(BATCH, STEPS).transpose(0, 1).contiguous()
            open_states = solve_periodic_qtip3_exact(
                phases.roll(STEPS // 2, dims=0), scalar_lut
            )
            open_values = open_states.transpose(0, 1).reshape(BATCH, 64, 4)
            overlap = (open_values[:, 32, 0] >> 3).to(torch.int32).contiguous()
            closed_states = solve_periodic_qtip3_exact(
                phases, scalar_lut, overlap=overlap
            )
            final_states = (
                closed_states.transpose(0, 1)
                .reshape(BATCH, 64, 4)[:, :, -1]
                .contiguous()
            )
            packed_parts.append(
                _pack_cuda_states_v4(
                    final_states, geometry=native_v4_geometry(3.0)
                ).cpu()
            )
            chunk_calls += 1
        packed_cells.append(
            np.ascontiguousarray(torch.cat(packed_parts).numpy())
        )
    after = counters()
    return packed_cells, {
        "cells": len(checked),
        "blocks": int(sum(len(value) for value in checked)),
        "contiguous_b256_chunks": chunk_calls,
        "graph_replay_calls": after["graph_replay_calls"]
        - before["graph_replay_calls"],
        "fallback_calls": after["fallback_calls"] - before["fallback_calls"],
    }


__all__ = [
    "counters",
    "geometry",
    "solve_periodic_qtip3_cells_exact",
    "solve_periodic_qtip3_exact",
]
