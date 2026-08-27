"""Package-owned full-row exact CUDA Viterbi for public QTIP L16/K2/V2.

One CUDA CTA owns one complete source row. FP32 prefix costs remain resident while
exact four-bit q winners are written to a memory-admitted packed traceback. There
is no proposal, pruning, approximation, host recurrence, or slower fallback.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any

import torch

PREFIXES = 4096
BRANCHES = 16
STATES = 65536
STEPS = 128
MAX_CHUNK = 8192
_PRODUCTION_LINEAGE_SHA256 = (
    "379a24289514ead53de1415fdddc9cf77026d46c7b8d9ffef783fe5632a9319b"
)
_PARITY_REFERENCE_SHA256 = (
    "96b83c837a017c36f6630ac7b6b7a3be16888ea15a03593f3c4709b0675c3a50"
)


@lru_cache(maxsize=1)
def prepare_exact_cuda() -> Any:
    """Build/load the SHA-named full-row exact packed-backpointer producer."""
    from torch.utils.cpp_extension import load

    csrc = Path(__file__).with_name("csrc")
    sources = [csrc / "binding_exact.cpp", csrc / "trellis_v2_exact.cu"]
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.read_bytes())
    build_directory = Path.home() / ".cache/banana-smasher/trellis-v2" / digest.hexdigest()
    build_directory.mkdir(parents=True, exist_ok=True)
    return load(
        name=f"banana_smasher_trellis_v2_fullrow_{digest.hexdigest()[:16]}",
        sources=[str(source) for source in sources],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo", "--fmad=false"],
        build_directory=str(build_directory),
        with_cuda=True,
        verbose=False,
    )


def geometry(cb: Any) -> dict[str, int | str | bool]:
    got = (int(cb.L), int(cb.K), int(cb.V))
    if got != (16, 2, 2):
        raise ValueError(f"trellis-v2 is sealed for L16/K2/V2, got {got}")
    return {
        "implementation": "full-row-packed-backpointer-cuda-exact-v1",
        "production_lineage_sha256": _PRODUCTION_LINEAGE_SHA256,
        "parity_reference_sha256": _PARITY_REFERENCE_SHA256,
        "L": 16,
        "K": 2,
        "V": 2,
        "steps": STEPS,
        "full_states": STATES,
        "retained_prefix_costs": PREFIXES,
        "branches_per_prefix": BRANCHES,
        "full_branches_per_prefix": BRANCHES,
        "branch_sampling": "full",
        "strict_tie_order": "ascending-q-strict-less-than",
        "chunk_sequences": MAX_CHUNK,
        "backpointer_dtype": "packed-uint4-q",
        "ordering": "one-full-row-cta-with-resident-fp32-costs",
        "minimum_ctas_per_sm": 2,
        "production_default": True,
        "fallback": 0,
    }


def trellis_v2_exact(
    cb: Any,
    x: torch.Tensor,
    overlap: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return canonical all-16-branch states for CUDA input ``[2*T,B]``."""
    geometry(cb)
    if (
        not x.is_cuda
        or x.ndim != 2
        or x.shape[0] % 2
        or x.shape[0] < 8
    ):
        raise ValueError(
            f"x must be CUDA [2*T,B] with T>=4, got {tuple(x.shape)} {x.device}"
        )
    if x.dtype not in {torch.float16, torch.float32}:
        raise ValueError(
            f"public smash solve must produce float16/float32 trellis input; got {x.dtype}"
        )
    batch = int(x.shape[1])
    if batch < 1 or batch > MAX_CHUNK:
        raise ValueError(
            f"exact public QTIP2 path requires B in 1..{MAX_CHUNK}, got {batch}"
        )
    if not cb.lut.is_cuda or tuple(cb.lut.shape) != (2, STATES):
        raise ValueError(
            "canonical LUT must be CUDA [2,65536], got "
            f"{tuple(cb.lut.shape)} {cb.lut.device}"
        )
    if cb.lut.device != x.device:
        raise ValueError(f"canonical LUT device differs from x: {cb.lut.device} != {x.device}")
    if overlap is not None:
        if (
            not overlap.is_cuda
            or overlap.device != x.device
            or overlap.ndim != 1
            or overlap.numel() != batch
            or overlap.dtype not in {torch.int32, torch.int64}
        ):
            raise ValueError("overlap must be integral CUDA [B] on the input device")
        if bool(((overlap < 0) | (overlap >= PREFIXES)).any()):
            raise ValueError(f"overlap prefixes must be in [0, {PREFIXES})")
    x = x.to(dtype=torch.float32).contiguous()
    overlap_arg = (
        None if overlap is None else overlap.to(dtype=torch.int32).contiguous()
    )
    module = prepare_exact_cuda()
    lut_aos = cb.lut.T.to(dtype=torch.float32).contiguous()
    return module.viterbi(x, lut_aos, overlap_arg)[0]
