from __future__ import annotations

import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys
from typing import Any

_EXTENSION: Any | None = None


def periodic_cuda_geometry() -> dict[str, Any]:
    """Describe the specialized full-branch PERIODIC producer."""
    return {
        "implementation": "periodic-k2-k3-cuda-exact-v1",
        "L": 16,
        "V": 2,
        "steps": 128,
        "transition_bits": [4, 6],
        "retained_prefix_costs": [4096, 1024],
        "branches_per_prefix": [16, 64],
        "branch_sampling": "full",
        "passes": ["open", "cyclic-closure"],
        "backpointer_dtype": "uint8",
        "fallback": None,
    }


def _load_extension() -> Any:
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    raw_path = os.environ.get("BANANA_SMASHER_PERIODIC_CUDA_EXTENSION")
    if not raw_path:
        raise RuntimeError(
            "PERIODIC exact CUDA producer is not configured; build periodic_cuda "
            "and set BANANA_SMASHER_PERIODIC_CUDA_EXTENSION"
        )
    extension = Path(raw_path).expanduser().resolve()
    if not extension.is_file():
        raise RuntimeError(f"PERIODIC exact CUDA producer is missing: {extension}")
    name = "periodic_qtip_cuda_exact"
    loader = importlib.machinery.ExtensionFileLoader(name, str(extension))
    spec = importlib.util.spec_from_file_location(name, extension, loader=loader)
    if spec is None:
        raise ImportError(f"cannot load PERIODIC exact CUDA producer: {extension}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    _EXTENSION = module
    return module


def solve_periodic_cuda(targets: Any, lut: Any) -> Any:
    """Solve CUDA ``[B,128,2]`` targets with full alternating K2/K3 search."""
    import torch

    if (
        not targets.is_cuda
        or targets.dtype != torch.float32
        or targets.ndim != 3
        or tuple(targets.shape[1:]) != (128, 2)
    ):
        raise ValueError("PERIODIC targets must be CUDA float32 [B,128,2]")
    batch = int(targets.shape[0])
    if batch < 256 or batch % 256:
        raise ValueError("PERIODIC exact CUDA batch must be divisible by 256")
    if (
        not lut.is_cuda
        or lut.device != targets.device
        or lut.dtype != torch.float32
        or tuple(lut.shape) != (65536, 2)
    ):
        raise ValueError("PERIODIC LUT must be CUDA float32 [65536,2]")
    extension = _load_extension()
    x = targets.permute(1, 2, 0).reshape(256, batch).contiguous()
    lut = lut.contiguous()
    outputs = []
    for start in range(0, batch, 256):
        source = x[:, start : start + 256].contiguous()
        open_states = extension.viterbi(source, lut, None)[0]
        closing = (open_states[-1] & 1023).to(torch.int32).contiguous()
        outputs.append(extension.viterbi(source, lut, closing)[0])
    return torch.cat(outputs, dim=1).transpose(0, 1).contiguous()
