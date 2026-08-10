from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ._native_q2_data import (
    DUPLICATE_CHILD_PARENT_PAIRS,
    PARENT_LUT_DATA_SHA256,
    SEEDED_LUT_DATA_SHA256,
    _PARENT_LUT_U16_HEX,
)

__all__ = [
    "DUPLICATE_CHILD_PARENT_PAIRS",
    "NativeQ2Quantizer",
    "PARENT_LUT_DATA_SHA256",
    "SEEDED_LUT_DATA_SHA256",
    "canonical_parent_lut",
    "decode_states",
    "ldlq_transformed",
    "pack_trellis",
    "procedural_parent_indices",
    "seeded_parent_lut",
    "sha256_bytes",
    "state_levels",
    "tensor_core_permutation",
    "unpack_trellis",
]

_K2_MULTIPLIER = np.uint64(0x83DCD12D)
_K2_EDGES = 1 << 14


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_parent_lut() -> np.ndarray:
    """Return the exact 1024 parent values defined by the K2 half decoder."""
    bits = np.frombuffer(bytes.fromhex(_PARENT_LUT_U16_HEX), dtype=np.uint16)
    return bits.copy().view(np.float16)


def seeded_parent_lut() -> np.ndarray:
    """Return the sole native runtime LUT, with 111 child slots parent-seeded."""
    lut = canonical_parent_lut()
    for child, parent in DUPLICATE_CHILD_PARENT_PAIRS:
        lut[child] = lut[parent]
    return lut


def procedural_parent_indices(states: np.ndarray) -> np.ndarray:
    """Map uint16 trellis states to the native 1024-entry K2 LUT."""
    values = np.asarray(states, dtype=np.uint16).astype(np.uint64, copy=False)
    products = (values * _K2_MULTIPLIER) & np.uint64(0xFFFFFFFF)
    return (
        (products & 0xFF)
        + ((products >> 8) & 0xFF)
        + ((products >> 16) & 0xFF)
        + ((products >> 24) & 0xFF)
    ).astype(np.uint16)


def state_levels(states: np.ndarray) -> np.ndarray:
    levels = procedural_parent_indices(states)
    aliases = np.arange(1024, dtype=np.uint16)
    for child, parent in DUPLICATE_CHILD_PARENT_PAIRS:
        aliases[child] = parent
    return aliases[levels]


def decode_states(
    states: np.ndarray, lut: np.ndarray | None = None
) -> np.ndarray:
    values = seeded_parent_lut() if lut is None else np.asarray(lut)
    if values.shape != (1024,) or values.dtype != np.float16:
        raise ValueError("native Q2 LUT must be float16[1024]")
    return values[procedural_parent_indices(states)]


def pack_trellis(encoded: np.ndarray) -> np.ndarray:
    """Pack low two state bits in the native 32-int16-per-tile wire order."""
    states = np.asarray(encoded)
    if states.dtype != np.int16 or states.shape[-1] != 256:
        raise ValueError("encoded trellis must be int16[...,256]")
    low = states.view(np.uint16) & np.uint16(3)
    flat = low.reshape(-1, 16, 16).astype(np.uint32)
    shifts = np.arange(30, -1, -2, dtype=np.uint32)
    words32 = np.bitwise_or.reduce(flat << shifts, axis=-1)
    packed = np.empty((words32.shape[0], 32), dtype=np.uint16)
    packed[:, 0::2] = (words32 & np.uint32(0xFFFF)).astype(np.uint16)
    packed[:, 1::2] = (words32 >> np.uint32(16)).astype(np.uint16)
    return packed.view(np.int16).reshape(*states.shape[:-1], 32)


def unpack_trellis(packed: np.ndarray) -> np.ndarray:
    values = np.asarray(packed)
    if values.dtype != np.int16 or values.shape[-1] != 32:
        raise ValueError("packed Q2 trellis must be int16[...,32]")
    words = values.view(np.uint16).reshape(-1, 16, 2).astype(np.uint32)
    words32 = words[..., 0] | (words[..., 1] << np.uint32(16))
    shifts = np.arange(30, -1, -2, dtype=np.uint32)
    states = ((words32[..., None] >> shifts) & np.uint32(3)).astype(np.uint16)
    return states.reshape(*values.shape[:-1], 256).view(np.int16)


def tensor_core_permutation(device: Any):
    import torch

    permutation = [0] * 256
    for thread in range(32):
        r0 = (thread % 4) * 2
        rows = (r0, r0 + 1, r0 + 8, r0 + 9)
        c0 = thread // 4
        columns = (c0, c0 + 8)
        offset = thread * 8
        permutation[offset : offset + 8] = [
            rows[0] * 16 + columns[0],
            rows[1] * 16 + columns[0],
            rows[2] * 16 + columns[0],
            rows[3] * 16 + columns[0],
            rows[0] * 16 + columns[1],
            rows[1] * 16 + columns[1],
            rows[2] * 16 + columns[1],
            rows[3] * 16 + columns[1],
        ]
    return torch.tensor(permutation, dtype=torch.int64, device=device)


def load_native_q2_extension(
    *, build_directory: str | Path | None = None, verbose: bool = False
):
    from torch.utils.cpp_extension import load

    module_dir = Path(__file__).parent
    sources = [
        module_dir / "_native_q2_ext.cpp",
        module_dir / "_native_q2_ext.cu",
    ]
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in sources)).hexdigest()[:12]
    name = f"banana_native_q2_{digest}"
    kwargs: dict[str, Any] = {}
    if build_directory is not None:
        directory = Path(build_directory)
        directory.mkdir(parents=True, exist_ok=True)
        kwargs["build_directory"] = str(directory)
    os.environ.setdefault("MAX_JOBS", "4")
    return load(
        name=name,
        sources=[str(path) for path in sources],
        extra_cuda_cflags=["-O3", "--std=c++17"],
        verbose=verbose,
        with_cuda=True,
        **kwargs,
    )


class NativeQ2Quantizer:
    def __init__(
        self,
        lut,
        *,
        build_directory: str | Path | None = None,
        verbose_build: bool = False,
    ) -> None:
        import torch

        if lut.device.type != "cuda" or lut.dtype != torch.float16 or tuple(lut.shape) != (1024,):
            raise ValueError("native Q2 LUT must be CUDA float16[1024]")
        self.lut = lut.contiguous()
        self.extension: Any = load_native_q2_extension(
            build_directory=build_directory, verbose=verbose_build
        )
        self._edge_history = None
        self._cost_scratch = None
        self.cuda_calls = 0
        self.fallback_calls = 0

    def quantize(self, tiles):
        import torch

        if tiles.device != self.lut.device or tiles.dtype != torch.float32:
            raise ValueError("tiles must be CUDA float32 on the LUT device")
        tiles = tiles.contiguous()
        required = (tiles.shape[0], 256, _K2_EDGES)
        if self._edge_history is None or tuple(self._edge_history.shape) != required:
            self._edge_history = torch.empty(
                required, dtype=torch.int16, device=tiles.device
            )
            self._cost_scratch = torch.empty(
                (tiles.shape[0], 2, _K2_EDGES),
                dtype=torch.float16,
                device=tiles.device,
            )
        values = torch.empty_like(tiles)
        indices = torch.empty_like(tiles, dtype=torch.int16)
        self.extension.quantize_tiles_q2(
            tiles,
            values,
            indices,
            self._cost_scratch,
            self._edge_history,
            self.lut,
        )
        self.cuda_calls += 1
        return values, indices


def ldlq_transformed(
    weight,
    ldl,
    quantizer: NativeQ2Quantizer,
    *,
    buffer_rows: int = 128,
    resume: dict[str, Any] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
):
    """Run the exact reverse-buffer/reverse-target LDLQ recurrence."""
    import torch

    if weight.device.type != "cuda" or ldl.device != weight.device:
        raise ValueError("transformed weight and LDL matrix must share a CUDA device")
    if weight.dtype != torch.float32 or ldl.dtype != torch.float32:
        raise ValueError("LDLQ inputs must be float32")
    size_k, size_n = weight.shape
    if size_k % buffer_rows or buffer_rows % 16 or size_n % 128:
        raise ValueError("LDLQ geometry is not divisible by the native tile/buffer sizes")
    permutation = tensor_core_permutation(weight.device)
    inverse = torch.argsort(permutation)
    if resume is None:
        weight_q = torch.zeros_like(weight)
        encoded = torch.zeros(
            (size_k // 16, size_n // 16, 256),
            dtype=torch.int16,
            device=weight.device,
        )
        product_cache = torch.zeros_like(weight)
        completed_buffers = 0
    else:
        weight_q = resume["weight_q"].to(weight.device)
        encoded = resume["encoded"].to(weight.device)
        product_cache = resume["product_cache"].to(weight.device)
        completed_buffers = int(resume["completed_buffers"])
    for ordinal, upper in enumerate(range(size_k, 0, -buffer_rows)):
        if ordinal < completed_buffers:
            continue
        lower = upper - buffer_rows
        buffer_weight = weight[lower:upper]
        buffer_quantized = weight_q[lower:upper]
        buffer_encoded = encoded[lower // 16 : upper // 16]
        buffer_product = product_cache[lower:upper]
        buffer_ldl = ldl[lower:upper]
        for target_upper in range(buffer_rows, 0, -16):
            target_lower = target_upper - 16
            error = buffer_weight[target_upper:] - buffer_quantized[target_upper:]
            ldl_slice = buffer_ldl[
                target_upper:, lower + target_lower : lower + target_upper
            ]
            compensation = buffer_product[target_lower:target_upper]
            compensation.addmm_(ldl_slice.T, error, alpha=1.0, beta=1.0)
            rows = buffer_weight[target_lower:target_upper] + compensation
            tiles = (
                rows.reshape(16, size_n // 16, 16)
                .permute(1, 0, 2)
                .reshape(size_n // 16, 256)
            )
            values, indices = quantizer.quantize(tiles[:, permutation])
            values = values[:, inverse]
            values = (
                values.reshape(size_n // 16, 16, 16)
                .permute(1, 0, 2)
                .reshape(16, size_n)
            )
            buffer_quantized[target_lower:target_upper] = values
            buffer_encoded[target_lower // 16 : target_upper // 16] = indices.unsqueeze(0)
        buffer_error = buffer_weight - buffer_quantized
        product_cache.addmm_(buffer_ldl.T, buffer_error, alpha=1.0, beta=1.0)
        completed_buffers = ordinal + 1
        if checkpoint is not None:
            checkpoint(
                {
                    "completed_buffers": completed_buffers,
                    "weight_q": weight_q,
                    "encoded": encoded,
                    "product_cache": product_cache,
                    "cuda_calls": quantizer.cuda_calls,
                    "fallback_calls": quantizer.fallback_calls,
                }
            )
    return weight_q, encoded
