from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenAttentionDescriptor:
    """Hash-bound lazy descriptor for a file-backed frozen attention member."""

    path: Path
    key: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    member_sha256: str
    index_path: Path
    index_sha256: str
    execution_device: str

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        key: str,
        member_sha256: str | None,
        index_path: str | Path,
        index_sha256: str,
        execution_device: str,
    ) -> FrozenAttentionDescriptor:
        source = Path(path)
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"frozen attention member must be a regular file: {source}")
        index_source = Path(index_path)
        if index_source.is_symlink() or not index_source.is_file():
            raise ValueError(
                f"frozen attention index must be a regular file: {index_source}"
            )
        if len(index_sha256) != 64:
            raise ValueError("index_sha256 must be a SHA-256 hex digest")
        if _sha256(index_source) != index_sha256:
            raise RuntimeError("frozen attention index SHA-256 mismatch")
        array = np.load(source, mmap_mode="r", allow_pickle=False)
        actual_sha = _sha256(source)
        if member_sha256 is not None and actual_sha != member_sha256:
            raise RuntimeError("frozen attention member SHA-256 mismatch")
        return cls(
            path=source,
            key=str(key),
            shape=tuple(int(value) for value in array.shape),
            dtype=str(array.dtype),
            nbytes=int(array.nbytes),
            member_sha256=actual_sha,
            index_path=index_source,
            index_sha256=index_sha256,
            execution_device=str(execution_device),
        )

    def open(self) -> np.ndarray[Any, Any]:
        if self.path.is_symlink() or not self.path.is_file():
            raise RuntimeError("frozen attention member is missing or not a regular file")
        if _sha256(self.path) != self.member_sha256:
            raise RuntimeError("frozen attention member SHA-256 mismatch")
        if self.index_path.is_symlink() or not self.index_path.is_file():
            raise RuntimeError("frozen attention index is missing or not a regular file")
        if _sha256(self.index_path) != self.index_sha256:
            raise RuntimeError("frozen attention index SHA-256 mismatch")
        array = np.load(self.path, mmap_mode="r", allow_pickle=False)
        if tuple(array.shape) != self.shape or str(array.dtype) != self.dtype or int(array.nbytes) != self.nbytes:
            raise RuntimeError("frozen attention descriptor geometry mismatch")
        return array


class ProductionTrainableSurface:
    """Small manifest-shaped full-depth trainable seam used by update runtimes."""

    def __init__(self, *, depth: int, width: int) -> None:
        import torch

        if depth <= 0 or width <= 0:
            raise ValueError("depth and width must be positive")
        self.depth = int(depth)
        self.width = int(width)
        self._module = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "projection": torch.nn.Linear(width, width, bias=False),
                        "norm": torch.nn.LayerNorm(width),
                    }
                )
                for _ in range(depth)
            ]
        )

    def __call__(self, value: Any) -> Any:
        current = value
        for layer in self._module:
            current = layer["norm"](current + layer["projection"](current))
        return current

    def parameters(self) -> Any:
        return self._module.parameters()

    def to(self, device: Any) -> ProductionTrainableSurface:
        self._module.to(device)
        return self

    def trainable_census(self) -> dict[str, Any]:
        import torch

        layers_with_gradients = []
        finite = True
        parameter_tensors = 0
        for index, layer in enumerate(self._module):
            gradients = [parameter.grad for parameter in layer.parameters()]
            parameter_tensors += len(gradients)
            if gradients and all(gradient is not None for gradient in gradients):
                layers_with_gradients.append(index)
            finite = finite and all(
                gradient is not None and bool(torch.isfinite(gradient).all())
                for gradient in gradients
            )
        return {
            "depth": self.depth,
            "parameter_tensors": parameter_tensors,
            "layers_with_gradients": layers_with_gradients,
            "all_gradients_finite": finite,
        }
