from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from safetensors import safe_open
from torch import nn


_SAFETENSOR_DTYPES: dict[str, tuple[str, int]] = {
    "BOOL": ("torch.bool", 1),
    "U8": ("torch.uint8", 1),
    "I8": ("torch.int8", 1),
    "I16": ("torch.int16", 2),
    "U16": ("torch.uint16", 2),
    "F16": ("torch.float16", 2),
    "BF16": ("torch.bfloat16", 2),
    "I32": ("torch.int32", 4),
    "U32": ("torch.uint32", 4),
    "F32": ("torch.float32", 4),
    "F64": ("torch.float64", 8),
    "I64": ("torch.int64", 8),
    "U64": ("torch.uint64", 8),
    "F8_E4M3": ("torch.float8_e4m3fn", 1),
    "F8_E5M2": ("torch.float8_e5m2", 1),
}


def _fd_alias(fd: int) -> str:
    for root in (Path("/proc/self/fd"), Path("/dev/fd")):
        if root.is_dir():
            return str(root / str(fd))
    raise RuntimeError("verified file-descriptor loading is unavailable")


@contextmanager
def _verified_open(path: Path, expected_sha256: str | None = None):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        digest = hashlib.sha256()
        while block := os.read(fd, 8 << 20):
            digest.update(block)
        observed = digest.hexdigest()
        if expected_sha256 is not None and observed != expected_sha256:
            raise RuntimeError(
                f"source member identity drift: {observed} != {expected_sha256}"
            )
        os.lseek(fd, 0, os.SEEK_SET)
        yield _fd_alias(fd), observed, fd
    finally:
        os.close(fd)


def _verified_bytes(
    path: Path, expected_sha256: str, *, identity_name: str
) -> tuple[bytes, str]:
    try:
        with _verified_open(path, expected_sha256) as (_, observed, fd):
            data = bytearray()
            while block := os.read(fd, 8 << 20):
                data.extend(block)
            return bytes(data), observed
    except RuntimeError as exc:
        raise RuntimeError(
            f"source {identity_name} identity drift: {exc}"
        ) from exc


def _relative_path(value: str, *, field_name: str) -> str:
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise ValueError(f"{field_name} must be a confined relative path")
    normalized = candidate.as_posix()
    if normalized in {"", "."}:
        raise ValueError(f"{field_name} must identify a file")
    return normalized


def _confined(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    if path == resolved_root or resolved_root not in path.parents:
        raise RuntimeError(f"file-backed path escapes source root: {relative}")
    return path


def _index_member(document_bytes: bytes, tensor_key: str) -> str:
    try:
        document = json.loads(document_bytes)
    except json.JSONDecodeError as exc:
        raise RuntimeError("cannot parse file-backed index") from exc
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or tensor_key not in weight_map:
        raise RuntimeError(f"tensor key is missing from source index: {tensor_key}")
    member = weight_map[tensor_key]
    if not isinstance(member, str):
        raise RuntimeError(f"source index member is not a string: {tensor_key}")
    return _relative_path(member, field_name="member_relative_path")


def _tensor_metadata(
    path: Path, tensor_key: str, expected_sha256: str | None = None
) -> tuple[tuple[int, ...], str, int, str]:
    with _verified_open(path, expected_sha256) as (alias, observed_sha256, _):
        with safe_open(alias, framework="pt", device="cpu") as handle:
            if tensor_key not in handle.keys():
                raise RuntimeError(f"tensor key is missing from source member: {tensor_key}")
            source_slice = handle.get_slice(tensor_key)
            shape = tuple(int(item) for item in source_slice.get_shape())
            safetensor_dtype = str(source_slice.get_dtype())
    if safetensor_dtype not in _SAFETENSOR_DTYPES:
        raise RuntimeError(f"unsupported SafeTensor dtype: {safetensor_dtype}")
    dtype, element_size = _SAFETENSOR_DTYPES[safetensor_dtype]
    return shape, dtype, int(math.prod(shape) * element_size), observed_sha256


@dataclass(frozen=True, slots=True)
class FileBackedTensorDescriptor:
    """Relocatable immutable SafeTensor identity with no resident tensor fields."""

    index_relative_path: str
    member_relative_path: str
    tensor_key: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    index_sha256: str
    member_sha256: str
    execution_device: torch.device

    @classmethod
    def from_index(
        cls,
        *,
        root: str | Path,
        index_relative_path: str,
        expected_index_sha256: str,
        tensor_key: str,
        execution_device: str | torch.device,
    ) -> "FileBackedTensorDescriptor":
        source_root = Path(root).resolve()
        index_relative_path = _relative_path(
            index_relative_path, field_name="index_relative_path"
        )
        index_path = _confined(source_root, index_relative_path)
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        index_bytes, observed_index_sha256 = _verified_bytes(
            index_path,
            expected_index_sha256,
            identity_name="index",
        )
        member_relative_path = _index_member(index_bytes, tensor_key)
        member_path = _confined(source_root, member_relative_path)
        if not member_path.is_file():
            raise FileNotFoundError(member_path)
        shape, dtype, nbytes, member_sha256 = _tensor_metadata(
            member_path, tensor_key
        )
        return cls(
            index_relative_path=index_relative_path,
            member_relative_path=member_relative_path,
            tensor_key=str(tensor_key),
            shape=shape,
            dtype=dtype,
            nbytes=nbytes,
            index_sha256=observed_index_sha256,
            member_sha256=member_sha256,
            execution_device=torch.device(execution_device),
        )

    @property
    def persistent_tensor_count(self) -> int:
        return 0

    @property
    def persistent_tensor_bytes(self) -> int:
        return 0

    def bind(self, root: str | Path) -> "BoundFileBackedTensor":
        bound = BoundFileBackedTensor(descriptor=self, root=str(Path(root).resolve()))
        bound.verify()
        return bound


@dataclass(slots=True)
class BoundFileBackedTensor:
    descriptor: FileBackedTensorDescriptor
    root: str
    _load_counter: list[int] = field(default_factory=lambda: [0], repr=False)

    @property
    def execution_device(self) -> torch.device:
        return self.descriptor.execution_device

    @property
    def load_count(self) -> int:
        return self._load_counter[0]

    def _paths(self) -> tuple[Path, Path]:
        source_root = Path(self.root).resolve()
        return (
            _confined(source_root, self.descriptor.index_relative_path),
            _confined(source_root, self.descriptor.member_relative_path),
        )

    def verify(self) -> None:
        index_path, member_path = self._paths()
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        if not member_path.is_file():
            raise FileNotFoundError(member_path)
        index_bytes, _ = _verified_bytes(
            index_path,
            self.descriptor.index_sha256,
            identity_name="index",
        )
        member = _index_member(index_bytes, self.descriptor.tensor_key)
        if member != self.descriptor.member_relative_path:
            raise RuntimeError("source index member binding drift")
        with _verified_open(member_path, self.descriptor.member_sha256):
            pass

    def load(self) -> torch.Tensor:
        index_path, member_path = self._paths()
        index_bytes, _ = _verified_bytes(
            index_path,
            self.descriptor.index_sha256,
            identity_name="index",
        )
        member = _index_member(index_bytes, self.descriptor.tensor_key)
        if member != self.descriptor.member_relative_path:
            raise RuntimeError("source index member binding drift")
        with _verified_open(member_path, self.descriptor.member_sha256) as (
            alias,
            _,
            _,
        ):
            with safe_open(alias, framework="pt", device="cpu") as handle:
                if self.descriptor.tensor_key not in handle.keys():
                    raise RuntimeError(
                        f"tensor key is missing from source member: "
                        f"{self.descriptor.tensor_key}"
                    )
                value = handle.get_tensor(self.descriptor.tensor_key)
        observed = (
            tuple(int(item) for item in value.shape),
            str(value.dtype),
            int(value.numel() * value.element_size()),
        )
        expected = (
            self.descriptor.shape,
            self.descriptor.dtype,
            self.descriptor.nbytes,
        )
        if observed != expected:
            raise RuntimeError(
                f"source tensor geometry/dtype/bytes drift: {observed} != {expected}"
            )
        self._load_counter[0] += 1
        return value.to(device=self.execution_device)

    def rebind(
        self, root: str | Path
    ) -> tuple["BoundFileBackedTensor", dict[str, str]]:
        rebound = self.descriptor.bind(root)
        return rebound, {
            "status": "PASS_HASH_BOUND_REBIND",
            "index_relative_path": self.descriptor.index_relative_path,
            "member_relative_path": self.descriptor.member_relative_path,
            "index_sha256": self.descriptor.index_sha256,
            "member_sha256": self.descriptor.member_sha256,
        }


class _FileBackedFrozenLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, source: BoundFileBackedTensor):
        ctx.source = source
        weight = source.load()
        if weight.ndim != 2:
            raise RuntimeError("file-backed frozen linear weight must be rank two")
        if weight.dtype != inputs.dtype:
            weight = weight.to(dtype=inputs.dtype)
        output = torch.nn.functional.linear(inputs, weight)
        del weight
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        weight = ctx.source.load()
        if weight.dtype != grad_output.dtype:
            weight = weight.to(dtype=grad_output.dtype)
        grad_input = torch.matmul(grad_output, weight)
        del weight
        return grad_input, None


class FileBackedFrozenLinear(nn.Module):
    """Frozen linear that lazy-opens immutable weights in forward and backward."""

    def __init__(self, source: BoundFileBackedTensor) -> None:
        super().__init__()
        if len(source.descriptor.shape) != 2:
            raise ValueError("file-backed frozen linear weight must be rank two")
        self.source = source
        self.out_features, self.in_features = source.descriptor.shape
        self.execution_device = source.execution_device

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if int(inputs.shape[-1]) != self.in_features:
            raise RuntimeError(
                f"file-backed frozen linear input drift: {inputs.shape[-1]} != "
                f"{self.in_features}"
            )
        if (
            inputs.device.type != self.execution_device.type
            or self.execution_device.index not in (None, inputs.device.index)
        ):
            raise RuntimeError(
                f"file-backed frozen linear execution-device drift: {inputs.device} "
                f"!= {self.execution_device}"
            )
        return _FileBackedFrozenLinearFn.apply(inputs, self.source)
