from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import banana_smasher.file_backed as file_backed
from banana_smasher.file_backed import (
    FileBackedFrozenLinear,
    FileBackedTensorDescriptor,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_tree(root: Path) -> tuple[Path, torch.Tensor]:
    root.mkdir(parents=True)
    weight = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 2.0, 1.0]], dtype=torch.float32
    )
    member = root / "model-00001.safetensors"
    save_file({"layers.0.attn.proj.weight": weight}, member)
    index = root / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "layers.0.attn.proj.weight": member.name,
                }
            },
            sort_keys=True,
        )
        + "\n"
    )
    return index, weight


def test_descriptor_is_relative_hash_bound_and_has_zero_persistent_tensors(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    index, weight = _source_tree(root)

    descriptor = FileBackedTensorDescriptor.from_index(
        root=root,
        index_relative_path=index.name,
        expected_index_sha256=_sha256(index),
        tensor_key="layers.0.attn.proj.weight",
        execution_device="cpu",
    )

    assert descriptor.index_relative_path == "model.safetensors.index.json"
    assert descriptor.member_relative_path == "model-00001.safetensors"
    assert descriptor.tensor_key == "layers.0.attn.proj.weight"
    assert descriptor.shape == (2, 4)
    assert descriptor.dtype == "torch.float32"
    assert descriptor.nbytes == weight.numel() * weight.element_size()
    assert descriptor.index_sha256 == _sha256(index)
    assert descriptor.member_sha256 == _sha256(root / descriptor.member_relative_path)
    assert descriptor.execution_device == torch.device("cpu")
    assert descriptor.persistent_tensor_count == 0
    assert descriptor.persistent_tensor_bytes == 0
    assert not any(isinstance(value, torch.Tensor) for value in dataclasses.astuple(descriptor))


def test_hash_bound_rebind_relocates_and_old_binding_fails(tmp_path: Path) -> None:
    first = tmp_path / "first"
    index, weight = _source_tree(first)
    descriptor = FileBackedTensorDescriptor.from_index(
        root=first,
        index_relative_path=index.name,
        expected_index_sha256=_sha256(index),
        tensor_key="layers.0.attn.proj.weight",
        execution_device="cpu",
    )
    bound = descriptor.bind(first)

    second = tmp_path / "relocated"
    shutil.copytree(first, second)
    relocated, receipt = bound.rebind(second)
    shutil.rmtree(first)

    torch.testing.assert_close(relocated.load(), weight)
    assert receipt == {
        "status": "PASS_HASH_BOUND_REBIND",
        "index_relative_path": descriptor.index_relative_path,
        "member_relative_path": descriptor.member_relative_path,
        "index_sha256": descriptor.index_sha256,
        "member_sha256": descriptor.member_sha256,
    }
    with pytest.raises((FileNotFoundError, RuntimeError)):
        bound.load()


def test_rebind_refuses_member_substitution_even_when_index_is_unchanged(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    index, _ = _source_tree(source)
    descriptor = FileBackedTensorDescriptor.from_index(
        root=source,
        index_relative_path=index.name,
        expected_index_sha256=_sha256(index),
        tensor_key="layers.0.attn.proj.weight",
        execution_device="cpu",
    )
    relocated = tmp_path / "relocated"
    shutil.copytree(source, relocated)
    (relocated / descriptor.member_relative_path).write_bytes(b"substituted")

    with pytest.raises(RuntimeError, match="member identity drift"):
        descriptor.bind(relocated)


def test_frozen_linear_reopens_for_forward_and_backward_without_storage(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    index, weight = _source_tree(source)
    descriptor = FileBackedTensorDescriptor.from_index(
        root=source,
        index_relative_path=index.name,
        expected_index_sha256=_sha256(index),
        tensor_key="layers.0.attn.proj.weight",
        execution_device="cpu",
    )
    bound = descriptor.bind(source)
    module = FileBackedFrozenLinear(bound)

    assert list(module.parameters()) == []
    assert list(module.buffers()) == []
    assert not any(isinstance(value, torch.Tensor) for value in vars(module).values())
    inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    output = module(inputs)
    expected = torch.nn.functional.linear(inputs, weight)
    torch.testing.assert_close(output, expected)
    assert tuple(output.grad_fn.saved_tensors) == ()
    output.sum().backward()
    torch.testing.assert_close(inputs.grad, torch.ones_like(output) @ weight)
    assert bound.load_count == 2
    assert module.execution_device == torch.device("cpu")


def test_load_uses_the_verified_open_member_not_a_replaced_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    index, weight = _source_tree(source)
    descriptor = FileBackedTensorDescriptor.from_index(
        root=source,
        index_relative_path=index.name,
        expected_index_sha256=_sha256(index),
        tensor_key="layers.0.attn.proj.weight",
        execution_device="cpu",
    )
    bound = descriptor.bind(source)
    member = source / descriptor.member_relative_path
    replacement = source / "replacement.safetensors"
    save_file({descriptor.tensor_key: torch.full_like(weight, 99)}, replacement)
    original_safe_open = file_backed.safe_open
    swapped = False

    def swap_after_verified_open(path, *args, **kwargs):
        nonlocal swapped
        if not swapped:
            assert str(path).startswith(("/dev/fd/", "/proc/self/fd/"))
            os.replace(replacement, member)
            swapped = True
        return original_safe_open(path, *args, **kwargs)

    monkeypatch.setattr(file_backed, "safe_open", swap_after_verified_open)
    torch.testing.assert_close(bound.load(), weight)
