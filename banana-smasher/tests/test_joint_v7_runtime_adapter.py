from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


def _runtime_module():
    path = Path(__file__).resolve().parents[2] / "runtime/v7/runner/joint_v7_runtime_adapter.py"
    spec = importlib.util.spec_from_file_location("joint_v7_runtime_adapter_indexer_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Indexer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kv_norm = nn.LayerNorm(4, elementwise_affine=True)


class _Compressor(nn.Module):
    def __init__(self, *, has_indexer: bool) -> None:
        super().__init__()
        self.outer_norm = nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        if has_indexer:
            self.indexer = _Indexer()

    def forward(self, value: torch.Tensor):
        compressed = value * self.outer_norm.reshape(1, 1, 1, -1)
        return compressed, torch.zeros((1, 1, 1, 1), dtype=value.dtype)


def _student() -> SimpleNamespace:
    layers = []
    for layer in range(43):
        compressor = _Compressor(has_indexer=layer in range(2, 43, 2))
        layers.append(SimpleNamespace(self_attn=SimpleNamespace(compressor=compressor)))
    return SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(layers=layers)),
    )


def test_indexer_norm_access_bridge_preserves_forward_and_existing_norm_gradient() -> None:
    runtime = _runtime_module()
    student = _student()
    target = student.model.model.layers[2].self_attn.compressor
    value = torch.tensor([[[[1.0, -2.0, 3.0, -4.0]]]], requires_grad=True)

    baseline, _ = target(value)
    baseline.sum().backward()
    ordinary_control_grad = target.outer_norm.grad.detach().clone()
    target.outer_norm.grad = None
    value.grad = None

    names = runtime._install_indexer_norm_gradient_access(student)
    observed, _ = target(value)
    observed.sum().backward()

    assert names == tuple(
        f"model.layers.{layer}.self_attn.compressor.indexer.kv_norm"
        for layer in range(2, 43, 2)
    )
    assert torch.equal(observed.detach(), baseline.detach())
    assert target.outer_norm.grad is not None
    assert torch.equal(target.outer_norm.grad, ordinary_control_grad)
    indexer_grad = target.indexer.kv_norm.weight.grad
    assert indexer_grad is not None
    assert torch.isfinite(indexer_grad).all()
    assert torch.count_nonzero(indexer_grad).item() > 0
