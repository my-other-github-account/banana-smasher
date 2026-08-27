from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import torch


ROOT = Path(__file__).parents[2]
GROUPED_SOURCE = ROOT / "control-receipts" / "t_7c13ff03" / "diagnosis" / "fast_k2_grouped.py"
CUDA_SOURCE = ROOT / "fast_k2_grouped_kernel.cu"


def _normalized_hadamard_128(device, dtype):
    matrix = torch.tensor([[1.0]])
    while matrix.shape[0] < 128:
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    return (matrix / (128**0.5)).to(device=device, dtype=dtype)


def _load_grouped_module():
    codec = types.ModuleType("banana_smasher.q2_codec")
    codec.tensor_core_permutation = lambda: list(range(256))
    qtip = types.ModuleType("banana_smasher.qtip_k2")
    qtip.normalized_hadamard_128 = _normalized_hadamard_128
    package = types.ModuleType("banana_smasher")
    package.__path__ = []
    sys.modules["banana_smasher"] = package
    sys.modules["banana_smasher.q2_codec"] = codec
    sys.modules["banana_smasher.qtip_k2"] = qtip
    spec = importlib.util.spec_from_file_location("fast_k2_grouped_tile_bf16_test", GROUPED_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_post_inverse_transform_weight_rounds_to_bf16_before_gemm():
    grouped = _load_grouped_module()
    packed = torch.zeros((1, 8, 8, 32), dtype=torch.int16)
    lut = torch.zeros(1024, dtype=torch.float32)
    lut[0] = 1.0039
    su = torch.linspace(0.75, 1.25, 128).reshape(1, 128).to(torch.float16)
    sv = torch.linspace(1.25, 0.75, 128).reshape(1, 128).to(torch.float16)

    observed = grouped.sealed_bf16_weight_slab(packed, lut, su, sv, 0)
    q = grouped.direct_decode_matrix(packed[0], lut)
    hadamard = _normalized_hadamard_128(torch.device("cpu"), torch.float32)
    expected = (
        su[0].float().unsqueeze(1)
        * (hadamard @ q @ hadamard)
        * sv[0].float().unsqueeze(0)
    ).to(torch.bfloat16)
    assert torch.equal(observed[0], expected)

    x = (torch.arange(128, dtype=torch.float32) / 128).reshape(1, 128).to(torch.bfloat16)
    assert torch.equal(x @ observed[0], x @ expected)


def test_dequant_lut_precision_remains_fp16_until_full_weight_boundary():
    source = GROUPED_SOURCE.read_text()
    assert "wire_lut = lut if lut.dtype == torch.float16 else lut.to(torch.float16)" in source
    assert "sealed_bf16_weight_slab" in source
