from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

from repair_api import modern_green_resident as resident


WRAPPER_SHA256 = "0d4ece20b602fc59ffef349183db2bea0861b4a7f7c0ef93e50fd728310e7371"
EXPERT_SHA256 = "fc612f7863ad9d09a9faf11e203a9d20739b7dbb273b982fc3d36ee01d15a9b4"


def _install_independent_codec_fixture() -> None:
    package = ModuleType("banana_smasher")
    package.__path__ = []  # type: ignore[attr-defined]
    codec = ModuleType("banana_smasher.q2_codec")
    qtip = ModuleType("banana_smasher.qtip_k2")

    def tensor_core_permutation() -> np.ndarray:
        permutation = np.empty(256, dtype=np.int32)
        for thread in range(32):
            rows = (
                (thread % 4) * 2,
                (thread % 4) * 2 + 1,
                (thread % 4) * 2 + 8,
                (thread % 4) * 2 + 9,
            )
            column = thread // 4
            permutation[thread * 8 : thread * 8 + 8] = tuple(
                row * 16 + offset for offset in (column, column + 8) for row in rows
            )
        return permutation

    def normalized_hadamard_128(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        matrix = torch.ones((1, 1), device=device, dtype=torch.float64)
        kernel = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device, dtype=torch.float64)
        for _ in range(7):
            matrix = torch.kron(matrix, kernel)
        return (matrix / (128.0**0.5)).to(dtype=dtype)

    codec.tensor_core_permutation = tensor_core_permutation  # type: ignore[attr-defined]
    qtip.normalized_hadamard_128 = normalized_hadamard_128  # type: ignore[attr-defined]
    sys.modules["banana_smasher"] = package
    sys.modules["banana_smasher.q2_codec"] = codec
    sys.modules["banana_smasher.qtip_k2"] = qtip


def test_historical_score19_pair_loads_and_matches_known_dense_projection() -> None:
    _install_independent_codec_fixture()
    assets = Path(resident.__file__).resolve().parent / "assets"
    wrapper_path = assets / "static_w28_fast_k2_grouped.py"
    expert_path = assets / "static_w28_fast_v7_expert_base.py"

    assert hashlib.sha256(wrapper_path.read_bytes()).hexdigest() == WRAPPER_SHA256
    assert hashlib.sha256(expert_path.read_bytes()).hexdigest() == EXPERT_SHA256
    assert resident.STATIC_W28_GROUPED_WRAPPER_SHA256 == WRAPPER_SHA256
    assert resident.STATIC_W28_GROUPED_EXPERT_SHA256 == EXPERT_SHA256

    wrapper = resident._load_source_module("fast_k2_grouped", wrapper_path)
    expert = resident._load_source_module("fast_v7_expert_base", expert_path)
    assert expert.grouped_packed_projection is wrapper.grouped_packed_projection

    generator = torch.Generator(device="cpu").manual_seed(19)
    x = torch.randn((3, 128), generator=generator, dtype=torch.float32)
    assignments = torch.tensor([1, 0, 1], dtype=torch.int64)
    packed = torch.randint(
        -(2**15), 2**15, (2, 8, 8, 32), generator=generator, dtype=torch.int16
    )
    lut_master = torch.linspace(-1.0, 1.0, 1024, dtype=torch.float32)
    su = torch.randn((2, 128), generator=generator, dtype=torch.float32)
    sv = torch.randn((2, 128), generator=generator, dtype=torch.float32)

    observed = wrapper.grouped_packed_projection_reference(
        x, assignments, packed, lut_master, su, sv
    )
    transformed = wrapper.block_hadamard_128(x * su[assignments])
    wire_lut = lut_master.to(torch.float16)
    dense_rows = torch.stack(
        [
            row @ wrapper.direct_decode_matrix(packed[int(which)], wire_lut)
            for row, which in zip(transformed, assignments)
        ]
    )
    expected = wrapper.block_hadamard_128(dense_rows) * sv[assignments]

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
