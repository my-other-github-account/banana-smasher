from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from banana_smasher.qtip1 import gaussian_tlut
from banana_smasher.qtip25_native_v4_cuda_cell import run_cuda_cell, validate_input


def test_native_v4_cuda_cell_preflight_binds_exact_basis_and_geometry(tmp_path) -> None:
    target_path = tmp_path / "target.npy"
    tlut_path = tmp_path / "tlut.npy"
    np.save(target_path, np.zeros((3, 64, 4), dtype=np.float32), allow_pickle=False)
    np.save(tlut_path, gaussian_tlut(bits=9, columns=2), allow_pickle=False)

    target, tlut, identity = validate_input(
        target_path,
        tlut_path,
        intended_basis_sha256="9" * 64,
        observed_basis_sha256="9" * 64,
    )
    assert target.shape == (3, 64, 4)
    assert tlut.shape == (512, 2)
    assert identity["basis_sha256"] == "9" * 64
    with pytest.raises(ValueError, match="basis mismatch"):
        validate_input(
            target_path,
            tlut_path,
            intended_basis_sha256="9" * 64,
            observed_basis_sha256="8" * 64,
        )


def test_native_v4_cuda_cell_passes_geometry_to_public_decoder() -> None:
    tree = ast.parse(inspect.getsource(run_cuda_cell))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dequantize_native_v4_blocks"
    ]
    assert len(calls) == 2
    assert all(
        len(call.args) == 2
        and [keyword.arg for keyword in call.keywords] == ["bpw"]
        for call in calls
    )

def test_native_v4_cuda_cell_releases_allocator_cache_before_free_memory_gate() -> None:
    source = inspect.getsource(run_cuda_cell)

    assert "torch.cuda.empty_cache()" in source
    assert source.index("torch.cuda.empty_cache()") < source.index("torch.cuda.mem_get_info()")
