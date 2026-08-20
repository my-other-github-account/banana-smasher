from __future__ import annotations

import numpy as np
import pytest

from banana_smasher.qtip1 import gaussian_tlut
from banana_smasher.qtip25_native_v4_cuda_cell import validate_input


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