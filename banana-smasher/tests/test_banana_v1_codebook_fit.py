from __future__ import annotations

import numpy as np
import pytest

from banana_smasher import banana_v1
from banana_smasher.banana_v1_train import fit_source_matrix_once


def test_fit_banana_v1_codebook_is_deterministic_and_preserves_fp16() -> None:
    original = banana_v1.banana_v1_gaussian_codebook()
    levels = np.asarray([0, 0, 7, 1023], dtype=np.int64)
    targets = np.asarray([1.0, 3.0, -2.0, 4.0], dtype=np.float64)
    fitted, counts = banana_v1.fit_banana_v1_codebook(
        original, levels, targets, alpha=1.0
    )
    assert fitted.shape == (1024,)
    assert fitted.dtype == np.float16
    assert fitted.flags.c_contiguous
    assert counts.sum() == 4
    assert fitted[0] == np.float16(2.0)
    assert fitted[7] == np.float16(-2.0)
    assert fitted[1023] == np.float16(4.0)
    assert fitted[1] == original[1]
    streamed = banana_v1.fit_banana_v1_codebook_from_statistics(
        original,
        counts,
        np.bincount(levels, weights=targets, minlength=1024),
        alpha=1.0,
    )
    np.testing.assert_array_equal(streamed, fitted)


def test_fit_source_matrix_once_runs_assignment_then_centroid_update() -> None:
    source = np.arange(256, dtype=np.float32).reshape(16, 16) / np.float32(256.0)
    result, evidence = fit_source_matrix_once(source, seed=1701)
    assert result.codebook.shape == (1024,)
    assert result.codebook.dtype == np.float16
    assert evidence["assignment_count"] == 256
    assert evidence["level_count_sum"] == 256
    assert 0 < evidence["occupied_levels"] <= 1024
    assert evidence["fitted_distortion"] <= evidence["initial_distortion"]


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan")])
def test_fit_banana_v1_codebook_rejects_invalid_alpha(alpha: float) -> None:
    original = banana_v1.banana_v1_gaussian_codebook()
    with pytest.raises(ValueError, match="alpha"):
        banana_v1.fit_banana_v1_codebook_from_statistics(
            original,
            np.zeros(1024, dtype=np.int64),
            np.zeros(1024, dtype=np.float64),
            alpha=alpha,
        )
