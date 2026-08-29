import hashlib

import numpy as np

from banana_smasher.banana_v1_all43_train import (
    _statistics_for_source,
    build_shared_results,
    fit_shared_codebook_from_sources,
)
from banana_smasher.banana_v1 import banana_v1_gaussian_codebook, banana_v1_state_levels


def test_all_members_reassign_against_one_frozen_shared_codebook() -> None:
    sources = [
        np.arange(256, dtype=np.float32).reshape(16, 16) / np.float32(128.0),
        np.arange(255, -1, -1, dtype=np.float32).reshape(16, 16) / np.float32(96.0),
    ]
    codebook, statistics = fit_shared_codebook_from_sources(sources, seeds=[0, 1])
    results = build_shared_results(sources, seeds=[0, 1], codebook=codebook)

    assert codebook.dtype == np.float16
    assert codebook.shape == (1024,)
    assert statistics["assignment_count"] == 512
    assert 0 < statistics["occupied_levels"] <= 1024
    shared_sha = hashlib.sha256(codebook.tobytes()).hexdigest()
    assert len(results) == 2
    assert all(hashlib.sha256(result.codebook.tobytes()).hexdigest() == shared_sha for result in results)
    assert all(result.accounting["code_bpw"] == 2.0 for result in results)


def test_full_projection_statistics_cover_every_position_not_only_corner() -> None:
    source = np.arange(1024, dtype=np.float32).reshape(32, 32) / np.float32(128.0)
    original = banana_v1_gaussian_codebook()
    levels = banana_v1_state_levels()

    full_counts, _full_sums, _ = _statistics_for_source(
        source, seed=3, original=original, state_levels=levels
    )
    corner_counts, _corner_sums, _ = _statistics_for_source(
        source[:16, :16], seed=3, original=original, state_levels=levels
    )

    assert int(full_counts.sum()) == 32 * 32
    assert int(corner_counts.sum()) == 16 * 16
    assert not np.array_equal(full_counts, corner_counts)
