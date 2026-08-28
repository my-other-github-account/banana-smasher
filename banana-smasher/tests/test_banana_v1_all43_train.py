import hashlib

import numpy as np

from banana_smasher.banana_v1_all43_train import (
    build_shared_results,
    fit_shared_codebook_from_sources,
)


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
