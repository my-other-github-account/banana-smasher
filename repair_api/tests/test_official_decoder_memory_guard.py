from __future__ import annotations

import pytest

from repair_api.api import ArtifactError
from repair_api.modern_green_resident import _validation_attention_query_chunk_size


def test_official_decoder_dispatch_requires_bounded_eager_query_chunks() -> None:
    with pytest.raises(ArtifactError, match="requires attention_query_chunk_size"):
        _validation_attention_query_chunk_size(
            {"resident_validation_official_decoder_dispatch": True}
        )

    assert _validation_attention_query_chunk_size(
        {
            "resident_validation_official_decoder_dispatch": True,
            "attention_query_chunk_size": 64,
        }
    ) == 64
    assert _validation_attention_query_chunk_size({}) == 0
