from __future__ import annotations

import numpy as np
import pytest

from banana_smasher.qtip_periodic import (
    PERIODIC_QTIP25_FORMAT,
    decode_packed,
    decode_symbols,
    pack_symbols,
    periodic_wire_accounting,
    unpack_symbols,
)


def test_periodic_qtip25_symbol_wire_roundtrips_and_decodes_exactly() -> None:
    symbols = np.array([0, 7, 3, 4, 1, 6, 2, 5], dtype=np.uint8)
    packed = pack_symbols(symbols)

    assert PERIODIC_QTIP25_FORMAT == {
        "codec_form": "qtip25_periodic_23",
        "rate_num": 5,
        "rate_den": 2,
        "transition_bits": [2, 3],
        "bit_order": "msb-first",
    }
    assert packed.tolist() == [0x3F, 0x1D, 0x50]
    assert np.array_equal(unpack_symbols(packed, len(symbols)), symbols)

    lut = np.arange(1 << 17, dtype=np.float32).reshape(1 << 16, 2)
    assert np.array_equal(
        decode_packed(packed, len(symbols), lut),
        decode_symbols(symbols, lut),
    )

    accounting = periodic_wire_accounting(
        position_count=len(symbols),
        transform_bytes=20,
        scale_bytes=4,
        shared_tlut_bytes=4096,
    )
    assert accounting == {
        "codec_form": "qtip25_periodic_23",
        "rate_num": 5,
        "rate_den": 2,
        "position_count": 8,
        "code_bits": 20,
        "code_payload_bytes": 3,
        "code_padding_bits": 4,
        "selected_indices_bytes": 0,
        "assignment_map_bytes": 0,
        "routing_bytes": 0,
        "transform_bytes": 20,
        "scale_bytes": 4,
        "auxiliary_bytes": 24,
        "logical_expert_plane_bytes": 27,
        "deduplicated_shared_tlut_bytes": 4096,
        "code_bpw": 2.5,
        "auxiliary_bpw": 24.0,
        "logical_expert_plane_bpw": 27.0,
    }


def test_periodic_qtip25_rejects_unpaired_or_out_of_range_symbols() -> None:
    with pytest.raises(ValueError, match="even position count"):
        pack_symbols(np.array([0], dtype=np.uint8))
    with pytest.raises(ValueError, match="2-bit transition"):
        pack_symbols(np.array([4, 0], dtype=np.uint8))
    with pytest.raises(ValueError, match="3-bit transition"):
        pack_symbols(np.array([0, 8], dtype=np.uint8))
