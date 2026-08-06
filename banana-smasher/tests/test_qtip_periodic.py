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
from banana_smasher.qtip_periodic_provider import periodic_qtip25_provider


def test_periodic_qtip25_symbol_wire_roundtrips_and_decodes_exactly() -> None:
    # QTIP V=2 emits two coded values per transition.  K2/K3 therefore consume
    # four/six branch bits per transition, or 20 bits for eight coded values.
    symbols = np.array([0, 63, 15, 32], dtype=np.uint8)
    packed = pack_symbols(symbols)

    assert PERIODIC_QTIP25_FORMAT == {
        "codec_form": "qtip25_periodic_23",
        "rate_num": 5,
        "rate_den": 2,
        "transition_k": [2, 3],
        "values_per_transition": 2,
        "transition_bits": [4, 6],
        "bit_order": "msb-first",
    }
    assert packed.tolist() == [0x0F, 0xFE, 0x00]
    assert np.array_equal(unpack_symbols(packed, len(symbols)), symbols)

    lut = np.arange(1 << 17, dtype=np.float32).reshape(1 << 16, 2)
    assert np.array_equal(
        decode_packed(packed, len(symbols), lut),
        decode_symbols(symbols, lut),
    )

    accounting = periodic_wire_accounting(
        position_count=len(symbols) * 2,
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
    with pytest.raises(ValueError, match="even transition count"):
        pack_symbols(np.array([0], dtype=np.uint8))
    with pytest.raises(ValueError, match="4-bit transition"):
        pack_symbols(np.array([16, 0], dtype=np.uint8))
    with pytest.raises(ValueError, match="6-bit transition"):
        pack_symbols(np.array([0, 64], dtype=np.uint8))


def test_periodic_provider_generates_prices_materializes_and_verifies(tmp_path) -> None:
    provider = periodic_qtip25_provider()
    symbols = np.array([0, 63, 15, 32], dtype=np.uint8)

    receipt = provider.generate(
        tmp_path / "candidate",
        symbols=symbols,
        intended_basis_sha256="a" * 64,
    )

    assert provider.provider_id == "qtip25-periodic"
    assert provider.kind == "qtip_periodic"
    assert provider.runtime_family == "qtip25_periodic"
    assert receipt["status"] == "PASS"
    assert receipt["codec_form"] == "qtip25_periodic_23"
    assert receipt["rate_num"] == 5
    assert receipt["rate_den"] == 2
    assert receipt["position_count"] == 8
    assert receipt["cell_payload_bytes"] == 3
    assert receipt["assignment_map_bytes"] == 0
    assert receipt["routing_bytes"] == 0
    assert provider.price(tmp_path / "candidate").cell_payload_bytes == 3
    assert provider.verify(tmp_path / "candidate") is True
    materialized = provider.materialize(tmp_path / "candidate")
    assert materialized["runtime_family"] == "qtip25_periodic"
    assert materialized["codes"].tolist() == [0x0F, 0xFE, 0x00]
