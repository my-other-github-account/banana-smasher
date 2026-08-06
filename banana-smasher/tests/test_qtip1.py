from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from banana_smasher.qtip1 import (
    QTIP1_GEOMETRY,
    QTIP2_GEOMETRY,
    QtipGeometry,
    QtipProviderDeclaration,
    QtipWireConsumer,
    _state_lut,
    assign_qtip_provider_components,
    decode_qtip,
    encode_qtip,
    gaussian_tlut,
    pack_qtip_states,
    qtip1_5_provider_declaration,
    qtip1_provider_declaration,
    qtip_provider_counts,
    unpack_qtip_states,
    verify_qtip_wire,
    write_qtip_wire,
)


def test_canonical_k1v1_recognizable_matrix_encode_pack_decode_parity() -> None:
    """Fixture generated from Cornell-RelaxML/qtip e90c668 bitshift.py."""
    geometry = QtipGeometry(L=2, K=1, V=1, tlut_bits=2, decode_mode="lut")
    tlut = np.asarray(
        [[-1.0, 99.0], [-0.5, 99.0], [0.5, 99.0], [1.0, 99.0]],
        dtype=np.float32,
    )
    matrix = np.asarray(
        [
            [-1.0, -0.8, -0.4, 0.2, 0.7, 1.0, 0.4, -0.7],
            [1.0, 0.8, 0.4, -0.2, -0.7, -1.0, -0.4, 0.7],
        ],
        dtype=np.float32,
    )
    encoded = encode_qtip(
        matrix,
        geometry=geometry,
        tlut=tlut,
        scales=np.ones(2, dtype=np.float32),
    )

    expected_states = np.asarray(
        [
            [0, 0, 1, 3, 3, 3, 2, 0],
            [3, 3, 2, 0, 0, 0, 1, 3],
        ],
        dtype=np.int32,
    )
    expected_decoded = np.asarray(
        [
            [-1.0, -1.0, -0.5, 1.0, 1.0, 1.0, 0.5, -1.0],
            [1.0, 1.0, 0.5, -1.0, -1.0, -1.0, -0.5, 1.0],
        ],
        dtype=np.float32,
    )
    assert np.array_equal(encoded.states, expected_states)
    assert encoded.packed.dtype == np.uint16
    assert encoded.packed.tolist() == [[7680], [57600]]
    assert np.array_equal(
        unpack_qtip_states(encoded.packed, steps=8, geometry=geometry),
        expected_states,
    )
    assert np.array_equal(decode_qtip(encoded, tlut=tlut), expected_decoded)


def test_qtip1_l16_roundtrip_shape_and_exact_code_byte_accounting() -> None:
    matrix = np.linspace(-2.0, 2.0, 64, dtype=np.float32).reshape(2, 32)
    tlut = gaussian_tlut(bits=9, columns=2)
    encoded = encode_qtip(matrix, geometry=QTIP1_GEOMETRY, tlut=tlut)
    decoded = decode_qtip(encoded, tlut=tlut)

    assert encoded.states.shape == (2, 32)
    assert encoded.packed.shape == (2, 2)
    assert encoded.packed.nbytes == 8
    assert encoded.scales.shape == (2,)
    assert encoded.scales.nbytes == 8
    assert encoded.code_bpw == 1.0
    assert decoded.shape == matrix.shape
    assert np.isfinite(decoded).all()
    assert np.array_equal(
        unpack_qtip_states(encoded.packed, steps=32, geometry=QTIP1_GEOMETRY),
        encoded.states,
    )


def test_qtip1_l16_matches_pinned_public_canonical_source_fixture() -> None:
    """Exact fixture from qtip bitshift.py at e90c6688c8dfae326a3a81b5eb032db7c6680ec0."""
    matrix = np.linspace(-2.0, 2.0, 32, dtype=np.float32).reshape(1, 32)
    tlut = gaussian_tlut(bits=9, columns=2)
    encoded = encode_qtip(
        matrix,
        geometry=QTIP1_GEOMETRY,
        tlut=tlut,
        scales=np.ones(1, dtype=np.float32),
    )
    decoded = decode_qtip(encoded, tlut=tlut)

    assert encoded.packed.tolist() == [[50608, 48565]]
    assert hashlib.sha256(encoded.states.tobytes()).hexdigest() == (
        "2c6b6f7def6a0a2b1f013955bfb157f9d930947c9f95bace99ce636e0551ffaf"
    )
    assert hashlib.sha256(encoded.packed.tobytes()).hexdigest() == (
        "8900941f18dd7c64618669e63e765f93be195235d9f55b23a673f4d6817eb408"
    )
    assert hashlib.sha256(decoded.tobytes()).hexdigest() == (
        "fdf4b61c76eed55d8cd45de050021c93b38267849d940e8aff0480855d37a668"
    )


def test_reduced_geometry_uses_l_relative_hash_and_requires_cyclic_paths() -> None:
    geometry = QtipGeometry(L=4, K=2, V=1, tlut_bits=2, decode_mode="quantlut")
    tlut = np.asarray([[-1.0], [-0.25], [0.25], [1.0]], dtype=np.float32)
    encoded = encode_qtip(
        np.asarray([[-0.5, 0.5]], dtype=np.float32),
        geometry=geometry,
        tlut=tlut,
        scales=np.ones(1, dtype=np.float32),
    )

    assert encoded.states.shape == (1, 2)
    assert np.array_equal(
        unpack_qtip_states(encoded.packed, steps=2, geometry=geometry),
        encoded.states,
    )
    noncyclic = np.asarray([[0b0000, 0b0001, 0b0010, 0b0100]], dtype=np.int32)
    try:
        pack_qtip_states(
            noncyclic,
            QtipGeometry(L=4, K=1, V=1, tlut_bits=4, decode_mode="lut"),
        )
    except ValueError as exc:
        assert "does not close" in str(exc)
    else:
        raise AssertionError("non-cyclic QTIP path was accepted")


def test_qtip1_minimal_cycle_roundtrips_and_noncycle_is_refused() -> None:
    geometry = QtipGeometry(L=2, K=1, V=1, tlut_bits=2, decode_mode="lut")
    cyclic = np.asarray([[0, 0]], dtype=np.int32)
    packed = pack_qtip_states(cyclic, geometry)
    assert np.array_equal(
        unpack_qtip_states(packed, steps=2, geometry=geometry), cyclic
    )

    try:
        pack_qtip_states(np.asarray([[0, 1]], dtype=np.int32), geometry)
    except ValueError as exc:
        assert "does not close" in str(exc)
    else:
        raise AssertionError("non-cyclic two-step K1 path was accepted")


def test_reduced_geometry_quantlut_uses_canonical_16bit_hash_shift() -> None:
    geometry = QtipGeometry(L=4, K=1, V=1, tlut_bits=2, decode_mode="quantlut")
    tlut = np.asarray([[-3.0], [-1.0], [1.0], [3.0]], dtype=np.float32)
    state_lut = _state_lut(geometry, tlut)

    # Cornell QTIP quantlut always hashes through the 16-bit lane, even when L is
    # reduced for a source-parity fixture. States 0..15 therefore all select row 0.
    assert np.array_equal(state_lut, np.full((1, 16), -3.0, dtype=np.float32))


def test_qtip1_and_qtip15_declarations_parse_and_report_exact_counts() -> None:
    qtip1 = QtipProviderDeclaration.from_mapping(qtip1_provider_declaration().as_mapping())
    qtip15 = QtipProviderDeclaration.from_mapping(
        qtip1_5_provider_declaration().as_mapping()
    )
    identities = [
        (0, expert, projection)
        for projection in ("fused13", "down")
        for expert in range(256)
    ]

    assert qtip1.tier == "qtip@1.00"
    assert [row.geometry for row in qtip1.components] == [QTIP1_GEOMETRY]
    assert qtip_provider_counts(qtip1, identities) == {"qtip1-k1v1": 512}
    assert qtip15.tier == "qtip@1.50"
    assert qtip_provider_counts(qtip15, identities) == {
        "qtip1-k1v1": 256,
        "qtip2-k2v2": 256,
    }
    assigned = assign_qtip_provider_components(qtip15, reversed(identities))
    for projection in ("fused13", "down"):
        assert Counter(
            assigned[(0, expert, projection)].geometry for expert in range(256)
        ) == {QTIP1_GEOMETRY: 128, QTIP2_GEOMETRY: 128}
    try:
        assign_qtip_provider_components(qtip15, [(0, 0, "../escape")])
    except ValueError as exc:
        assert "invalid QTIP provider identity" in str(exc)
    else:
        raise AssertionError("unsafe QTIP projection was accepted")
    malformed = qtip15.as_mapping()
    malformed["schema"] = "not-qtip"
    try:
        QtipProviderDeclaration.from_mapping(malformed)
    except ValueError as exc:
        assert "unsupported QTIP provider schema" in str(exc)
    else:
        raise AssertionError("malformed QTIP declaration was accepted")


def test_qtip15_wire_has_exact_indices_scales_one_shared_tlut_and_reads_both_formats(
    tmp_path: Path,
) -> None:
    declaration = qtip1_5_provider_declaration()
    identities = [(0, 0, "down"), (0, 1, "down")]
    assignments = assign_qtip_provider_components(declaration, identities)
    matrices = {
        identities[0]: np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(2, 32),
        identities[1]: np.linspace(1.0, -1.0, 64, dtype=np.float32).reshape(2, 32),
    }
    tlut = gaussian_tlut(bits=9, columns=2)
    root = tmp_path / "wire"
    receipt = write_qtip_wire(
        root,
        declaration=declaration,
        matrices=matrices,
        tlut=tlut,
    )

    accounting = receipt["accounting"]
    assert accounting["weights"] == 128
    assert accounting["index_data_bytes"] == 24
    assert accounting["scale_data_bytes"] == 16
    assert accounting["shared_tlut_data_bytes"] == tlut.nbytes
    assert accounting["wire_data_bytes"] == 24 + 16 + tlut.nbytes
    assert accounting["code_bpw"] == 1.5
    members = receipt["members"]
    assert {row["tlut"]["path"] for row in members} == {"shared_tlut.npy"}
    assert {row["tlut"]["sha256"] for row in members} == {
        receipt["shared_tlut"]["sha256"]
    }
    assert sum(row["trellis"]["data_bytes"] for row in members) == 24
    assert sum(row["scales"]["data_bytes"] for row in members) == 16

    consumer = QtipWireConsumer(root)
    assert consumer.counts == {"qtip1-k1v1": 1, "qtip2-k2v2": 1}
    assert {assignments[identity].geometry for identity in identities} == {
        QTIP1_GEOMETRY,
        QTIP2_GEOMETRY,
    }
    for identity in identities:
        component = assignments[identity]
        direct = encode_qtip(
            matrices[identity], geometry=component.geometry, tlut=tlut
        )
        expected = decode_qtip(direct, tlut=tlut)
        observed = consumer.decode(identity)
        assert observed.shape == matrices[identity].shape
        assert np.array_equal(observed, expected)

    wire_path = root / "QTIP_WIRE.json"
    malformed_wire = json.loads(wire_path.read_text())
    malformed_wire["members"][0]["shape"][0] += 1
    wire_path.write_text(json.dumps(malformed_wire))
    malformed_consumer = QtipWireConsumer(root)
    try:
        malformed_consumer.decode(identities[0])
    except ValueError as exc:
        assert "row/scale shape drift" in str(exc)
    else:
        raise AssertionError("malformed QTIP wire member was accepted")


def test_qtip_wire_write_is_transactional_for_invalid_empty_member(tmp_path: Path) -> None:
    root = tmp_path / "wire"
    try:
        write_qtip_wire(
            root,
            declaration=qtip1_provider_declaration(),
            matrices={(0, 0, "down"): np.empty((0, 32), dtype=np.float32)},
            tlut=gaussian_tlut(bits=9, columns=2),
        )
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("empty QTIP wire member was accepted")
    assert not root.exists()


def test_qtip_wire_verify_rejects_fail_status_and_tampered_scales(tmp_path: Path) -> None:
    declaration = qtip1_provider_declaration()
    matrix = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(1, 32)
    tlut = gaussian_tlut(bits=9, columns=2)

    failed_root = tmp_path / "failed-wire"
    write_qtip_wire(
        failed_root,
        declaration=declaration,
        matrices={(0, 0, "down"): matrix},
        tlut=tlut,
    )
    failed_receipt_path = failed_root / "QTIP_WIRE.json"
    failed_receipt = json.loads(failed_receipt_path.read_text())
    failed_receipt["status"] = "FAIL"
    failed_receipt_path.write_text(json.dumps(failed_receipt))
    try:
        QtipWireConsumer(failed_root)
    except ValueError as exc:
        assert "status" in str(exc)
    else:
        raise AssertionError("FAIL QTIP wire receipt was accepted")

    tampered_root = tmp_path / "tampered-wire"
    write_qtip_wire(
        tampered_root,
        declaration=declaration,
        matrices={(0, 0, "down"): matrix},
        tlut=tlut,
    )
    receipt_path = tampered_root / "QTIP_WIRE.json"
    receipt = json.loads(receipt_path.read_text())
    scale_row = receipt["members"][0]["scales"]
    scale_path = tampered_root / scale_row["path"]
    scales = np.load(scale_path, allow_pickle=False)
    scales[0] = np.nan
    np.save(scale_path, scales, allow_pickle=False)
    scale_row["bytes"] = scale_path.stat().st_size
    scale_row["data_bytes"] = scales.nbytes
    scale_row["sha256"] = hashlib.sha256(scale_path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    try:
        QtipWireConsumer(tampered_root).decode((0, 0, "down"))
    except ValueError as exc:
        assert "finite and positive" in str(exc)
    else:
        raise AssertionError("tampered non-finite QTIP scales were accepted")

    verified = verify_qtip_wire(tampered_root)
    assert verified["status"] == "FAIL"
