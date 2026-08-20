from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import banana_smasher.qtip_v7_wire as wire_module
from banana_smasher.cli import main
from banana_smasher.qtip_v7_wire import (
    QtipV7LayerMapping,
    WireGeometry,
    account_qtip_v7_model,
    pack_qtip_v7_layer,
    verify_qtip_v7_layer,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixed_envelope_round_trip_reconstructs_exact_order(tmp_path: Path) -> None:
    geometry = WireGeometry(
        experts=2,
        projections=("w1", "w2", "w3"),
        packed_bytes=32,
        control_bytes=512,
        lut_bytes=16,
        header_bytes=2048,
        envelope_bytes=3264,
    )
    source = tmp_path / "members"
    source.mkdir()
    originals: dict[str, bytes] = {}
    for expert in range(geometry.experts):
        for projection_index, projection in enumerate(geometry.projections):
            name = f"E{expert:03d}_{projection}.q2v7wire"
            payload = bytes([expert + projection_index + 1]) * geometry.packed_bytes
            payload += bytes([projection_index]) * geometry.control_bytes
            (source / name).write_bytes(payload)
            originals[name] = payload
    lut = tmp_path / "L037.tlut.f16"
    lut.write_bytes(bytes(range(geometry.lut_bytes)))
    wire = tmp_path / "L037.qtip-v7-wire"
    receipt = tmp_path / "L037.qtip-v7-wire.receipt.json"

    packed = pack_qtip_v7_layer(
        source_root=source,
        lut=lut,
        layer=37,
        output=wire,
        receipt=receipt,
        _geometry=geometry,
    )

    assert wire.stat().st_size == geometry.envelope_bytes
    assert packed["physical_bytes"] == geometry.envelope_bytes
    assert packed["wire_size_delta"] == 0
    assert packed["lut_sha256"] == _sha(lut)
    reconstructed = tmp_path / "reconstructed"
    verified = verify_qtip_v7_layer(
        wire=wire,
        receipt=receipt,
        reconstructed_output=reconstructed,
        _geometry=geometry,
    )
    assert verified["status"] == "PASS"
    assert verified["reconstructed_stream_authenticated"] is True
    assert verified["embedded_lut_authenticated"] is True
    assert verified["roster_authenticated"] is True
    assert verified["physical_bytes_authenticated"] is True
    assert verified["roster"] == list(originals)
    assert sorted(path.name for path in reconstructed.iterdir()) == sorted(
        [*originals, "embedded_lut.f16"]
    )
    assert all((reconstructed / name).read_bytes() == payload for name, payload in originals.items())
    assert (reconstructed / "embedded_lut.f16").read_bytes() == lut.read_bytes()

    with QtipV7LayerMapping(wire, _geometry=geometry) as mapping:
        packed_view = mapping.packed_view(1, "w2")
        lut_view = mapping.lut_view()
        assert packed_view.obj is mapping.buffer
        assert lut_view.obj is mapping.buffer
        assert bytes(packed_view) == originals["E001_w2.q2v7wire"][: geometry.packed_bytes]
        assert bytes(lut_view) == lut.read_bytes()
        assert mapping.transient_controls() == b"".join(
            payload[geometry.packed_bytes :] for payload in originals.values()
        )
        assert mapping.transient_workspace_peak_bytes == (
            geometry.member_count * geometry.control_bytes
        )
        packed_view.release()
        lut_view.release()


def test_public_cli_exposes_wire_one_liners(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["qtip-v7-wire", "--help"])
    assert raised.value.code == 0
    parent_help = capsys.readouterr().out
    for command in ("pack-layer", "verify-layer", "account-model"):
        assert command in parent_help

    with pytest.raises(SystemExit) as raised:
        main(["qtip-v7-residency", "--help"])
    assert raised.value.code == 0
    residency_help = capsys.readouterr().out
    assert "--accounting" in residency_help
    assert "--hardware-readback" in residency_help
    assert "--capture-hardware" in residency_help

    with pytest.raises(SystemExit) as raised:
        main(["qtip-v7-layer-smoke", "--help"])
    assert raised.value.code == 0
    layer_help = capsys.readouterr().out
    assert "--wire" in layer_help
    assert "--output" in layer_help

    # Parser coverage stays cheap; the physical implementation is exercised above.
    with pytest.raises(SystemExit) as raised:
        main(["qtip-v7-wire", "pack-layer", "--help"])
    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    assert "--source-root" in help_text
    assert "--lut" in help_text


def test_model_accounting_is_derived_from_exact_43_verified_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts = []
    for layer in range(43):
        path = tmp_path / f"L{layer:03d}.receipt.json"
        path.write_text(json.dumps({
            "schema": "banana-smasher-qtip-v7-layer-wire-receipt-v1",
            "status": "PASS",
            "layer": layer,
            "member_count": 768,
            "physical_bytes": 1_620_052_992,
            "wire_size_delta": 0,
            "reconstructed_stream_authenticated": True,
            "embedded_lut_authenticated": True,
            "roster_authenticated": True,
            "physical_bytes_authenticated": True,
            "wire_sha256": f"{layer + 1:064x}",
            "wire": str(tmp_path / f"L{layer:03d}.qtip-v7-wire"),
        }, sort_keys=True) + "\n")
        receipts.append(path)
    output = tmp_path / "QTIP_V7_MODEL_ACCOUNTING.json"

    def verified_readback(*, wire: str | Path, receipt: str | Path) -> dict[str, object]:
        row = json.loads(Path(receipt).read_text())
        assert wire == row["wire"]
        return row

    monkeypatch.setattr(wire_module, "verify_qtip_v7_layer", verified_readback)

    result = account_qtip_v7_model(
        receipts=receipts,
        output=output,
        weight_denominator=671_000_000_000,
        weight_denominator_label="declared total model weight parameters",
    )

    assert result["status"] == "PASS"
    assert result["verified_layer_receipts"] == 43
    assert result["qtip_routed_stored_bytes"] == 69_662_278_656
    assert result["exl_k2_routed_stored_bytes"] == 69_662_278_656
    assert result["routed_gap_bytes"] == 0
    assert result["native_base_bytes"] == 19_708_797_688
    assert result["qtip_full_stored_bytes"] == 89_371_076_344
    assert result["exl_full_stored_bytes"] == 89_371_076_344
    assert result["full_gap_bytes"] == 0
    assert result["stored_wire_bpw"]["numerator_bits"] == 89_371_076_344 * 8
    assert result["stored_wire_bpw"]["weight_denominator"] == 671_000_000_000
    assert result["stored_wire_bpw"]["meaning"] == "stored wire bits per declared weight"
    assert result["decoded_dtype_claim"] is None
    assert json.loads(output.read_text()) == result


def test_public_wire_surface_contains_no_obsolete_stale_total() -> None:
    root = Path(__file__).parents[1]
    surfaces = [
        root / "src/banana_smasher/qtip_v7_wire.py",
        root / "src/banana_smasher/qtip_v7_repair.py",
        root / "src/banana_smasher/qtip_v7_joint_workflow.py",
        root / "src/banana_smasher/qtip_v7_residency.py",
        root / "src/banana_smasher/cli.py",
        root.parent / "archive/notes/qtip-v7-joint-repair-one-line-workflow.md",
    ]
    stale = "100" + ",636,011,256"
    assert all(stale not in path.read_text() for path in surfaces)
