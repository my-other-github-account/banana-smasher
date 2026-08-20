from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from unittest.mock import Mock

from banana_smasher.contract import PackValidationError
from banana_smasher.hf_deepseek_v4_backpack_adapter import DeepseekV4BackpackRuntime
from banana_smasher.qtip_v7_routes import (
    QTIP_V7_MEMBER_BYTES,
    QtipV7RouteCensus,
    load_qtip2_v7_dense_roster,
    load_qtip2_v7_wire,
)

BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
KINDS = (
    "nas_sftp",
    "ssh",
    "nas_shell",
    "nas_shell_stream",
    "nas_sftp_tranches",
    "local",
    "dense_roster",
    "ssh_transfer_manifest_full_tree",
    "split",
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _route(kind: str, layer: int) -> dict:
    base = {"kind": kind}
    if kind in {"nas_sftp", "ssh", "nas_shell", "local"}:
        base.update({"source": f"/providers/L{layer:03d}", "layout": "flat", "ext": "q2v7wire"})
    elif kind in {"nas_shell_stream", "nas_sftp_tranches"}:
        base.update({"root": f"/providers/L{layer:03d}"})
    elif kind == "ssh_transfer_manifest_full_tree":
        base.update(
            {
                "root": f"/providers/L{layer:03d}",
                "transfer_manifest_path": f"/providers/L{layer:03d}/TRANSFER.json",
                "transfer_manifest_sha256": _sha(f"transfer-{layer}"),
            }
        )
    elif kind == "split":
        base["parts"] = [
            {"root": f"/providers/L{layer:03d}/a", "files": 384, "bytes": 384 * 2_109_444},
            {"root": f"/providers/L{layer:03d}/b", "files": 384, "bytes": 384 * 2_109_444},
        ]
    else:
        base.update({"roster_sha256": _sha(f"roster-{layer}"), "hosts": {"spark-1": "local"}})
    return base


def _document() -> dict:
    layers = []
    for layer in range(43):
        kind = KINDS[layer % len(KINDS)]
        layers.append(
            {
                "layer": layer,
                "producer_task_id": f"producer-{layer}",
                "terminal_sha256": _sha(f"terminal-{layer}"),
                "route": _route(kind, layer),
                "physical_census": {
                    "file_count": 768,
                    "wire_bytes": 768 * 2_109_444,
                    "sentinels": [],
                },
            }
        )
    return {
        "schema": "banana-smasher.qtip2_v7.final_43_route_census.v1",
        "status": "PASS_43_OF_43",
        "basis_sha256": BASIS,
        "frozen_layer_count": 43,
        "frozen_layers": list(range(43)),
        "complete_layers": 43,
        "complete_members": 43 * 768,
        "complete_wire_bytes": 43 * 768 * 2_109_444,
        "gaps": 0,
        "duplicates": 0,
        "fallback_calls": 0,
        "layers": layers,
    }


def test_current_all43_route_census_loads_without_inventing_one_layout(tmp_path: Path) -> None:
    path = tmp_path / "FINAL_43_ROUTE_CENSUS.json"
    path.write_text(json.dumps(_document()))
    census = QtipV7RouteCensus.load(path, expected_basis_sha256=BASIS)
    assert census.layers == tuple(range(43))
    assert census.complete_members == 33_024
    assert {row.kind for row in census.routes} == set(KINDS)
    assert census.route(34).layer == 34
    assert census.route(34).terminal_sha256 == _sha("terminal-34")
    assert census.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_current_route_census_fails_closed_on_basis_and_coverage(tmp_path: Path) -> None:
    value = _document()
    value["layers"].pop()
    path = tmp_path / "FINAL_43_ROUTE_CENSUS.json"
    path.write_text(json.dumps(value))
    with pytest.raises(PackValidationError, match="exactly layers 0..42"):
        QtipV7RouteCensus.load(path, expected_basis_sha256=BASIS)
    value = _document()
    path.write_text(json.dumps(value))
    with pytest.raises(PackValidationError, match="basis"):
        QtipV7RouteCensus.load(path, expected_basis_sha256=_sha("wrong"))


def test_dense_roster_is_explicitly_not_relabelled_as_qtip_wire(tmp_path: Path) -> None:
    path = tmp_path / "FINAL_43_ROUTE_CENSUS.json"
    path.write_text(json.dumps(_document()))
    census = QtipV7RouteCensus.load(path, expected_basis_sha256=BASIS)
    dense = next(row for row in census.routes if row.kind == "dense_roster")
    assert dense.wire_format == "dense_bf16_roster"
    assert dense.member_count == 768
    ordinary = next(row for row in census.routes if row.kind == "nas_sftp")
    assert ordinary.wire_format == "qtip2_v7_fixed_wire"


@pytest.mark.parametrize(
    ("projection", "packed_shape", "su_shape", "sv_shape", "weight_shape"),
    [
        ("w1", (256, 128, 32), (4096,), (2048,), (2048, 4096)),
        ("w2", (128, 256, 32), (2048,), (4096,), (4096, 2048)),
        ("w3", (256, 128, 32), (4096,), (2048,), (2048, 4096)),
    ],
)
def test_raw_v7_member_uses_current_full_row_geometry(
    tmp_path: Path,
    projection: str,
    packed_shape: tuple[int, ...],
    su_shape: tuple[int, ...],
    sv_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
) -> None:
    path = tmp_path / f"E000_{projection}.q2v7wire"
    path.write_bytes(bytes(QTIP_V7_MEMBER_BYTES))
    member = load_qtip2_v7_wire(path, projection=projection)
    assert member["packed"].shape == packed_shape
    assert member["SU"].shape == su_shape
    assert member["SV"].shape == sv_shape
    assert member["weight_shape"] == weight_shape
    assert member["Wscale"].shape == ()


def test_raw_v7_member_rejects_truncation(tmp_path: Path) -> None:
    path = tmp_path / "E000_w1.q2v7wire"
    path.write_bytes(bytes(QTIP_V7_MEMBER_BYTES - 1))
    with pytest.raises(PackValidationError, match="byte geometry"):
        load_qtip2_v7_wire(path, projection="w1")


def test_dense_l034_roster_resolves_declared_selected_wire_paths(tmp_path: Path) -> None:
    members = []
    for expert in range(256):
        for projection in ("w1", "w2", "w3"):
            path = tmp_path / "run" / "staged_wire" / f"E{expert:03d}" / f"{projection}.wire.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
            members.append(
                {
                    "expert": expert,
                    "projection": projection,
                    "path": str(path.relative_to(tmp_path)),
                    "bytes": 1,
                    "sha256": _sha(f"{expert}-{projection}"),
                }
            )
    roster = tmp_path / "L034_SELECTED_WIRE_PROVIDER_ROSTER.json"
    roster.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-qtip2-v7-l034-selected-wire-roster-v1",
                "basis_sha256": BASIS,
                "layer": 34,
                "member_count": 768,
                "members": members,
            }
        )
    )
    resolved = load_qtip2_v7_dense_roster(
        roster,
        expected_basis_sha256=BASIS,
        expected_roster_sha256=hashlib.sha256(roster.read_bytes()).hexdigest(),
        expected_member_bytes=1,
    )
    assert resolved[(0, "w1")][0] == tmp_path / "run/staged_wire/E000/w1.wire.bin"
    assert resolved[(255, "w3")][1] == _sha("255-w3")


def test_dense_l034_roster_rejects_escaping_member_path(tmp_path: Path) -> None:
    roster = tmp_path / "roster.json"
    roster.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-qtip2-v7-l034-selected-wire-roster-v1",
                "basis_sha256": BASIS,
                "layer": 34,
                "member_count": 768,
                "members": [
                    {
                        "expert": expert,
                        "projection": projection,
                        "path": "../escape" if expert == 0 and projection == "w1" else f"E{expert}/{projection}",
                        "bytes": QTIP_V7_MEMBER_BYTES,
                        "sha256": _sha(f"{expert}-{projection}"),
                    }
                    for expert in range(256)
                    for projection in ("w1", "w2", "w3")
                ],
            }
        )
    )
    with pytest.raises(PackValidationError, match="escapes"):
        load_qtip2_v7_dense_roster(
            roster,
            expected_basis_sha256=BASIS,
            expected_roster_sha256=hashlib.sha256(roster.read_bytes()).hexdigest(),
        )


def test_backpack_adapter_maps_v7_wires_to_existing_logical_projections() -> None:
    runtime = object.__new__(DeepseekV4BackpackRuntime)
    runtime.torch = torch
    runtime._decode_qtip2_v7_part = Mock(
        side_effect=[torch.ones(2, 3), torch.full((2, 3), 2.0)]
    )
    fused = runtime._decode_qtip2_v7(4, 7, "fused13")
    assert fused.shape == (4, 3)
    assert fused[:2].eq(1).all() and fused[2:].eq(2).all()
    assert runtime._decode_qtip2_v7_part.call_args_list[0].args == (4, 7, "w1")
    assert runtime._decode_qtip2_v7_part.call_args_list[1].args == (4, 7, "w3")
    runtime._decode_qtip2_v7_part = Mock(return_value=torch.ones(3, 2))
    assert runtime._decode_qtip2_v7(4, 7, "down").shape == (3, 2)
    runtime._decode_qtip2_v7_part.assert_called_once_with(4, 7, "w2")
